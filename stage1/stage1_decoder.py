"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Lite version: stripped of blink prediction head, CDN denoising, and @register dependency.
Adds transform_expansion parameter for lightweight eye branch.
"""

import math
import copy
import functools
from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List
from torchvision.ops import roi_align

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.zoo.rtdetr.utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from src.zoo.rtdetr.utils import bias_init_with_prob


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention"""
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    """Lite version: removed pred_blink, added transform_expansion parameter."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1,
                 num_heads=8, transform_expansion=4):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

        attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.temproal_attention = nn.ModuleList([copy.deepcopy(attention_layer) for _ in range(num_layers)])
        self.temproal_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # transform_expansion controls the eye branch MLP size (default 4 = original, 2 = lite)
        self.transform_eye = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, transform_expansion * hidden_dim),
                nn.LayerNorm(transform_expansion * hidden_dim),
                nn.Linear(transform_expansion * hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def spatial_temproal_attention(self, tgt, B, T, stage):
        """Cross-frame attention: each query attends to its own history across T frames."""
        _, N, D = tgt.size()
        tgt = tgt.reshape(B, T, N, D)
        tgt = tgt.transpose(1, 2)
        tgt = tgt.reshape(B*N, T, D).transpose(0, 1)
        tgt1, _ = self.temproal_attention[stage](tgt, tgt, tgt)
        tgt = tgt1 + tgt
        tgt = self.temproal_norm[stage](tgt)
        tgt = tgt.transpose(0, 1).reshape(B, N, T, D)
        tgt = tgt.transpose(1, 2)
        tgt = tgt.reshape(B*T, N, D)
        return tgt

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                B,
                T,
                dec_bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                test=False,
                track=False,
                ori_feats=None):

        dec_out_logits = []
        dec_out_head_bboxes = []
        dec_out_eye_bboxes = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        output = target
        head_content = None
        eye_content = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes,
                          attn_mask, memory_mask, query_pos_embed)
            output = self.spatial_temproal_attention(output, B, T, i)
            eye_content = self.transform_eye[i](output)
            head_content = output

            inter_ref_bbox_head = F.sigmoid(dec_bbox_head[i](head_content) + inverse_sigmoid(ref_points_detach))
            inter_ref_bbox_eye = F.sigmoid(dec_bbox_head[i](eye_content) + inverse_sigmoid(ref_points_detach))

            inter_ref_bbox = inter_ref_bbox_head

            dec_out_logits.append(score_head[i](head_content))
            if i == 0:
                dec_out_head_bboxes.append(inter_ref_bbox_head)
                dec_out_eye_bboxes.append(inter_ref_bbox_eye)
            else:
                dec_out_head_bboxes.append(F.sigmoid(dec_bbox_head[i](head_content) + inverse_sigmoid(ref_points)))
                dec_out_eye_bboxes.append(F.sigmoid(dec_bbox_head[i](eye_content) + inverse_sigmoid(ref_points)))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        if test:
            return (torch.stack(dec_out_head_bboxes), torch.stack(dec_out_eye_bboxes),
                    torch.stack(dec_out_logits), None, head_content, eye_content)
        else:
            return (torch.stack(dec_out_head_bboxes), torch.stack(dec_out_eye_bboxes),
                    torch.stack(dec_out_logits), None, None, None)


class RTDETRTransformerv2Lite(nn.Module):
    """Lite version of RTDETRTransformerv2 for face + eye localization only.

    Stripped of: pred_blink (BlinkTransformerDecoder), CDN denoising, @register dependency.
    Added: transform_expansion parameter for lightweight eye branch MLP.

    Config for lite model:
        num_classes=1, hidden_dim=256, num_queries=50, num_layers=3, nhead=4,
        transform_expansion=2
    """

    def __init__(self,
                 num_classes=2,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=None,
                 feat_strides=None,
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 learn_query_content=False,
                 eval_spatial_size=None,
                 track=True,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 transform_expansion=4):
        super().__init__()

        if feat_channels is None:
            feat_channels = [512, 1024, 2048]
        if feat_strides is None:
            feat_strides = [8, 16, 32]

        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.track = track
        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.transform_expansion = transform_expansion

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx,
            num_heads=nhead, transform_expansion=transform_expansion)

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, B, T, memory, spatial_shapes,
                           denoising_logits=None, denoising_bbox_unact=None):
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(B, T, output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, B, T, memory, outputs_logits, outputs_coords_unact, topk):
        N = outputs_logits.size(1)
        outputs_logits_index = outputs_logits.reshape(B, T, N, -1)
        outputs_logits_index = torch.mean(outputs_logits_index, dim=1)
        outputs_logits_index = outputs_logits_index.reshape(B, 1, N, -1).repeat(1, T, 1, 1)
        outputs_logits_index = outputs_logits_index.reshape(B*T, N, -1)

        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits_index.max(-1).values, topk, dim=-1)
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits_index.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits_index.squeeze(-1), topk, dim=-1)

        topk_coords = outputs_coords_unact.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        topk_memory = memory.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_coords

    def forward(self, feats, B, T, targets=None, test=False):
        """Forward pass.

        Args:
            feats: List of encoder feature maps (3 levels, all 256ch)
            B: batch size
            T: number of frames
            targets: training targets (optional)
            test: if True, returns head_query and eye_query for Stage 2

        Returns:
            out: dict with pred_logits, pred_head_boxes, pred_eye_boxes
            head_query: [B*T, N, 256] head branch features (test mode only)
            eye_query: [B*T, N, 256] eye branch features (test mode only)
        """
        memory, spatial_shapes = self._get_encoder_input(feats)

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(B, T, memory, spatial_shapes)

        out_head_bboxes, out_eye_bboxes, out_logits, _, head_query, eye_query = self.decoder(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
            B, T, self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            test=test, track=self.track, ori_feats=feats)

        out = {'pred_logits': out_logits[-1],
               'pred_head_boxes': out_head_bboxes[-1],
               'pred_eye_boxes': out_eye_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_head_bboxes[:-1], out_eye_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

        return out, head_query, eye_query

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_head_coord, outputs_eye_coord=None):
        if outputs_eye_coord is not None:
            return [{'pred_logits': a, 'pred_head_boxes': b, 'pred_eye_boxes': c}
                    for a, b, c in zip(outputs_class, outputs_head_coord, outputs_eye_coord)]
        else:
            return [{'pred_logits': a, 'pred_head_boxes': b}
                    for a, b in zip(outputs_class, outputs_head_coord)]
