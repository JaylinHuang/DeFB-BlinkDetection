"""Lightweight Track Model for Face + Eye Localization.

Assembles PResNet-18 backbone + HybridEncoder + RTDETRTransformerv2Lite decoder.
Self-contained model class — no dependency on @register() or __inject__.
"""

import copy
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from newtest.rtdetrv2_decoder_lite import RTDETRTransformerv2Lite


def build_lite_model(
    # Backbone
    backbone_depth=18,
    backbone_variant='d',
    backbone_return_idx=(1, 2, 3),
    backbone_pretrained=True,
    # Encoder
    encoder_in_channels=None,       # auto-computed from backbone depth
    encoder_hidden_dim=256,
    encoder_use_encoder_idx=None,   # [] = disable transformer encoder
    encoder_num_encoder_layers=1,
    encoder_nhead=8,
    encoder_dim_feedforward=1024,
    encoder_depth_mult=0.33,        # lite: fewer RepVggBlocks
    encoder_expansion=1.0,
    encoder_act='silu',
    # Decoder
    decoder_num_classes=1,
    decoder_hidden_dim=256,
    decoder_num_queries=50,         # lite: 50
    decoder_num_layers=3,           # lite: 3
    decoder_nhead=4,                # lite: 4
    decoder_dim_feedforward=1024,
    decoder_num_points=4,
    decoder_learn_query_content=True,
    decoder_track=True,
    decoder_transform_expansion=2,  # lite: 2x instead of 4x
    decoder_eval_spatial_size=(720, 1280),
    # Misc
    eval_spatial_size=(720, 1280),
    device=None,
):
    """Build the lightweight Track Model with configurable parameters.

    Returns:
        TrackModelLite instance
    """
    # ---- Backbone ----
    backbone = PResNet(
        depth=backbone_depth,
        variant=backbone_variant,
        return_idx=list(backbone_return_idx),
        num_stages=4,
        freeze_norm=True,
        pretrained=backbone_pretrained,
    )

    # ---- Encoder ----
    if encoder_in_channels is None:
        # PResNet-18: BasicBlock expansion=1 → [128, 256, 512]
        # PResNet-50: BottleNeck expansion=4 → [512, 1024, 2048]
        expansion = 1 if backbone_depth in (18, 34) else 4
        base_channels = [64, 128, 256, 512]
        encoder_in_channels = [expansion * base_channels[i] for i in backbone_return_idx]

    encoder = HybridEncoder(
        in_channels=encoder_in_channels,
        feat_strides=[8, 16, 32],
        hidden_dim=encoder_hidden_dim,
        use_encoder_idx=encoder_use_encoder_idx or [],
        num_encoder_layers=encoder_num_encoder_layers,
        nhead=encoder_nhead,
        dim_feedforward=encoder_dim_feedforward,
        dropout=0.,
        enc_act='gelu',
        expansion=encoder_expansion,
        depth_mult=encoder_depth_mult,
        act=encoder_act,
    )

    # ---- Decoder ----
    decoder = RTDETRTransformerv2Lite(
        num_classes=decoder_num_classes,
        hidden_dim=decoder_hidden_dim,
        num_queries=decoder_num_queries,
        feat_channels=[encoder_hidden_dim] * 3,
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[decoder_num_points] * 3 if isinstance(decoder_num_points, int) else decoder_num_points,
        nhead=decoder_nhead,
        num_layers=decoder_num_layers,
        dim_feedforward=decoder_dim_feedforward,
        dropout=0.,
        activation="relu",
        learn_query_content=decoder_learn_query_content,
        eval_spatial_size=decoder_eval_spatial_size,
        track=decoder_track,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        transform_expansion=decoder_transform_expansion,
    )

    model = TrackModelLite(backbone, encoder, decoder)
    model.eval_spatial_size = eval_spatial_size

    if device is not None:
        model = model.to(device)

    return model


class TrackModelLite(nn.Module):
    """Standalone Track Model for face + eye localization.

    Equivalent to the original RTDETR class but without @register() dependency.
    Handles B×T dimension reshaping internally.
    """

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.eval_spatial_size = None

    def forward(self, x, targets=None, test=False):
        """Forward pass.

        Args:
            x: [B, T, C, H, W] input video tensor
            targets: training targets (optional)
            test: if True, returns memory + head_query + eye_query for Stage 2

        Returns:
            If test=False: output dict {pred_logits, pred_head_boxes, pred_eye_boxes, ...}
            If test=True: (output_dict, memory_list, head_query, eye_query)
        """
        B, T, C, H, W = x.size()
        x = x.reshape(B * T, C, H, W)

        x = self.backbone(x)
        x = self.encoder(x)

        memory = copy.deepcopy(x) if test else None

        out, head_query, eye_query = self.decoder(x, B, T, targets, test=test)

        if test:
            return out, memory, head_query, eye_query
        else:
            return out

    def deploy(self):
        """Convert to deployment mode (RepVgg fusion)."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def build_nano_model(
    backbone_pretrained=True,
    eval_spatial_size=(720, 1280),
    device=None,
):
    """Ultra-light RT-DETRv2 variant — ~5M params with MobileNetV3 backbone.

    Changes from build_lite_model (21.8M):
      - Backbone: MobileNetV3-Small (~1.5M) instead of PResNet-18 (~11.7M)
      - hidden_dim: 128 (vs 256)
      - num_queries: 20 (vs 50)
      - num_layers: 2 (vs 3)
      - dim_feedforward: 512 (vs 1024)
      - nhead: 2 (vs 4)

    Returns:
        TrackModelLite instance (~5M params)
    """
    import torchvision.models as tv_models

    # ---- MobileNetV3-Small backbone (~1.5M params) ----
    mbn = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
    )
    # Extract intermediate features at strides 8, 16, 32
    # MobileNetV3-Small stages:
    #   features[0]:  stride 2  (H/2)
    #   features[1]:  stride 2  (H/4)
    #   features[2-3]: stride 2  (H/8)  ← return_idx[0]
    #   features[4-5]: stride 2  (H/16) ← return_idx[1]
    #   features[6-12]:stride 2  (H/32) ← return_idx[2]
    # Output channels at each stage: 24→16→24→40→48→96→576
    # After block 2: 24ch, after block 5: 48ch, after block 12: 576ch

    class MobileNetBackbone(nn.Module):
        def __init__(self, mbn):
            super().__init__()
            # MobileNetV3-Small layer breakdown:
            #   features[0]: Conv 3→16, s=2 (H/2)
            #   features[1]: IR 16→16, s=2 (H/4)
            #   features[2]: IR 16→24, s=2 (H/8)   → 24ch
            #   features[3]: IR 24→24, s=1
            #   features[4]: IR 24→40, s=2 (H/16)
            #   features[5]: IR 40→40, s=1          → 40ch
            #   features[6]: IR 40→48, s=1
            #   features[7]: IR 48→48, s=1
            #   features[8]: IR 48→96, s=2 (H/32)
            #   features[9]: IR 96→96, s=1
            #   features[10]: IR 96→96, s=1
            #   features[11]: Conv 96→576, s=1     → 576ch
            self.stage0 = mbn.features[:3]   # stride 8,  output 24ch
            self.stage1 = mbn.features[3:6]  # stride 16, output 40ch
            self.stage2 = mbn.features[6:]   # stride 32, output 576ch
            self.out_channels = [24, 40, 576]
            self.feat_strides = [8, 16, 32]

        def forward(self, x):
            f0 = self.stage0(x)
            f1 = self.stage1(f0)
            f2 = self.stage2(f1)
            return [f0, f1, f2]

    backbone = MobileNetBackbone(mbn)
    encoder_in_channels = backbone.out_channels

    # ---- Tiny Encoder ----
    hidden_dim = 128
    encoder = HybridEncoder(
        in_channels=encoder_in_channels,
        feat_strides=[8, 16, 32],
        hidden_dim=hidden_dim,
        use_encoder_idx=[],
        num_encoder_layers=1,
        nhead=2,
        dim_feedforward=512,
        dropout=0.,
        enc_act='gelu',
        expansion=1.0,
        depth_mult=0.15,
        act='silu',
    )

    # ---- Tiny Decoder ----
    decoder = RTDETRTransformerv2Lite(
        num_classes=1,
        hidden_dim=hidden_dim,
        num_queries=20,
        feat_channels=[hidden_dim] * 3,
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[4, 4, 4],
        nhead=2,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.,
        activation="relu",
        learn_query_content=True,
        eval_spatial_size=eval_spatial_size,
        track=True,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        transform_expansion=2,
    )

    model = TrackModelLite(backbone, encoder, decoder)
    model.eval_spatial_size = eval_spatial_size

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Count all parameters (including frozen)."""
    return sum(p.numel() for p in model.parameters())
