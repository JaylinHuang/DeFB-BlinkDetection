"""Train Lite Model + Extract 64×64 Eye Crops — 完整Pipeline

训练5 epochs + 眼部裁剪提取(GT + Model Prediction) + Loss数据汇总.

Usage:
    python newtest/train_and_extract.py              # 完整流程
    python newtest/train_and_extract.py --skip-train  # 仅提取 (需要已有checkpoint)
    python newtest/train_and_extract.py --gt-only     # 仅GT裁剪 (不训练)

流程:
  1. 训练: N epoch快速训练Lite Track Model (with_eye_bbox=True, 眼部bbox由landmark自动生成)
  2. Loss收集: 汇总loss_detail.json数据
  3. GT裁剪: 从landmark计算眼部区域 → 裁剪64×64
  4. 模型裁剪: 用训练好的模型预测eye_bbox → 裁剪64×64
  5. 报告: 生成PIPELINE_REPORT.md

Author: DeFB Lite Team
Created: 2026-06-28
"""

import sys
import os
import json
import time
import argparse
import copy
import glob as glob_mod

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add parent project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import YAMLConfig
from src.solver import TASKS
from src.misc import dist_utils
from newtest.track_model_lite import build_lite_model, count_parameters, count_all_parameters
from newtest.eye_crop_extractor import (
    EyeCropExtractor,
    get_eye_region_from_landmarks,
    face_bbox_to_eye_region,
    cxcywh_to_xyxy,
)
import newtest.collate  # register LiteBatchImageCollate


# ============================================================
# Paths
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPEBLINK_DIR = "E:/documents/mssb code/mpeblink2.0/mpeblink_lite"
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'eye_crop_pipeline')


def setup_dirs():
    """Create output directory structure."""
    dirs = {
        'root': OUTPUT_DIR,
        'ckpt': os.path.join(OUTPUT_DIR, 'checkpoints'),
        'crops_gt': os.path.join(OUTPUT_DIR, 'crops_gt'),
        'crops_pred': os.path.join(OUTPUT_DIR, 'crops_pred'),
        'loss': os.path.join(OUTPUT_DIR, 'loss'),
        'grids': os.path.join(OUTPUT_DIR, 'grids'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# Step 1: Training
# ============================================================
def train_model(dirs, epochs=5, device=None):
    """训练Lite Track Model.

    使用train_mini数据集(5视频), with_eye_bbox=True (眼部bbox从landmark自动计算).

    Returns:
        (ckpt_path, loss_summary_dict)
    """
    print("\n" + "=" * 60)
    print(f"Step 1: Training Lite Track Model ({epochs} epochs)")
    print("=" * 60)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Pretrained backbone ----
    pretrained = os.path.join(ROOT_DIR, 'ResNet18_vd_pretrained_from_paddle.pth')
    if not os.path.exists(pretrained):
        print("  [WARN] No pretrained backbone found, training from scratch")
        pretrained = None

    # ---- Build model ----
    model = build_lite_model(backbone_pretrained=pretrained if pretrained else False)
    total = count_all_parameters(model)
    trainable = count_parameters(model)
    print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable ({trainable/total*100:.1f}%)")

    # ---- Config ----
    config_path = os.path.join(os.path.dirname(__file__), 'config_lite_quick.yml')
    cfg = YAMLConfig(config_path)

    # Override for 5-epoch training
    cfg.epoches = epochs
    cfg.output_dir = dirs['ckpt']
    cfg.sync_bn = False
    cfg.find_unused_parameters = False

    # Dataset paths (use existing train_mini which has landmarks for eye_bbox)
    cfg.train_dataloader.dataset.ann_file = os.path.join(
        MPEBLINK_DIR, 'annotations', 'train_mini.json')
    cfg.train_dataloader.dataset.img_fold = os.path.join(
        MPEBLINK_DIR, 'train_rawframes')
    cfg.train_dataloader.total_batch_size = 1
    cfg.train_dataloader.num_workers = 0

    cfg.val_dataloader.dataset.ann_file = os.path.join(
        MPEBLINK_DIR, 'annotations', 'val_mini.json')
    cfg.val_dataloader.dataset.img_fold = os.path.join(
        MPEBLINK_DIR, 'val_rawframes')
    cfg.val_dataloader.total_batch_size = 1
    cfg.val_dataloader.num_workers = 0

    print(f"  Config:      {config_path}")
    print(f"  Train ann:   {cfg.train_dataloader.dataset.ann_file}")
    print(f"  Val ann:     {cfg.val_dataloader.dataset.ann_file}")
    print(f"  Epochs:      {epochs}")
    print(f"  Output:      {dirs['ckpt']}")

    # Override dataset to ensure with_eye_bbox=True
    if not hasattr(cfg.train_dataloader.dataset, 'with_eye_bbox'):
        cfg.train_dataloader.dataset.with_eye_bbox = True

    # ---- Inject model ----
    cfg._model = model
    cfg._device = device

    # ---- Train ----
    print("\n  Training... (this may take 30-40 min for 5 epochs)")
    t0 = time.time()

    solver_class = TASKS[cfg.task]
    solver = solver_class(cfg)
    solver.fit()

    elapsed = time.time() - t0
    print(f"\n  Training done: {elapsed/60:.1f} min")

    # ---- Find checkpoint ----
    ckpt_path = os.path.join(dirs['ckpt'], 'last.pth')
    if not os.path.exists(ckpt_path):
        ckpts = glob_mod.glob(os.path.join(dirs['ckpt'], '*.pth'))
        ckpt_path = ckpts[0] if ckpts else None
        if ckpt_path:
            print(f"  Using checkpoint: {ckpt_path}")
    else:
        print(f"  Checkpoint: {ckpt_path}")

    dist_utils.cleanup()
    return ckpt_path


# ============================================================
# Step 2: Collect loss data
# ============================================================
def collect_loss(dirs):
    """收集训练loss数据并汇总."""
    print("\n" + "=" * 60)
    print("Step 2: Collecting loss data")
    print("=" * 60)

    summary = {}
    loss_dir = dirs['ckpt']

    for fname in ['loss_detail.json', 'loss_data.json']:
        src = os.path.join(loss_dir, fname)
        if not os.path.exists(src):
            print(f"  [WARN] {fname} not found")
            continue

        with open(src, 'r') as f:
            data = json.load(f)

        # Copy
        dst = os.path.join(dirs['loss'], fname)
        with open(dst, 'w') as f:
            json.dump(data, f)
        print(f"  {fname}: {len(data.get('step', data.get('steps', [])))} steps → {dst}")

        # Summary stats
        if fname == 'loss_detail.json':
            for key in data:
                if key == 'step' or not data[key]:
                    continue
                vals = data[key]
                summary[key] = {
                    'initial': round(vals[0], 4) if vals else None,
                    'final': round(vals[-1], 4) if vals else None,
                    'min': round(min(vals), 4) if vals else None,
                    'max': round(max(vals), 4) if vals else None,
                }

    # Save summary
    summary_path = os.path.join(dirs['loss'], 'loss_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")

    if summary:
        print("\n  Loss Summary:")
        for k, v in summary.items():
            print(f"    {k}: {v['initial']} → {v['final']} (min={v['min']}, max={v['max']})")

    return summary


# ============================================================
# Step 3: GT eye crop extraction
# ============================================================
def extract_gt_crops(dirs, max_persons=10, max_frames=30):
    """从GT landmark提取64×64眼部裁剪."""
    print("\n" + "=" * 60)
    print("Step 3: Extracting GT eye crops (Landmark → 64×64)")
    print("=" * 60)

    from newtest.eye_crop_extractor import extract_eye_crops_gt

    ann_file = os.path.join(MPEBLINK_DIR, 'annotations', 'train_mini.json')
    rawframes = os.path.join(MPEBLINK_DIR, 'train_rawframes')

    meta = extract_eye_crops_gt(
        ann_file, rawframes, dirs['crops_gt'],
        max_persons=max_persons,
        max_frames_per_person=max_frames,
        method='face_first',  # 两阶段: 人脸上采样→眼部, 眼睛占比更大
    )
    return meta


# ============================================================
# Step 4: Model-prediction eye crop extraction
# ============================================================
def extract_pred_crops(ckpt_path, dirs, device=None, max_frames=16):
    """用训练好的模型预测eye_bbox并提取64×64眼部裁剪."""
    print("\n" + "=" * 60)
    print("Step 4: Extracting eye crops (Model Prediction → 64×64)")
    print("=" * 60)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not ckpt_path or not os.path.exists(ckpt_path):
        print("  [SKIP] No checkpoint available")
        return {'total_crops': 0, 'results': []}

    # ---- Load model ----
    print(f"  Loading checkpoint: {ckpt_path}")
    model = build_lite_model(backbone_pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in state:
        state = state['model']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    print("  Model loaded")

    # ---- Load annotations ----
    ann_file = os.path.join(MPEBLINK_DIR, 'annotations', 'train_mini.json')
    with open(ann_file, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    rawframes = os.path.join(MPEBLINK_DIR, 'train_rawframes')

    extractor = EyeCropExtractor()

    # ---- Import transforms for preprocessing ----
    from torchvision import transforms as Tf

    all_results = []
    total_crops = 0

    for anno in annotations:
        video_id = anno.get('video_id', 0)
        person_id = anno.get('id', 0)
        landmarks = anno.get('landmark', [])
        face_bboxes = anno.get('bboxes', [])

        video_dir = os.path.join(rawframes, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        frame_files = sorted(os.listdir(video_dir))
        if not frame_files:
            continue

        n_available = min(len(frame_files), len(landmarks), 60)

        # ---- Load and prepare frames ----
        frames_rgb = []
        valid_idx = []
        for i in range(n_available):
            if not landmarks[i]:
                continue
            frame = cv2.imread(os.path.join(video_dir, frame_files[i]))
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            valid_idx.append(i)

        if len(frames_rgb) < 4:
            continue

        # Take clips of clip_length=4 (matching training config)
        clip_len = 4
        n_clips = len(frames_rgb) // clip_len

        person_crops = []
        person_dir = os.path.join(dirs['crops_pred'], str(video_id),
                                  f"person_{person_id:04d}")
        os.makedirs(person_dir, exist_ok=True)

        for clip_i in range(n_clips):
            start = clip_i * clip_len
            clip_frames = frames_rgb[start:start + clip_len]
            clip_indices = valid_idx[start:start + clip_len]

            # To tensor: [T, C, H, W]
            to_tensor = Tf.ToTensor()
            tensors = [to_tensor(f) for f in clip_frames]
            x = torch.stack(tensors, dim=0)  # [T, 3, 720, 1280]
            x = x.unsqueeze(0).to(device)    # [1, T, 3, 360, 640]

            with torch.no_grad():
                out, memory, head_query, eye_query = model(x, test=True)

            # pred_eye_boxes: [T, N, 4] cxcywh normalized [0,1]
            pred_eye = out['pred_eye_boxes']  # [T, N, 4]
            pred_logits = out['pred_logits']  # [T, N, 1]
            T_out, N, _ = pred_eye.shape

            # Select best person (highest avg confidence across frames)
            scores = pred_logits.squeeze(-1).sigmoid()  # [T, N]
            avg_score = scores.mean(dim=0)  # [N]
            best_n = avg_score.argmax().item()

            # Get eye bboxes for best person
            eye_norm = pred_eye[:, best_n, :]  # [T, 4] cxcywh normalized

            # Denormalize to pixel coordinates
            eye_pixel = eye_norm.clone()
            eye_pixel[:, 0] *= 1280  # cx (1280x720 frame)
            eye_pixel[:, 1] *= 720   # cy
            eye_pixel[:, 2] *= 1280  # w
            eye_pixel[:, 3] *= 720   # h
            eye_pixel = eye_pixel.cpu().numpy()

            # Extract crops
            for t in range(T_out):
                if total_crops >= max_frames * len(annotations):
                    break

                frame = cv2.imread(os.path.join(video_dir,
                                                frame_files[clip_indices[t]]))
                if frame is None:
                    continue

                eye_xyxy = cxcywh_to_xyxy(eye_pixel[t])
                crop = extractor.extract(frame, eye_xyxy)

                crop_path = os.path.join(person_dir,
                                         f"frame_{clip_indices[t]:05d}.png")
                extractor.save(crop, crop_path)
                person_crops.append(crop)
                total_crops += 1

                all_results.append({
                    'video_id': video_id,
                    'person_id': person_id,
                    'frame_idx': clip_indices[t],
                    'crop_path': crop_path,
                    'eye_bbox_pixel_xyxy': eye_xyxy.tolist(),
                    'eye_bbox_norm_cxcywh': eye_norm[t].cpu().tolist(),
                })

            if total_crops >= max_frames * len(annotations):
                break

        # Save grid
        if person_crops:
            grid_path = os.path.join(person_dir, 'grid.png')
            extractor.save_grid(person_crops, grid_path, cols=8)
            print(f"  Video {video_id} Person {person_id}: "
                  f"{len(person_crops)} pred crops")

    # ---- Save metadata ----
    meta = {
        'total_crops': total_crops,
        'output_size': [64, 64],
        'source': 'model_prediction',
        'checkpoint': ckpt_path,
        'results': all_results,
    }
    meta_path = os.path.join(dirs['crops_pred'], 'extraction_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Total pred crops: {total_crops}")
    print(f"  Meta: {meta_path}")
    return meta


# ============================================================
# Step 5: Generate report
# ============================================================
def generate_report(dirs, loss_summary, gt_meta, pred_meta, epochs):
    """生成Markdown格式的完整报告."""
    print("\n" + "=" * 60)
    print("Step 5: Generating report")
    print("=" * 60)

    report_path = os.path.join(dirs['root'], 'PIPELINE_REPORT.md')

    lines = [
        "# DeFB Lite — 人眼定位网络 64×64眼部图像提取报告",
        "",
        f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**训练Epochs**: {epochs}",
        "",
        "---",
        "",
        "## 1. 实验配置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        "| 模型 | TrackModelLite (PResNet-18 + HybridEncoder + Lite Decoder) |",
        "| 参数量 | ~21.8M (11.2M backbone + 5.2M encoder + 5.4M decoder) |",
        "| 训练数据 | train_mini: 5 videos |",
        "| 验证数据 | val_mini: 3 videos |",
        "| Epochs | 5 |",
        "| Clip length | 4 |",
        "| Batch size | 1 |",
        "| 输入尺寸 | 1280×720 |",
        "| 眼部输出尺寸 | 64×64 |",
        "| 眼部GT来源 | 68点facial landmark (与_parse_ann_info同算法) |",
        "| 设备 | NVIDIA GeForce RTX 4070 Laptop (8GB) |",
        "",
        "## 2. Loss 数据",
        "",
    ]

    if loss_summary:
        lines.append("| Loss 分量 | 初始值 | 最终值 | 最小值 | 最大值 |")
        lines.append("|-----------|--------|--------|--------|--------|")
        for key, stats in loss_summary.items():
            if isinstance(stats, dict):
                lines.append(
                    f"| {key} | {stats.get('initial', 'N/A')} | "
                    f"{stats.get('final', 'N/A')} | "
                    f"{stats.get('min', 'N/A')} | "
                    f"{stats.get('max', 'N/A')} |")
        lines.append("")
    else:
        lines.append("_Loss数据将在训练完成后自动填充_")
        lines.append("")

    lines.extend([
        "## 3. 眼部图像提取结果",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| GT裁剪总数 | {gt_meta.get('total_crops', 0) if gt_meta else 0} |",
        f"| 模型预测裁剪总数 | {pred_meta.get('total_crops', 0) if pred_meta else 0} |",
        f"| 输出尺寸 | 64×64 |",
        f"| GT bbox算法 | mpeblink._parse_ann_info landmark索引 |",
        f"| 模型预测 | pred_eye_boxes → 反归一化 → xyxy → crop+resize |",
        "",
        "## 4. 输出目录结构",
        "",
        "```",
        f"output/eye_crop_pipeline/",
        f"├── checkpoints/         # 模型权重 (.pth)",
        f"├── loss/                # Loss数据 (loss_detail.json + loss_summary.json)",
        f"├── crops_gt/            # GT眼部裁剪 (landmark → 64×64)",
        f"│   └── {{video_id}}/person_{{id}}/",
        f"│       ├── frame_00000.png",
        f"│       ├── ...",
        f"│       └── grid.png     # 网格可视化",
        f"├── crops_pred/          # 模型预测眼部裁剪 (pred_eye_boxes → 64×64)",
        f"│   └── {{video_id}}/person_{{id}}/",
        f"└── PIPELINE_REPORT.md   # 本报告",
        "```",
        "",
        "## 5. 后续工作",
        "",
        "- [ ] 架构B: PResNet-50 + decoder expansion=4 变体",
        "- [ ] 架构C: YOLO-based 或 MediaPipe 眼部检测方案",
        "- [ ] 参数调优: lr (1e-5/1e-4/5e-4), dim (128/256/512), dropout (0/0.1/0.2), 数据增强×3",
        "- [ ] 对比实验: 收集开源项目, HUSTLEBW指标测试",
        "",
        "---",
        f"*Report generated by train_and_extract.py on {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  Report → {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train Lite Model + Extract 64×64 Eye Crops')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, use existing checkpoint')
    parser.add_argument('--gt-only', action='store_true',
                        help='GT extraction only, skip training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Use specific checkpoint')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("DeFB Lite — Train & Extract 64×64 Eye Crops Pipeline")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {'Yes' if torch.cuda.is_available() else 'No'}")

    dirs = setup_dirs()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---- Determine checkpoint ----
    ckpt_path = args.checkpoint
    if not args.skip_train and not args.gt_only:
        ckpt_path = train_model(dirs, args.epochs, device)

    if not ckpt_path and (args.skip_train or args.gt_only):
        ckpts = glob_mod.glob(os.path.join(dirs['ckpt'], '*.pth'))
        ckpt_path = ckpts[0] if ckpts else None
        if ckpt_path:
            print(f"Found existing checkpoint: {ckpt_path}")

    # ---- Collect loss ----
    loss_summary = collect_loss(dirs) if not args.gt_only else {}

    # ---- GT extraction ----
    gt_meta = extract_gt_crops(dirs)

    # ---- Model prediction extraction ----
    pred_meta = {}
    if not args.gt_only:
        pred_meta = extract_pred_crops(ckpt_path, dirs, device)
    else:
        print("\n  [SKIP] Model prediction extraction (--gt-only)")

    # ---- Report ----
    generate_report(dirs, loss_summary, gt_meta, pred_meta, args.epochs)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"  Output: {dirs['root']}")
    print(f"  Report: {os.path.join(dirs['root'], 'PIPELINE_REPORT.md')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
