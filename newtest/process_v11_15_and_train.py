"""Process test/11-15 videos -> mpeblink_lite -> train 15 epochs.

Pipeline:
  1. Process videos 11-15: mp4->frames + WFLW->COCO JSON
  2. Extract eye samples (64x64 single-eye, 13-frame window)
  3. Train TrackModelLite 15 epochs

Usage:
  python newtest/process_v11_15_and_train.py
  python newtest/process_v11_15_and_train.py --epochs 15 --skip-process

中文: 处理test/11-15视频, 提取帧+标注, 加入mpeblink_lite, 提取眼部样本, 训练15轮。
"""

import sys, os, json, time, argparse, subprocess, shutil, copy

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import YAMLConfig
from src.solver import TASKS
from src.misc import dist_utils
from newtest.track_model_lite import build_lite_model, count_parameters, count_all_parameters
from newtest.eye_crop_extractor import EyeCropExtractor
import newtest.collate

# Paths
TEST_ROOT = "E:/documents/mssb code/mpeblink2.0/test"
LITE_ROOT = "E:/documents/mssb code/mpeblink2.0/mpeblink_lite"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'train_v11_15')


# ============================================================
# Step 1: Process videos 11-15
# ============================================================
def process_videos(video_ids, target_w=1280, target_h=720):  # 1280x720
    """Process test videos: extract frames + create COCO annotations.

    中文: 处理test目录下的视频, 提取640x360帧, 生成COCO格式标注。
    """
    print("=" * 60)
    print(f"Step 1: Processing videos {video_ids} -> {LITE_ROOT}")
    print("=" * 60)

    rawframes_dir = os.path.join(LITE_ROOT, 'train_rawframes')
    annotations_dir = os.path.join(LITE_ROOT, 'annotations')
    os.makedirs(rawframes_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    dataset = {
        'info': {'description': 'MPEblink Lite videos 11-15', 'version': '1.0',
                 'year': '2026', 'date_created': time.strftime('%Y-%m-%d %H:%M:%S')},
        'licenses': {'licenses': 'research only'},
        'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person_face'}],
        'videos': [],
        'annotations': [],
    }

    anno_id = 0

    for vid in video_ids:
        video_dir = os.path.join(TEST_ROOT, str(vid))
        video_path = os.path.join(video_dir, 'video.mp4')
        anno_path = os.path.join(video_dir, 'annotation_WFLW.json')

        if not os.path.exists(video_path):
            print(f"  [SKIP] {vid}: no video.mp4")
            continue
        if not os.path.exists(anno_path):
            print(f"  [SKIP] {vid}: no annotation_WFLW.json")
            continue

        with open(anno_path, 'r') as f:
            origin_anno = json.load(f)

        src_h = origin_anno.pop('height', 1040)
        src_w = origin_anno.pop('width', 1920)
        length = origin_anno.pop('length', 0)

        scale_w = target_w / src_w
        scale_h = target_h / src_h

        # Extract frames
        save_dir = os.path.join(rawframes_dir, str(vid))
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        file_names = []
        img_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (target_w, target_h))
            cv2.imwrite(os.path.join(save_dir, f"{img_idx:05d}.png"), frame)
            file_names.append(f"{vid}/{img_idx:05d}.png")
            img_idx += 1
        cap.release()

        actual_len = min(len(file_names), length)

        # Video record
        dataset['videos'].append({
            'height': target_h, 'width': target_w,
            'length': actual_len, 'file_names': file_names[:actual_len],
            'id': vid,
        })

        # Process each person
        person_keys = [k for k in origin_anno.keys() if k.startswith('person')]
        for pk in person_keys:
            pd = origin_anno[pk]
            anno_id += 1

            # Scale bboxes
            new_bboxes = []
            for i in range(actual_len):
                bbox = pd['bbox'][i] if i < len(pd['bbox']) else None
                if bbox is None:
                    new_bboxes.append(None)
                else:
                    new_bboxes.append([bbox[0]*scale_w, bbox[1]*scale_h,
                                       bbox[2]*scale_w, bbox[3]*scale_h])

            # Scale landmarks (68-point)
            new_landmarks = []
            lm_key = 'landmark'
            if lm_key not in pd:
                lm_key = 'landmark_WFLW'
            if lm_key in pd and pd[lm_key]:
                for i in range(actual_len):
                    if i >= len(pd[lm_key]):
                        new_landmarks.append(None)
                        continue
                    frame_lm = pd[lm_key][i]
                    if frame_lm is None:
                        new_landmarks.append(None)
                    else:
                        new_landmarks.append([[p[0]*scale_w, p[1]*scale_h]
                                             for p in frame_lm])

            # Binary blink labels
            blink_intervals = pd.get('blink', [])
            blink_binary = []
            for i in range(actual_len):
                in_blink = 0
                for bl in blink_intervals:
                    if i >= bl[0] and i <= bl[1]:
                        in_blink = 1
                        break
                blink_binary.append(in_blink)

            dataset['annotations'].append({
                'height': target_h, 'width': target_w,
                'length': 1, 'category_id': 1,
                'bboxes': new_bboxes,
                'landmark': new_landmarks,
                'blinks': blink_intervals,
                'blinks_binary': blink_binary,
                'video_id': vid,
                'id': anno_id,
            })

        print(f"  Video {vid}: {actual_len} frames, {len(person_keys)} persons, "
              f"{len(file_names)} PNGs")

    # Save annotation
    ann_path = os.path.join(annotations_dir, 'train_v11_15.json')
    with open(ann_path, 'w') as f:
        json.dump(dataset, f)
    print(f"\n  Annotation saved: {ann_path}")
    print(f"  Videos: {len(dataset['videos'])}, Persons: {len(dataset['annotations'])}")
    return ann_path


# ============================================================
# Step 2: Extract eye samples from new videos
# ============================================================
def extract_eye_samples_v11_15(ann_file, output_subdir='v11_15'):
    """Extract best single-eye samples from videos 11-15.

    中文: 从11-15视频提取最佳单眼样本 (每视频最佳左眼+右眼)。
    """
    print("\n" + "=" * 60)
    print("Step 2: Extracting eye samples from videos 11-15")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(__file__))
    from extract_eye_samples import (extract_all_eye_candidates,
                                     select_best_per_video,
                                     save_eye_output_v2)

    with open(ann_file, 'r') as f:
        data = json.load(f)
    video_map = {v['id']: v for v in data.get('videos', [])}

    rawframes_dir = os.path.join(LITE_ROOT, 'train_rawframes')
    extractor = EyeCropExtractor(output_size=(64, 64))
    eye_output_dir = os.path.join(OUTPUT_DIR, 'eye_samples', output_subdir)
    os.makedirs(eye_output_dir, exist_ok=True)

    window_size = 13
    face_ref_height = 256

    candidates = extract_all_eye_candidates(
        ann_file, rawframes_dir, extractor,
        window_size=window_size, face_ref_height=face_ref_height,
        max_frames_per_video=200, video_map=video_map)

    print(f"\n  Candidates: {len(candidates)} "
          f"({sum(1 for c in candidates if c['side']=='left')}L + "
          f"{sum(1 for c in candidates if c['side']=='right')}R)")

    selected = select_best_per_video(candidates)

    # Save
    manifest_videos = {}
    total_eyes = 0
    for vid in sorted(selected.keys()):
        eyes = selected[vid]
        manifest_eyes = []
        for i, cand in enumerate(eyes):
            ei = i + 1
            save_eye_output_v2(extractor, cand['crops'], eye_output_dir,
                              vid, ei, cand, window_size)
            manifest_eyes.append({
                'eye_name': f"eye_{ei:03d}",
                'person_id': cand['person_id'],
                'side': cand['side'],
                'side_cn': cand['side_cn'],
                'start_frame': cand['start_frame'],
                'avg_sharpness': round(cand['score'], 2),
            })
            total_eyes += 1
            print(f"  Saved: {vid}/eye_{ei:03d} ({cand['side_cn']}, "
                  f"p{cand['person_id']}, score={cand['score']:.1f})")
        manifest_videos[str(vid)] = {'video_id': vid, 'n_eyes': len(eyes),
                                      'eyes': manifest_eyes}

    manifest = {
        'total_eyes': total_eyes, 'total_videos': len(selected),
        'window_size': window_size, 'videos': manifest_videos,
    }
    with open(os.path.join(eye_output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  Eye samples: {total_eyes} eyes -> {eye_output_dir}")
    return eye_output_dir


# ============================================================
# Step 3: Train TrackModelLite
# ============================================================
def train_model(ann_file, epochs=15, device=None):
    """Train TrackModelLite on the new video data.

    中文: 用11-15视频数据训练TrackModelLite, 目标15 epochs, ~30分钟。
    """
    print("\n" + "=" * 60)
    print(f"Step 3: Training TrackModelLite ({epochs} epochs)")
    print("=" * 60)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Use the quick config as base
    config_path = os.path.join(os.path.dirname(__file__), 'config_lite_quick.yml')
    cfg = YAMLConfig(config_path)

    # Override for training on v11-15
    cfg.epoches = epochs
    ckpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    cfg.output_dir = ckpt_dir
    cfg.sync_bn = False
    cfg.find_unused_parameters = False

    # Dataset: use the new v11-15 annotations
    # No validation (use same data)
    cfg.train_dataloader.dataset.ann_file = ann_file
    cfg.train_dataloader.dataset.img_fold = os.path.join(LITE_ROOT, 'train_rawframes')
    cfg.train_dataloader.dataset.clip_length = 4
    cfg.train_dataloader.dataset.infer_length = 4
    cfg.train_dataloader.dataset.stride = 3  # skip frames for speed
    cfg.train_dataloader.total_batch_size = 1
    cfg.train_dataloader.num_workers = 0

    # Val: use a subset of the same data
    cfg.val_dataloader.dataset.ann_file = ann_file
    cfg.val_dataloader.dataset.img_fold = os.path.join(LITE_ROOT, 'train_rawframes')
    cfg.val_dataloader.dataset.clip_length = 4
    cfg.val_dataloader.dataset.infer_length = 4
    cfg.val_dataloader.dataset.stride = 3
    cfg.val_dataloader.total_batch_size = 1
    cfg.val_dataloader.num_workers = 0

    # Build model
    pretrained = os.path.join(ROOT_DIR, 'ResNet18_vd_pretrained_from_paddle.pth')
    if not os.path.exists(pretrained):
        pretrained = None

    model = build_lite_model(backbone_pretrained=pretrained if pretrained else False)
    total = count_all_parameters(model)
    trainable = count_parameters(model)
    print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    cfg._model = model
    cfg._device = device

    print(f"  Config:     {config_path}")
    print(f"  Train ann:  {ann_file}")
    print(f"  Epochs:     {epochs}")
    print(f"  Clip len:   4, stride: 3")
    print(f"  Output:     {ckpt_dir}")

    # Train
    print("\n  Training...")
    t0 = time.time()
    solver_class = TASKS[cfg.task]
    solver = solver_class(cfg)
    solver.fit()
    elapsed = time.time() - t0

    dist_utils.cleanup()

    # Find checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'last.pth')

    # Collect loss
    loss_summary = {}
    for fname in ['loss_detail.json', 'loss_data.json']:
        src = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            with open(src, 'r') as f:
                ld = json.load(f)
            if fname == 'loss_detail.json':
                for k in ld:
                    if k != 'step' and ld[k]:
                        loss_summary[k] = {
                            'initial': round(ld[k][0], 2),
                            'final': round(ld[k][-1], 2),
                            'min': round(min(ld[k]), 2),
                        }

    print(f"\n  Training done: {elapsed/60:.1f} min")
    return ckpt_path, elapsed, loss_summary


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--skip-process', action='store_true',
                        help='Skip video processing (videos already processed)')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip eye extraction')
    parser.add_argument('--videos', type=int, nargs='+', default=[11, 12, 13, 14, 15])
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline: Process v11-15 -> Extract Eyes -> Train 15 Epochs")
    print("=" * 60)
    t_total = time.time()

    ann_file = os.path.join(LITE_ROOT, 'annotations', 'train_v11_15.json')

    # ---- Step 1: Process videos ----
    if not args.skip_process:
        ann_file = process_videos(args.videos)
    else:
        print(f"  [SKIP] Video processing (using existing: {ann_file})")

    # ---- Step 2: Extract eye samples ----
    if not args.skip_extract:
        eye_dir = extract_eye_samples_v11_15(ann_file)

    # ---- Step 3: Train ----
    ckpt_path, train_time, loss_summary = train_model(ann_file, args.epochs)

    # ---- Report ----
    t_total_elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("Pipeline Complete — 完成")
    print("=" * 60)
    print(f"  Videos processed: {len(args.videos)} (test/{args.videos[0]}-{args.videos[-1]})")
    print(f"  Training epochs:  {args.epochs}")
    print(f"  Training time:    {train_time/60:.1f} min")
    print(f"  Total time:       {t_total_elapsed/60:.1f} min")
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Output dir:       {OUTPUT_DIR}")

    if loss_summary:
        print(f"\n  Loss Summary (last epoch):")
        for k, v in loss_summary.items():
            print(f"    {k}: {v['initial']} -> {v['final']} (min={v['min']})")

    print()
    print("  Eye samples dir:")
    print(f"    {OUTPUT_DIR}/eye_samples/v11_15/")
    print("  To use numpy arrays:")
    print(f"    import numpy as np")
    print(f"    eye = np.load('{OUTPUT_DIR}/eye_samples/v11_15/11/eye_001.npy')")
    print(f"    # shape: [13, 64, 64, 3], dtype: uint8")


if __name__ == '__main__':
    main()
