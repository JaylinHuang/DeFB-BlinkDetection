"""Extract Best Single-Eye Samples — 最佳单眼样本提取

从标注数据中提取最佳质量的13帧连续单眼裁剪样本。
输出两种接口:
  1. 图片文件夹: output/eye_samples/{video_id}/eye_001/frame_00.png ... frame_12.png
  2. NumPy数组:  output/eye_samples/{video_id}/eye_001.npy [13, 64, 64, 3] uint8

目录结构 (按视频分组, 每视频最多2个eye):
  output/eye_samples/
  ├── 1/                      # video_id=1 (数据集1)
  │   ├── eye_001/            # 该视频最佳左眼 (13张PNG)
  │   ├── eye_001.npy         # numpy数组
  │   ├── eye_002/            # 该视频最佳右眼 (13张PNG)
  │   └── eye_002.npy
  ├── 2/                      # video_id=2
  │   ├── eye_001/
  │   ├── eye_001.npy
  │   ├── eye_002/
  │   └── eye_002.npy
  ├── ...
  └── manifest.json           # 全局汇总

选取规则:
  - 每个视频内, 从所有person中选出最佳左眼×1 + 最佳右眼×1
  - 按Laplacian variance清晰度评分, 选最高分
  - 每个视频文件夹最多2个eye序列

Usage:
  python newtest/extract_eye_samples.py
  python newtest/extract_eye_samples.py --window 13 --face-ref-height 256

中文翻译:
  按视频(1,2,3,4,5文件夹)分组, 每组最多2个eye序列(eye_001/eye_002)。
  每个eye序列包含13帧连续单眼裁剪, 同时输出PNG图片和.npy数组。

Author: DeFB Lite Team
Created: 2026-06-28
Updated: 2026-06-28 — Restructured to video-based folders, max 2 eyes per video
"""

import sys
import os
import json
import time
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from newtest.eye_crop_extractor import EyeCropExtractor


# ============================================================
# Default paths
# ============================================================
DEFAULT_ANN = 'E:/documents/mssb code/mpeblink2.0/mpeblink_lite/annotations/train_mini.json'
DEFAULT_RAWFRAMES = 'E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes'
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'output', 'eye_samples')


def load_frames(video_dir: str, frame_files: List[str],
                max_frames: int = None) -> List[np.ndarray]:
    """Load video frames — 加载视频帧.

    Args:
        video_dir: rawframes目录 (e.g. .../train_rawframes/1/)
        frame_files: 帧文件名列表 (可能带video_id前缀如 '1/00000.png')
        max_frames: 最多加载帧数

    Returns:
        [frame_0, frame_1, ...] BGR格式numpy数组列表
    """
    frames = []
    files = frame_files[:max_frames] if max_frames else frame_files
    for fname in files:
        basename = os.path.basename(fname)  # strip '1/' prefix
        path = os.path.join(video_dir, basename)
        frame = cv2.imread(path)
        frames.append(frame if frame is not None else None)
    return frames


# ============================================================
# Core extraction: collect candidates, then select best per video
# ============================================================

def extract_all_eye_candidates(
    ann_file: str,
    rawframes_dir: str,
    extractor: EyeCropExtractor,
    window_size: int,
    face_ref_height: int,
    max_frames_per_video: int,
    video_map: Dict,
) -> List[dict]:
    """Extract all eye candidates from all persons — 从所有person提取全部候选眼.

    Returns list of candidate dicts:
      {video_id, person_id, side, start_frame, crops[np.array], score}
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    candidates = []

    for anno in annotations:
        video_id = anno.get('video_id', 0)
        person_id = anno.get('id', 0)
        face_bboxes = anno.get('bboxes', [])
        landmarks = anno.get('landmark', [])

        if not landmarks or not face_bboxes:
            continue

        video_dir = os.path.join(rawframes_dir, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        video_info = video_map.get(video_id, {})
        file_names = video_info.get('file_names', [])
        if not file_names:
            file_names = sorted(os.listdir(video_dir))

        n_available = min(len(file_names), len(landmarks), len(face_bboxes),
                         max_frames_per_video)

        if n_available < window_size:
            print(f"  [SKIP] Video {video_id} Person {person_id}: "
                  f"only {n_available} frames (< {window_size})")
            continue

        # Load frames
        print(f"  Video {video_id} Person {person_id}: loading {n_available} frames...")
        frames = load_frames(video_dir, file_names, n_available)
        landmarks_t = landmarks[:n_available]
        bboxes_t = face_bboxes[:n_available]

        valid_count = sum(1 for i in range(n_available)
                         if frames[i] is not None
                         and landmarks_t[i] is not None
                         and bboxes_t[i] is not None)
        if valid_count < window_size:
            print(f"    [SKIP] Only {valid_count} valid frames")
            continue

        # Extract LEFT eye — 左眼
        print(f"    Left eye (左眼): scanning...")
        start_l, crops_l, score_l = extractor.find_best_window(
            frames, bboxes_t, landmarks_t,
            side='left', window_size=window_size, face_ref_height=face_ref_height)
        candidates.append({
            'video_id': video_id, 'person_id': person_id,
            'side': 'left', 'side_cn': '左眼',
            'start_frame': start_l, 'crops': crops_l, 'score': score_l,
        })
        print(f"      -> left score={score_l:.1f} start={start_l}")

        # Extract RIGHT eye — 右眼
        print(f"    Right eye (右眼): scanning...")
        start_r, crops_r, score_r = extractor.find_best_window(
            frames, bboxes_t, landmarks_t,
            side='right', window_size=window_size, face_ref_height=face_ref_height)
        candidates.append({
            'video_id': video_id, 'person_id': person_id,
            'side': 'right', 'side_cn': '右眼',
            'start_frame': start_r, 'crops': crops_r, 'score': score_r,
        })
        print(f"      -> right score={score_r:.1f} start={start_r}")

    return candidates


def select_best_per_video(candidates: List[dict]) -> Dict[int, List[dict]]:
    """Select best 2 eyes per video — 每视频选最佳的2个eye (最优左眼+最优右眼).

    Returns: {video_id: [best_left, best_right]}
    """
    # Group by video_id
    by_video: Dict[int, List[dict]] = {}
    for c in candidates:
        vid = c['video_id']
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(c)

    selected: Dict[int, List[dict]] = {}

    for vid in sorted(by_video.keys()):
        all_cands = by_video[vid]

        # Separate left and right
        left_cands = [c for c in all_cands if c['side'] == 'left']
        right_cands = [c for c in all_cands if c['side'] == 'right']

        # Pick best left and best right
        best_left = max(left_cands, key=lambda c: c['score']) if left_cands else None
        best_right = max(right_cands, key=lambda c: c['score']) if right_cands else None

        selected[vid] = []
        if best_left:
            selected[vid].append(best_left)
        if best_right:
            selected[vid].append(best_right)

        # Summary for this video
        n_left = len(left_cands)
        n_right = len(right_cands)
        sel_parts = []
        if best_left:
            sel_parts.append(f"eye_001(L,p{best_left['person_id']},s={best_left['score']:.1f})")
        if best_right:
            sel_parts.append(f"eye_002(R,p{best_right['person_id']},s={best_right['score']:.1f})")
        sel_str = ' + '.join(sel_parts) if sel_parts else '(none)'
        print(f"\n  Video {vid}: {n_left}L + {n_right}R candidates -> selected {sel_str}")

    return selected


# ============================================================
# Save output
# ============================================================

def save_eye_output_v2(
    extractor: EyeCropExtractor,
    crops: np.ndarray,
    output_dir: str,
    video_id: int,
    eye_idx: int,
    candidate: dict,
    window_size: int,
):
    """Save one eye to video-based folder — 保存单眼到视频子文件夹.

    Args:
        extractor: EyeCropExtractor
        crops: [window_size, 64, 64, 3]
        output_dir: 输出根目录
        video_id: 视频编号
        eye_idx: eye序号 (1-based: eye_001, eye_002)
        candidate: 候选眼信息 {person_id, side, side_cn, start_frame, score}
        window_size: 窗口帧数
    """
    eye_name = f"eye_{eye_idx:03d}"
    video_dir = os.path.join(output_dir, str(video_id))
    img_dir = os.path.join(video_dir, eye_name)
    os.makedirs(img_dir, exist_ok=True)

    # ---- Output 1: Image folder (13 PNGs) ----
    # 输出1: 图片文件夹, 13张PNG
    for i in range(window_size):
        frame_path = os.path.join(img_dir, f"frame_{i:02d}.png")
        extractor.save(crops[i], frame_path)

    # Grid for visual inspection
    # 网格图用于肉眼检查
    grid_path = os.path.join(img_dir, 'grid.png')
    extractor.save_grid([crops[i] for i in range(window_size)],
                        grid_path, cols=window_size, spacing=2)

    # ---- Output 2: NumPy array (.npy) ----
    # 输出2: NumPy数组
    npy_path = os.path.join(video_dir, f"{eye_name}.npy")
    np.save(npy_path, crops.astype(np.uint8))

    # ---- Info file ----
    info_path = os.path.join(img_dir, 'info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"eye_name: {eye_name}\n")
        f.write(f"video_id: {video_id}\n")
        f.write(f"person_id: {candidate['person_id']}\n")
        f.write(f"side: {candidate['side']} ({candidate['side_cn']})\n")
        f.write(f"start_frame: {candidate['start_frame']}\n")
        f.write(f"window_size: {window_size}\n")
        f.write(f"avg_sharpness: {candidate['score']:.2f}\n")
        f.write(f"output_shape: [{window_size}, 64, 64, 3]\n")


# ============================================================
# Verification
# ============================================================

def verify_output_v2(output_dir: str, window_size: int):
    """Verify output — 验证输出完整性."""
    print()
    print("=" * 60)
    print("Verification — 输出验证")
    print("=" * 60)

    manifest_path = os.path.join(output_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print("  [FAIL] manifest.json not found!")
        return False

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    errors = []
    for vid_name, vid_info in manifest.get('videos', {}).items():
        for eye_info in vid_info.get('eyes', []):
            eye_name = eye_info['eye_name']
            img_dir = os.path.join(output_dir, vid_name, eye_name)
            npy_path = os.path.join(output_dir, vid_name, f"{eye_name}.npy")

            # Check image folder
            if not os.path.isdir(img_dir):
                errors.append(f"{vid_name}/{eye_name}: folder missing")
                continue

            png_count = len([f for f in os.listdir(img_dir)
                           if f.endswith('.png') and f != 'grid.png'])
            if png_count != window_size:
                errors.append(f"{vid_name}/{eye_name}: "
                             f"expected {window_size} PNGs, got {png_count}")

            # Check grid
            if not os.path.exists(os.path.join(img_dir, 'grid.png')):
                errors.append(f"{vid_name}/{eye_name}: grid.png missing")

            # Check .npy
            if not os.path.exists(npy_path):
                errors.append(f"{vid_name}/{eye_name}: .npy missing")
                continue

            arr = np.load(npy_path)
            expected = (window_size, 64, 64, 3)
            if arr.shape != expected:
                errors.append(f"{vid_name}/{eye_name}: "
                             f"shape {arr.shape}, expected {expected}")
            if arr.dtype != np.uint8:
                errors.append(f"{vid_name}/{eye_name}: "
                             f"dtype {arr.dtype}, expected uint8")

    if errors:
        print(f"  [FAIL] {len(errors)} errors:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        total = manifest.get('total_eyes', 0)
        n_videos = len(manifest.get('videos', {}))
        print(f"  [PASS] All checks passed")
        print(f"    Videos: {n_videos}")
        print(f"    Total eyes: {total}")
        print(f"    Each: {window_size} PNGs (64x64) + .npy [{window_size},64,64,3]")
        return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract best single-eye 13-frame samples')
    parser.add_argument('--ann', type=str, default=DEFAULT_ANN)
    parser.add_argument('--rawframes', type=str, default=DEFAULT_RAWFRAMES)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--window', type=int, default=13)
    parser.add_argument('--face-ref-height', type=int, default=256)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()

    if args.verify_only:
        verify_output_v2(args.output, args.window)
        return

    print("=" * 60)
    print("Single-Eye Sample Extraction — 最佳单眼样本提取")
    print("=" * 60)
    print(f"  Annotation:  {args.ann}")
    print(f"  Rawframes:   {args.rawframes}")
    print(f"  Output:      {args.output}")
    print(f"  Window:      {args.window} frames")
    print(f"  Face ref:    {args.face_ref_height}px")
    print(f"  Structure:   {{video_id}}/eye_XXX/ (max 2 per video)")
    print()

    # ---- Load metadata ----
    with open(args.ann, 'r') as f:
        data = json.load(f)
    video_map = {v['id']: v for v in data.get('videos', [])}
    n_persons = len(data.get('annotations', []))
    n_videos = len(data.get('videos', []))
    print(f"  Dataset: {n_videos} videos, {n_persons} persons")

    # ---- Setup ----
    extractor = EyeCropExtractor(output_size=(64, 64))
    os.makedirs(args.output, exist_ok=True)

    t_start = time.time()

    # ---- Phase 1: Extract all candidates ----
    # 阶段1: 提取所有候选眼
    print(f"\n{'='*60}")
    print("Phase 1: Extract all eye candidates — 提取所有候选眼")
    print("=" * 60)
    candidates = extract_all_eye_candidates(
        args.ann, args.rawframes, extractor,
        args.window, args.face_ref_height, args.max_frames, video_map)

    print(f"\n  Total candidates: {len(candidates)} "
          f"({sum(1 for c in candidates if c['side']=='left')}L + "
          f"{sum(1 for c in candidates if c['side']=='right')}R)")

    # ---- Phase 2: Select best per video ----
    # 阶段2: 每视频选最佳
    print(f"\n{'='*60}")
    print("Phase 2: Select best per video — 每视频选最佳")
    print("=" * 60)
    selected = select_best_per_video(candidates)

    # ---- Phase 3: Save ----
    # 阶段3: 保存输出
    print(f"\n{'='*60}")
    print("Phase 3: Save outputs — 保存输出")
    print("=" * 60)

    manifest_videos = {}
    total_eyes = 0

    for vid in sorted(selected.keys()):
        eyes = selected[vid]
        manifest_eyes = []

        for i, cand in enumerate(eyes):
            eye_idx = i + 1  # eye_001, eye_002
            save_eye_output_v2(
                extractor, cand['crops'], args.output,
                vid, eye_idx, cand, args.window)
            manifest_eyes.append({
                'eye_name': f"eye_{eye_idx:03d}",
                'person_id': cand['person_id'],
                'side': cand['side'],
                'side_cn': cand['side_cn'],
                'start_frame': cand['start_frame'],
                'end_frame': cand['start_frame'] + args.window - 1,
                'avg_sharpness': round(cand['score'], 2),
                'npy_file': f"eye_{eye_idx:03d}.npy",
                'image_dir': f"eye_{eye_idx:03d}",
            })
            total_eyes += 1
            print(f"  Saved: {vid}/eye_{eye_idx:03d} "
                  f"({cand['side_cn']}, p{cand['person_id']}, "
                  f"score={cand['score']:.1f})")

        manifest_videos[str(vid)] = {
            'video_id': vid,
            'n_eyes': len(eyes),
            'eyes': manifest_eyes,
        }

    # ---- Manifest ----
    t_elapsed = time.time() - t_start
    manifest = {
        'total_eyes': total_eyes,
        'total_videos': len(selected),
        'window_size': args.window,
        'face_ref_height': args.face_ref_height,
        'output_size': [64, 64],
        'extraction_time_seconds': round(t_elapsed, 1),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'videos': manifest_videos,
    }
    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- Verify ----
    verify_output_v2(args.output, args.window)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Extraction Complete — 提取完成")
    print("=" * 60)
    print(f"  Videos:  {len(selected)}")
    print(f"  Eyes:    {total_eyes} (max 2 per video)")
    print(f"  Window:  {args.window} frames")
    print(f"  Output:  {args.output}")
    print(f"  Time:    {t_elapsed:.1f}s")
    print()
    print("  目录结构 (Directory structure):")
    for vid in sorted(selected.keys()):
        eyes = selected[vid]
        print(f"    {vid}/")
        for i, cand in enumerate(eyes):
            ei = i + 1
            print(f"      eye_{ei:03d}/  ({cand['side_cn']}, p{cand['person_id']}, "
                  f"score={cand['score']:.1f})")
            print(f"      eye_{ei:03d}.npy")

    # ---- Usage ----
    print()
    print("=" * 60)
    print("How to use — 如何使用")
    print("=" * 60)
    print()
    print("  Image interface (图片接口):")
    print(f"    {args.output}/1/eye_001/frame_00.png")
    print(f"    ... (13 frames)")
    print(f"    {args.output}/1/eye_001/frame_12.png")
    print()
    print("  NumPy interface (数组接口, 给组长眨眼检测模型):")
    print(f"    import numpy as np")
    print(f"    eye = np.load('{args.output}/1/eye_001.npy')")
    print(f"    # shape: [13, 64, 64, 3], dtype: uint8")
    print()
    print("  Metadata (元数据):")
    print(f"    {args.output}/manifest.json")
    print()


if __name__ == '__main__':
    main()
