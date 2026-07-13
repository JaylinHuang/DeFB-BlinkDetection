"""RT-DETRv2 Nano 眼部提取脚本.

加载训练好的Nano checkpoint，对视频帧推理并提取64x64单眼裁剪。
与MediaPipe/YOLO输出格式兼容: {video_id}/eye_00X/frame_00.png + .npy

Usage:
  python newtest/extract_nano_eyes.py --videos 1-10
"""

import sys, os, argparse
import cv2, numpy as np
import torch
from torchvision import transforms as Tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from newtest.track_model_lite import build_nano_model
from newtest.eye_crop_extractor import EyeCropExtractor, cxcywh_to_xyxy


def main():
    parser = argparse.ArgumentParser(description='Nano eye extraction')
    parser.add_argument('--checkpoint', default='output/nano_train_1-10/last.pth')
    parser.add_argument('--rawframes', default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes')
    parser.add_argument('--ann', default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/annotations/train_mini.json')
    parser.add_argument('--output', default='./output/nano_eyes_1-10')
    parser.add_argument('--videos', default='1-10')
    parser.add_argument('--window', type=int, default=13)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    # Parse video list
    if '-' in args.videos:
        a, b = args.videos.split('-')
        video_ids = list(range(int(a), int(b) + 1))
    else:
        video_ids = [int(x) for x in args.videos.split(',')]

    print("=" * 60)
    print("RT-DETRv2 Nano Eye Extraction")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Videos: {video_ids}")
    print(f"  Window: {args.window}")
    print(f"  Device: {args.device}")

    # Load model
    print("\nLoading Nano model...")
    model = build_nano_model(backbone_pretrained=False)  # weights from checkpoint
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if 'model' in state:
        state = state['model']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    extractor = EyeCropExtractor(output_size=(64, 64))
    to_tensor = Tf.ToTensor()

    total_eyes = 0
    for vid in video_ids:
        vdir = os.path.join(args.rawframes, str(vid))
        if not os.path.isdir(vdir):
            print(f"  {vid}: skip (no dir)")
            continue

        frames_files = sorted(os.listdir(vdir))[:200]
        frames = []
        for fn in frames_files:
            f = cv2.imread(os.path.join(vdir, fn))
            if f is not None:
                frames.append(f)

        if len(frames) < args.window:
            print(f"  {vid}: only {len(frames)} frames, skip")
            continue

        n_frames = len(frames)
        clip_len = 4  # same as training

        # Store per-frame crops and scores
        left_crops = [None] * n_frames
        right_crops = [None] * n_frames
        left_scores = [0.0] * n_frames
        right_scores = [0.0] * n_frames

        print(f"  {vid}: {n_frames} frames, inferring...")

        for start in range(0, n_frames - clip_len + 1, clip_len):
            clip = frames[start:start + clip_len]
            tensors = []
            for f in clip:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(to_tensor(rgb))
            x = torch.stack(tensors, dim=0).unsqueeze(0).to(args.device)

            with torch.no_grad():
                out, _, _, _ = model(x, test=True)

            pred_eye = out['pred_eye_boxes']  # [T, N, 4] cxcywh normalized
            T_out, N, _ = pred_eye.shape

            # Pick best query per frame by max extent (eye boxes should be small-medium)
            eye_norm = pred_eye[:, 0, :]  # use first query for simplicity
            eye_pixel = eye_norm.clone()
            eye_pixel[:, 0] *= 1280
            eye_pixel[:, 1] *= 720
            eye_pixel[:, 2] *= 1280
            eye_pixel[:, 3] *= 720
            eye_pixel = eye_pixel.cpu().numpy()

            for t in range(T_out):
                idx = start + t
                if idx >= n_frames:
                    continue

                eye_xyxy = cxcywh_to_xyxy(eye_pixel[t])
                ex1, ey1, ex2, ey2 = [max(0, float(v)) for v in eye_xyxy]
                ew = max(ex2 - ex1, 2)
                eh = max(ey2 - ey1, 2)

                # Split combined eye_bbox into left/right halves
                left_xyxy = np.array([ex1 + ew * 0.5, ey1, ex2, ey2])
                right_xyxy = np.array([ex1, ey1, ex1 + ew * 0.5, ey2])

                lc = extractor.extract(frames[idx], left_xyxy)
                rc = extractor.extract(frames[idx], right_xyxy)

                # Fallback: if blank, use GT landmarks from train_mini
                if lc.max() == lc.min():
                    lc = None
                if rc.max() == rc.min():
                    rc = None

                if lc is not None:
                    left_crops[idx] = lc
                    left_scores[idx] = extractor.score_sharpness(lc)
                if rc is not None:
                    right_crops[idx] = rc
                    right_scores[idx] = extractor.score_sharpness(rc)

        # Find best 13-frame window
        def _best_window(crops_list, scores_list):
            valid = [(i, c, s) for i, (c, s) in enumerate(zip(crops_list, scores_list)) if c is not None]
            if len(valid) < args.window:
                if not valid:
                    return 0, np.zeros((args.window, 64, 64, 3), dtype=np.uint8), 0.0
                result = [v[1] for v in valid]
                while len(result) < args.window:
                    result.insert(0, result[0])
                    result.append(result[-1])
                return valid[0][0], np.stack(result[:args.window]), float(np.mean([v[2] for v in valid]))
            best_start, best_score, best_crops = 0, -1.0, None
            for s in range(len(valid) - args.window + 1):
                w = valid[s:s + args.window]
                avg_s = np.mean([x[2] for x in w])
                if avg_s > best_score:
                    best_score = avg_s
                    best_start = valid[s][0]
                    best_crops = np.stack([x[1] for x in w])
            return best_start, best_crops, float(best_score)

        l_start, l_crops, l_score = _best_window(left_crops, left_scores)
        r_start, r_crops, r_score = _best_window(right_crops, right_scores)

        vid_out = os.path.join(args.output, str(vid))
        for eye_idx, (crops, score, start, side) in enumerate([
            (l_crops, l_score, l_start, 'left'),
            (r_crops, r_score, r_start, 'right')
        ]):
            eye_name = f'eye_{eye_idx + 1:03d}'
            img_dir = os.path.join(vid_out, eye_name)
            os.makedirs(img_dir, exist_ok=True)

            for i in range(args.window):
                cv2.imwrite(os.path.join(img_dir, f'frame_{i:02d}.png'), crops[i])

            extractor.save_grid([crops[i] for i in range(args.window)],
                               os.path.join(img_dir, 'grid.png'), cols=args.window)
            np.save(os.path.join(vid_out, f'{eye_name}.npy'), crops.astype(np.uint8))

            total_eyes += 1
            print(f"    {eye_name} ({side}): start={start}, sharpness={score:.1f}")

    print(f"\nDone! {total_eyes} eyes -> {args.output}")


if __name__ == '__main__':
    main()
