"""YOLOv8-Nano 眼部提取脚本.

用训练好的YOLOv8-Nano模型检测视频帧中的左眼/右眼, 输出64x64裁剪.
与原TrackModelLite格式完全兼容: {video_id}/eye_001/ (left), eye_002/ (right),
每个eye文件夹含13帧PNG + .npy数组 [13,64,64,3] uint8.

Usage:
  python alternative_yolo/extract_eyes.py
  python alternative_yolo/extract_eyes.py --videos 1-10 --model output/yolo_eye_training/eye_detect/weights/best.pt
"""

import sys, os, json, time, argparse
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO
from newtest.eye_crop_extractor import EyeCropExtractor


def extract_eyes_yolo(frame_bgr, model, conf=0.3, iou=0.5):
    """YOLO推理 → 提取左眼/右眼64x64裁剪."""
    H, W = frame_bgr.shape[:2]
    results = model(frame_bgr, conf=conf, iou=iou, verbose=False)

    left_crop, right_crop = None, None
    left_conf, right_conf = 0.0, 0.0

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, cls_id in enumerate(boxes.cls):
            cls_int = int(cls_id.item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_val = float(boxes.conf[i].item())

            x1, y1, x2, y2 = [max(0, int(float(v))) for v in xyxy]
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Make square by expanding shorter side (preserve aspect ratio, no stretch)
            cw, ch = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = max(cw, ch)
            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(W, int(cx + side / 2))
            y2 = min(H, int(cy + side / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (64, 64),
                            interpolation=cv2.INTER_LINEAR)

            if cls_int == 0 and conf_val > left_conf:  # left_eye
                left_crop = crop
                left_conf = conf_val
            elif cls_int == 1 and conf_val > right_conf:  # right_eye
                right_crop = crop
                right_conf = conf_val

    return left_crop, right_crop


def score_sharpness(crop):
    """Laplacian方差清晰度评分."""
    if crop is None:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_best_window(frames_bgr, model, window_size=13, conf=0.3):
    """滑动窗口选择最佳13帧连续序列."""
    n = len(frames_bgr)
    left_data = [(None, 0.0)] * n
    right_data = [(None, 0.0)] * n

    for i, f in enumerate(frames_bgr):
        lc, rc = extract_eyes_yolo(f, model, conf=conf)
        if lc is not None:
            left_data[i] = (lc, score_sharpness(lc))
        if rc is not None:
            right_data[i] = (rc, score_sharpness(rc))

    def _best(data, wsize):
        valid = [(i, c, s) for i, (c, s) in enumerate(data) if c is not None]
        if not valid or len(valid) < wsize:
            # Not enough valid frames
            if not valid:
                return 0, np.zeros((wsize, 64, 64, 3), dtype=np.uint8), 0.0
            result = [v[1] for v in valid]
            while len(result) < wsize:
                result.insert(0, result[0])
                result.append(result[-1])
            avg_s = np.mean([v[2] for v in valid])
            return valid[0][0], np.stack(result[:wsize]), float(avg_s)

        best_start, best_score, best_crops = 0, -1.0, None
        for s in range(len(valid) - wsize + 1):
            w = valid[s:s + wsize]
            avg_s = np.mean([x[2] for x in w])
            if avg_s > best_score:
                best_score = avg_s
                best_start = valid[s][0]
                best_crops = np.stack([x[1] for x in w])
        return best_start, best_crops, float(best_score)

    return _best(left_data, window_size), _best(right_data, window_size)


def process_video(video_dir, video_id, output_dir, model, extractor, window_size=13, conf=0.3):
    """处理单个视频, 提取最佳眼部窗口."""
    frame_files = sorted(os.listdir(video_dir))
    frames = []
    for fn in frame_files[:200]:
        f = cv2.imread(os.path.join(video_dir, fn))
        if f is not None:
            frames.append(f)

    if len(frames) < window_size:
        print(f"  {video_id}: only {len(frames)} frames, skip")
        return

    print(f"  {video_id}: {len(frames)} frames, scanning...")
    (l_start, l_crops, l_score), (r_start, r_crops, r_score) = find_best_window(
        frames, model, window_size, conf)

    vid_out = os.path.join(output_dir, str(video_id))
    for eye_idx, (crops, score, start, side) in enumerate([
        (l_crops, l_score, l_start, 'left'),
        (r_crops, r_score, r_start, 'right')
    ]):
        eye_name = f'eye_{eye_idx + 1:03d}'
        img_dir = os.path.join(vid_out, eye_name)
        os.makedirs(img_dir, exist_ok=True)

        for i in range(window_size):
            cv2.imwrite(os.path.join(img_dir, f'frame_{i:02d}.png'), crops[i])

        # Grid preview
        extractor.save_grid([crops[i] for i in range(window_size)],
                           os.path.join(img_dir, 'grid.png'), cols=window_size)

        # NumPy array
        np.save(os.path.join(vid_out, f'{eye_name}.npy'), crops.astype(np.uint8))

        print(f"    {eye_name} ({side}): start={start}, sharpness={score:.1f}")


def main():
    parser = argparse.ArgumentParser(description='YOLO-Nano eye extraction')
    parser.add_argument('--rawframes',
                        default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes')
    parser.add_argument('--output', default='./output/yolo_eyes')
    parser.add_argument('--videos', default=None,
                        help='e.g. 1-10 or 1,3,5')
    parser.add_argument('--window', type=int, default=13)
    parser.add_argument('--model', default=None,
                        help='Path to trained best.pt')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    # Load model
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(ROOT, 'output', 'yolo_eye_training',
                                  'eye_detect', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("Train first: python alternative_yolo/train_yolo_eye.py")
        print("Or specify: --model path/to/best.pt")
        return

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    model.to(args.device)

    # Video list
    if args.videos:
        if '-' in args.videos:
            a, b = args.videos.split('-')
            video_ids = list(range(int(a), int(b) + 1))
        else:
            video_ids = [int(x) for x in args.videos.split(',')]
    else:
        video_ids = sorted([int(d) for d in os.listdir(args.rawframes)
                           if os.path.isdir(os.path.join(args.rawframes, d))])[:10]

    extractor = EyeCropExtractor()
    print(f'YOLOv8-Nano Eye Extraction')
    print(f'  Model:   {model_path}')
    print(f'  Videos:  {video_ids}')
    print(f'  Window:  {args.window} frames')
    print(f'  Conf:    {args.conf}')
    print(f'  Output:  {args.output}')
    print()

    total = 0
    for vid in video_ids:
        vdir = os.path.join(args.rawframes, str(vid))
        if os.path.isdir(vdir):
            process_video(vdir, vid, args.output, model, extractor,
                         args.window, args.conf)
            total += 2

    print(f'\nDone! {total} eyes -> {args.output}')


if __name__ == '__main__':
    main()
