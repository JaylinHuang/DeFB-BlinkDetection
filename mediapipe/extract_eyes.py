"""MediaPipe Face Landmarker — 人眼定位备选方案 B

原理: MediaPipe Face Landmarker (478点3D landmark) → 精确眼周关键点 → 单眼裁剪
参数量: ~3M (Face Landmarker模型)
速度:   ~50 FPS (CPU), ~150 FPS (GPU)
特点:   零训练, 预训练模型开箱即用, 478点中眼周关键点精度高

输入: 1280x720视频帧
输出: 64x64单眼裁剪 (13帧窗口, eye_001/eye_002, PNG + .npy)
      与原TrackModelLite完全兼容

MediaPipe Face Landmarker 478点眼周关键点:
  左眼(iris): 468-472  plus contour: 33,133,155,154,153,145,144,163,7,173,157,158,159,160,161,246
  右眼(iris): 473-477  plus contour: 362,263,382,381,380,374,373,390,249,466,386,387,388,389,385,384

Usage:
  python alternative_mediapipe/extract_eyes.py
  python alternative_mediapipe/extract_eyes.py --videos 1-5 --output ./output/mp_eyes
"""

import sys, os, json, time, argparse
import cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from newtest.eye_crop_extractor import EyeCropExtractor

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")


# MediaPipe Face Landmarker eye contour indices (478-point format)
# These match the canonical MediaPipe face mesh topology
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 382, 381, 380, 374, 373, 390, 249]
# Iris centers for eye center reference
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def extract_eyes_mp(frame_bgr, detector, extractor):
    """用MediaPipe Face Landmarker提取单眼64x64裁剪."""
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    # Take first face
    face_lm = result.face_landmarks[0]
    lm_px = [(int(lm.x * W), int(lm.y * H)) for lm in face_lm]

    def get_eye_crop(indices):
        pts = np.array([lm_px[i] for i in indices if i < len(lm_px)])
        if len(pts) == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        # Add 25% padding
        pad = 0.25
        x1 = x1 - pad * w
        y1 = y1 - pad * h
        x2 = x2 + pad * w
        y2 = y2 + pad * h

        # Make square by expanding shorter side (preserve aspect ratio, no stretch)
        cw, ch = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(cw, ch)
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = int(cx + side / 2)
        y2 = int(cy + side / 2)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return cv2.resize(frame_bgr[y1:y2, x1:x2], (64, 64))

    left = get_eye_crop(LEFT_EYE_CONTOUR)
    right = get_eye_crop(RIGHT_EYE_CONTOUR)
    return left, right


def score_sharpness(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_best_window(frames_bgr, detector, extractor, window_size=13):
    n = len(frames_bgr)
    left_data, right_data = [(None, 0.0)] * n, [(None, 0.0)] * n

    for i, f in enumerate(frames_bgr):
        lc, rc = extract_eyes_mp(f, detector, extractor)
        if lc is not None:
            left_data[i] = (lc, score_sharpness(lc))
            right_data[i] = (rc, score_sharpness(rc))

    def _best(data, wsize):
        valid = [(i, c, s) for i, (c, s) in enumerate(data) if c is not None]
        if len(valid) < wsize:
            result = [v[1] for v in valid]
            while len(result) < wsize:
                result.insert(0, result[0] if result else np.zeros((64, 64, 3), dtype=np.uint8))
                result.append(result[-1])
            return 0, np.stack(result[:wsize]), (np.mean([v[2] for v in valid]) if valid else 0.0)

        best_start, best_score, best_crops = 0, -1.0, None
        for s in range(len(valid) - wsize + 1):
            w = valid[s:s + wsize]
            avg_s = np.mean([x[2] for x in w])
            if avg_s > best_score:
                best_score = avg_s
                best_start = valid[s][0]
                best_crops = np.stack([x[1] for x in w])
        return best_start, best_crops, best_score

    return _best(left_data, window_size), _best(right_data, window_size)


def process_video(video_dir, video_id, output_dir, detector, extractor, window_size=13):
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
        frames, detector, extractor, window_size)

    vid_out = os.path.join(output_dir, str(video_id))
    for eye_idx, (crops, score, start, side) in enumerate([
        (l_crops, l_score, l_start, 'left'),
        (r_crops, r_score, r_start, 'right')
    ]):
        eye_name = f'eye_{eye_idx+1:03d}'
        img_dir = os.path.join(vid_out, eye_name)
        os.makedirs(img_dir, exist_ok=True)
        for i in range(window_size):
            cv2.imwrite(os.path.join(img_dir, f'frame_{i:02d}.png'), crops[i])
        extractor.save_grid([crops[i] for i in range(window_size)],
                           os.path.join(img_dir, 'grid.png'), cols=window_size)
        np.save(os.path.join(vid_out, f'{eye_name}.npy'), crops.astype(np.uint8))
        print(f"    {eye_name} ({side}): start={start}, sharpness={score:.1f}")


def main():
    parser = argparse.ArgumentParser(description='MediaPipe Face Landmarker eye extraction')
    parser.add_argument('--rawframes', default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes')
    parser.add_argument('--output', default='./output/mediapipe_eyes')
    parser.add_argument('--videos', default=None)
    parser.add_argument('--window', type=int, default=13)
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("Please install: pip install mediapipe")
        return

    # Create detector with downloaded model
    model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        return
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

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
    print(f'MediaPipe Face Landmarker Eye Extraction')
    print(f'  Videos: {video_ids}')
    print(f'  Window: {args.window} frames')
    print(f'  Output: {args.output}')

    total = 0
    for vid in video_ids:
        vdir = os.path.join(args.rawframes, str(vid))
        if os.path.isdir(vdir):
            process_video(vdir, vid, args.output, detector, extractor, args.window)
            total += 2

    detector.close()
    print(f'\nDone! {total} eyes -> {args.output}')


if __name__ == '__main__':
    main()
