"""Update eye_samples with model-predicted crops using trained checkpoint.

Usage:
  python newtest/update_eye_samples_pred.py
"""

import sys, os, json, time, shutil
import cv2, numpy as np
import torch
from torchvision import transforms as Tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from newtest.track_model_lite import build_lite_model
from newtest.eye_crop_extractor import EyeCropExtractor, cxcywh_to_xyxy

CKPT = 'output/train_v11_15/checkpoints/last.pth'
ANN = 'E:/documents/mssb code/mpeblink2.0/mpeblink_lite/annotations/train_v11_15.json'
RAW = 'E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes'
OUT = 'output/train_v11_15/eye_samples/v11_15'
WINDOW = 13
DEVICE = 'cuda:0'

def main():
    print('=' * 60)
    print('Model-Predicted Eye Crop Extraction')
    print('=' * 60)

    # Load model
    print('Loading model from', CKPT)
    model = build_lite_model(backbone_pretrained=False)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if 'model' in state:
        state = state['model']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    with open(ANN) as f:
        data = json.load(f)
    video_map = {v['id']: v for v in data['videos']}
    extractor = EyeCropExtractor()
    to_tensor = Tf.ToTensor()

    def run_model_on_frames(frames_bgr, face_bboxes, landmarks, clip_len=4):
        n_frames = len(frames_bgr)
        left_crops = [None] * n_frames
        right_crops = [None] * n_frames
        left_scores = [0.0] * n_frames
        right_scores = [0.0] * n_frames

        for start in range(0, n_frames - clip_len + 1, clip_len):
            clip = frames_bgr[start:start + clip_len]
            tensors = []
            for f in clip:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(to_tensor(rgb))
            x = torch.stack(tensors, dim=0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out, memory, head_query, eye_query = model(x, test=True)

            pred_eye = out['pred_eye_boxes']
            pred_logits = out['pred_logits']
            T_out, N, _ = pred_eye.shape

            scores = pred_logits.squeeze(-1).sigmoid()
            best_n = scores.mean(dim=0).argmax().item()

            eye_norm = pred_eye[:, best_n, :]
            eye_pixel = eye_norm.clone()
            eye_pixel[:, 0] *= 1280  # 1280x720 frame
            eye_pixel[:, 1] *= 720
            eye_pixel[:, 2] *= 1280
            eye_pixel[:, 3] *= 720
            eye_pixel = eye_pixel.cpu().numpy()

            for t in range(T_out):
                idx = start + t
                if idx >= n_frames:
                    continue

                # Model predicts ONE combined eye_bbox per person (both eyes)
                # Split into left/right halves geometrically
                eye_xyxy = cxcywh_to_xyxy(eye_pixel[t])
                ex1, ey1, ex2, ey2 = [float(v) for v in eye_xyxy]
                ew = max(ex2 - ex1, 2)
                eh = max(ey2 - ey1, 2)

                # Left half + Right half (each at least 1px)
                left_xyxy = np.array([ex1 + ew * 0.5, ey1, ex2, ey2])
                right_xyxy = np.array([ex1, ey1, ex1 + ew * 0.5, ey2])

                # Clamp to image bounds
                for arr in [left_xyxy, right_xyxy]:
                    arr[0] = max(0, min(arr[0], 639))
                    arr[1] = max(0, min(arr[1], 359))
                    arr[2] = max(arr[0] + 1, min(arr[2], 1280))  # clamp to 1280x720
                    arr[3] = max(arr[1] + 1, min(arr[3], 720))

                left_c = extractor.extract(frames_bgr[idx], left_xyxy)
                right_c = extractor.extract(frames_bgr[idx], right_xyxy)

                # Fallback: if model bbox gives blank, use GT face-first
                if left_c.max() == left_c.min():
                    fb = face_bboxes[idx]
                    lm = landmarks[idx] if idx < len(landmarks) else None
                    if fb and lm:
                        face_xyxy = np.array([fb[0], fb[1], fb[0]+fb[2], fb[1]+fb[3]])
                        left_c = extractor.extract_single_eye(frames_bgr[idx], face_xyxy, lm, 'left')
                if right_c.max() == right_c.min():
                    fb = face_bboxes[idx]
                    lm = landmarks[idx] if idx < len(landmarks) else None
                    if fb and lm:
                        face_xyxy = np.array([fb[0], fb[1], fb[0]+fb[2], fb[1]+fb[3]])
                        right_c = extractor.extract_single_eye(frames_bgr[idx], face_xyxy, lm, 'right')

                left_crops[idx] = left_c
                left_scores[idx] = extractor.score_sharpness(left_c)
                right_crops[idx] = right_c
                right_scores[idx] = extractor.score_sharpness(right_c)

        return left_crops, left_scores, right_crops, right_scores

    def find_best_window(crops_list, scores_list, window=WINDOW):
        valid = [(i, c, s) for i, (c, s) in enumerate(zip(crops_list, scores_list)) if c is not None]
        if len(valid) < window:
            result = [v[1] for v in valid]
            while len(result) < window:
                result.insert(0, result[0])
                result.append(result[-1])
            result = result[:window]
            avg_s = np.mean([v[2] for v in valid]) if valid else 0.0
            return 0, np.stack(result), float(avg_s)

        best_start, best_score, best_crops = 0, -1.0, None
        for start in range(len(valid) - window + 1):
            w = valid[start:start + window]
            avg_s = np.mean([x[2] for x in w])
            if avg_s > best_score:
                best_score = avg_s
                best_start = valid[start][0]
                best_crops = np.stack([x[1] for x in w])
        return best_start, best_crops, float(best_score)

    # Process
    manifest_eyes = []
    total_eyes = 0

    for anno in data['annotations']:
        vid = anno['video_id']
        pid = anno['id']
        face_bboxes = anno['bboxes']
        landmarks = anno['landmark']

        video_dir = os.path.join(RAW, str(vid))
        if not os.path.isdir(video_dir):
            continue

        file_names = video_map[vid].get('file_names', sorted(os.listdir(video_dir)))
        n_fr = min(len(file_names), len(landmarks), len(face_bboxes), 100)

        frames_bgr = []
        for i in range(n_fr):
            f = cv2.imread(os.path.join(video_dir, os.path.basename(file_names[i])))
            frames_bgr.append(f)

        print(f'Video {vid} Person {pid}: model inference on {n_fr} frames...')
        left_crops, left_scores, right_crops, right_scores = run_model_on_frames(
            frames_bgr, face_bboxes, landmarks)

        for side, crops_list, scores_list in [
            ('left', left_crops, left_scores),
            ('right', right_crops, right_scores)
        ]:
            start, best_crops, score = find_best_window(crops_list, scores_list)

            eye_idx = 1 if side == 'left' else 2
            eye_name = f'eye_{eye_idx:03d}'
            video_out = os.path.join(OUT, str(vid))
            img_dir = os.path.join(video_out, eye_name)

            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            os.makedirs(img_dir, exist_ok=True)

            for i in range(WINDOW):
                cv2.imwrite(os.path.join(img_dir, f'frame_{i:02d}.png'), best_crops[i])

            extractor.save_grid(
                [best_crops[i] for i in range(WINDOW)],
                os.path.join(img_dir, 'grid.png'), cols=WINDOW)

            np.save(os.path.join(video_out, f'{eye_name}.npy'), best_crops.astype(np.uint8))

            side_cn = 'left' if side == 'left' else 'right'
            with open(os.path.join(img_dir, 'info.txt'), 'w', encoding='utf-8') as f:
                f.write(f'eye_name: {eye_name}\n')
                f.write(f'video_id: {vid}\n')
                f.write(f'person_id: {pid}\n')
                f.write(f'side: {side} ({side_cn})\n')
                f.write(f'start_frame: {start}\n')
                f.write(f'window_size: {WINDOW}\n')
                f.write(f'avg_sharpness: {score:.2f}\n')
                f.write(f'source: model_prediction_5epochs_v11_15\n')

            total_eyes += 1
            print(f'  -> {vid}/{eye_name} ({side_cn}): start={start}, sharpness={score:.1f}')

    # Manifest
    manifest = {
        'total_eyes': total_eyes,
        'total_videos': len(data['videos']),
        'window_size': WINDOW,
        'source': 'model_prediction_trained_5epochs_on_v11_15',
        'checkpoint': CKPT,
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'videos': {},
    }
    for vid in sorted(set(a['video_id'] for a in data['annotations'])):
        manifest['videos'][str(vid)] = {'video_id': vid, 'eyes': []}
        vdir = os.path.join(OUT, str(vid))
        for ename in ['eye_001', 'eye_002']:
            ipath = os.path.join(vdir, ename, 'info.txt')
            if os.path.exists(ipath):
                with open(ipath, 'r') as f:
                    info = {}
                    for line in f.read().strip().split('\n'):
                        if ': ' in line:
                            k, v = line.split(': ', 1)
                            info[k] = v
                manifest['videos'][str(vid)]['eyes'].append({
                    'eye_name': ename,
                    'side': info.get('side', ''),
                    'person_id': int(info.get('person_id', 0)),
                    'start_frame': int(info.get('start_frame', 0)),
                    'avg_sharpness': float(info.get('avg_sharpness', 0)),
                    'source': 'model_prediction',
                })

    with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\nDone! {total_eyes} eyes -> {OUT}')
    print(f'Manifest: {OUT}/manifest.json')


if __name__ == '__main__':
    main()
