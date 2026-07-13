"""Prepare JPEG training data from train/ and val/ directories.

Converts raw video.mp4 + annotation_WFLW.json into:
  1. Raw frames (640x360 JPEG, quality=90)
  2. COCO-style annotation JSON (same format as mpeblink v2)

Overwrites mpeblink_lite/ with new data.

Usage:
  python newtest/prepare_jpeg_data.py
  python newtest/prepare_jpeg_data.py --train-count 150 --val-count 50
"""

import os, sys, json, time, shutil, argparse
from tqdm import tqdm
import cv2

JPEG_QUALITY = 90

def process_videos(source_dir, output_dir, video_ids, split_name,
                   target_width=1280, target_height=720):  # 1280x720 for better eye detail
    """Process video list into JPEG rawframes + COCO annotations.

    Args:
        source_dir: e.g. 'E:/.../mpeblink2.0/train'
        output_dir: e.g. 'E:/.../mpeblink2.0/mpeblink_lite'
        video_ids: list of int video IDs
        split_name: 'train' or 'val'
    """
    rawframes_dir = os.path.join(output_dir, f'{split_name}_rawframes')
    # Clear old frames for this split
    if os.path.exists(rawframes_dir):
        shutil.rmtree(rawframes_dir)
    os.makedirs(rawframes_dir, exist_ok=True)

    dataset = {
        'info': {
            'description': f'MPEblink Lite {split_name} (JPEG)',
            'version': '2.0', 'year': '2026',
            'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'licenses': {'licenses': 'research only'},
        'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person_face'}],
        'videos': [],
        'annotations': [],
    }

    anno_id = 1

    for vid in tqdm(video_ids, desc=f'Processing {split_name}'):
        video_dir = os.path.join(source_dir, str(vid))
        video_path = os.path.join(video_dir, 'video.mp4')
        anno_path = os.path.join(video_dir, 'annotation_WFLW.json')

        if not os.path.exists(video_path) or not os.path.exists(anno_path):
            print(f'  [SKIP] {vid}: missing video or annotation')
            continue

        with open(anno_path, 'r') as f:
            origin_anno = json.load(f)

        src_h = origin_anno.pop('height', 1040)
        src_w = origin_anno.pop('width', 1920)
        length = origin_anno.pop('length', 0)

        scale_w = target_width / src_w
        scale_h = target_height / src_h

        # Extract frames as JPEG
        save_dir = os.path.join(rawframes_dir, str(vid))
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        file_names = []
        img_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (target_width, target_height))
            fname = f'{img_idx:05d}.jpg'
            cv2.imwrite(os.path.join(save_dir, fname),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            file_names.append(f'{vid}/{fname}')
            img_idx += 1
        cap.release()

        actual_len = min(len(file_names), length)
        if actual_len < length:
            print(f'  [WARN] {vid}: expected {length} frames, got {actual_len}')

        # Video record
        dataset['videos'].append({
            'height': target_height, 'width': target_width,
            'length': actual_len,
            'file_names': file_names[:actual_len],
            'id': vid,
        })

        # Process each person
        person_keys = [k for k in origin_anno.keys() if k.startswith('person')]
        for pk in person_keys:
            pd = origin_anno[pk]
            anno_id += 1

            # Scale bboxes (xywh pixel)
            new_bboxes = []
            for i in range(actual_len):
                bbox = pd['bbox'][i] if i < len(pd['bbox']) else None
                if bbox is None:
                    new_bboxes.append(None)
                else:
                    new_bboxes.append([
                        bbox[0] * scale_w, bbox[1] * scale_h,
                        bbox[2] * scale_w, bbox[3] * scale_h])

            # Scale landmarks (68-point)
            lm_key = 'landmark'
            if lm_key not in pd:
                lm_key = 'landmark_WFLW'
            new_landmarks = []
            if lm_key in pd and pd[lm_key]:
                for i in range(actual_len):
                    if i >= len(pd[lm_key]) or pd[lm_key][i] is None:
                        new_landmarks.append(None)
                    else:
                        new_landmarks.append([
                            [p[0] * scale_w, p[1] * scale_h]
                            for p in pd[lm_key][i]])

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
                'height': target_height, 'width': target_width,
                'length': 1, 'category_id': 1,
                'bboxes': new_bboxes,
                'landmark': new_landmarks,
                'blinks': blink_intervals,
                'blinks_binary': blink_binary,
                'video_id': vid,
                'id': anno_id,
            })

    # Save annotation JSON
    anno_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    json_path = os.path.join(anno_dir, f'{split_name}.json')
    with open(json_path, 'w') as f:
        json.dump(dataset, f)
    print(f'  Saved: {json_path} ({len(dataset["videos"])} videos, '
          f'{len(dataset["annotations"])} persons)')

    # Calculate size
    total_size = 0
    for root, dirs, files in os.walk(rawframes_dir):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
    print(f'  Size: {total_size / 1024**3:.2f} GB ({total_size / 1024**2:.0f} MB)')

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare JPEG training data')
    parser.add_argument('--root', default='E:/documents/mssb code/mpeblink2.0',
                        help='mpeblink2.0 root directory')
    parser.add_argument('--output', default=None,
                        help='Output dir (default: root/mpeblink_lite)')
    parser.add_argument('--train-count', type=int, default=150,
                        help='Number of train videos (from train/1 to train/N)')
    parser.add_argument('--val-count', type=int, default=50,
                        help='Number of val videos (from val/1 to val/N)')
    parser.add_argument('--target-width', type=int, default=1280)   # 1280x720
    parser.add_argument('--target-height', type=int, default=720)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    args = parser.parse_args()

    global JPEG_QUALITY
    JPEG_QUALITY = args.jpeg_quality

    if args.output is None:
        args.output = os.path.join(args.root, 'mpeblink_lite')

    print('=' * 60)
    print('Prepare JPEG Training Data')
    print('=' * 60)
    print(f'  Source:        {args.root}')
    print(f'  Output:        {args.output}')
    print(f'  Train videos:  1-{args.train_count} (from train/)')
    print(f'  Val videos:    1-{args.val_count} (from val/)')
    print(f'  Target size:   {args.target_width}x{args.target_height}')
    print(f'  JPEG quality:  {args.jpeg_quality}')
    print(f'  Format:        .jpg (JPEG)')
    print()

    # ---- Train ----
    train_ids = list(range(1, args.train_count + 1))
    source_train = os.path.join(args.root, 'train')
    print('=' * 60)
    process_videos(source_train, args.output, train_ids, 'train',
                   args.target_width, args.target_height)

    # ---- Val ----
    val_ids = list(range(1, args.val_count + 1))
    source_val = os.path.join(args.root, 'val')
    print()
    process_videos(source_val, args.output, val_ids, 'val',
                   args.target_width, args.target_height)

    # ---- Summary ----
    train_dir = os.path.join(args.output, 'train_rawframes')
    val_dir = os.path.join(args.output, 'val_rawframes')
    train_size = sum(os.path.getsize(os.path.join(r,f))
                     for r,_,fs in os.walk(train_dir) for f in fs)
    val_size = sum(os.path.getsize(os.path.join(r,f))
                   for r,_,fs in os.walk(val_dir) for f in fs)

    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)
    print(f'  train_rawframes: {train_size/1024**3:.2f} GB')
    print(f'  val_rawframes:   {val_size/1024**3:.2f} GB')
    print(f'  Total frames:    {(train_size+val_size)/1024**3:.2f} GB')
    print(f'  Annotations:')
    print(f'    {args.output}/annotations/train.json')
    print(f'    {args.output}/annotations/val.json')


if __name__ == '__main__':
    main()
