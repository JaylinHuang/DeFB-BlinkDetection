"""Preprocess first N test folders from mpeblink2.0 into training format.

Converts raw video.mp4 + annotation_WFLW.json into:
  1. Raw frames (640×360 PNGs)
  2. COCO-style annotation JSON (same format as mpeblink v2)

Usage:
  python newtest/prepare_data.py --root "E:/documents/mssb code/mpeblink2.0" --num-videos 50 --train-ratio 0.8
"""

import os
import sys
import json
import time
import cv2
import argparse
from tqdm import tqdm


def process_videos(root, output_dir, video_ids, split_name, target_width=640, target_height=360):
    """Process a list of video IDs into rawframes + COCO annotations."""

    rawframes_dir = os.path.join(output_dir, f'{split_name}_rawframes')
    os.makedirs(rawframes_dir, exist_ok=True)

    dataset = {}

    # Info
    dataset['info'] = {
        'description': 'MPEblink Lite Test Dataset',
        'url': '',
        'version': '1.0',
        'year': '2026',
        'contributor': '',
        'date_created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    dataset['licenses'] = {'licenses': 'research only'}
    dataset['categories'] = [
        {'supercategory': 'object', 'id': 1, 'name': 'person_face'}
    ]

    videos = []
    annotations = []
    anno_id = 1

    for video_id in tqdm(video_ids, desc=f'Processing {split_name}'):
        video_dir = os.path.join(root, 'test', str(video_id))
        video_path = os.path.join(video_dir, 'video.mp4')
        anno_path = os.path.join(video_dir, 'annotation_WFLW.json')

        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_id}: no video.mp4")
            continue
        if not os.path.exists(anno_path):
            print(f"  [SKIP] {video_id}: no annotation_WFLW.json")
            continue

        # Load annotations
        with open(anno_path, 'r') as f:
            origin_anno = json.load(f)

        height = origin_anno.pop('height')
        width = origin_anno.pop('width')
        length = origin_anno.pop('length')

        scale_w = target_width / width
        scale_h = target_height / height

        # Extract frames
        save_dir = os.path.join(rawframes_dir, str(video_id))
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        file_names = []
        img_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            relative_path = f'{video_id}/{str(img_index).rjust(5,"0")}.png'
            frame = cv2.resize(frame, (target_width, target_height))
            cv2.imwrite(os.path.join(rawframes_dir, relative_path), frame)
            file_names.append(relative_path)
            img_index += 1
        cap.release()

        actual_length = len(file_names)
        if actual_length < length:
            print(f"  [WARN] Video {video_id}: expected {length} frames, got {actual_length}")
            length = actual_length

        # Build video record
        video_rec = {
            'height': target_height,
            'width': target_width,
            'length': length,
            'file_names': file_names,
            'id': video_id
        }
        videos.append(video_rec)

        # Process each person
        person_keys = [k for k in origin_anno.keys() if k.startswith('person')]

        for person_key in person_keys:
            person_data = origin_anno[person_key]
            anno = {
                'height': target_height,
                'width': target_width,
                'length': 1,
                'category_id': 1
            }

            # Scale bboxes
            new_bboxes = []
            for i in range(length):
                bbox = person_data['bbox'][i] if i < len(person_data['bbox']) else None
                if bbox is None:
                    new_bboxes.append(None)
                else:
                    new_bboxes.append([
                        bbox[0] * scale_w,
                        bbox[1] * scale_h,
                        bbox[2] * scale_w,
                        bbox[3] * scale_h
                    ])

            # Scale landmarks
            new_landmarks = []
            if 'landmark' in person_data and person_data['landmark']:
                for i in range(length):
                    if i >= len(person_data['landmark']):
                        new_landmarks.append(None)
                        continue
                    frame_lm = person_data['landmark'][i]
                    if frame_lm is None:
                        new_landmarks.append(None)
                    else:
                        scaled_lm = []
                        for lm in frame_lm:
                            # WFLW format: landmark has 3 elements (x, y, confidence?)
                            # Standard format expects 2 elements (x, y)
                            scaled_lm.append([lm[0] * scale_w, lm[1] * scale_h])
                        new_landmarks.append(scaled_lm)

            # Build binary blink labels
            blink_intervals = person_data.get('blink', [])
            blink_binary = []
            for i in range(length):
                in_blink = 0
                for blink in blink_intervals:
                    # WFLW format: [start, end, ?] — use first 2 elements
                    if i >= blink[0] and i <= blink[1]:
                        in_blink = 1
                        break
                blink_binary.append(in_blink)

            anno['bboxes'] = new_bboxes
            anno['landmark'] = new_landmarks
            anno['blinks'] = blink_intervals
            anno['blinks_binary'] = blink_binary
            anno['video_id'] = video_id
            anno['id'] = anno_id
            anno_id += 1

            annotations.append(anno)

    dataset['videos'] = videos
    dataset['annotations'] = annotations

    # Write JSON
    anno_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    json_path = os.path.join(anno_dir, f'{split_name}.json')
    with open(json_path, 'w') as f:
        json.dump(dataset, f)

    print(f"  Saved: {json_path} ({len(videos)} videos, {len(annotations)} persons)")
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess mpeblink test data for training')
    parser.add_argument('--root', default='E:/documents/mssb code/mpeblink2.0',
                        help='Path to mpeblink2.0 root')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: root/mpeblink_lite)')
    parser.add_argument('--num-videos', type=int, default=50,
                        help='Number of test videos to use')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train split ratio')
    parser.add_argument('--target-width', type=int, default=640)
    parser.add_argument('--target-height', type=int, default=360)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.root, 'mpeblink_lite')

    print(f"Root: {args.root}")
    print(f"Output: {args.output}")
    print(f"Videos: {args.num_videos}, Train ratio: {args.train_ratio}")
    print(f"Target size: {args.target_width}×{args.target_height}")

    # Get all available test video IDs
    test_dir = os.path.join(args.root, 'test')
    all_ids = []
    for name in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, name)):
            try:
                all_ids.append(int(name))
            except ValueError:
                pass
    all_ids.sort()
    print(f"Found {len(all_ids)} test videos total")

    # Take first N
    selected_ids = all_ids[:args.num_videos]
    print(f"Using first {len(selected_ids)}: {selected_ids[:5]}...{selected_ids[-3:]}")

    # Split
    split_idx = int(len(selected_ids) * args.train_ratio)
    train_ids = selected_ids[:split_idx]
    val_ids = selected_ids[split_idx:]

    print(f"Train: {len(train_ids)} videos (ids: {train_ids[0]}...{train_ids[-1]})")
    print(f"Val:   {len(val_ids)} videos (ids: {val_ids[0]}...{val_ids[-1]})")

    # Process train
    print("\n" + "=" * 60)
    process_videos(args.root, args.output, train_ids, 'train',
                   args.target_width, args.target_height)

    # Process val
    print()
    process_videos(args.root, args.output, val_ids, 'val',
                   args.target_width, args.target_height)

    print("\nDone! Dataset ready at:", args.output)
    print("  Train:", os.path.join(args.output, 'annotations', 'train.json'))
    print("  Val:  ", os.path.join(args.output, 'annotations', 'val.json'))


if __name__ == '__main__':
    main()
