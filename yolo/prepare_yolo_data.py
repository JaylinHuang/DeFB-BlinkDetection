"""从WFLW 68点landmark生成YOLO格式眼部检测标注.

原理: 利用现有GT landmark计算左右眼bbox → YOLO class+xywh归一化标注
左眼: landmark[42:48], 右眼: landmark[36:42]
YOLO格式: class cx cy w h (归一化0-1, class 0=left_eye, 1=right_eye)

Usage:
  python alternative_yolo/prepare_yolo_data.py
  python alternative_yolo/prepare_yolo_data.py --stride 3
"""

import sys, os, json, argparse
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "alternative_yolo", "datasets", "eye_detection")

# 68-point WFLW eye landmark indices
LEFT_EYE_IDX = list(range(42, 48))   # 42,43,44,45,46,47
RIGHT_EYE_IDX = list(range(36, 42))  # 36,37,38,39,40,41
PAD_RATIO = 0.25   # 25% padding around eye landmarks


def landmarks_to_eye_bbox(lm_68, eye_indices, img_w, img_h):
    """从68点landmark提取单眼bbox, 返回YOLO格式 (cx,cy,w,h) 归一化."""
    pts = []
    for idx in eye_indices:
        if idx < len(lm_68) and lm_68[idx] is not None:
            x, y = lm_68[idx][0], lm_68[idx][1]
            if x > 0 and y > 0:
                pts.append((x, y))

    if len(pts) < 3:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    ew, eh = x2 - x1, y2 - y1

    if ew <= 1 or eh <= 1:
        return None

    # Add padding
    px1 = max(0, int(x1 - PAD_RATIO * ew))
    py1 = max(0, int(y1 - PAD_RATIO * eh))
    px2 = min(img_w, int(x2 + PAD_RATIO * ew))
    py2 = min(img_h, int(y2 + PAD_RATIO * eh))

    # YOLO format: center_x, center_y, width, height (normalized)
    cx = ((px1 + px2) / 2.0) / img_w
    cy = ((py1 + py2) / 2.0) / img_h
    w = (px2 - px1) / img_w
    h = (py2 - py1) / img_h

    return (cx, cy, w, h)


def process_annotations(ann_file, img_fold, split, stride=4):
    """处理标注文件, 为每一帧生成YOLO标签文件.

    Args:
        ann_file: COCO-format annotation JSON path
        img_fold: rawframes 目录 (train_rawframes 或 val_rawframes)
        split: 'train' 或 'val'
        stride: 每隔N帧取1帧 (减少数据量)
        copy_images: True=复制图片, False=记录源路径(节省空间)
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)

    img_w, img_h = 1280, 720

    images_dir = os.path.join(OUT_DIR, "images", split)
    labels_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Accumulate labels per image (multiple persons may share same frame)
    image_labels = {}  # key: (vid, frame_file) → [(class, cx, cy, w, h), ...]

    n_eyes, n_frames, n_skipped = 0, 0, 0

    for anno in data['annotations']:
        vid = anno['video_id']
        landmarks_all = anno['landmark']
        if not landmarks_all:
            continue

        file_names = None
        for vrec in data.get('videos', []):
            if vrec['id'] == vid:
                file_names = vrec.get('file_names', [])
                break

        if not file_names:
            # Fallback: list directory
            vdir = os.path.join(img_fold, str(vid))
            if os.path.isdir(vdir):
                file_names = sorted(os.listdir(vdir))
            else:
                continue

        total_frames = min(len(landmarks_all), len(file_names))

        for i in range(0, total_frames, stride):
            lm = landmarks_all[i]
            if lm is None:
                n_skipped += 1
                continue

            fn = file_names[i]
            img_path = os.path.join(img_fold, fn)  # e.g. train_rawframes/1/00000.jpg

            # Check source exists
            full_path = os.path.join(img_fold, str(vid), os.path.basename(fn)) \
                if not os.path.isabs(fn) else fn
            if not os.path.exists(full_path):
                full_path = os.path.join(img_fold, fn)

            key = (vid, os.path.basename(fn))
            if key not in image_labels:
                image_labels[key] = []
                n_frames += 1

            # Left eye
            left_bbox = landmarks_to_eye_bbox(lm, LEFT_EYE_IDX, img_w, img_h)
            if left_bbox:
                image_labels[key].append((0,) + left_bbox)
                n_eyes += 1

            # Right eye
            right_bbox = landmarks_to_eye_bbox(lm, RIGHT_EYE_IDX, img_w, img_h)
            if right_bbox:
                image_labels[key].append((1,) + right_bbox)
                n_eyes += 1

    # Write YOLO label files
    image_list_path = os.path.join(OUT_DIR, f"{split}.txt")
    with open(image_list_path, 'w') as img_list:

        for (vid, fname), labels in sorted(image_labels.items()):
            if not labels:
                continue

            # Source image full path
            src_path = os.path.join(img_fold, str(vid), fname)
            if not os.path.exists(src_path):
                continue

            # 创建硬链接（零拷贝，瞬间完成，不占额外空间，不需管理员权限）
            dst_name = f"{vid}_{fname}"
            dst_path = os.path.join(images_dir, dst_name)

            if not os.path.exists(dst_path):
                try:
                    os.link(os.path.abspath(src_path), dst_path)
                except OSError:
                    # 硬链接失败时回退到复制
                    import shutil
                    shutil.copy2(src_path, dst_path)

            # Write label file (YOLO format: class cx cy w h, space-separated)
            label_name = dst_name.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            with open(label_path, 'w') as lf:
                for lbl in labels:
                    lf.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            img_list.write(f"{dst_path}\n")

    # Summary
    print(f"  [{split}] {len(image_labels)} unique frames, {n_eyes} eye labels")
    print(f"       skipped {n_skipped} frames (no landmarks), stride={stride}")
    return len(image_labels), n_eyes


def create_data_yaml():
    """生成 YOLO 训练用的 data.yaml."""
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    yaml_content = f"""# YOLOv8 Eye Detection Dataset
# Auto-generated from WFLW 68-point landmarks

path: {OUT_DIR}
train: {os.path.join(OUT_DIR, 'train.txt')}
val: {os.path.join(OUT_DIR, 'val.txt')}

nc: 2
names:
  0: left_eye
  1: right_eye
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n  data.yaml -> {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO eye detection dataset")
    parser.add_argument('--stride', type=int, default=3,
                        help='Sample every Nth frame (default: 3)')
    parser.add_argument('--rawframes', type=str, required=True,
                        help='Root dir containing train_rawframes/ and val_rawframes/')
    args = parser.parse_args()

    raw_root = args.rawframes if args.rawframes else '.'
    train_ann = os.path.join(raw_root, 'annotations', 'train.json')
    val_ann = os.path.join(raw_root, 'annotations', 'val.json')

    print("=" * 60)
    print("YOLO Eye Detection Dataset Preparation")
    print("=" * 60)
    print(f"  Stride: every {args.stride}rd frame")
    print(f"  Data root: {raw_root}")
    print(f"  Output: {OUT_DIR}")
    print(f"  68-pt landmarks: left eye [42:48], right eye [36:42]")
    print(f"  Padding: {PAD_RATIO*100:.0f}% around eye landmarks")

    # Process train
    print(f"\n[Train] {train_ann}")
    n_train_img, n_train_eyes = process_annotations(
        train_ann,
        os.path.join(raw_root, 'train_rawframes'),
        'train',
        stride=args.stride,
    )

    # Process val
    print(f"\n[Val] {val_ann}")
    n_val_img, n_val_eyes = process_annotations(
        val_ann,
        os.path.join(raw_root, 'val_rawframes'),
        'val',
        stride=args.stride,
    )

    # Create data.yaml
    create_data_yaml()

    print(f"\nDone!")
    print(f"  Train: {n_train_img} images, {n_train_eyes} eye labels")
    print(f"  Val:   {n_val_img} images, {n_val_eyes} eye labels")
    print(f"  Total: {n_train_img + n_val_img} images, {n_train_eyes + n_val_eyes} eyes")
    print(f"\n  Dataset dir: {OUT_DIR}")
    print(f"  Next: python alternative_yolo/train_yolo_eye.py")


if __name__ == '__main__':
    main()
