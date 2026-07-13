"""训练YOLOv8-Nano眼部检测模型.

基于 prepare_yolo_data.py 生成的YOLO格式数据集训练。
模型: YOLOv8n (nano, ~3.2M params), 检测左眼/右眼两个类别。

内存估算:
  默认 (--server): batch=4, imgsz=416, workers=2 → ~8-12GB RAM
  本地 (no flag):  batch=8, imgsz=640, workers=4 → ~20-25GB RAM

Usage:
  python alternative_yolo/train_yolo_eye.py                    # 本地训练
  python alternative_yolo/train_yolo_eye.py --server           # 服务器低内存模式
  python alternative_yolo/train_yolo_eye.py --epochs 100 --batch 4 --imgsz 416
"""

import os, sys, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Nano for eye detection")
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (default: 8 local, 4 server)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Training image size (default: 640 local, 416 server)')
    parser.add_argument('--workers', type=int, default=None,
                        help='DataLoader workers (default: 4 local, 2 server)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (cuda:0 / cpu)')
    parser.add_argument('--server', action='store_true',
                        help='Server low-memory mode: batch=4, imgsz=416, workers=2, '
                             'no mosaic/mixup (~8-12GB RAM)')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    # ---- Memory mode ----
    if args.server:
        batch = args.batch or 4
        imgsz = args.imgsz or 416
        workers = args.workers or 2
        # Disable memory-heavy augmentations on server
        mosaic = 0.0
        mixup = 0.0
        copy_paste = 0.0
        close_mosaic = 0
        cache = False
        print("[Server Mode] Low memory config (~8-12GB RAM)")
    else:
        batch = args.batch or 8
        imgsz = args.imgsz or 640
        workers = args.workers or 4
        mosaic = 0.5
        mixup = 0.1
        copy_paste = 0.1
        close_mosaic = 10
        cache = False
        print("[Local Mode] Standard config (~20-25GB RAM)")

    data_yaml = os.path.join(ROOT, 'alternative_yolo', 'datasets', 'eye_detection', 'data.yaml')

    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found.")
        print("Run prepare_yolo_data.py first!")
        return

    output_dir = os.path.join(ROOT, 'output', 'yolo_eye_training')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("YOLOv8-Nano Eye Detection Training")
    print("=" * 60)
    print(f"  Model:      yolov8n.pt (nano, ~3.2M params)")
    print(f"  Data:       {data_yaml}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Workers:    {workers}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {output_dir}")

    # Load pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=args.lr,
        device=args.device,
        workers=workers,
        project=output_dir,
        name='eye_detect',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=close_mosaic,
        cache=cache,
        val=True,
        save=True,
        save_period=10,
        # Data augmentation (memory-safe on server)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    # Evaluate on val set
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    metrics = model.val(data=data_yaml, device=args.device)
    print(f"\n  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")

    # Export best model path
    best_pt = os.path.join(output_dir, 'eye_detect', 'weights', 'best.pt')
    print(f"\n  Best model: {best_pt}")
    print(f"  To extract eyes:")
    print(f"    python alternative_yolo/extract_eyes.py --model {best_pt}")


if __name__ == '__main__':
    main()
