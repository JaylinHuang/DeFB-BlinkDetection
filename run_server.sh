#!/bin/bash
# DeFB — 服务器一键脚本
# 用法: bash run_server.sh /mnt/workspace/mpeblink_lite
set -e

DATA_ROOT="${1:-}"
if [ -z "${DATA_ROOT}" ]; then
    echo "用法: bash $0 /mnt/workspace/mpeblink_lite"
    exit 1
fi

cd "$(dirname "$0")"

echo "=== DeFB Pipeline ==="
echo "Data: ${DATA_ROOT}"

# ---- Step 0: 系统库（MediaPipe 需要） ----
echo "[0] apt install libGL..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libegl1-mesa 2>/dev/null || true

# ---- Step 0b: pip 依赖 ----
echo "[0b] pip install..."
python -m pip install ultralytics mediapipe opencv-python-headless tqdm tensorboard \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --default-timeout=600 -q 2>&1 | grep -v "WARNING\|Requirement\|already\|notice" || true

# ---- Step 0.5: MediaPipe 模型 ----
if [ ! -f mediapipe/face_landmarker.task ]; then
    echo "[0.5] 下载 MediaPipe 模型..."
    wget -q "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
        -O mediapipe/face_landmarker.task || true
fi

# ---- Step 1: YOLO 数据集 ----
if [ ! -f yolo/datasets/eye_detection/data.yaml ]; then
    echo "[1] 删除旧数据并重新生成 YOLO 数据集..."
    rm -rf yolo/datasets/eye_detection/
    python yolo/prepare_yolo_data.py \
        --stride 3 --rawframes "${DATA_ROOT}"
else
    echo "[1] YOLO 数据集已存在, 跳过"
fi

# ---- Step 2: 训练 ----
if [ -f "output/yolo_eye_training/eye_detect/weights/best.pt" ]; then
    echo "[2] 模型已存在, 跳过训练"
else
    echo "[2] 训练 YOLOv8-Nano 100 epochs..."
    python yolo/train_yolo_eye.py --epochs 100 --device cuda:0
fi

# ---- Step 3: 验证 ----
if [ -f "output/yolo_eye_training/eye_detect/weights/best.pt" ]; then
    echo "[3] 验证..."
    python -c "
from ultralytics import YOLO
m = YOLO('output/yolo_eye_training/eye_detect/weights/best.pt')
r = m.val(data='yolo/datasets/eye_detection/data.yaml', device='cuda:0')
print(f'  mAP@50:    {r.box.map50:.4f}')
print(f'  mAP@50-95: {r.box.map:.4f}')
print(f'  Precision: {r.box.mp:.4f}')
print(f'  Recall:    {r.box.mr:.4f}')
"
fi

# ---- Step 4: MediaPipe 验证 ----
echo "[4] MediaPipe 验证 (val 1-3)..."
python mediapipe/extract_eyes.py \
    --videos 1-3 \
    --rawframes "${DATA_ROOT}/val_rawframes" \
    --output ./output/verify_mp_eyes 2>&1

echo ""
echo "=== Done ==="
echo "模型: output/yolo_eye_training/eye_detect/weights/best.pt"
echo "指标: output/yolo_eye_training/eye_detect/results.csv"
echo "验证: output/verify_mp_eyes/"
echo ""
echo "下载:"
echo "  tar czf ~/DeFB_results.tar.gz \\"
echo "    output/yolo_eye_training/eye_detect/weights/best.pt \\"
echo "    output/yolo_eye_training/eye_detect/results.csv \\"
echo "    output/yolo_eye_training/eye_detect/labels.jpg \\"
echo "    output/verify_mp_eyes/"
