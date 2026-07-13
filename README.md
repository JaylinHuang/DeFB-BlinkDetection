# 非受限条件下的眨眼检测方法的研究

> **眼部定位裁剪与眨眼检测两级分离设计**  
> 模式识别课程设计 · 2026年7月

## 项目简介

本项目针对非受限条件下的眨眼检测任务，采用**"眼部定位裁剪 + 眨眼检测"两级分离**的系统设计架构。第一阶段从原始视频帧中检测人脸并精确裁剪出眼部区域；第二阶段基于裁剪出的眼部帧序列进行时序建模，判断是否存在眨眼事件。

### 两级分离设计优势
- 各阶段可独立优化，模块化程度高
- 便于针对不同任务特性选择最适合的算法方案
- 支持多种方案横向对比与替换

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入: 视频帧 (1280×720)                 │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │     Stage 1: 眼部定位裁剪       │
         │  ┌─────────┐ ┌──────────┐     │
         │  │RT-DETRv2│ │MediaPipe │     │
         │  │ (废弃)  │ │ (对照)   │     │
         │  └─────────┘ └──────────┘     │
         │  ┌─────────────────────────┐  │
         │  │  YOLOv8-Nano (主方案)   │  │
         │  │  + 超参数调优 (Task2)   │  │
         │  └─────────────────────────┘  │
         └───────────────┬───────────────┘
                         │ 64×64 眼部裁剪帧 (×13帧)
         ┌───────────────┴───────────────┐
         │     Stage 2: 眨眼检测          │
         │  ┌─────────────────────────┐  │
         │  │   BiMamba (核心创新)    │  │
         │  │  双向多尺度时序Mamba     │  │
         │  │  + MIL聚合框架          │  │
         │  └─────────────────────────┘  │
         │  对照: MS-LSTM / BiLSTM+Attn  │
         │       / 3D-CNN / BlinkFormer  │
         └───────────────┬───────────────┘
                         │
                  输出: 眨眼/非眨眼
```

## Stage 1: 人脸检测和眼部定位

### 方案对比

| 方案 | 架构 | 参数量 | 检测成功率 | 结论 |
|------|------|--------|-----------|------|
| RT-DETRv2 | Transformer (端到端) | 2.6M~21.8M | 0% | loss_vfl=0.0 框架级bug，废弃 |
| MediaPipe | 预训练 Face Landmarker | - | 76% | 零训练，sharpness高但失败率高 |
| **YOLOv8-Nano** | **CNN (Anchor-Free)** | **3.01M** | **100%** | **主方案，可调参改进** |

### YOLOv8-Nano 超参数调优 (Task 2)

在基线配置上设计了 **6 组对照实验**，系统探索学习率、图像分辨率、模型规模和批量大小的影响：

| 实验 | 变量 | Best mAP50 | Final mAP50 | Best Epoch |
|------|------|:----------:|:-----------:|:----------:|
| exp0 基线 | lr0=0.01, batch=8, imgsz=640, yolov8n | 0.2347 | 0.1139 | 12 |
| exp1 | lr0=0.001 | 0.2107 | 0.1581 | 26 |
| exp2 | lr0=0.0005 | 0.2052 | 0.1046 | 14 |
| **exp3** | **imgsz=1280, batch=4** | **0.2355** | 0.1562 | 14 |
| exp4 | yolov8s (11.14M params) | 0.2272 | 0.1625 | 6 |
| exp5 | batch=16 | 0.2305 | **0.2281** | 83 |

**核心发现：**
- imgsz=1280 取得最高 Best mAP50 (0.2355)，大图输入保留更多人眼细节
- batch=16 训练最稳定，Final mAP50=0.2281，过拟合程度最低 (Gap=0.0024)
- 学习率与性能正相关：lr0=0.01 > 0.001 > 0.0005
- 大模型 yolov8s 在有限数据下反而不如 yolov8n

## Stage 2: 眨眼检测

### 核心创新：BiMamba (双向多尺度时序 Mamba)

基于 Mamba 状态空间模型的眨眼检测网络，主要创新点：

1. **双向 SSM 机制** — 正向+反向两条独立状态空间路径，消除眨眼边界定位延迟
2. **多尺度空洞卷积** — dilation=(1,3,5) 并行分支，捕获不同速度的眨眼模式
3. **空间-时序解耦** — 2D CNN 逐帧提取空间特征 → Mamba 时序建模
4. **MIL 框架** — 多示例学习，仅用序列级标签实现帧级定位

### 对照方法
- MS-LSTM (多尺度 LSTM)
- BiLSTM + Attention
- 差分特征 + Focal Loss
- A-Softmax 集成
- BlinkFormer (Transformer)
- 3D-CNN

## 项目结构

```
DeFB/
├── src/                        # 主框架代码
│   ├── blink/                  # 眨眼检测模块
│   │   ├── model.py            #   BiMamba 等网络定义
│   │   ├── dataset.py          #   数据加载
│   │   ├── train_blink_detector.py  # 训练脚本
│   │   ├── test_eval.py        #   测试评估
│   │   └── loss.py             #   损失函数
│   ├── zoo/rtdetr/             # RT-DETRv2 模型实现
│   ├── data/                   # 数据集与数据加载
│   ├── nn/                     # 神经网络模块 (backbone/neck/head)
│   ├── configs/                # 配置文件 (YAML)
│   ├── solver/                 # 训练/评估引擎
│   └── tools/                  # 工具脚本
├── yolo/                       # YOLOv8 眼部检测
│   ├── train_yolo_eye.py       #   训练脚本
│   ├── extract_eyes.py         #   眼部裁剪提取
│   └── prepare_yolo_data.py    #   数据集准备
├── api/                        # REST API 服务 (FastAPI)
│   ├── main.py                 #   API 入口 + 路由定义
│   ├── model.py                #   YOLO 推理封装 (单例模式)
│   ├── schemas.py              #   Pydantic 请求/响应模型
│   ├── config.py               #   环境变量配置
│   └── requirements.txt        #   API 独立依赖
├── Dockerfile                  # 容器化部署
├── docker-compose.yml          # Docker Compose 编排
├── mediapipe/                  # MediaPipe 眼部提取 (对照方案)
├── newtest/                    # RT-DETRv2 Lite 实验脚本
├── alternative_yolo/           # YOLO 数据集配置
│   └── datasets/
│       ├── eye_detection/      #   完整数据集配置
│       └── eye_detection_small/#   小数据集配置
├── output/                     # 训练输出 (曲线/CSV)
├── docs/                       # 项目报告 (PDF)
├── requirements.txt
└── run_server.sh
```

## 环境配置

### 硬件环境
- GPU: NVIDIA A10 (23GB VRAM)
- 平台: ModelScope DSW (魔搭社区)

### 软件环境
- OS: Ubuntu 22.04 / CUDA 12.8.1
- Python: 3.12 / PyTorch: 2.10.0
- ultralytics >= 8.4.88 (需支持 numpy 2.x)

### 安装

```bash
# 1. 创建环境
conda create -n defb python=3.9
conda activate defb

# 2. 安装 PyTorch
pip install torch>=2.0.1 torchvision>=0.15.2

# 3. 安装依赖
pip install -r requirements.txt
```

## 快速开始

### Stage 1: YOLOv8-Nano 眼部检测训练

```bash
# 训练 (默认: batch=8, imgsz=640, lr=0.001)
python yolo/train_yolo_eye.py

# 服务器低内存模式
python yolo/train_yolo_eye.py --server

# 自定义超参数 (对应 exp3: imgsz=1280)
python yolo/train_yolo_eye.py --imgsz 1280 --batch 4 --epochs 100

# 眼部裁剪提取
python yolo/extract_eyes.py --model output/yolo_eye_training/eye_detect/weights/best.pt
```

### Stage 2: 眨眼检测训练

```bash
# 使用统一流水线
bash run_server.sh

# 或分步执行
python src/tools/train.py -c src/configs/rtdetrv2/detrs-blink_len=10_mpeblinkv1.yml
python src/blink/train_blink_detector.py --config src/configs/BlinkModule/full_v1.py
python src/blink/test_eval.py
```

## REST API 服务 (FastAPI)

项目提供基于 **FastAPI** 的推理服务，支持 HTTP 接口调用眼部检测模型。

### API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 / 模型状态 |
| `POST` | `/api/v1/detect` | 上传图片 → 返回 JSON 检测结果 |
| `POST` | `/api/v1/detect/image` | 上传图片 → 返回标注后的图片 |
| `GET` | `/docs` | Swagger UI 交互式文档 |

### 本地运行

```bash
# 1. 安装 API 依赖
pip install -r api/requirements.txt

# 2. 放置模型权重到 output/yolo_eye_training/eye_detect/weights/best.pt
#    或通过环境变量指定路径
export MODEL_PATH=/path/to/best.pt

# 3. 启动服务
python -m api.main
# 或
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. 打开浏览器访问 http://localhost:8000/docs
```

### 调用示例

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 检测图片 (返回 JSON)
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@test.jpg" \
  -F "conf=0.15"

# 检测图片 (返回标注图片)
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@test.jpg" \
  -o result.png
```

### JSON 响应示例

```json
{
  "success": true,
  "image_width": 640,
  "image_height": 480,
  "detection_count": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "left_eye",
      "confidence": 0.852,
      "bbox": {"x1": 120.5, "y1": 80.3, "x2": 180.2, "y2": 130.7}
    },
    {
      "class_id": 1,
      "class_name": "right_eye",
      "confidence": 0.615,
      "bbox": {"x1": 420.1, "y1": 82.0, "x2": 480.5, "y2": 131.4}
    }
  ],
  "inference_time_ms": 12.3,
  "model_name": "best.pt"
}
```

### Docker 部署

```bash
# 1. 创建权重目录并放入模型文件
mkdir -p weights
cp output/yolo_eye_training/eye_detect/weights/best.pt weights/

# 2. 构建并启动
docker-compose up --build -d

# 3. 查看日志
docker logs -f debf-eye-detection

# 4. 停止
docker-compose down
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `output/.../best.pt` | 模型权重路径 |
| `CONF_THRESHOLD` | `0.15` | 置信度阈值 |
| `IOU_THRESHOLD` | `0.45` | NMS IoU 阈值 |
| `IMAGE_SIZE` | `640` | 推理输入尺寸 |
| `API_HOST` | `0.0.0.0` | 服务监听地址 |
| `API_PORT` | `8000` | 服务监听端口 |

## 数据集

### MPEblink / HUST-LBEW
- 来源: [MPEblink Dataset](https://doi.org/10.5281/zenodo.7754768)
- 预处理: mpeblink_lite 版本
- 标注: YOLO格式 (left_eye, right_eye)
- 训练集: 11,870 张 | 验证集: 5,327 张

> 数据集文件较大，未包含在仓库中。请从上述链接下载并放置于 `alternative_yolo/datasets/` 目录下。

## 主要结果

### YOLOv8-Nano 训练收敛曲线

训练曲线和评估图表位于 `output/yolo_eye_training/eye_detect/` 目录：
- `results.png` — 完整训练曲线 (loss + mAP)
- `BoxPR_curve.png` — PR 曲线
- `confusion_matrix.png` — 混淆矩阵
- `val_batch*_pred.jpg` — 验证集预测可视化

### BiMamba 眨眼检测

7个网络变体在统一数据集上进行对比，所有模型报告 Recall / Precision / F1 三项指标。

## 技术报告

完整项目报告见 [`docs/非受限条件下的眨眼检测方法的研究.pdf`](docs/)，包含：
- 三方案 (RT-DETRv2 / MediaPipe / YOLOv8) 详细对比分析
- YOLOv8-Nano 超参数调优完整实验报告
- BiMamba 网络设计与消融实验
- 20 篇参考文献

## 致谢

本项目基于以下开源工作构建：
- [DeFB (AAAI 2026)](https://github.com/jinfanggan/DeFB) — 原始框架
- [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) — 检测 Transformer
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 目标检测
- [MediaPipe](https://developers.google.com/mediapipe) — 面部关键点检测
- [MPEblink Dataset](https://github.com/wenzhengzeng/MPEblink) — 眨眼数据集

## License

本项目仅供学习和研究使用。
