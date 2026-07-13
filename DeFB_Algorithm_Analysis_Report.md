# DeFB 算法分析报告 & 调试手册

> 日期: 2026-06-30
> 项目: 人眼定位网络设计 (Task 1) + 参数调优 (Task 2)
> 环境: ModelScope GPU Notebook (A10 23GB, PyTorch 2.10.0, CUDA 12.4)

---

## 目录

1. [算法分析总览](#1-算法分析总览)
2. [方案A: RT-DETRv2 失败分析](#2-方案a-rt-detrv2-失败分析)
3. [方案B: MediaPipe Face Landmarker](#3-方案b-mediapipe-face-landmarker)
4. [方案C: YOLOv8-Nano](#4-方案c-yolov8-nano)
5. [训练日志解读 (终端内容)](#5-训练日志解读-终端内容)
6. [调试注意事项](#6-调试注意事项)
7. [常用命令速查](#7-常用命令速查)
8. [最终推荐](#8-最终推荐)

---

## 1. 算法分析总览

### 1.1 任务定义

输入: 1280×720 视频帧 → 输出: 左右眼独立 64×64 裁剪, 13帧窗口
两种输出接口: PNG图片 (可视化审查) + NumPy数组 [13,64,64,3] (模型输入)

### 1.2 三架构最终对比

| 维度 | A: RT-DETRv2 | B: MediaPipe | C: YOLOv8-Nano |
|:-----|:-----------:|:------------:|:--------------:|
| 参数量 | 21.8M / 2.6M | ~3M | 3.0M |
| 原理 | Transformer检测器 | 478点 face mesh | Anchor-free CNN |
| 训练 | 需100+ epoch | 零训练 | 100 epoch |
| 训练耗时 | — | 0 | ~5h (A10) |
| val成功率 | ❌ 框架bug | 76% (76/100) | **100% (100/100)** |
| avg sharpness | ❌ | **20.1** | 18.7 |
| **结论** | **❌ 否决** | **基线对照** | **✅ 主方案** |

### 1.3 训练收敛对比 (YOLO vs RT-DETRv2)

```
RT-DETRv2:  loss_vfl=0.0 (不变)  →  cls_loss未学习  →  ❌
YOLOv8:     cls_loss 1.53→0.39    →  分类头正常学习  →  ✅
```

这是两者最根本的区别。YOLO 的 loss 每一轮都在稳步下降,而 RT-DETRv2 的 VFL loss 从第1轮到第100轮始终为 0。

---

## 2. 方案A: RT-DETRv2 失败分析

### 2.1 根因: loss_vfl = 0.0

```python
# src/criterion/rtdetr_criterion.py 中
loss_vfl = vfl_loss(pred_scores, target_scores)
# → 始终为 0.0
```

**分类头从未激活**,导致:
- HungarianMatcher 无法正确分配正负样本
- 回归头虽然下降,但不会学到"在哪里检测"
- 模型输出默认值 `[1,1,1,1]` → 整张图

### 2.2 验证复现

| 变体 | backbone | 参数量 | 训练轮数 | loss_vfl | pred_eye_boxes | 可用 |
|:----:|:--------:|:------:|:--------:|:--------:|:--------------:|:----:|
| Lite | PResNet-18 | 21.8M | 5 | 0.0 | [1,1,1,1] | ❌ |
| Lite | PResNet-18 | 50 | 0.0 | [1,1,1,1] | ❌ |
| Nano | MobileNetV3 | 50 | 0.0 | [1,1,1,1] | ❌ |

**结论**: 框架级 bug,与模型大小、训练轮数无关。Lite(21.8M)和Nano(2.6M)完全复现。

### 2.3 无法修复的原因

- 框架代码量大 (`src/zoo/rtdetr/` + `src/criterion/`)
- HungarianMatcher 的 cost_class 计算需深入 debug
- VFL 的 alpha/gamma 参数可能不匹配
- 修复周期 > 论文验收 deadline

---

## 3. 方案B: MediaPipe Face Landmarker

### 3.1 工作原理

```
输入帧 → FaceLandmarker → 478点 face mesh
  → 取眼周16轮廓点 (LEFT_EYE_CONTOUR / RIGHT_EYE_CONTOUR)
  → 计算最小包围盒 + 25% padding
  → 扩边为正方形 → resize 到 64×64
```

### 3.2 关键代码

```python
# mediapipe/extract_eyes.py
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173,
                    133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466,
                     263, 382, 381, 380, 374, 373, 390, 249]
```

### 3.3 val 1-50 结果

| 指标 | 值 |
|:-----|:---|
| 成功检测 | **76/100眼 (76%)** |
| 平均sharpness | **20.1** |
| 失败视频 | 1,2,3,10,12,20,23,28,32,37,39,41 (共12个) |
| 失败原因 | 人脸遮挡/大角度/模糊 |

### 3.4 优缺点

- ✅ 零训练, pip install 即用
- ✅ 清晰度高于 YOLO (20.1 vs 18.7)
- ❌ 24% 视频检测失败
- ❌ 无法通过训练改进

---

## 4. 方案C: YOLOv8-Nano

### 4.1 整体架构

```
输入 640×640
  → Conv + C2f + SPPF (backbone)
  → FPN + PAN (neck)  
  → Detect head (left_eye / right_eye 二分类)
  → 预测 bbox → 扩边正方形 → resize 64×64
```

参数量: 3,006,038 | GFLOPs: 8.1

### 4.2 数据集生成逻辑

```python
# alternative_yolo/prepare_yolo_data.py
# 从 WFLW 68点 landmark 直接计算眼bbox
LEFT_EYE_IDX  = [42,43,44,45,46,47]  # 左眼6点
RIGHT_EYE_IDX = [36,37,38,39,40,41]  # 右眼6点

# 取最小包围盒 + 25% padding → YOLO格式 (cx,cy,w,h)
```

### 4.3 完整训练日志解读

以下从服务器终端日志中提取关键信息:

#### 4.3.1 硬件信息

```
GPU: NVIDIA A10, 22716MiB (约22.7GB显存)
实际占用: 仅 1.07-1.16GB (imgsz=640 时)
PyTorch: 2.10.0+cu128
Ultralytics: 8.4.83
```

#### 4.3.2 训练参数

```
epochs=100, batch=8, imgsz=640
optimizer=AdamW(lr=0.001), cos_lr=True
mosaic=0.5, mixup=0.1, copy_paste=0.1
close_mosaic=10  (最后10轮关闭马赛克)
workers=4
```

#### 4.3.3 数据规模

```
Train: 23656 images, 113460 labels (平均每帧4.8个眼标注)
Val:   10626 images, 43075 labels
```

#### 4.3.4 训练时间

```
每轮: ~2分40秒 (训练) + ~32秒 (验证) = ~3分12秒
100轮总计: ~5小时20分
最终28轮 (73→100, 续跑): 1.465小时
```

#### 4.3.5 收敛曲线关键节点

| Epoch | box_loss | cls_loss | mAP@50 | Precision | Recall | 备注 |
|:-----:|:--------:|:--------:|:------:|:---------:|:------:|:----|
| 1 | 2.140 | 1.534 | 0.310 | 0.388 | 0.673 | 起点 |
| 10 | 1.430 | 0.677 | 0.378 | 0.413 | 0.576 | |
| 20 | 1.306 | 0.592 | 0.287 | 0.379 | 0.496 | 小目标波动 |
| 50 | 1.151 | 0.498 | 0.349 | 0.445 | 0.523 | 半程 |
| 73(续跑起点) | 1.091 | 0.466 | 0.332 | 0.438 | 0.510 | |
| 90(关闭mosaic) | 1.064 | 0.453 | 0.312 | 0.425 | 0.498 | mosaic关 |
| 100 | **0.944** | **0.390** | 0.309 | 0.421 | 0.499 | 最终轮 |

**best.pt (epoch 未知, 取val最优)**:
- mAP@50: **0.417**
- Precision: **0.488**
- Recall: **0.552**
- left_eye mAP: 0.389
- right_eye mAP: 0.445

> 注: 最终轮 mAP 0.309 低于 best.pt 的 0.417, 说明 mAP 在训练过程中有波动, best.pt 自动保存了最优权重。

### 4.4 val 1-50 推理结果

| 指标 | 值 |
|:-----|:---|
| 成功检测 | **100/100眼 (100%)** |
| 平均sharpness | 18.7 |
| 中位sharpness | 17.7 |
| 最佳视频 | 27 (140.7/53.4), 30 (154.2/144.2), 31 (147.5/135.1) |
| 较差视频 | 1 (8.9/20.9), 10 (13.0/25.0), 37 (11.8/25.6) |

### 4.5 改进方向

| 方向 | 方法 | 预期提升 | 内存影响 |
|:-----|:-----|:--------:|:--------:|
| 提高分辨率 | `--imgsz 1280` | 小目标检测↑ | ~4GB |
| 降低阈值 | `--conf 0.2` | 召回率↑ | 无 |
| 换大模型 | YOLOv8s (11M) | 精度↑ | ~3GB |
| 更多数据 | stride=2 或 1 | 样本量↑ | 磁盘 |

---

## 5. 训练日志解读 (终端内容)

### 5.1 关键日志行解读

```bash
# 1. AMP 检查
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
# → 混合精度训练, 加速且省显存

# 2. 缓存扫描 (首次运行)
train: Scanning /mnt/workspace/DeFB/alternative_yolo/datasets/eye_detection/labels/train.cache...
# → 构建缓存, 下次秒开

# 3. 重复标签警告
train: 144_00171.jpg: 1 duplicate labels removed
# → 同一帧多人标注合并时的正常现象

# 4. 续跑提示
Resuming training output/yolo_eye_training/eye_detect/weights/last.pt from epoch 73 to 100 total epochs
# → 从第73轮续跑

# 5. 关闭 Mosaic
Closing dataloader mosaic
# → 最后10轮关掉马赛克增强, 用原图精调

# 6. 保存 best.pt
Optimizer stripped from .../weights/best.pt, 6.2MB
# → 去除优化器状态, 纯推理权重

# 7. 最终验证 (best.pt)
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all      10626      43075      0.488      0.552      0.417      0.214
left_eye 10626      21538      0.481       0.54      0.389      0.197
right_eye10625      21537      0.496      0.563      0.445       0.23
# → 左右眼分别评估, right_eye 优于 left_eye
```

### 5.2 训练过程中的风险信号

```bash
# ⚠️ 信号1: loss_vfl=0.0 (RT-DETRv2) → 分类头死了
# ⚠️ 信号2: pred_eye_boxes=[1,1,1,1] → 模型在猜默认值
# ✅ 信号3: cls_loss 持续下降 → 分类头正常学习
# ✅ 信号4: mAP 波动但整体在 0.3-0.5 范围 → 正常
# ⚠️ 信号5: GPU 利用率 0% → 训练已崩, 需检查
```

---

## 6. 调试注意事项

### 6.1 ModelScope 环境特有问题

#### 1: Session 重启后包丢失
```
现象: ModuleNotFoundError: No module named 'ultralytics'
原因: pip 包在 session 重启后丢失
解决: 每次 session 重启先 pip install, 或建 conda 环境
命令: pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### 2: pip 镜像慢/超时
```
阿里云(推荐): -i https://mirrors.aliyun.com/pypi/simple/
清华(备选):   -i https://pypi.tuna.tsinghua.edu.cn/simple
中科大:       -i https://pypi.mirrors.ustc.edu.cn/simple/
备忘: 加 --default-timeout=600 --trusted-host XXX
```

#### 3: libGLESv2.so.2 找不到
```
现象: OSError: libGLESv2.so.2: cannot open shared object file
原因: opencv-python 依赖 GUI 库, 但服务器无显示器
解决: 用 opencv-python-headless 替代
       安装系统库: apt-get install -y libgl1-mesa-glx libegl1-mesa
```

#### 4: Conda 不持久化
```
现象: session 重启后 conda env 消失
解决: 用 --prefix 指定到 /mnt/workspace/
      conda create -y -p /mnt/workspace/conda/envs/defb python=3.12
      或直接 pip install (系统 Python 持久化)
```

#### 5: yolo26n.pt 下载失败
```
现象: WARNING ⚠️ Download failure, retrying...
解决: 手动下载并放入缓存:
      mkdir -p ~/.cache/ultralytics/
      cp yolo26n.pt ~/.cache/ultralytics/
      # 或直接用 yolov8n.pt 改名代替
```

#### 6: WebSocket 断连
```
现象: WebSocket close with status code 1006
解决: 不要关浏览器, 等待自动重连
       如果训练进程没了, 用 nvidia-smi 确认 GPU 空闲后重跑
       训练时可用 tail -f results.csv 监控进度
```

#### 7: `pkill -9 -f python` 误杀
```
现象: pip 包也消失了, 需要重装
原因: pkill 杀了所有 python 进程, 包括系统服务
解决: 用 pkill -f train_yolo 指定进程名
      或 ps aux | grep python 确认无误再杀
```

#### 8: 文件上传限制
```
现象: 文件数超限 / 上传中断
解决: 分多次上传 (每次30个以下视频目录)
      val_rawframes 只需 1-30 也能跑, 不影响训练
```

### 6.2 训练调优注意点

#### 学习率
```
--lr 0.001 (默认) 适合 YOLOv8n 正常收敛
--lr 0.01  大一点适合小目标, 但可能震荡
--lr 0.0001 稳妥, 但收敛慢
```

#### 图片尺寸 imgsz
```
640:  显存 1.1GB, 速度 2.5分/轮, 小目标精度一般
1280: 显存 ~4-5GB, 速度 ~5分/轮, 小目标精度提升
```

#### 训练轮数
```
30-50轮: 快速看趋势, 判断调参方向
100轮:   完整训练, 保证收敛
mAP 在 0.2-0.5 范围波动属正常(小目标)
```

#### 续跑
```
自动续跑: YOLO 检测到 results.csv 存在且未满 epoch 时自动续跑
手动续跑: YOLO('last.pt').train(resume=True)
          注意: resume=True 从 last.pt 继续, 从断点处接着跑
```

---

## 7. 常用命令速查

### 7.1 服务器部署

```bash
# 一键部署
bash run_server.sh /mnt/workspace/mpeblink_lite

# 或逐条执行:
cd /mnt/workspace/DeFB
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --default-timeout=600
python alternative_yolo/prepare_yolo_data.py --stride 3 --rawframes /mnt/workspace/mpeblink_lite
python yolo/train_yolo_eye.py --epochs 100 --device cuda:0
```

### 7.2 训练

```bash
# 从头训练
python yolo/train_yolo_eye.py --epochs 100 --device cuda:0

# 续跑
python -c "from ultralytics import YOLO; YOLO('output/yolo_eye_training/eye_detect/weights/last.pt').train(resume=True)"

# 调参
python yolo/train_yolo_eye.py --epochs 50 --lr 0.01 --imgsz 640

# 看进程
ps aux | grep train_yolo | grep -v grep
```

### 7.3 眼部裁剪推理

```bash
# YOLO 推理 (100%成功率)
python yolo/extract_eyes.py \
    --model output/yolo_eye_training/eye_detect/weights/best.pt \
    --videos 1-50 \
    --rawframes /mnt/workspace/mpeblink_lite/val_rawframes \
    --output ./output/yolo_eyes_val

# MediaPipe 推理 (零训练)
python mediapipe/extract_eyes.py \
    --videos 1-50 \
    --rawframes /mnt/workspace/mpeblink_lite/val_rawframes \
    --output ./output/mp_eyes_val
```

### 7.4 模型验证

```bash
# 查看训练进度
tail -f output/yolo_eye_training/eye_detect/results.csv

# 查看 GPU 状态
watch -n 2 nvidia-smi

# 统计推理结果 sharpness
python -c "
import numpy as np, os
for v in range(1,51):
    for e in ['eye_001','eye_002']:
        p = f'output/yolo_eyes_val1-50/{v}/{e}.npy'
        if os.path.exists(p):
            c = np.load(p)
            s = np.std(np.mean(c, axis=3))
            print(f'{v}/{e}: {s:.1f}')
"
```

### 7.5 下载结果

```bash
# 服务器上打包
tar czf ~/DeFB_results.tar.gz \
    output/yolo_eye_training/eye_detect/weights/best.pt \
    output/yolo_eye_training/eye_detect/results.csv \
    output/yolo_eye_training/eye_detect/labels.jpg \
    output/verify_mp_eyes/ 2>/dev/null || true

# 本地解压
tar xzf DeFB_results.tar.gz
```

### 7.6 清理

```bash
# 清理旧训练结果 (重新开始)
rm -rf output/yolo_eye_training/

# 清理 YOLO 数据集 (重新生成)
rm -rf alternative_yolo/datasets/eye_detection/

# 清理进程
pkill -f train_yolo

# 查看磁盘
df -h /mnt/workspace
```

---

## 8. 最终推荐

### 人眼定位 (Task 1)

| 用途 | 方案 | 参数量 | 推荐等级 |
|:-----|:----|:------:|:--------:|
| 主训练模型 | YOLOv8-Nano | 3.0M | ⭐⭐⭐ |
| 零训练基线 | MediaPipe | ~3M | ⭐⭐ |
| 否决方案 | RT-DETRv2 | — | ❌ |

### 参数调优 (Task 2)

| 优先级 | 参数 | 推荐范围 | 预期影响 |
|:------:|:-----|:--------|:--------|
| 1 | `--imgsz` | 640→1280 | 小目标精度↑, 显存↑ |
| 2 | `--lr` | 0.001→0.0005/0.01 | 收敛速度 |
| 3 | `--batch` | 8→16 | 梯度稳定 |
| 4 | `--model` | yolov8n→yolov8s | 11M参数, 精度↑ |

### 一句话总结

> **RT-DETRv2 的 eye_bbox 头存在框架级收敛问题 (loss_vfl=0.0), 不适用于人眼定位。YOLOv8-Nano 在 100 轮训练后实现 100% 检测成功率 (50/50 视频), sharpness 略低于 MediaPipe (18.7 vs 20.1) 但优势在于可调参改进。推荐 YOLOv8-Nano 作为主方案, MediaPipe 作为零训练对照方案。**

---

## 附录: 项目最终文件夹结构

```
E:\Documents\DeFB/
├── run_server.sh                   ← 一键部署脚本
├── requirements.txt                ← 依赖清单
├── README.md
├── DeFB_Algorithm_Analysis_Report.md   ← 本报告
├── DeFB_Architecture_Analysis.md       ← 训练失败总结
├── DeFB_Architecture_Comparison.md     ← 三架构对比报告
├── yolov8n.pt                          ← YOLO预训练权重
│
├── mediapipe/                          ← 方案B (人眼定位, 零训练)
│   ├── extract_eyes.py
│   └── face_landmarker.task
│
├── yolo/                               ← 方案C (人眼定位, 主训练方案)
│   ├── prepare_yolo_data.py            ← 从WFLW landmark生成YOLO标注
│   ├── train_yolo_eye.py              ← YOLOv8-Nano训练
│   └── extract_eyes.py                ← YOLO推理裁剪眼部
│
├── newtest/                            ← 失败算法保留 (RT-DETRv2 Lite/Nano)
│   ├── track_model_lite.py            ← Lite/Nano模型构建
│   ├── train_lite.py                  ← 训练入口
│   ├── config_nano_1-10.yml           ← Nano训练配置
│   ├── config_1280x720.yml            ← 服务器训练配置
│   ├── eye_crop_extractor.py          ← 眼部裁剪核心模块
│   ├── extract_eye_samples.py         ← GT landmark眼部提取
│   └── ... (共21个文件)
│
├── output/                              ← 训练 & 推理产出
│   ├── yolo_eye_training/               ← YOLO训练结果
│   │   └── eye_detect/weights/
│   │       ├── best.pt                  ← 最佳模型 (6.2MB)
│   │       └── last.pt                  ← 最后一轮模型
│   ├── mediapipe_eyes_val1-50/          ← MediaPipe val 1-50 裁剪
│   ├── yolo_eyes_val1-50/               ← YOLO val 1-50 裁剪
│   └── verify_mp_eyes/                  ← MediaPipe验证结果
│
└── src/                                  ← 核心框架
    ├── blink/                           ← Stage 2 眨眼检测模块
    │   ├── model.py                     ← BlinkTransformerDecoder
    │   ├── dataset.py                   ← 眨眼数据集
    │   ├── loss.py                      ← 损失函数
    │   └── train_blink_detector.py      ← 训练脚本
    ├── configs/                         ← RT-DETRv2 训练配置
    │   ├── BlinkModule/
    │   ├── dataset/
    │   └── rtdetrv2/
    ├── core/                            ← 核心工具
    ├── data/                            ← 数据集 & DataLoader
    ├── misc/                            ← 工具函数
    ├── nn/                              ← 神经网络模块
    ├── optim/                           ← 优化器
    ├── solver/                          ← 训练引擎
    ├── tools/                           ← 框架工具脚本
    └── zoo/                             ← 模型库 (RTDETR)
```

### 文件说明

| 路径 | 说明 | 职责 |
|:-----|:-----|:-----|
| `mediapipe/` | 方案B | 零训练人眼定位基线 |
| `yolo/` | 方案C | 可训练的人眼定位主方案 |
| `src/blink/` | Stage 2 | 眨眼检测模块 (组长完成) |
| `src/zoo/rtdetr/` | 框架 | RT-DETRv2 检测器实现 |
| `newtest/` | 实验 | Lite/Nano 训练脚本 |
| `output/` | 结果 | 所有训练产物和眼部裁剪 |
