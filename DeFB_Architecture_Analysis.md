# DeFB 训练失败总结与最终方案定型

> 撰写日期: 2026-06-29
> 所属任务: Task 1 (人眼定位网络设计, 3+架构比对)

---

## 一、总览

历时多轮实验，已尝试 **5 种方案** (含变体两套)，结论如下:

| 架构 | 参数量 | 状态 | 核心原因 |
|------|--------|:----:|----------|
| A: RT-DETRv2 Lite | 21.8M | ❌ 不可用 | loss_vfl=0.0，分类头未激活 |
| A2: RT-DETRv2 Nano (MobileNetV3) | 2.6M | ❌ 不可用 | 继承同一框架 bug |
| ~ BlazeFace ~ | 0.2M | ❌ 已否决 | 6点关键点无法精确框定眼睛 |
| **B: MediaPipe Face Landmarker** | ~3M | ✅ **启用** | 零训练, 478点眼周轮廓 |
| **C: YOLOv8-Nano** | 3.0M | ✅ **启用** | 训练正常, 内存已优化 |

---

## 二、方案A: RT-DETRv2 系列失败分析

### 2.1 现象

```
loss_vfl      = 0.0 (始终为零)
loss_bbox_eye = 下降 (100+)
loss_giou_eye = 下降 (400+)
pred_eye_boxes = [1, 1, 1, 1]  (归一化默认最大值)
```

所有训练日志一致 (Lite 5轮 / Lite 第一轮 / Nano 20轮)：

| Epoch | loss_vfl | loss_bbox_eye | loss_giou_eye | loss_total |
|-------|----------|---------------|---------------|------------|
| 1 | 0.000 | 191 | 795 | 80734 |
| 2 | 0.000 | 152 | 632 | 72611 |
| 3 | 0.000 | 131 | 556 | — |
| 4 | 0.000 | 114 | 516 | — |
| 5 | 0.000 | 101 | 496 | 61047 |

### 2.2 根因: loss_vfl=0.0

**VFL (Varifocal Loss)** 是分类头的损失函数，控制模型是否"检测到"目标。

```python
# src/criterion/rtdetr_criterion.py
loss_vfl = vfl_loss(pred_scores, target_scores)  
# → 始终为 0.0
```

**分类头从未激活**，因此:
- 匈牙利匹配 (HungarianMatcher) 无法正确分配正负样本
- 回归头 (bbox_eye / giou_eye) 虽然下降，但无法学习"在哪里检测眼睛"
- 模型输出默认值 `[1,1,1,1]` = 整张 1280×720 画面

### 2.3 输出结果

```
模型预测 eye_bbox: [1.0, 1.0, 1.0, 1.0]  (cxcywh 归一化)
反归一化:           [640, 360, 1280, 720]   (像素坐标)
                → 实际 = 整张帧画面的半边
                → "裁剪"后 = 躯干、背景等不相关内容
```

验证确认:

```
Nano 预测: center=(1280, 720), size=1280×720 (整张图)
GT eye_bbox:  size=101×45 (双眼区域)
左眼 GT:      size=25×7  (真正需要的单眼)
```

### 2.4 与模型大小无关

| 变体 | Backbone | 总参数量 | loss_vfl |
|------|----------|:--------:|:--------:|
| Lite | PResNet-18 (11.7M) | **21.8M** | 0.0 |
| Nano | MobileNetV3-Small (1.5M) | **2.6M** | 0.0 |

结论：**这是 RT-DETRv2 框架的通用 bug，与模型大小无关。**

### 2.5 定位为: 不可修复（当前阶段）

修复需要:
1. 深入 HungarianMatcher 检查 cost_class 计算是否生效
2. 调试 alpha/gamma 配置
3. 修改 criterion 源码（需理解整个 loss 体系）

框架代码量大、耦合深，修复周期远超论文验收 deadline。**不建议继续投入。**

### 2.6 务实结论

RT-DETRv2 在 DeFB 中的实际角色是:
- ✅ **特征提取**: 给 Stage 2 (Blink Model) 提供 multi-scale feature maps
- ❌ **眼部定位**: eye_bbox head 未收敛，无法直接输出可用裁剪

> **正式停用 RT-DETRv2 作为人眼定位（Task 1）方案。**

---

## 三、BlazeFace 失败分析（已删除）

### 3.1 现象

```
sharpness = 0.3-2.5  (vs MediaPipe 4.8-72.7)
输出图像为纯色模糊块, 仅20-30种像素值
```

### 3.2 根因

**Bug 1 — 坐标单位错误:**

```python
# NormalizedKeypoint.x/y 是 [0,1] 归一化
left_eye = (kps[0].x, kps[0].y)    # 0.42, 0.35
# 被当作像素坐标使用
→ estimate_eye_bbox 算出 x1=-44 → clamp 到 (0,0,44,19)
→ 裁剪左上角极小区域

# 修复后:
left_eye = (kps[0].x * W, kps[0].y * H)  # 538, 252 像素
```

**Bug 2 — 几何估算不精确:**
即使坐标修复，BlazeFace 只有 6 个面部关键点:

```python
eye_w = face_w * 0.22
eye_h = eye_w * 0.55
# 固定比例无法适应不同人脸姿态/角度
```

### 3.3 否决原因

BlazeFace 本质是**人脸检测器 + 粗估眼部位置**，而非专用眼部检测器。6 点关键点远不如 MediaPipe 478 点精确。**已删除全部代码和输出。**

---

## 四、方案C: YOLOv8-Nano 训练问题 & 解决方案

### 4.1 问题: 内存占用过高

```
默认配置 → 内存峰值 30GB+
服务器仅有 32GB 内存 → OOM / swap 抖动风险
```

> 注意: **这是 YOLO 训练配置问题，不是模型 bug。**

### 4.2 排查

| 配置项 | 默认值 | 内存消耗原因 |
|--------|--------|-------------|
| batch_size | 8 | 4× 加载数据 + 梯度累积 |
| imgsz | 640 | 图像大 → tensor 大 → pin_memory 多 |
| workers | 4 | 每个 worker 缓存数据 |
| mosaic/mixup | 0.5/0.1 | 拼接图像, RAM 翻倍 |
| cache | True (默认) | 将全部图片缓存到 RAM |

### 4.3 修复

新增 `--server` 低内存模式:

```bash
python alternative_yolo/train_yolo_eye.py --server --epochs 100
```

| 参数 | 本地模式 | --server 模式 |
|------|:-------:|:-------------:|
| batch | 8 | **4** |
| imgsz | 640 | **416** |
| workers | 4 | **2** |
| mosaic | 0.5 | **0.0** |
| mixup | 0.1 | **0.0** |
| cache | True | **False** |
| **预估 RAM** | 20-25GB | **8-12GB** ✅ |

模型本身训练收敛正常（5 轮 mAP50=0.455），**训练没有问题。**

---

## 五、最终启用的方案

### B: MediaPipe Face Landmarker

```bash
# 零训练, 直接推理
python alternative_mediapipe/extract_eyes.py \
    --videos 1-150 \
    --output ./output/mp_eyes
```

- 预训练模型: 478 点 face mesh
- 眼部定位: 16 个眼周 contour landmark → padded bbox → 64×64
- 优点: 开箱即用, 清晰度 4.8-72.7
- 缺点: 约 30% 视频检测失败（人脸角度/遮挡）

### C: YOLOv8-Nano

```bash
# 服务器训练 (安全内存)
python alternative_yolo/train_yolo_eye.py --server --epochs 100

# 推理提取
python alternative_yolo/extract_eyes.py \
    --model output/yolo_eye_training/eye_detect/weights/best.pt
```

- 架构: YOLOv8n (3.0M 参数)
- 训练数据: 从 WFLW 68 landmark 自动生成眼 bbox 标注
- 优势: 可提升、可调参，适合参数调优（Task 2）
- 注意: imgsz=416 可能影响小眼检测精度，建议服务器改成 imgsz=640

---

## 六、命令行速查

```bash
# === 训练 ===
# YOLO训练 (服务器 32GB 安全模式)
python alternative_yolo/train_yolo_eye.py --server --epochs 100

# YOLO训练 (如果服务器内存充足)
python alternative_yolo/train_yolo_eye.py --server --epochs 100 --imgsz 640

# === 推理提取 ===
# MediaPipe (全部 150 视频)
python alternative_mediapipe/extract_eyes.py \
    --videos 1-150 \
    --rawframes E:/.../train_rawframes \
    --output ./output/mp_eyes

# YOLO (用训练好的模型)
python alternative_yolo/extract_eyes.py \
    --videos 1-150 \
    --model output/yolo_eye_training/eye_detect/weights/best.pt \
    --output ./output/yolo_eyes

# === 参数调优 (Task 2) ===
# 改 batch_size | imgsz | lr | 数据增强
python alternative_yolo/train_yolo_eye.py --server --epochs 100 --lr 0.01
```
