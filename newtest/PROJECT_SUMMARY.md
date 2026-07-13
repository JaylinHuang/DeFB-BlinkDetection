# DeFB Lite Track Model — 人脸 + 眼部定位模块 项目总结

## 1. 项目概述

从 DeFB 完整 blink detection 项目中，"蒸馏"出一个轻量级的人脸+眼部定位模块（Stage 1），输出可直接对接 Stage 2 的 time-mamba blink 检测模块。

**核心目标：**
- 参数从原版 ~60M 缩减至 ~22M（减少 62%）
- 保持与原 DeFB Stage 1 完全兼容的输出接口
- 单卡可训练（batch_size=2, ~8GB VRAM）

---

## 2. 环境配置

| 组件 | 版本 |
|------|------|
| Python | 3.14.6 |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |
| torchvision | 0.27.1+cu130 |
| GPU | NVIDIA GeForce RTX 4070 Laptop (8GB) |
| 环境路径 | `D:\app\anaconda3\envs\defb` |

---

## 3. 代码架构

### 3.1 newtest/ 独立文件（5个，共 ~3400 行）

```
newtest/
├── README.md                        # 使用说明
├── track_model_lite.py              # 模型组装 + TrackModelLite 类
├── rtdetrv2_decoder_lite.py         # 自定义 Lite Decoder（核心定制代码）
├── train_lite.py                    # 训练入口（单卡，无需 torchrun）
├── test_lite.py                     # 可行性验证（5 项测试）
├── config_lite.yml                  # 原始配置模板（需 mpeblink v2 数据集）
├── config_lite_test50.yml           # 测试数据配置（前 50 个视频）
├── config_lite_quick.yml            # 快速测试配置（5 视频 / 1 epoch）
├── prepare_data.py                  # 数据预处理（video → rawframes + COCO JSON）
├── collate.py                       # 自定义 collate（@register 注册）
├── visualize_training.py            # 训练曲线可视化
└── PROJECT_SUMMARY.md               # 本文档
```

### 3.2 依赖的 src/ 模块（19个文件）

newtest 代码**直接 import** 了以下 `src/` 模块：

| 调用方 | 依赖模块 | 用途 |
|--------|---------|------|
| `track_model_lite.py` | `src.nn.backbone.presnet` | PResNet-18/34/50 backbone |
| `track_model_lite.py` | `src.zoo.rtdetr.hybrid_encoder` | HybridEncoder（FPN + Transformer encoder） |
| `rtdetrv2_decoder_lite.py` | `src.zoo.rtdetr.utils` | `deformable_attention_core_func_v2`, `get_activation`, `inverse_sigmoid`, `bias_init_with_prob` |
| `train_lite.py` | `src.core.YAMLConfig` | YAML 配置加载 |
| `train_lite.py` | `src.solver.TASKS` | 注册的 solver（DetSolver） |
| `train_lite.py` | `src.misc.dist_utils` | 分布式工具（单卡模式自动 fallback） |
| `collate.py` | `src.core.workspace` | `@register()` 注册机制 |

YAML 配置触发实例化的 `src/` 模块：

| 类别 | 模块 | 用途 |
|------|------|------|
| 训练循环 | `src.solver.det_solver`, `det_engine`, `_solver` | 训练/验证循环 |
| 数据集 | `src.data.dataset.mpeblink`, `mpeblink_api` | MPEblink v2 数据集加载 |
| 数据加载 | `src.data.dataloader` | `BatchImageCollateFuncion` collate |
| 图像变换 | `src.data.transforms._transforms`, `container` | `Compose`, `ConvertPILImage` 等 |
| 损失函数 | `src.zoo.rtdetr.rtdetrv2_criterion` | `RTDETRCriterionv2`（含 head+eye bbox 损失） |
| 匹配器 | `src.zoo.rtdetr.matcher` | `HungarianMatcher`（匈牙利匹配） |
| 后处理 | `src.zoo.rtdetr.rtdetr_postprocessor` | 输出后处理 |
| 优化器 | `src.optim` | AdamW, MultiStepLR, LinearWarmup |
| 日志 | `src.misc.logger` | MetricLogger |
| 配置 | `src.core.yaml_config`, `yaml_utils`, `_config`, `workspace` | 配置系统基础设施 |
| Box 操作 | `src.zoo.rtdetr.box_ops` | `box_cxcywh_to_xyxy`, `generalized_box_iou` |

### 3.3 外部依赖

```
torch, torchvision (>=0.27), numpy, opencv-python (cv2), pyyaml, scipy, tqdm, matplotlib, h5py
```

---

## 4. 模型架构

### 4.1 参数分布

| 组件 | 参数 | 说明 |
|------|------|------|
| PResNet-18 Backbone | 11.2M (51%) | depth=18, variant='d', frozen_norm=True, **pretrained** |
| HybridEncoder | 5.2M (24%) | hidden_dim=256, depth_mult=0.33 (lite), num_encoder_layers=1 |
| RTDETRTransformerv2Lite Decoder | 5.4M (25%) | 3 layers, 50 queries, 4 heads, expansion=2× |
| **总计** | **21.8M** | 原版 ~60M，缩减 **62%** |

### 4.2 Lite Decoder 参数详析

| 子模块 | 参数量 | 说明 |
|--------|--------|------|
| Transformer Layers ×3 | 4.5M | Self-attn + Cross-attn (MSDeformable) + FFN + Temporal-attn |
| Score + BBox Heads ×3 | 0.4M | 1 class + 4 coord (head + eye 双分支) |
| Input Projection | 0.2M | 3 级特征投影到 256ch |
| Query / Enc Embedding | 0.2M | Learnable query + encoder output projection |

**与原版 Decoder 的关键差异：**
- 移除了 CDN denoising 模块
- 移除了 `pred_blink` 眨眼分类头
- 移除了 `@register()` 依赖（self-contained class）
- `transform_expansion=2`（原版 4）：眼部分支 MLP 减半

### 4.3 数据流

```
输入: [B, T, 3, 360, 640]
  │
  ├─ backbone (PResNet-18): 独立处理每帧
  │   └─ [B*T, C, H/8, W/8], [B*T, 2C, H/16, W/16], [B*T, 4C, H/32, W/32]
  │
  ├─ encoder (HybridEncoder): FPN + RepVGG blocks + Transformer encoder
  │   └─ memory: 3× [B*T, 256, H/s, W/s]
  │
  ├─ decoder (RTDETRTransformerv2Lite):
  │   ├─ encoder top-k selection → 50 queries
  │   ├─ ×3 layers: self-attn → cross-attn (deformable) → FFN → temporal-attn → eye-branch
  │   └─ 输出: {pred_logits, pred_head_boxes, pred_eye_boxes}
  │
  └─ Stage 1 输出 → 桥接 (RoIAlign) → Stage 2 输入:
      blink_features [K×W, 16, 15360]  ← 3 scales × 5×4 grid × 256ch
      head_query     [K×W, 16, 256]    ← decoder head branch 特征
      eye_query      [K×W, 16, 256]    ← decoder eye branch 特征
```

---

## 5. 训练结果

### 5.1 快速验证训练（5 视频 / 1 epoch）

| 指标 | 值 |
|------|------|
| 配置 | `config_lite_quick.yml` |
| 训练数据 | 5 videos, 2,387 clips (clip_length=4) |
| 验证数据 | 3 videos, 2,065 clips |
| Batch size | 1 |
| 训练时间 | **7 分 15 秒** |
| GPU 内存 | 峰值 1,153 MB (~14%) |
| 最终 avg loss | 37,773 |
| 最终 avg bbox_head L1 | 5,539 |
| 最终 avg bbox_eye L1 | 5,476 |
| 最终 avg giou_head | 2.51 |
| 最终 avg giou_eye | 2.91 |

### 5.2 全量预处理数据（已就绪，未训练）

| 数据集 | 视频数 | 人数 | 状态 |
|--------|--------|------|------|
| train (full) | 40 | 78 | 已预处理 ✅ |
| val (full) | 10 | 18 | 已预处理 ✅ |

```bash
# 完整训练命令
python newtest/train_lite.py -c newtest/config_lite_test50.yml -d cuda:0 \
    --pretrained_backbone ResNet18_vd_pretrained_from_paddle.pth
```

### 5.3 保存的输出文件

```
output/lite_quick/
├── last.pth                (250 MB)  最终模型
├── checkpoint0000.pth      (250 MB)  epoch 0 checkpoint
├── 0_iter_999.pth          (250 MB)  中间检查点
├── 0_iter_1999.pth         (250 MB)  中间检查点
├── loss_detail.json                  loss 分量数据
├── loss_data.json                    简化 loss 数据
├── training_curves.png               训练曲线图
├── blink_loss.json                   (空，该 loss 未激活)
└── summary/                          TensorBoard 事件文件
```

---

## 6. 已知问题与待办

### 6.1 已修复

| 问题 | 修复 |
|------|------|
| torchvision 0.27 API 不兼容 | `_transform` → `transform`（`src/data/transforms/_transforms.py`） |
| YAML 配置参数名错误 | `img_folder→img_fold`, `anno_file→ann_file`, 移除 `sample_len`/`sample_stride` |
| transforms 缺少 type 字段 | 增加 `type: Compose` |
| `${data_path}` 未被解析 | 改为绝对路径 |
| COCO JSON categories 格式 | 修复为 list 格式 |
| collate key 名分析错误 | 确认链：`head_bbox`(collate)→`transform_targets`→`head_boxes`(loss)，原 collate 正确 |

### 6.2 待修复

| 问题 | 说明 | 建议 |
|------|------|------|
| `loss_vfl: 0.0` | 分类损失全程为零 | 检查 criterion 中 num_boxes 计数逻辑 |
| `loss_blinks` 未激活 | 原 blink 分类头已移除 | 如需 blink 损失需重新添加 |
| 仅 1 epoch 验证 | loss 趋势不明显 | 需要 20-30 epoch 看收敛 |

### 6.3 代码耦合

`newtest/` 深度依赖 `src/` 共 19 个模块（详见 §3.2）。如需完全独立，还需将：
- `src.solver.*` → 独立训练循环
- `src.data.dataset.mpeblink*` → 独立数据集类
- `src.zoo.rtdetr.*` → 独立 criterion / matcher / postprocessor
- `src.data.transforms.*` → 独立图像变换
- `src.nn.backbone.presnet` → 独立 backbone
- `src.zoo.rtdetr.hybrid_encoder` → 独立 encoder

迁移到 newtest。当前状态下，newtest 是**最小可行产品**（MVP），以最少新代码实现了完整训练 pipeline。

---

## 7. 快速参考

```bash
# 环境激活
D:\app\anaconda3\envs\defb\python.exe

# 可行性验证（无需数据）
python newtest/test_lite.py --device cuda:0

# 数据预处理
python newtest/prepare_data.py --root "E:/documents/mssb code/mpeblink2.0" --num-videos 50

# 快速训练测试（5 视频 / 1 epoch）
python newtest/train_lite.py -c newtest/config_lite_quick.yml -d cuda:0 \
    --pretrained_backbone ResNet18_vd_pretrained_from_paddle.pth

# 完整训练（40 train + 10 val / 30 epochs）
python newtest/train_lite.py -c newtest/config_lite_test50.yml -d cuda:0 \
    --pretrained_backbone ResNet18_vd_pretrained_from_paddle.pth

# 可视化
python newtest/visualize_training.py
```
