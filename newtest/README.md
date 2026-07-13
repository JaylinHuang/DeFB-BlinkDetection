# DeFB Lite Track Model — 人脸 + 眼部定位模块

## 快速开始

### 1. 可行性测试（无需数据集）

```bash
# 仅验证模型架构、形状兼容性、time-mamba 对接
python newtest/test_lite.py --device cuda:0
```

### 2. 训练（需要 mpeblink v2 数据集）

```bash
# 单卡训练，batch_size=2，直接运行（不用 torchrun）
python newtest/train_lite.py -c newtest/config_lite.yml -d cuda:0
```

### 3. 配置调整

所有架构参数在 `newtest/track_model_lite.py` 的 `build_lite_model()` 函数中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone_depth | 18 | PResNet 深度 (18/34/50) |
| decoder_num_layers | 3 | Decoder 层数 |
| decoder_num_queries | 50 | Query 数量 |
| decoder_nhead | 4 | 注意力头数 |
| decoder_transform_expansion | 2 | 眼部分支 MLP 展开比 |

训练参数在 `newtest/config_lite.yml` 中：
- `total_batch_size: 2` — 单卡小 batch
- `num_workers: 0` — 主进程加载数据
- `sync_bn: False` — 单卡不需要
- `use_amp: False` — 如需混合精度可改为 True

### 4. 预期效果

- 模型总参数 ~23M（原版 ~60M，缩减 62%）
- 输出接口与原 DeFB Stage 1 完全兼容
- `blink_features [K×W, 16, 15360]` + `head_query [K×W, 16, 256]` + `eye_query [K×W, 16, 256]`
- 可直接对接 time-mamba 模块
