# Stage 1 完整训练包（本地补推）

DeFB 轻量化人脸/人眼定位：PResNet-18 替换 PResNet-50（约 59.5M → 21.8M），RT-DETRv2 Decoder 双分支 `head_bbox` + `eye_bbox`。

原先只在本地 `E:\stage1_package`，未进仓库。入口已从旧的 `newtest.*` 引用改到本目录。

## 运行

```bash
# 可行性验证（无需数据集）
python stage1/test_stage1.py --device cuda:0

# 训练
python stage1/train_stage1.py -c stage1/config_stage1.yml -d cuda:0
```

## 权重

仓库忽略 `*.pth`。请把 `ResNet18_vd_pretrained_from_paddle.pth` 放到 `weights/` 或训练脚本会搜索的路径。

## 数据

HUST-LEBW 配置见 `src/configs/dataset/hust.yml`（需改成本机数据路径）。
