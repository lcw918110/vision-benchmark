# 训练实验记录

从 `/mnt/bak/CoreYoloU_runs` 提取的**训练参数与训练曲线**。权重本体（.pt）体积达
2.5 GB 且为二进制，不入库；这里只保存可复现训练所需的文本记录。

## 内容

每个实验目录包含两个文件：

| 文件 | 内容 |
|---|---|
| `args.yaml` | 完整训练参数（模型结构、数据集、超参、增强策略、优化器设置） |
| `results.csv` | 逐 epoch 训练曲线（损失、precision/recall、mAP50、mAP50-95、学习率） |

## 实验分组

| 分组 | 实验数 | 说明 |
|---|---:|---|
| `full/` | 12 | 11 个业务数据集的全量训练，命名 `<数据集>-g<显卡号>` |
| `unified_100k/` | 7 | 统一模型，10 万样本规模 |
| `mix/` | 4 | 混合数据训练 |
| `s640/` | 4 | 640 分辨率训练 |
| `ft320/` | 2 | 320 分辨率微调 |
| `full_s640/` | 4 | 全量数据 + 640 分辨率 |
| `unified_20k/` | 3 | 统一模型，2 万样本 |
| `unified_50k/` | 2 | 统一模型，5 万样本 |
| `unified_50k_strat/` | 1 | 5 万样本分层采样 |
| `unified_full/` | 2 | 全量统一模型基准 |

## 对应的权重位置

权重不在仓库内，本机路径为：

```
/mnt/bak/CoreYoloU_runs/<分组>/<实验名>/weights/best.pt
/mnt/bak/CoreYoloU_runs/<分组>/<实验名>/weights/last.pt
```

`_keep_best/` 下另有 19 个精选权重，命名格式 `<方案>__<数据集>_<策略>.pt`，
例如 `micro__PersonSleep_s640.pt`、`p2__PiDaiYuanBan-mix-g1.pt`。

全部 91 个权重的路径、大小与 sha256 见
[`output/dataset/weights-manifest.json`](../output/dataset/weights-manifest.json)。

## 未纳入的内容

- `_trash/`（912 KB）：废弃实验的中间产物
- `*.jpg`（训练批次可视化图、混淆矩阵、PR 曲线图）：二进制，可由 `results.csv` 重绘
- `weights/*.pt`：权重本体，见上

## 用途

`results.csv` 可直接用于对比不同训练策略的收敛速度与最终精度，例如：

```bash
python3 - <<'EOF'
import csv, pathlib
for f in sorted(pathlib.Path('training-runs').rglob('results.csv')):
    rows = list(csv.DictReader(f.open()))
    if not rows: continue
    best = max(float(r['metrics/mAP50-95(B)']) for r in rows)
    print(f'{best:.4f}  {len(rows):3d}ep  {f.parent.name}')
EOF
```
