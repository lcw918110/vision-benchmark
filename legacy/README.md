# 历史数据准备脚本与类别定义

本目录保存早期（`~/ZJH`）的数据集准备脚本与各业务类别定义，原位于家目录、
不受版本控制。机器重装前迁入仓库统一管理。

## `zhj-scripts/` — 数据准备脚本

| 脚本 | 用途 |
|---|---|
| `CheckSameNameInTxt.py` | 检查 txt 清单中是否存在重名文件 |
| `CompareTwoTxtKeepSame.py` | 对比两个 txt 清单，保留交集 |
| `CountImageAnno.py` | 统计图片与标注数量 |
| `label.py` | 标注格式转换 |
| `MoveJpgAndOtherInTxtToSignPath1.py` | 按 txt 清单批量复制图片与标签到指定目录 |
| `SelectSignClassToTxt.py` | 按类别筛选标注并输出清单 |

这些脚本使用硬编码绝对路径（`/home/jykj/Dataset/...`），换机后需先改路径。

## `class-names/` — 类别定义

各数据集的 `*.names` 文件，每行一个类别名，行号即类别 ID。与
`data/benchmark/standard/*/data.yaml` 中的 `names` 对应。

| 文件 | 数据集 |
|---|---|
| `人安全帽labels.names` | PersonHelmet |
| `人睡觉labels.names` | PersonSleep |
| `人工作服安全帽labels.names` | WorkclothesHelmet |
| `人乘猴车labels.names` | RenChengHouChe |
| `皮带原版labels.names` | PiDaiYuanBan |
| `护帮板labels.names` | HuBangBan |
| `探水前上杆labels.names` / `探水后上杆labels.names` | TanShuiHouShangGan |
| `风门开关labels.names` | FengMenKaiGuan |
| `传感器labels.names` | ChuanGanQi |
| `产运销煤种labels.names` | ChanYunXiaoMeiZhong |
| `obj.names` | 通用/早期定义 |

## 路径清单

- `total-train.txt.gz` — 训练集全量图片路径清单（169,687 行，压缩自 19 MB）
- `batch_violation_old.txt.gz` — 违规样本清单（15,667 行）
- `batch_8_in_testdataset_same.txt` — 与测试集重叠的样本清单（137 行）
- `train.txt`、`dataset_dir.txt` — 早期训练清单与数据根目录记录

清单内是绝对路径，仅作历史记录与去重依据，不保证换机后仍有效。
