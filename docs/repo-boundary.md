# 仓库边界清单：哪些不入库

**原则：数据集、模型权重、凭据一律不进 git。**

原因不是体积大小，而是 git 的两个特性让误入库的代价很高：

1. 二进制文件无法有效增量存储，每次改动都会完整存一份新副本
2. **一旦提交并推送到远程，文件就永久留在历史里**。后续 `git rm` 只是不再引用，
   克隆仓库的人仍能完整取回。要真正删除必须重写历史并强推，会影响所有协作者

因此仓库只存放**体积很小的文本类资产**：代码、配置、文档、类别定义，以及
体积以 KB 计的统计结论（`manifest.json`、`excluded.txt`、`report.json` 等）。

`.gitignore` 已改为**按扩展名全局拦截**，不依赖具体目录名，新增数据集或模型时不会漏。

---

## 一、数据集（严禁入库）

| 路径 | 体积 | 内容 |
|---|---:|---|
| `data/benchmark/standard/*/images/` | 3.1 GB | 11 个标准测试集的 5,700 张原始图片 |
| `data/benchmark/standard/*/labels/` | 21 MB | 对应标注文件 |
| `data/benchmark/videos/` | 234 MB | 测试视频 || `data/benchmark/train_sample/` | — | 训练抽样 |
| `data/raw/` | — | 原始数据 |
| `~/Dataset/`（仓库外） | 878 GB | 42 个业务数据集全量原始数据 |

**入库的只有**：每个标准集目录下的 `data.yaml`、`manifest.json`、`excluded.txt`
（抽样溯源清单，合计 33 个文件、约 1.1 MB）。

## 二、模型权重（严禁入库）

| 路径 | 体积 | 内容 |
|---|---:|---|
| `models/yolov8/custom/**/*.pt` | 527 MB | 9 个业务模型训练成果 |
| `models/yolov8/{detect,segment,pose,obb,classify}/` | 6.2 MB | 官方预训练权重 |
| `models/**/weights/` | — | 训练过程权重 |
| `/mnt/bak/CoreYoloU_runs/**/*.pt`（仓库外） | 2.5 GB | 82 个训练产物，11 个数据集 + 统一模型 |

**入库的只有**：每个模型目录下的 `args.yaml`（训练参数）与模型结构 `.yaml`
（如 `yolov11.yaml`），合计 12 个文件，用于复现训练配置。

单个权重最大 111.8 MB（`探水后上杆-yolov9e-960.pt`），超过 GitHub 100 MB
单文件限制，即便想入库也必须走 Git LFS。

**入库的例外**：`output/dataset/weights-manifest.json`（28 KB）只含 91 个权重的
路径与 sha256，不含权重本体，用于校验备份是否完整。

## 三、凭据与密钥（严禁入库）

| 文件 | 说明 |
|---|---|
| `~/.ssh/id_ed25519_102` | GitHub 账号级私钥，无口令保护 |
| `~/.local/share/mihomo/config.yaml` | 代理节点与订阅凭据 |
| `*.pem`、`*.key`、`.env*` | 通用私钥与环境变量 |
| `~/.git-credentials`、`.netrc` | git/HTTP 凭据 |

## 四、可重建的中间产物（不入库）

| 路径 | 体积 | 重建方式 |
|---|---:|---|
| `output/dataset/benchmark-quality-v1/*/*.json` | 9.6 MB | `scripts/15` 重跑质检 |
| `output/dataset/benchmark-audit-v1/` | 3.2 MB | `scripts/14` 重跑审计 |
| `data/benchmark/standard/*/labels.cache` | 1.1 MB | ultralytics 首次训练自动生成 |
| `output/dataset/*.log`、`logs/` | 0.7 MB | 运行日志 |

**入库的只有**：`report.json`、`issues.csv`、`CHECKSUMS.txt` 与 11 个
`summary.json`（合计约 400 KB），这些是质检结论，重建需要数小时磁盘 I/O。

## 五、当前仓库实际内容

截至最新提交，仓库共跟踪文件 134 个，最大的 5 个是：

| 文件 | 体积 |
|---|---:|
| `legacy/class-names/total-train.txt.gz` | 811 KB |
| `output/dataset/benchmark-quality-v1/report.json` | 216 KB |
| `output/dataset/benchmark-quality-v1/PersonHelmet/summary.json` | 101 KB |
| `data/benchmark/standard/TS_Front_V2/excluded.txt` | 92 KB |
| `data/benchmark/standard/PersonHelmet/excluded.txt` | 82 KB |

已核查 git 全部历史：**从未提交过任何图片、视频或权重文件**，历史上最大的
blob 是 811 KB 的压缩文本清单。仓库没有需要清理的历史包袱。

## 六、校验方法

随时可以确认没有误入库：

```bash
# 查看仓库内所有二进制/数据类文件（应为空）
git ls-files | grep -iE '\.(jpg|png|mp4|pt|onnx|engine|zip|db|log|cache)$'

# 查看误入库的文件的大小排名
git ls-tree -r -l HEAD | sort -k4 -nr | head -10

# 检查某个新文件是否会被拦下
git check-ignore -v path/to/new/file.jpg
```
