# 矿山视觉算法横向评测与部署验证平台 (vision-benchmark)

本项目面向煤矿安全生产视觉场景，提供以 Ultralytics YOLO 水平框和旋转框检测为主的容器化评测，以及标准测试集、精度和执行后端对比工具。现有脚本并非任意算法或所有视觉任务的通用适配器；同卷可比性、结果溯源及任务扩展的限制与整体建设方案见 [`docs/metrics-and-conventions.md`](./docs/metrics-and-conventions.md) 的「一附」。

---

## 一、 文档与台账职责分工（单一归属）

每份文档只负责各自维度的数据或规范，互不重复存放：

| 文档 / 配置路径 | 职责边界 |
|---|---|
| [`docs/benchmark-full-table.md`](./docs/benchmark-full-table.md) | **全业务算法评测整表台账**（A～Z 共 26 列 × 45 行全量评测数据，唯一结果明细表） |
| [`docs/metrics-and-conventions.md`](./docs/metrics-and-conventions.md) | **评测环境与指标定义规范**（统一容器环境、目标尺度划分、告警门限、INT8 量化实验及各列物理定义） |
| [`docs/benchmark-dataset-inventory.md`](./docs/benchmark-dataset-inventory.md) | **11 个台账基准数据集地址与维度清单**（原始集、标准集路径与统计口径差异） |
| [`docs/benchmark-quality-report.md`](./docs/benchmark-quality-report.md) | **11 个基准数据集全量质检报告**（1,602,513 张逐张解码 / 4,670,797 框逐行校验，含全部异常定位、一键修复与 sha256 校验清单） |
| [`docs/dataset-standard-report.md`](./docs/dataset-standard-report.md) | **标准测试集来源、抽取方法与评价报告**（5,700 张 / 17,037 框的逐集实测统计、抽样做法、7 层防泄漏防线与已知缺口） |
| [`docs/dataset-full-inventory.md`](./docs/dataset-full-inventory.md) | **全量原始数据集统计清单**（`/home/jykj/Dataset/` 下 **42 个业务 / 1,979,306 张 / 5,831,781 框 / 878.3 GB** 的逐集详细统计、类别分布、尺度分布与数据质量异常清单） |
| [`docs/backup-checklist.md`](./docs/backup-checklist.md) | **重装前备份清单**（哪些内容已由 git 保护、哪些必须另存、可重建项与敏感凭据处理方式） |
| [`docs/repo-boundary.md`](./docs/repo-boundary.md) | **仓库边界清单**（数据集、权重、凭据为何不入库及 `.gitignore` 拦截规则，含校验命令） |
| [`configs/registry.py`](./configs/registry.py) | **全项目代码唯一配置真源**（集中声明 11 个数据集路径、8 个模型权重及基准参数，供所有评测与同步脚本调用） |

---

## 二、 基准软硬件环境与目录结构

- **基准硬件**：`NVIDIA GeForce RTX 4090` (24 GB GDDR6X)，Ubuntu Linux `7.0.0-38-generic`，Driver `595.58.03`
- **统一全栈镜像**：基于 `CUDA 12.8.1` + `PyTorch 2.9.0+cu128` + `Ultralytics 8.4.99` 构建（`docker/yolov8/requirements.lock.txt` 锁定全部依赖），原生支持 `YOLOv8 / YOLOv8-OBB / YOLOv9 / YOLO11` 全系列架构，映射为 `vision/yolo-unified:8.4.99-cu12.8`（及 `yolov8`、`yolov9`、`yolo11` 标签）。

```text
vision-benchmark/
├── docker-compose.yml              # 统一容器编排（支持 yolov8 / yolov9 / yolo11 / yolo-unified 服务）
├── docs/                           # 项目标准结果与规范文档目录
│   ├── benchmark-full-table.md     # 业务数据集与算法评测完整 Markdown 台账（A~Z 列 × 45 行）
│   ├── benchmark-dataset-inventory.md # 11 个基准数据集地址与维度清单
│   ├── benchmark-quality-report.md # 11 个基准数据集全量质检报告（全量解码 + 标注校验）
│   ├── metrics-and-conventions.md  # 评测环境与全指标定义规范
│   ├── dataset-standard-report.md  # 11 个完备标准测试集（5,700 张）来源、抽取方法与评价报告
│   └── dataset-full-inventory.md   # 全量原始数据集统计清单（42 个业务 / 197.9 万张 / 583.2 万框 / 878 GB）
├── configs/                        # 容器内评测核心代码（挂载为 /configs）
│   ├── registry.py                 # 唯一事实源：11 个数据集与 8 个模型的配置及实测基准
│   ├── algo_arena.py               # 统一评测引擎（架构矩阵 / 同卷PK / C++静态图 / 多路视频流压测）
│   ├── fast_infer.py               # 多线程异步解码 + Pinned Memory + 动态批处理推理引擎
│   ├── accel_compare.py            # 官方基线 vs FastInfer 端到端吞吐对比
│   └── verify_fast.py              # FP32 vs FP16 逐框 IoU 零漏检一致性校验
├── scripts/                        # 宿主机统一入口与工具脚本（挂载为 /scripts）
│   ├── run.sh                      # 【全项目唯一启动入口】统一管理全部权重与评测任务
│   ├── 05-sync-benchmark-table.py  # 自动刷新 docs/benchmark-full-table.md 台账 J~Z 列
│   ├── 07-extract-v9e-weights.py   # YOLOv9e DualDDetect 权重无损转换脚本
│   ├── 08-build-standard-set.py    # 标准测试集补齐器（多矿区注水轮询，目标 5,700 张）
│   ├── 09-dataset-full-inventory.py # 全量原始数据集统计扫描（生成 docs/dataset-full-inventory.md）
│   ├── 10-scan-dataset-inventory.sh # 全量数据集统计的后台守护式增量扫描入口
│   ├── 11-dataset-shard-scan.py    # 单业务分片扫描（供 10/12 调用）
│   ├── 12-dataset-shard-daemon.sh  # 分片扫描守护进程
│   ├── 14-benchmark-dataset-audit.py # 11 个基准数据集文件级审计（配对 / 框数 / 类别 / 尺度）
│   ├── 15-benchmark-quality-audit.py # 11 个基准数据集内容级质检（全量解码 + 标注校验）
│   ├── 16-quality-audit-daemon.sh   # 内容级质检的后台守护式入口
│   ├── 19-benchmark-quality-report.py # 汇总质检结果并生成 docs/benchmark-quality-report.md
│   ├── 20-benchmark-quality-checksums.py # 生成/校验质检产物的 sha256 清单
│   ├── 21-fix-bad-labels.py         # 清理已定位的 14 个异常标注（默认 dry-run + 备份）
│   ├── 22-push-with-token.sh        # 备用推送方式（HTTPS + token；本机默认走 SSH）
├── models/yolov8/custom/           # 8 个业务模型权重 (.pt) 及配套训练/类别配置 .yaml
├── data/benchmark/standard/        # 11 个业务标准测试集（共 5,700 张图片、labels、data.yaml、manifest.json）
├── output/benchmark/arena/         # 自动化评测结构化产物（JSON / Markdown 报告）
└── output/dataset/benchmark-quality-v1/ # 全量质检产物（report.json、issues.csv、CHECKSUMS.txt、逐数据集 summary.json）
```

---

## 三、 统一启动入口 (`scripts/run.sh`)

所有算法架构、模型权重与评测流程均通过 `bash scripts/run.sh` 统一启动管理：

```bash
# 1. 查看已纳管的全部 11 个数据集、算法架构、分辨率与权重状态
bash scripts/run.sh list

# 2. 对单张图片或图片目录执行推理（自动匹配权重与最佳分辨率，输出画框图与 predictions.json）
bash scripts/run.sh infer PersonHelmet data/benchmark/standard/PersonHelmet/images/xxx.jpg
bash scripts/run.sh infer 人睡觉 /path/to/your_image.jpg 0.35
bash scripts/run.sh infer models/yolov8/custom/belt/yolov11m_belt.pt /path/to/your_image.jpg

# 3. 一键运行全量评测擂台并自动更新 docs/benchmark-full-table.md
bash scripts/run.sh all

# 4. 按需运行专项评测模块
bash scripts/run.sh benchmark --help # 新版可追溯同卷评测及可比性检查，详见指标规范「一附」
bash scripts/run.sh matrix          # 历史矩阵（不同业务数据集，不能直接用于算法排名）
bash scripts/run.sh backend         # 跨推理后端对比 (PyTorch Eager vs TorchScript C++ JIT 静态图)
bash scripts/run.sh stream          # 1/2/4/8/16 路 1080p 视频流并发阶梯压测
bash scripts/run.sh accel           # 官方基线 (b1) vs FastInfer (b16/FP16) 端到端加速对比
bash scripts/run.sh verify          # FP32 vs FP16 逐框 IoU 一致性校验

# 5. 新模型一键准入或单数据集重测（支持指定服务名 YOLO_SERVICE=yolov8|yolov9|yolo11）
YOLO_SERVICE=yolo11 bash scripts/run.sh admit ChuanGanQi

# 6. 将最新评测结果同步刷新至 docs/benchmark-full-table.md
bash scripts/run.sh sync-table
```

---

## 四、 核心评测与选型结论总结

> 各数据集与各标签的具体评测数值详见唯一结果台账 [`docs/benchmark-full-table.md`](./docs/benchmark-full-table.md)。

1. **中量级主力架构选型（`YOLO11m` vs `YOLOv8m`）**：
   - `YOLO11m` 参数量（`20.03M` vs `25.84M`）**减少 22.5%**，计算量（`67.7` vs `78.7 GFLOPs`）**降低 14.0%**，权重体积缩小 11 MB，在多路视频流微批并发下吞吐更高（8 路 1080p 并发下总吞吐 `193.8 FPS` vs `134.3 FPS`），常规 `640` 分辨率业务优先选用 `YOLO11m`。
2. **重型高精与小目标架构选型（`YOLO11x-960` vs `YOLOv9e-960`）**：
   - 在 ~57M 参数、~430 GFLOPs@960 同等算力规模下，**`YOLO11x` 的纯 GPU 吞吐（`125.7 FPS`）比 `YOLOv9e`（`90.9 FPS`）快 38.3%**，单帧 P99 延迟（`27.5ms` vs `53.2ms`）降低近一半；跨 330 个矿区的广域泛化测试中 `YOLO11x` 达到 `mAP50 = 0.9778`，显著优于同配置 `YOLOv8m`（`0.4291`）。
3. **量化部署结论（`FP16` 与跨硬件 `INT8`）**：
   - **GPU `FP16`**：8 个模型 `mAP50` 与 `FP32` 差值 `≤ 0.0001`，逐框校验 0 漏检，生产环境默认开启；
   - **边缘异构硬件 `INT8 (W8A8 PTQ)`**：8 个模型的 `mAP50` 衰减全部控制在 **`0.00% ~ 0.67%`** 以内；转换为昇腾 OM / 瑞芯微 RKNN / TensorRT INT8 时，直接使用 `data/benchmark/standard/<业务>/images`（400～600 张）作为校准集，并对最后一层 Detect 检测头保持 `FP16` 混合精度。
4. **单路低延迟与多路高并发策略**：
   - **单路实时告警（`batch=1`）**：导出为 **TorchScript C++ JIT 静态图**消除 Python 算子派发开销，单帧速度提升 **`1.53x ~ 2.01x`**；
   - **多路 RTSP 视频流（`1080p@25fps`）**：启用跨路动态微批聚合（Micro-Batching）后，单张 RTX 4090 在 `4 路 25fps 满帧` 或 `16 路 15fps 抽帧` 下显存占用仅 `444 ~ 529 MB`（不足总显存的 3%）。
5. **待交付权重准入机制**：
   - `人乘猴车`、`护帮板`、`卡车空满载` 3 个业务已完成 500 张/集的标准测试集抽样落盘，后续拿到 `.pt` 权重后填入 `configs/registry.py` 并执行 `bash scripts/run.sh admit <dataset_key>` 即可自动完成评测与台账回填。

---

## 五、 Git 仓库版本控制约定

本项目通过 [`.gitignore`](./.gitignore) 分离**代码/配置/台账/溯源清单（入库，总计 `< 2 MB`）**与**大体积权重/图片视频二进制（不入库）**：

- **纳入 Git 仓库跟踪**：
  - 核心代码与脚本：`configs/*.py`、`scripts/*.sh`、`scripts/*.py`、`docker/yolov8/*`、`docker-compose.yml`
  - 结果台账与文档：`README.md`、`docs/*.md`、`output/benchmark/arena/*.json`、`output/benchmark/arena/*.md`
  - 模型配套训练与类别配置：`models/yolov8/custom/**/args.yaml`、`models/yolov8/custom/**/*.yaml`
  - 标准测试集抽样溯源清单：`data/benchmark/standard/*/data.yaml`、`manifest.json`、`excluded.txt`
- **排除在 Git 仓库之外**：
  - 二进制模型权重（`models/**/*.pt`，528 MB）：仅保存在本机 `models/` 目录，评测直接使用本地权重；
  - 标准测试集图片与标签（`data/benchmark/standard/*/{images,labels}`，3.1 GB）：由 `scripts/08-build-standard-set.py` 依据 `manifest.json` 复现；
  - 运行日志：`logs/`。
