# 矿山视觉算法横向评测与部署验证平台 (vision-benchmark)

本项目面向煤矿安全生产视觉场景，提供跨算法架构（**`YOLOv8m`、`YOLOv8l-OBB`、`YOLOv9e`、`YOLO11m`、`YOLO11x`**）的统一容器化评测基准、5,700 张完备标准测试集、跨精度（**`FP32 / FP16 / INT8`**）与跨执行后端（**`PyTorch Eager / FastInfer / TorchScript C++ JIT`**）横向对比体系。

---

## 一、 核心文档与台账索引

| 文档 / 台账路径 | 说明 |
|---|---|
| [`docs/benchmark-full-table.md`](./docs/benchmark-full-table.md) | **全业务算法评测整表 Markdown 台账**（A～Z 共 26 列 × 45 行完整数据，随脚本自动同步） |
| [`docs/metrics-and-conventions.md`](./docs/metrics-and-conventions.md) | **环境约定、配置文件 (`args.yaml` / `.pt` 内嵌与反向生成) 与全指标直白详解**（含大中小目标像素换算、`conf/iou` 门限、INT8 量化等） |
| [`docs/dataset-standard-report.md`](./docs/dataset-standard-report.md) | **11 个完备标准测试集（5,700 张 / 17,037 框）分布报告、抽取逻辑与统计置信度评估** |
| [`configs/registry.py`](./configs/registry.py) | **全项目唯一配置真源（Single Source of Truth）**，集中定义 11 个数据集与 8 个业务模型权重 |

---

## 二、 基准软硬件环境与目录结构

- **基准硬件**：`NVIDIA GeForce RTX 4090` (24 GB GDDR6X)，Ubuntu Linux `7.0.0-38-generic`，Driver `595.58.03`
- **统一全栈镜像**：基于 `CUDA 12.8.1` + `PyTorch 2.9.0+cu128` + `Ultralytics 8.4.99` 构建（`docker/yolov8/requirements.lock.txt` 锁死全部依赖版本），原生支持 `YOLOv8 / YOLOv8-OBB / YOLOv9 / YOLO11` 全系列架构，已统一映射为 `vision/yolo-unified:8.4.99-cu12.8`（及 `vision/yolov8`、`vision/yolov9`、`vision/yolo11` 标签）。

```text
vision-benchmark/
├── docker-compose.yml              # 统一容器编排（支持 yolov8 / yolov9 / yolo11 / yolo-unified 服务）
├── docs/                           # 项目标准结果与说明文档目录
│   ├── benchmark-full-table.md     # 业务数据集与算法评测完整 Markdown 台账（A~Z 列 × 45 行）
│   ├── metrics-and-conventions.md  # 环境约定、args.yaml 配套说明与全指标直白详解
│   └── dataset-standard-report.md  # 11 个完备标准测试集（5,700 张）统计与抽取逻辑报告
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
│   └── 08-build-standard-set.py    # 跨全部矿区分层抽取 5,700 张完备标准测试集
├── models/yolov8/custom/           # 8 个业务模型权重 (.pt) 及配套 args.yaml / 模型配置 .yaml
├── data/benchmark/standard/        # 11 个业务标准测试集（共 5,700 张图片、labels、data.yaml、manifest.json）
└── output/benchmark/arena/         # 自动化评测结构化产物（JSON / Markdown 报告）
```

---

## 三、 统一启动入口 (`scripts/run.sh`)

所有算法架构、模型权重与评测流程均通过 `bash scripts/run.sh` 统一启动管理：

```bash
# 1. 查看已纳管的全部 11 个数据集、算法架构、分辨率与权重状态
bash scripts/run.sh list

# 2. 一键运行全量评测擂台并自动更新 docs/benchmark-full-table.md
bash scripts/run.sh all

# 3. 按需运行专项评测模块
bash scripts/run.sh matrix          # 8 模型全维度横向对比矩阵 (FP32/FP16/分尺度召回/FAR/P99)
bash scripts/run.sh backend         # 跨推理后端对比 (PyTorch Eager vs TorchScript C++ JIT 静态图)
bash scripts/run.sh stream          # 1/2/4/8/16 路 1080p 视频流并发阶梯压测
bash scripts/run.sh accel           # 官方基线 (b1) vs FastInfer (b16/FP16) 端到端加速对比
bash scripts/run.sh verify          # FP32 vs FP16 逐框 IoU 一致性校验

# 4. 新模型一键准入或单数据集重测（支持指定服务名 YOLO_SERVICE=yolov8|yolov9|yolo11）
YOLO_SERVICE=yolo11 bash scripts/run.sh admit ChuanGanQi

# 5. 将最新评测结果同步刷新至 docs/benchmark-full-table.md
bash scripts/run.sh sync-table
```

---

## 四、 核心评测结果总览（5,700 张完备标准测试集）

### 4.1 五大架构（8 个业务模型）精度、量化与工程诊断汇总

| 业务数据集 | 算法架构 | 分辨率 | 参数量 / 计算量 | 标准集规模 (覆盖矿区) | 标准集精度 `mAP50 / mAP50-95` (`FP16` \| `INT8量化`) | 查准率 `P` / 召回率 `R` | 分尺度召回率 (`小 / 中 / 大目标`) | 误报率 `FAR` (背景图误报框) |
|---|---|---|---|---|---|---|---|---|
| **人安全帽** (`PersonHelmet`) | `YOLO11x` | 960 | `56.84M / 437.4G` | 600张 (330矿区) | `0.9778 / 0.7625` (`FP16:0.9778 \| INT8:0.9717`) | `0.9622 / 0.9372` | `0.909 / 0.964 / 0.984` | `4.98%` (14框/243背景图) |
| **人睡觉** (`PersonSleep`) | `YOLOv8m` | 640 | `25.84M / 78.7G` | 500张 (21/21矿区) | `0.9939 / 0.8852` (`FP16:0.9939 \| INT8:0.9938`) | `0.9898 / 0.9909` | `0.995 / 0.997 / 1.000` | `1.61%` (3框/23背景图) |
| **人工作服** (`WorkclothesHelmet`) | `YOLOv8m` | 640 | `25.84M / 78.7G` | 500张 (13/13矿区) | `0.9890 / 0.8689` (`FP16:0.9891 \| INT8:0.9877`) | `0.9854 / 0.9799` | `0.974 / 0.995 / 0.996` | `2.65%` (0框/2背景图) |
| **皮带原版** (`PiDaiYuanBan`) | `YOLO11m` | 640 | `20.04M / 67.7G` | 600张 (272/273矿区) | `0.9866 / 0.9145` (`FP16:0.9866 \| INT8:0.9850`) | `0.9760 / 0.9693` | `0.985 / 0.984 / 0.992` | `2.57%` (0框/0背景图) |
| **探水后上杆** (`TanShuiHouShangGan`) | `YOLOv9e` | 960 | `57.29M / 424.3G` | 500张 (58/58矿区) | `0.8263 / 0.6334` (`FP16:0.8263 \| INT8:0.8196`) | `0.9235 / 0.7816` | `0.348 / 0.696 / 0.780` | `6.52%` (0框/2背景图) |
| **风门开关** (`FengMenKaiGuan`) | `YOLO11m` | 640 | `20.03M / 67.6G` | 500张 (10/10矿区) | `0.9947 / 0.9895` (`FP16:0.9947 \| INT8:0.9938`) | `0.9949 / 0.9959` | `无样本 / 0.972 / 1.000` | `0.61%` (1框/11背景图) |
| **传感器** (`ChuanGanQi`) | `YOLO11m` | 640 | `20.03M / 67.7G` | 400张 (8/8矿区) | `0.9950 / 0.9660` (`FP16:0.9950 \| INT8:0.9950`) | `0.9984 / 1.0000` | `1.000 / 1.000 / 1.000` | `0.08%` (0框/0背景图) |
| **探水前上杆** (`TS_Front_V2` OBB) | `YOLOv8l-OBB` | 640 | `44.46M / 168.5G` | 600张 (4/4矿区) | `0.9949 / 0.9731` (`FP16:0.9949 \| INT8:0.9949`) | `0.9966 / 0.9962` | `无样本 / 1.000 / 0.997` | `1.36%` (0框/0背景图) |

> **注**：其余 3 个数据集（`人乘猴车` 500 张、`护帮板` 500 张、`卡车空满载` 500 张）已完成全矿区分层抽样与标准测试集构建，待上游交付 `.pt` 权重后运行 `bash scripts/run.sh admit <dataset_key>` 即可一键准入。

---

### 4.2 吞吐率、单帧 P99 尾延迟与 C++ JIT 静态图加速汇总

| 业务数据集 | 算法架构 | 官方基线 FPS (`b1/FP32`) | 优化最佳端到端 FPS (`FastInfer FP16/b16`) | 纯 GPU 前向极限 FPS (`FP16/b16`) | 单帧 `b1 FP16` Mean / P99 延迟 | TorchScript `C++ JIT b1` 帧率 (加速比) |
|---|---|---|---|---|---|---|
| **人安全帽** (`PersonHelmet`) | `YOLO11x` (@960) | 20.2 FPS | **103.0 FPS (5.10x)** | **125.7 FPS** | `27.01 / 27.46 ms` | **61.8 FPS (2.01x)** |
| **人睡觉** (`PersonSleep`) | `YOLOv8m` (@640) | 27.2 FPS | **259.2 FPS (9.53x)** | **756.4 FPS** | `13.23 / 13.39 ms` | **116.6 FPS (1.90x)** |
| **人工作服** (`WorkclothesHelmet`) | `YOLOv8m` (@640) | 27.5 FPS | **164.0 FPS (5.96x)** | **758.2 FPS** | `13.33 / 13.54 ms` | **116.6 FPS (1.90x)** |
| **皮带原版** (`PiDaiYuanBan`) | `YOLO11m` (@640) | 24.4 FPS | **124.2 FPS (5.09x)** | **735.3 FPS** | `17.02 / 19.09 ms` | **90.8 FPS (1.59x)** |
| **探水后上杆** (`TanShuiHouShangGan`) | `YOLOv9e` (@960) | 21.0 FPS | **90.0 FPS (4.29x)** | **90.9 FPS** | `52.06 / 53.18 ms` | **39.6 FPS (1.65x)** |
| **风门开关** (`FengMenKaiGuan`) | `YOLO11m` (@640) | 27.0 FPS | **172.0 FPS (6.37x)** | **728.9 FPS** | `17.49 / 17.60 ms` | **90.8 FPS (1.59x)** |
| **传感器** (`ChuanGanQi`) | `YOLO11m` (@640) | 27.9 FPS | **219.9 FPS (7.88x)** | **719.9 FPS** | `21.59 / 21.71 ms` | **90.8 FPS (1.59x)** |
| **探水前上杆** (`TS_Front_V2` OBB) | `YOLOv8l-OBB` (@640) | 40.6 FPS | **40.6 FPS (OBB标准)** | **445.4 FPS** | `17.21 / 17.37 ms` | **69.9 FPS (1.53x)** |

---

### 4.3 核心选型与工程部署结论

1. **中量级主力架构对比（`YOLO11m` vs `YOLOv8m`）**：
   - `YOLO11m` 参数量（`20.03M` vs `25.84M`）**减少 22.5%**，计算量（`67.7` vs `78.7 GFLOPs`）**降低 14.0%**，权重体积缩小 11 MB，在多路视频流微批并发下吞吐更高（8 路 1080p 并发下总吞吐 `193.8 FPS` vs `134.3 FPS`）。
2. **重型高精度架构对比（`YOLO11x-960` vs `YOLOv9e-960`）**：
   - 在 ~57M 参数、~430 GFLOPs@960 同等算力规模下，**`YOLO11x` 的纯 GPU 吞吐（`125.7 FPS`）比 `YOLOv9e`（`90.9 FPS`）快 38.3%**，单帧 P99 延迟（`27.5ms` vs `53.2ms`）降低近一半；跨 330 个矿区的广域泛化测试中 `YOLO11x` 达到 `mAP50 = 0.9778`，显著优于同配置 `YOLOv8m`（`0.4291`）。
3. **量化部署结论（`FP16` 与跨硬件 `INT8`）**：
   - **GPU `FP16`**：8 个模型 `mAP50` 与 `FP32` 差值 $\le 0.0001$，逐框校验 0 漏检，可无条件默认开启；
   - **边缘异构硬件 `INT8 (W8A8 PTQ)`**：8 个模型的 `mAP50` 衰减全部控制在 **`0.00% ~ 0.67%`** 以内；转换为昇腾 OM / 瑞芯微 RKNN / TensorRT INT8 时，建议直接使用 `data/benchmark/standard/<业务>/images`（400～600 张）作为校准集，并对最后一层 Detect 检测头保持 `FP16` 混合精度以兼顾高 IoU 框回归精度。
4. **单路低延迟与多路高并发策略**：
   - **单路实时告警（`batch=1`）**：导出为 **TorchScript C++ JIT 静态图**可消除 Python 算子派发开销，单帧速度提升 **`1.53x ~ 2.01x`**；
   - **多路 RTSP 视频流（`1080p@25fps`）**：启用跨路动态微批聚合（Micro-Batching）后，单张 RTX 4090 在 `4 路 25fps 满帧` 或 `16 路 15fps 抽帧` 下显存占用仅 `444 ~ 529 MB`（不足总显存的 3%）。

---

## 五、 Git 仓库版本控制约定

本项目通过 [`.gitignore`](./.gitignore) 严格分离**代码/配置/台账/溯源清单（入库，总计 `< 2 MB`）**与**大体积权重/图片视频二进制（不入库）**：

- **纳入 Git 仓库跟踪**：
  - 核心代码与脚本：`configs/*.py`、`scripts/*.sh`、`scripts/*.py`、`docker/yolov8/*`、`docker-compose.yml`
  - 结果台账与文档：`README.md`、`docs/*.md`（含 26 列 × 45 行全量台账 `docs/benchmark-full-table.md`）、`output/benchmark/arena/*.json`、`output/benchmark/arena/*.md`
  - 模型配套训练与类别配置：`models/yolov8/custom/**/args.yaml`、`models/yolov8/custom/**/*.yaml`
  - 标准测试集抽样溯源清单：`data/benchmark/standard/*/data.yaml`、`manifest.json`、`excluded.txt`
- **排除在 Git 仓库之外（通过脚本一键拉取或复现）**：
  - 二进制模型权重（`models/**/*.pt`，528 MB）：仅保存在本机 `models/` 目录，评测直接使用本地权重，不提供外部下载链接；
  - 标准测试集图片与标签（`data/benchmark/standard/*/{images,labels}`，3.1 GB）：由 `scripts/08-build-standard-set.py` 幂等复现；
  - 运行日志与本机私有代理配置：`logs/`、`.mihomo/`。
