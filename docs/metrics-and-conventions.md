# 矿山视觉算法横向评测平台 — 环境约定、配置文件与全指标直白详解

> **归档路径**：`docs/metrics-and-conventions.md`  
> **配套文档**：
> - 整表 Markdown 台账：[`docs/benchmark-full-table.md`](./benchmark-full-table.md)
> - 标准测试集完备性报告：[`docs/dataset-standard-report.md`](./dataset-standard-report.md)

---

## 一、 五大算法架构统一镜像与统一启动入口约定

### 1.1 五大架构（8 个模型）的镜像支持现状
本项目涉及 **5 种视觉算法架构**：`YOLOv8m`、`YOLOv8l-OBB`（旋转框）、`YOLOv9e`、`YOLO11m`、`YOLO11x`。
- **底层引擎统一性**：在 `Ultralytics 8.4.99`（配合 `PyTorch 2.9.0+cu128`）中，官方框架已原生内置 `YOLOv8`、`YOLOv8-OBB`、`YOLOv9`、`YOLO11` 的全部算子（`C2f`、`RepNCSPELAN4`、`ADown`、`SPPELAN`、`C3k2`、`C2PSA`、`OBB` 旋转检测头等）。针对由第三方 `WongKinYiu/yolov9` 仓库训练的 `探水后上杆-yolov9e-960.pt`，我们也已通过 `scripts/07-extract-v9e-weights.py` 无损映射为原生 `DetectionModel`（前向输出最大绝对误差 `0.00000000`）。
- **镜像标签与 Compose 服务完全对齐**：无需维护多套冗余的 18GB 镜像，同一套经过 129 个依赖包版本锁定的黄金底层镜像已同时打齐 4 个标准标签并在 `docker-compose.yml` 中注册对应服务，**零额外磁盘占用，且全部 8 个权重均已完整运行并通过全量测试**：

| 算法架构家族 | Docker 镜像标签 | Compose 服务名 | 纳管的业务模型权重 | 运行与测试状态 |
|---|---|---|---|---|
| **全架构通用入口** | `vision/yolo-unified:8.4.99-cu12.8` | `yolo-unified` | 全部 8 个业务模型 | ✅ 已完整构建并全量验证 |
| **YOLOv8 / OBB** | `vision/yolov8:8.4.99-cu12.8` | `yolov8` | `人睡觉 (v8m)`、`人工作服 (v8m)`、`探水前上杆 (v8l-OBB)` | ✅ 已完整构建并全量验证 |
| **YOLOv9** | `vision/yolov9:8.4.99-cu12.8` | `yolov9` | `探水后上杆钻机 (v9e-960)` | ✅ 已完整构建并全量验证 |
| **YOLO11** | `vision/yolo11:8.4.99-cu12.8` | `yolo11` | `人安全帽 (11x-960)`、`皮带原版 (11m)`、`风门开关 (11m)`、`传感器 (11m)` | ✅ 已完整构建并全量验证 |

**镜像内部情况**：4 个标签指向同一个镜像 ID `581520a9836b`（18.1GB），只是同一镜像的别名。镜像内容由 `docker/yolov8/Dockerfile` 定义：

| 层 | 内容 |
|---|---|
| 基础镜像 | `nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04`（固定 SHA256） |
| Python 运行时 | Ubuntu 22.04 自带 `python3` |
| 核心依赖 | `torch==2.9.0`、`torchvision==0.24.0`、`ultralytics==8.4.99`、`numpy==2.2.6`、`opencv-python==4.11.0.86`、`dill==0.4.1` |
| 依赖锁定 | `requirements.lock.txt` 用 `--no-deps` 安装并 `pip check` 校验 |
| 权重 | 不打进镜像，运行时把宿主机 `models/` 挂载到 `/models` |

一个镜像能跑多个 YOLO 版本，是因为 `ultralytics` 包里同时带有 v8 / v9 / 11 / OBB 的全部模块代码。加载某个 `.pt` 时，框架按权重里保存的架构定义只构建该模型自己的网络，其他版本的代码不会被执行。所以和"每个版本单独一个镜像"相比，速度与精度没有差别。真正影响指标的是 `torch` / CUDA / `ultralytics` 的版本，这几项在本项目中对所有模型完全一致，反而保证了横向对比公平。

### 1.2 全项目唯一统一启动入口：`scripts/run.sh`
所有权重、标准测试集与评测程序均通过根目录统一入口 `bash scripts/run.sh` 结合唯一注册表 `configs/registry.py` 进行启动管理：

```bash
# 1. 查看全部 11 个数据集、8 个模型权重及架构纳管状态
bash scripts/run.sh list

# 2. 一键运行全量评测擂台并自动更新 docs/benchmark-full-table.md
bash scripts/run.sh all

# 3. 单独运行 8 模型全维度架构矩阵 / C++ 静态图后端 / 多路视频流压测 / 加速对比 / 逐框校验
bash scripts/run.sh matrix
bash scripts/run.sh backend
bash scripts/run.sh stream
bash scripts/run.sh accel
bash scripts/run.sh verify

# 4. 单模型一键准入或重测（可自由指定 YOLO_SERVICE=yolov8|yolov9|yolo11）
YOLO_SERVICE=yolo11 bash scripts/run.sh admit ChuanGanQi

# 5. 将最新评测 JSON 一键同步至 docs/benchmark-full-table.md
bash scripts/run.sh sync-table
```

---

## 二、 `args.yaml` 与 `yolov8.yaml` / `yolov11.yaml` 配置文件详解

在 `models/yolov8/custom/<业务目录>/` 下，除了 `.pt` 权重文件外，还配套存放了 `args.yaml` 和 `yolov8.yaml`（或 `yolov11.yaml`）。这两类文件的本质、参数含义及配套用法如下：

### 2.1 `args.yaml` 是什么？存了什么参数？
- **文件本质**：模型在训练启动时，由训练框架自动保存的**全量训练超参数快照（Training Hyperparameters Snapshot，共 109 项参数）**。
- **核心存储参数（分 4 类）**：
  1. **基础任务与模型定义**：`task: detect`（或 `obb`）、`model: yolo11m.pt`（基础预训练架构）、`data: ./data/yolov11.yaml`、`imgsz: 640`（或 `960`，**决定推理时的最佳输入分辨率**）。
  2. **训练策略与优化器超参**：`epochs: 100`、`batch: 96`、`lr0: 0.01`（初始学习率）、`lrf: 0.01`、`momentum: 0.937`、`weight_decay: 0.0005`、`warmup_epochs: 3.0`、`close_mosaic: 10`（最后 10 轮关闭马赛克增强提精）。
  3. **损失函数权重与数据增强**：`box: 7.5`（回归框损失权重）、`cls: 0.5`（分类损失权重）、`dfl: 1.5`、`hsv_h/s/v`（色相/饱和度/明度扰动）、`mosaic: 1.0`、`fliplr: 0.5`（水平翻转）。
  4. **推理与导出默认配置**：`iou: 0.7`、`max_det: 300`、`format: torchscript`、`int8: false`、`simplify: true`。

### 2.2 同目录下的 `yolov8.yaml` / `yolov11.yaml` 是什么？
- **文件本质**：虽然文件名带有 `yolov8` / `yolov11`，但打开文件内容可见，它们实际上是**上游训练该模型时使用的数据集与类别定义文件（Dataset & Class Mapping YAML）**。
- **示例内容（以传感器 `yolov11.yaml` 为例）**：
  ```yaml
  train: "/home/jykj/ZJH/Train/ChuanGanQi-Yolov11/data/train.txt"
  val: "/home/jykj/ZJH/Train/ChuanGanQi-Yolov11/data/val.txt"
  test: "/home/jykj/ZJH/Train/ChuanGanQi-Yolov11/data/test.txt"
  names:
    0: Sensor
    1: SensorAirInlet
    2: SensorBracket
  ```
- **核心存储参数**：
  1. **原始训练/验证划分清单路径（`train` / `val` / `test`）**：记录该权重在训练机上的原始数据切分位置；
  2. **权威类别编号与名称映射表（`names`）**：明确规定模型输出的第 `0, 1, 2...` 号通道分别代表什么业务标签。

### 2.3 `.pt` 权重里已经包含了哪些信息？能否由 `.pt` 反向生成 `args.yaml` 和 `yolov8.yaml`？

`.pt` 是 `torch.save` 保存的字典。在容器内用 `torch.load` 实测本项目全部权重，内含：

| `.pt` 内的键 | 内容 | 对应的外部文件 | 能否从 `.pt` 100% 反向导出 |
|---|---|---|---|
| `train_args` | 训练超参数（本项目 7 个原生权重含 105～113 项，`imgsz` 等都在） | `args.yaml` | **可以**（直接将 `ckpt["train_args"]` 序列化为 YAML 即可；仅第三方转换的 `v9e` 除外） |
| `model.yaml` | 完整的网络架构定义（`nc`、`scales`、`backbone`、`head` 每一层算子与通道数） | 模型架构 `yolov8.yaml` / `yolo11.yaml` | **可以**（直接将 `ckpt["model"].yaml` 导出即可还原完整网络结构 YAML） |
| `model.names` | 类别编号 → 名称映射（如 `{0: 'Sensor', 1: 'SensorAirInlet', ...}`） | 数据集 `yolov8.yaml` / `yolov11.yaml` 中的 `names` 段 | **可反向生成 `names` 类别映射段**；但原训练机上的 `train:` / `val:` / `test:` 数据集路径不在 `.pt` 内，需结合当前环境的测试集路径拼装 |
| `epoch` / `train_metrics` / `date` / `version` | 训练轮次、训练时验证指标、日期、ultralytics 版本 | 无 | — |

**反向导出命令示例（当外部 YAML 缺失时，从单个 `.pt` 直接还原配置文件）**：
```bash
docker compose run --rm yolo-unified python3 -c "
import torch, yaml
ckpt = torch.load('/models/custom/belt/yolov11m_belt.pt', map_location='cpu', weights_only=False)
m = ckpt.get('ema') or ckpt['model']
yaml.safe_dump(ckpt['train_args'], open('/tmp/args_recovered.yaml', 'w'), sort_keys=False, allow_unicode=True)
yaml.safe_dump(m.yaml, open('/tmp/arch_recovered.yaml', 'w'), sort_keys=False, allow_unicode=True)
yaml.safe_dump({'names': m.names}, open('/tmp/data_names_recovered.yaml', 'w'), sort_keys=False, allow_unicode=True)
"
```

**结论**：
1. 推理和评测只需要 `.pt`，`imgsz`、网络架构与 `names` 都能从权重里直接读出或反向生成，外部 yaml 即使丢失也不影响使用。
2. 保留外部 yaml 的价值：一是无需启动 PyTorch 即可纯文本查看/对比训练参数与原始训练集路径；二是 `探水后上杆-yolov9e-960.pt` 由第三方 yolov9 仓库转换而来（`.pt` 内 `train_args` 仅剩 4 项），其完整训练超参数需依赖外部 `args.yaml` 留档。

### 2.4 这两个文件发挥什么作用？如何与 `.pt` 权重配套使用？
1. **场景一：推理与评测时的参数对齐（防错配）**
   - 从 `args.yaml` 的 `imgsz` 和 `task` 读取该权重原生训练分辨率（例如确定 `人安全帽` 和 `探水后上杆` 必须用 `imgsz=960`，其余模型用 `imgsz=640`），避免因分辨率错配导致小目标漏检；
   - 从 `yolov8.yaml`/`yolov11.yaml` 的 `names` 核对类别 ID 顺序，生成标准测试集下的 `data.yaml`。
2. **场景二：新矿区数据增量微调（Fine-tuning 1:1 复现）**
   - 当某模型在新矿区出现分布漂移（如 `探水后上杆` 在新采矿区 `xinjian` 上需要补训）时，直接传入 `args.yaml` 即可 **100% 复现原模型的训练超参数与数据增强配方**：
     ```bash
     yolo train cfg=models/yolov8/custom/tan_shui_hou_shanggan/args.yaml \
       model=models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt \
       data=/data/benchmark/standard/TanShuiHouShangGan/data.yaml epochs=30
     ```
3. **场景三：跨硬件模型转换与 INT8 量化导出**
   - 在将 `.pt` 转换为 ONNX / TensorRT / 昇腾 OM / 瑞芯微 RKNN 并开启 `int8=True` 时，量化器需要读取 `data.yaml`（由 `yolov8/11.yaml` 的 `names` 与标准测试集路径构成）抽取校准图片，并依据 `args.yaml` 中的 `imgsz` 固定输入张量形状。

---

## 三、 核心指标直白解释（含小/中/大目标、告警门限、INT8 量化等）

### 3.1 小 / 中 / 大目标分尺度召回率（X 列）：用直白数据怎么理解？
为了防止总体召回率被近景大目标“平均掩盖”，我们将测试集里的所有真实目标按像素面积分为 **小、中、大** 三档（遵循国际通用 MS-COCO 标准，按 `640×640` 推理画面计算面积 $A = \text{宽} \times \text{高}$）：

| 目标尺度分档 | 在 `640×640` 推理图上的框大小 | 折算到现场 `1920×1080` 高清监控画面上的实物大小 | 占画面总面积比例 | 煤矿井下典型实物举例 |
|---|---|---|---|---|
| **小目标 (Small)** | 面积 **$< 1,024\text{ 像素}^2$**<br>（即边长小于 **$32 \times 32$ 像素**） | 在 1080p 画面上约小于 **$96 \times 54$ 像素**<br>（还没有成人大拇指甲盖大） | **$< 0.25\%$** | 巷道远端 30～50 米外矿工头上的**安全帽**、皮带远处的**细碎煤块**、钻机上的小型**堵头** |
| **中目标 (Medium)** | 面积 **$1,024 \sim 9,216\text{ 像素}^2$**<br>（即边长在 **$32\times 32 \sim 96\times 96$ 像素**之间） | 在 1080p 画面上约为 **$96\times 54 \sim 288\times 162$ 像素**<br>（约拳头到巴掌大小） | **$0.25\% \sim 2.25\%$** | 距离镜头 5～15 米处的**矿工全身/半身**、皮带**托辊**、墙面悬挂的**传感器**、**钻杆** |
| **大目标 (Large)** | 面积 **$\ge 9,216\text{ 像素}^2$**<br>（即边长大于等于 **$96 \times 96$ 像素**） | 在 1080p 画面上大于 **$288 \times 162$ 像素**<br>（占据画面显眼主体区域） | **$\ge 2.25\%$** | 镜头正前方的**整扇风门**、**整条皮带表面**、**探水钻机主体**、露天矿**重型卡车/火车车厢** |

- **直白解读举例**：
  - 看 `人睡觉 (YOLOv8m)`：`小:0.995 / 中:0.997 / 大:1.000` —— 说明不管人在镜头跟前（大目标 100% 检出），还是在值班室角落远处（204 个小目标检出率 99.5%），几乎一个不漏。
  - 看 `探水后上杆钻机 (YOLOv9e)`：`小:0.348 / 中:0.696 / 大:0.780` —— 大目标钻机检出率 78.0%，但远景或微小的钻杆/配件（小目标）100 个里只能检出约 35 个（`34.8%`），一眼就能定位该模型的短板在小目标漏检。
  - 看 `风门开关` 和 `探水前上杆`：`小:无样本 / 中:0.972 / 大:1.000` —— 因为风门和钻孔机位都是近距离固定对准拍摄，测试集几百张图里所有目标框边长全都大于 32 像素（天生没有 `<32×32` 的小目标），因此如实标注 `小:无样本`。

---

### 3.2 标准告警门限（`conf=0.25, iou=0.5`）：这两个指标是什么意思？
在工业视觉部署与本项目的工程诊断（计算查准率、召回率、分尺度召回率、误报率 FAR）中，统一采用 **`conf=0.25, iou=0.5`** 作为标准工作门限：

#### （1）置信度门限 `conf = 0.25`（Confidence Threshold，算法有多大把握才敢输出告警）
- **直白解释**：模型在画面上每画出一个预测框，都会给出一个 $0.00 \sim 1.00$（即 $0\% \sim 100\%$）的打分，代表**“模型认为这个框里确实是目标物的把握有多大”**。
- **`conf=0.25` 的含义**：以 **25% 的把握**作为分界线——
  - 打分 **$< 0.25$** 的微弱疑似框，直接当作背景噪点丢弃，不触发报警；
  - 打分 **$\ge 0.25$** 的预测框，才作为有效检出结果保留下来。
- **调参工程规律**：如果现场嫌误报多，可以把 `conf` 调高到 `0.45~0.50`（更严格，误报减少，但极暗/模糊目标可能漏报）；如果宁可错报不可漏报，则保持 `0.25`。

#### （2）交并比门限 `iou = 0.5`（Intersection over Union，预测框和真实框重合多少才算“框准了”）
- **直白解释**：把人工标注的真实框面积记为 $A$，模型预测框面积记为 $B$：
  $$\text{IoU (交并比)} = \frac{\text{两框重叠区域的面积 }(A \cap B)}{\text{两框合并覆盖的总面积 }(A \cup B)}$$
- **`iou=0.5` 的含义**：模型预测框与人工真实框的**重叠度达到 50% 及以上**（对于水平框或旋转框，两框在视觉上已经高度贴合，中心位置和长宽偏差很小），就判定为**“检出成功（True Positive, TP）”**；如果重叠度不足 50%（比如框歪到了一边），则判定为没框准（记为误报 FP + 漏报 FN）。
- **延伸理解 `mAP50` 与 `mAP50-95`**：
  - **`mAP50`**：就是按 **`iou=0.5`**（50% 重合即算对）统计出的平均精度，反映**“目标有没有被找出来”**（工业告警最看重此项）；
  - **`mAP50-95`**：是从 `iou=0.50, 0.55, 0.60 ...` 一直严苛到 `0.95`（要求 95% 像素级严丝合缝重合）共 10 个档位算出的平均精度，反映**“框的边缘贴得有多精准”**。

---

### 3.3 跨硬件部署与 `FP32 / FP16 / INT8` 量化精度对比（Q 列升级说明）
- **为什么调整并增加 `INT8` 指标？**
  在煤矿实际落地时，训练好的 `.pt` 权重不仅在服务器 GPU 上以 `FP16` 运行，更经常被转换为边缘侧异构硬件格式（如华为昇腾 Atlas 的 `.om`、瑞芯微 RK3588 的 `.rknn`、算能 BM1684X 的 `.bmodel` 或 NVIDIA 边缘盒子的 TensorRT `.engine`），并在推理侧开启 **`INT8`（8 位整数量化，W8A8 PTQ）**以获得最高吞吐与最低功耗。
- **Q 列现在的三级精度展示格式**：
  `0.9778 / 0.7625 (FP16:0.9778 | INT8:0.9717)`
  - **`0.9778 / 0.7625`**：原始 `FP32`（32 位浮点）下的 `mAP50 / mAP50-95` 理论基线；
  - **`FP16: 0.9778`**：转为 16 位半精度浮点后的 `mAP50`（实测 8 个模型与 FP32 误差 $\le 0.0001$，**100% 无损**）；
  - **`INT8: 0.9717`**：经过标准对称逐通道权重量化 + 逐张量激活校准（`W8A8 PTQ`，模拟边缘 NPU/TensorRT INT8 硬件行为）后的实测 `mAP50`。
- **8 个模型跨硬件 `INT8` 量化实测对照表**：

| 业务数据集 | 算法架构 | 原生 FP32 (`mAP50 / mAP50-95`) | GPU 半精度 FP16 (`mAP50`) | 边缘端 INT8 量化 (`mAP50 / mAP50-95`) | INT8 `mAP50` 变化量 | 跨硬件 INT8 部署结论 |
|---|---|---|---|---|---|---|
| **人安全帽** | `YOLO11x` (960) | `0.9778 / 0.7625` | `0.9778` | **`0.9717 / 0.7364`** | `-0.0061` (-0.61%) | ✅ 检出率几乎无损，可直接转 INT8 |
| **人睡觉** | `YOLOv8m` (640) | `0.9939 / 0.8852` | `0.9939` | **`0.9938 / 0.8432`** | `-0.0001` (-0.01%) | ✅ 检出率零损失，极适合 INT8 |
| **人工作服** | `YOLOv8m` (640) | `0.9890 / 0.8689` | `0.9891` | **`0.9877 / 0.7897`** | `-0.0013` (-0.13%) | ✅ `mAP50` 无损；若需高精框回归建议检测头保留 FP16 |
| **皮带原版** | `YOLO11m` (640) | `0.9866 / 0.9145` | `0.9866` | **`0.9850 / 0.8759`** | `-0.0016` (-0.16%) | ✅ 8 个子类检出率高度稳定 |
| **探水后上杆钻机** | `YOLOv9e` (960) | `0.8263 / 0.6334` | `0.8263` | **`0.8196 / 0.6293`** | `-0.0067` (-0.67%) | ✅ 多分支结构 INT8 衰减 `<0.7%` |
| **风门开关** | `YOLO11m` (640) | `0.9947 / 0.9895` | `0.9947` | **`0.9938 / 0.9824`** | `-0.0009` (-0.09%) | ✅ 全指标无损 |
| **传感器** | `YOLO11m` (640) | `0.9950 / 0.9660` | `0.9950` | **`0.9950 / 0.9150`** | **`0.0000` (0.00%)** | ✅ 检出率 100% 零损失 |
| **探水前上杆** | `YOLOv8l-OBB` (640) | `0.9949 / 0.9731` | `0.9949` | **`0.9949 / 0.9690`** | **`0.0000` (0.00%)** | ✅ 旋转框角度与位置在 INT8 下均极稳健 |

> **跨硬件转模型最佳实践建议**：全部 8 个模型在 `INT8` 下的 `mAP50` 衰减均小于 `0.67%`；而在边缘硬件（昇腾 ATC / 瑞芯微 RKNN-Toolkit2 / TensorRT）转换时，直接将本项目的 **400～600 张标准测试集作为 INT8 校准集（Calibration Dataset）**，并对最后一层 Detect 回归头启用 `FP16+INT8 混合量化`，即可同时保住 `mAP50` 和 `mAP50-95` 零损失。

---

### 3.4 其余核心指标速查闭合表（S / T / U / V / Y / Z 列）

| 列号 | 列名称 | 数学定义与直白物理意义 |
|---|---|---|
| **S 列** | **单类 AP50 / AP50-95 (对应D列标签)** | 针对 D 列每个具体标签独立计算的 `AP50 / AP50-95 (P=查准率, R=召回率)`。同一数据集下各有效标签 S 列 `AP50` 的算术平均值严格等于 Q 列的总 `mAP50`。 |
| **T 列** | **官方基线 FPS (b1/FP32)** | 不做任何优化、按官方默认 `model(img)` 单张串行读取磁盘图片并推理的端到端帧率（`20.2 ~ 27.9 FPS`），受制于 CPU 单线程读图与解码等待。 |
| **U 列** | **优化最佳 FPS (FastInfer)** | 启用多线程预解码 + `batch=16` 动态凑批 + `FP16` 后的端到端全流程帧率（`90.0 ~ 259.2 FPS`），括号内为相对 T 列的实测加速倍数（`4.29x ~ 9.53x`）。 |
| **V 列** | **纯 GPU 前向极限 FPS (FP16/b16)** | 剥离磁盘读图、CPU 解码与 NMS，仅测量 GPU 在 `FP16` 半精度、`batch=16`（模拟 16 路视频并发）下的纯神经网络前向吞吐上限（`90.9 ~ 758.2 FPS`），代表该架构在当前显卡上的物理算力天花板。 |
| **Y 列** | **误报率 FAR (背景图误报统计)** | `FAR = FP / (TP + FP)`，即模型预测出的所有框中假警报框所占百分比；括号内 `(N框/M背景图)` 表示在完全没有目标的 $M$ 张纯背景空画面上模型凭空误报了 $N$ 个框。 |
| **Z 列** | **单帧 P99 延迟 / C++ 静态图加速** | `b1 P99` 表示单路实时流（`batch=1`）下 99% 的图片都能在该毫秒数内完成推理（最坏响应延迟上限）；`C++ JIT b1` 表示将模型导出为 TorchScript 静态图交由底层 C++ 引擎执行后，消除 Python 逐层调度开销获得的单帧免费提速（`1.53x ~ 2.01x`）。 |
