# 矿山视觉算法横向评测平台 — 评测环境与指标定义规范

> **归档路径**：`docs/metrics-and-conventions.md`  
> **配套文档**：
> - 整表 Markdown 台账：[`docs/benchmark-full-table.md`](./benchmark-full-table.md)
> - 标准测试集来源与评价报告：[`docs/dataset-standard-report.md`](./dataset-standard-report.md)

---

## 一、 统一评测镜像与容器环境实现

本项目评测涵盖 **5 种视觉算法架构**：`YOLOv8m`、`YOLOv8l-OBB`（旋转框）、`YOLOv9e`、`YOLO11m`、`YOLO11x`。

### 1.1 多架构统一镜像构建方案
- **算子与框架层统一**：基于 `Ultralytics 8.4.99` + `PyTorch 2.9.0+cu128` 构建，原生支持 `YOLOv8`、`YOLOv8-OBB`、`YOLOv9`、`YOLO11` 的全部算子（`C2f`、`RepNCSPELAN4`、`ADown`、`SPPELAN`、`C3k2`、`C2PSA`、`OBB` 旋转检测头等）。针对由第三方 `WongKinYiu/yolov9` 训练的 `探水后上杆-yolov9e-960.pt`，通过 `scripts/07-extract-v9e-weights.py` 将 `DualDDetect` 结构无损转换为原生 `DetectionModel`（前向输出最大绝对误差 `0.00000000`）。
- **单镜像多标签映射**：4 个架构标签指向同一镜像 ID `581520a9836b`（18.1GB），并在 `docker-compose.yml` 中注册对应服务，零额外磁盘占用，保证所有算法在完全一致的 CUDA / PyTorch / Ultralytics 版本下公平对比：

<table>
  <thead>
    <tr>
      <th>算法架构家族</th>
      <th>Docker 镜像标签</th>
      <th>Compose 服务名</th>
      <th>纳管的业务模型权重</th>
      <th>运行与测试状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>全架构通用入口</strong></td>
      <td><code>vision/yolo-unified:8.4.99-cu12.8</code></td>
      <td><code>yolo-unified</code></td>
      <td>全部 8 个业务模型</td>
      <td rowspan="4">✅ 已完整构建并全量验证</td>
    </tr>
    <tr>
      <td><strong>YOLOv8 / OBB</strong></td>
      <td><code>vision/yolov8:8.4.99-cu12.8</code></td>
      <td><code>yolov8</code></td>
      <td><code>人睡觉 (v8m)</code>、<code>人工作服 (v8m)</code>、<code>探水前上杆 (v8l-OBB)</code></td>
    </tr>
    <tr>
      <td><strong>YOLOv9</strong></td>
      <td><code>vision/yolov9:8.4.99-cu12.8</code></td>
      <td><code>yolov9</code></td>
      <td><code>探水后上杆钻机 (v9e-960)</code></td>
    </tr>
    <tr>
      <td><strong>YOLO11</strong></td>
      <td><code>vision/yolo11:8.4.99-cu12.8</code></td>
      <td><code>yolo11</code></td>
      <td><code>人安全帽 (11x-960)</code>、<code>皮带原版 (11m)</code>、<code>风门开关 (11m)</code>、<code>传感器 (11m)</code></td>
    </tr>
  </tbody>
</table>

### 1.2 镜像内部环境构成（`docker/yolov8/Dockerfile`）

| 层 | 内容 |
|---|---|
| 基础镜像 | `nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04`（固定 SHA256） |
| Python 运行时 | Ubuntu 22.04 自带 `python3` |
| 核心依赖 | `torch==2.9.0`、`torchvision==0.24.0`、`ultralytics==8.4.99`、`numpy==2.2.6`、`opencv-python==4.11.0.86`、`dill==0.4.1` |
| 依赖锁定 | `requirements.lock.txt`（共 129 个包）用 `--no-deps` 安装并经 `pip check` 校验 |
| 权重与数据挂载 | 不打进镜像，运行时把宿主机 `models/` 挂载到 `/models`、`data/` 挂载到 `/data` |

---

## 一附、现有能力边界与统一扩展方案

本项目只负责评测，不涉及训练或外部工程修改。以下区分**当前实现**和**待建设方案**；方案不代表已经支持。

### 当前能力与限制

- `configs/registry.py` 以业务数据集为中心记录模型、任务和历史指标；`configs/algo_arena.py` 使用 Ultralytics `YOLO` 加载权重，主要覆盖水平框检测（detect）和旋转框检测（obb）。它不是任意框架、任意格式的通用适配器。
- `arch_matrix` 汇总不同业务模型在各自数据集上的结果，可用于能力盘点，**不能直接作为跨算法同卷排名**。`head_to_head` 是固定的两组数据集、模型和类别；`custom` 可以指定候选权重及同一数据集，但尚无通用类别语义映射校验。
- 原生验证使用 Ultralytics `model.val()`，记录 mAP、P/R 和逐类结果；工程诊断另行计算分尺度召回及误报。OBB 工程诊断使用旋转框的水平外接框和水平 IoU，不能当作旋转 IoU 诊断。
- `false_alarm_ratio` 的分母是预测框数 TP+FP，表示预测框中的误报占比，不是背景图误报率或每小时告警数；背景图数量为零时不能推断现场误报水平。
- `admit` 按数据集替换矩阵中的结果，并可能沿用注册表或历史加速结果，尚不能保证同一业务多候选、不同版本及其全部指标的来源隔离。固定文件名也会覆盖旧结果；子进程失败时队列等待缺少超时处理。
- 矩阵中的验证使用 `model.val()` 默认参数；同卷模式传入 `conf` 和 NMS `iou`；工程诊断独立使用 `conf=0.25`、匹配 IoU=0.5、NMS IoU=0.7。二者不同于 AP 的多个匹配 IoU 阈值，现有 P/R 也不能一概声称由固定 `conf=0.25` 算出。
- 新版入口已提供基础结果可比性检查及少量无 GPU 单元测试；完整的适配器一致性、真实模型端到端及多任务回归测试尚未完成，自动执行脚本不等于已经实现可靠的自动化选型。

### 新版同卷评测入口（第一阶段）

`scripts/run.sh benchmark` 接入 `configs/benchmark_v1.py`：目前仅支持 Ultralytics detect/obb。每次运行生成独立 `vision-benchmark/run-v1` JSON，包含数据与权重内容摘要、类别、评测参数、环境、逐类结果、失败状态及风险提示；`compare` 对数据、类别、任务、参数、环境和评测脚本版本逐项检查，不一致时拒绝比较。结果仅由新入口读取，不混入旧台账，供下一轮可视化使用。

运行示例（容器内路径）：

```bash
bash scripts/run.sh benchmark run --dataset /data/benchmark/standard/PersonSleep --weights /models/custom/20261030-99人睡觉yolov8m/20261030-99.pt --task detect --imgsz 640
bash scripts/run.sh benchmark compare /output/benchmark/runs/运行ID1.json /output/benchmark/runs/运行ID2.json
```

模型类别 ID 和名称须与标准集完全一致，否则失败；新入口暂不做类别映射、固定工作点诊断或跨后端适配。数据集指纹扫描所有实际图片和标签，可能耗时；标签缺失会失败，空标签才表示背景图。新旧入口并存；旧矩阵不具备新协议的追溯及比较保障。旧台账同步已停止将缺少权重摘要的旧矩阵结果自动混入注册表数据，台账暂只展示注册表历史值；新协议结果尚未自动回填台账。比较结果可通过 `compare ... --out /output/benchmark/runs/comparison.json` 保存供界面读取。

### 后续整体方案（部分待实现）

1. **分离数据集、模型和运行配置**：为数据集版本/样本哈希、模型及权重哈希、任务类型、类别语义、推理后端、前后处理和运行环境分别设立标识；一份数据集可关联多个候选模型，禁止仅以业务名覆盖结果。
2. **建立适配协议**：定义加载、预处理、推理、后处理及标准化预测输出，并明确坐标系、置信度、类别映射与批处理。先把现有 Ultralytics detect/obb 封装为首个适配器；其它框架、格式及任务须逐一实现和验证，不能仅注册权重即宣称支持。
3. **任务化指标**：检测使用逐类 AP、固定工作点 P/R、漏检及分尺度统计；OBB 使用旋转 IoU 和旋转框匹配；分割、分类、跟踪等任务分别使用对应标注与指标插件，未实现或无样本标记为不支持/无数据，不填零。业务关键类别应单列召回、误报及样本数；视频事件类指标需要有事件级标注和时序规则，不能从图片指标推算。
4. **制定可比性契约**：同卷比较要求任务、样本版本、类别语义、指标计算、阈值和环境一致；同分辨率公平对照与各模型最佳配置工程对照分开报告。数据泄漏风险须随结果披露，不以存在排除清单推断模型训练未泄漏。
5. **结果与自动化**：每次运行产生独立、版本化的结构化结果，记录权重/数据/配置/代码版本、硬件和精度、预热次数、批次、耗时定义、状态及失败原因；比较前校验可比性。区分前向、端到端、批吞吐与单帧延迟，保留原始样本与分位数；缺测不可默认为零。
6. **贯通入口和展示**：统一 CLI、评测引擎、结果结构与台账的数据来源；历史数据明确标记来源及不可比状态，不与新运行结果混用。加上小型可复现样例、类别映射/OBB/无样本/子进程失败/历史隔离的自动化测试，再逐步接入新任务与后端。

## 二、 评测指标计算方法与工程定义

> 以下历史指标说明和数值需结合上述实际实现边界阅读；固定工作点诊断与 Ultralytics 验证指标的参数和计算方法并不完全相同。

### 2.1 小 / 中 / 大目标分尺度召回率（台账 X 列）
测试集中的所有真实目标按像素面积分为 **小、中、大** 三档（遵循 MS-COCO 标准，按 `640×640` 推理画面计算面积 `面积 = 宽 × 高`）：

| 目标尺度分档 | 在 `640×640` 推理图上的框大小 | 折算到现场 `1920×1080` 高清监控画面上的实物大小 | 占画面总面积比例 | 煤矿井下典型实物对应 |
|---|---|---|---|---|
| **小目标 (Small)** | 面积 **`< 1,024 像素²`**<br>（边长 **`< 32 × 32 像素`**） | 1080p 画面上约 **`< 96 × 54 像素`** | **`< 0.25%`** | 巷道远端 30～50 米外矿工头上的**安全帽**、皮带远处的**细碎煤块**、钻机上的小型**堵头** |
| **中目标 (Medium)** | 面积 **`1,024 ~ 9,216 像素²`**<br>（边长 **`32×32 ~ 96×96 像素`**） | 1080p 画面上约为 **`96×54 ~ 288×162 像素`** | **`0.25% ~ 2.25%`** | 距离镜头 5～15 米处的**矿工全身/半身**、皮带**托辊**、墙面悬挂的**传感器**、**钻杆** |
| **大目标 (Large)** | 面积 **`≥ 9,216 像素²`**<br>（边长 **`≥ 96 × 96 像素`**） | 1080p 画面上 **`≥ 288 × 162 像素`** | **`≥ 2.25%`** | 镜头正前方的**整扇风门**、**整条皮带表面**、**探水钻机主体**、露天矿**重型卡车/火车车厢** |

- **数据判读规则**：
  - `人睡觉 (YOLOv8m)` 为 `小:0.995 / 中:0.997 / 大:1.000`，代表远近目标检出率均衡；
  - `探水后上杆钻机 (YOLOv9e)` 为 `小:0.348 / 中:0.696 / 大:0.780`，直接反映远景/微小配件（如堵头）存在明显漏检短板；
  - `风门开关` 与 `探水前上杆` 为 `小:无样本`，因固定近景机位下真实目标框边长全部 `≥ 32` 像素。

---

### 2.2 标准告警门限设定（`conf=0.25, iou=0.5`）
当前自定义工程诊断采用 **`conf=0.25`、匹配 `IoU=0.5`**，预测时 NMS IoU 为 0.7。`model.val()` 输出的 P/R 和 AP 使用其验证流程，不能直接等同于此固定工作点；同卷模式的 `--iou` 是 NMS 参数。

1. **置信度门限 `conf = 0.25`（Confidence Threshold）**：
   - 模型输出的候选框得分在 `[0.0, 1.0]` 之间，仅保留得分 `≥ 0.25` 的预测框参与匹配与告警统计，得分 `< 0.25` 的背景噪点直接过滤。
   - 现场部署调优时，误报偏高可将 `conf` 上调至 `0.45~0.50`；强调零漏报则保持 `0.25`。
2. **交并比门限 `iou = 0.5`（Intersection over Union）**：
   - 预测框与真实标注框的重叠面积比 `IoU = 重叠面积 / 并集总面积 ≥ 0.50` 且类别一致时，判定为检出成功（`TP`）；否则记为误报（`FP`）或漏报（`FN`）。
   - **`mAP50`**：固定在 `iou=0.50` 下的全类平均精度，衡量“目标是否被检出”；
   - **`mAP50-95`**：从 `iou=0.50` 到 `0.95`（步长 `0.05` 共 10 档）的平均精度，衡量“边框贴合紧密度”。

---

### 2.3 跨硬件 `FP32 / FP16 / INT8` 量化评测方法（台账 Q 列）
针对煤矿现场将 `.pt` 权重转换为边缘异构硬件格式（昇腾 `.om`、瑞芯微 `.rknn`、算能 `.bmodel`、TensorRT `.engine`）并开启 `INT8` 推理的场景，台账 Q 列采用三级精度记录格式：
`0.9778 / 0.7625 (FP16:0.9778 | INT8:0.9717)`

- **第一项 `0.9778 / 0.7625`**：原生 `FP32` 下的 `mAP50 / mAP50-95` 基线；
- **第二项 `FP16: 0.9778`**：GPU 半精度浮点下的 `mAP50`；
- **第三项 `INT8: 0.9717`**：对模型全部 `Conv2d / Linear` 层执行标准对称逐通道权重量化（8-bit Per-Channel Symmetric `[-127, 127]`）+ 逐张量激活动态校准量化（8-bit Per-Tensor Activation Quantization，`W8A8 PTQ`）后的实测 `mAP50`。

**8 个业务模型 `FP32 / FP16 / INT8` 量化衰减对照**：

<table>
  <thead>
    <tr>
      <th>业务数据集</th>
      <th>算法架构</th>
      <th>原生 FP32 (<code>mAP50 / mAP50-95</code>)</th>
      <th>GPU 半精度 FP16 (<code>mAP50</code>)</th>
      <th>边缘端 INT8 量化 (<code>mAP50 / mAP50-95</code>)</th>
      <th>INT8 <code>mAP50</code> 变化量</th>
      <th>跨硬件 INT8 部署评价</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>人安全帽</strong></td>
      <td><code>YOLO11x</code> (960)</td>
      <td><code>0.9778 / 0.7625</code></td>
      <td><code>0.9778</code></td>
      <td><strong><code>0.9717 / 0.7364</code></strong></td>
      <td><code>-0.0061</code> (-0.61%)</td>
      <td>✅ 检出率无损，可直接转 INT8</td>
    </tr>
    <tr>
      <td><strong>人睡觉</strong></td>
      <td rowspan="2"><code>YOLOv8m</code> (640)</td>
      <td><code>0.9939 / 0.8852</code></td>
      <td><code>0.9939</code></td>
      <td><strong><code>0.9938 / 0.8432</code></strong></td>
      <td><code>-0.0001</code> (-0.01%)</td>
      <td>✅ 检出率零损失</td>
    </tr>
    <tr>
      <td><strong>人工作服</strong></td>
      <td><code>0.9890 / 0.8689</code></td>
      <td><code>0.9891</code></td>
      <td><strong><code>0.9877 / 0.7897</code></strong></td>
      <td><code>-0.0013</code> (-0.13%)</td>
      <td>✅ <code>mAP50</code> 无损；高精框回归建议检测头保留 FP16</td>
    </tr>
    <tr>
      <td><strong>皮带原版</strong></td>
      <td rowspan="3"><code>YOLO11m</code> (640)</td>
      <td><code>0.9866 / 0.9145</code></td>
      <td><code>0.9866</code></td>
      <td><strong><code>0.9850 / 0.8759</code></strong></td>
      <td><code>-0.0016</code> (-0.16%)</td>
      <td>✅ 8 个子类检出率稳定</td>
    </tr>
    <tr>
      <td><strong>风门开关</strong></td>
      <td><code>0.9947 / 0.9895</code></td>
      <td><code>0.9947</code></td>
      <td><strong><code>0.9938 / 0.9824</code></strong></td>
      <td><code>-0.0009</code> (-0.09%)</td>
      <td>✅ 全指标无损</td>
    </tr>
    <tr>
      <td><strong>传感器</strong></td>
      <td><code>0.9950 / 0.9660</code></td>
      <td><code>0.9950</code></td>
      <td><strong><code>0.9950 / 0.9150</code></strong></td>
      <td><strong><code>0.0000</code> (0.00%)</strong></td>
      <td>✅ 检出率零损失</td>
    </tr>
    <tr>
      <td><strong>探水后上杆钻机</strong></td>
      <td><code>YOLOv9e</code> (960)</td>
      <td><code>0.8263 / 0.6334</code></td>
      <td><code>0.8263</code></td>
      <td><strong><code>0.8196 / 0.6293</code></strong></td>
      <td><code>-0.0067</code> (-0.67%)</td>
      <td>✅ 多分支结构 INT8 衰减 <code>&lt;0.7%</code></td>
    </tr>
    <tr>
      <td><strong>探水前上杆</strong></td>
      <td><code>YOLOv8l-OBB</code> (640)</td>
      <td><code>0.9949 / 0.9731</code></td>
      <td><code>0.9949</code></td>
      <td><strong><code>0.9949 / 0.9690</code></strong></td>
      <td><strong><code>0.0000</code> (0.00%)</strong></td>
      <td>✅ 旋转框角度与位置在 INT8 下均稳健</td>
    </tr>
  </tbody>
</table>

---

### 2.4 台账其余扩展指标列定义（S / T / U / V / Y / Z 列）

| 列号 | 列名称 | 计算方式与物理意义 |
|---|---|---|
| **S 列** | **单类 AP50 / AP50-95 (对应D列标签)** | 针对 D 列每个具体标签独立计算的 `AP50 / AP50-95 (P=查准率, R=召回率)`。同一数据集下各有效标签 S 列 `AP50` 的算术平均值等于 Q 列的总 `mAP50`。 |
| **T 列** | **官方基线 FPS (b1/FP32)** | 官方默认 `model(img)` 单张串行读取磁盘图片并推理的端到端帧率（`20.2 ~ 27.9 FPS`），包含 CPU 单线程读图与解码耗时。 |
| **U 列** | **优化最佳 FPS (FastInfer)** | 启用多线程预解码 + Pinned Memory + `batch=16` 动态凑批 + `FP16` 后的端到端全流程帧率（`90.0 ~ 259.2 FPS`），括号内为相对 T 列的实测加速比（`4.29x ~ 9.53x`）。 |
| **V 列** | **纯 GPU 前向极限 FPS (FP16/b16)** | 剥离磁盘读图、CPU 解码与 NMS，仅测量 GPU 在 `FP16`、`batch=16` 下的纯神经网络前向吞吐上限（`90.9 ~ 758.2 FPS`），反映架构在当前显卡上的算力上限。 |
| **Y 列** | **误报率 FAR (背景图误报统计)** | `FAR = FP / (TP + FP)`，即全部预测框中误报框占比；括号内 `(N框/M背景图)` 统计在完全无目标的 `M` 张纯背景图上产生的误报框总数 `N`。 |
| **Z 列** | **单帧 P99 延迟 / C++ 静态图加速** | `b1 P99` 为单路实时流（`batch=1`）下 99% 分位的最坏单帧推理耗时；`C++ JIT b1` 为导出 TorchScript 静态图交由底层 C++ 引擎执行后的单帧帧率与加速比（`1.53x ~ 2.01x`）。 |
