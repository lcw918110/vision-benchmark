# 11 个基准数据集地址与维度清单

本清单按 [`benchmark-full-table.md`](./benchmark-full-table.md) 的业务范围和顺序整理：10 个水平框检测（HBB）数据集、1 个旋转框检测（OBB）数据集。原始图片及类别框数为**台账历史口径**，并非本次重新扫描结果；标准集统计引用 [`dataset-standard-report.md`](./dataset-standard-report.md) 和 `output/benchmark/arena/dataset_stats.json`。完整模型精度、单类 AP、FPS 与延迟以评测台账为准，旧指标的可比性限制见 [`metrics-and-conventions.md`](./metrics-and-conventions.md)。

**路径约定**：宿主机项目根目录为 `/home/jykj/vision-deploy`。`docker-compose.yml` 将 `/home/jykj/Dataset` 只读挂载至容器 `/Dataset`，将项目 `data/benchmark` 只读挂载至 `/data/benchmark`。下列标准集地址为宿主机绝对路径；其容器路径为 `/data/benchmark/standard/<标准集键>/`，评测配置为该目录下 `data.yaml`。标准集目录还包含 `images/`、`labels/`、`manifest.json` 和 `excluded.txt`。

**尺度口径**：下表的小/中/大均指标准集标注框数量，按归一化面积折算至 640×640：小 `<32²`，中 `32² ≤ 面积 <96²`，大 `≥96²`；OBB 使用四点多边形面积。这是目标尺度分布，并非原始图片分辨率或模型输入分辨率。原始图片分辨率未全量统计，不以模型输入尺寸代替。

## 1. 人安全帽（`PersonHelmet`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；3 类：2 人、3 戴安全帽、8 未戴安全帽 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/PersonHelmet`；容器 `/Dataset/Hbb/PersonHelmet` |
| 台账原始规模 | 309,422 张；1,370,100 框 |
| 台账类别框数 | 2: 790,421；3: 278,960；8: 300,719 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/PersonHelmet` |
| 标准集规模 | 600 张，1,696 框，空标注 243 张；覆盖 330/373 个来源目录 |
| 标准集尺度 小/中/大 | 684 / 703 / 309 框 |
| 模型 / 默认输入 | YOLO11x；960 |

## 2. 人睡觉（`PersonSleep`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；2 类：0 人睡觉、2 人 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/PersonSleep`；容器 `/Dataset/Hbb/PersonSleep` |
| 台账原始规模 | 31,216 张；50,738 框 |
| 台账类别框数 | 0: 6,224；2: 44,514 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/PersonSleep` |
| 标准集规模 | 500 张，1,222 框，空标注 23 张；覆盖 21/21 个来源目录 |
| 标准集尺度 小/中/大 | 204 / 722 / 296 框 |
| 模型 / 默认输入 | YOLOv8m；640 |

## 3. 人工作服（`WorkclothesHelmet`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；4 类：0 穿工作服的人、2 未穿工作服、3 戴安全帽、8 未戴安全帽 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/WorkclothesHelmet`；容器 `/Dataset/Hbb/WorkclothesHelmet` |
| 台账原始规模 | 5,158 张；16,716 框 |
| 台账类别框数 | 0: 4,730；2: 3,935；3: 5,944；8: 2,107 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/WorkclothesHelmet` |
| 标准集规模 | 500 张，2,200 框，空标注 2 张；覆盖 13/13 个来源目录 |
| 标准集尺度 小/中/大 | 1,007 / 933 / 260 框 |
| 模型 / 默认输入 | YOLOv8m；640 |

## 4. 人乘猴车（`RenChengHouChe`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；8 类：0 人乘猴车、1 睡觉、2 人、3 戴安全帽、4 超长异物、5 脚蹬地、8 未戴安全帽、9 打电话 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/RenChengHouChe`；容器 `/Dataset/Hbb/RenChengHouChe` |
| 台账原始规模 | 8,386 张；28,651 框 |
| 台账类别框数 | 0: 2,467；1: 828；2: 11,571；3: 8,098；4: 596；5: 3,832；8: 675；9: 584 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/RenChengHouChe` |
| 标准集规模 | 500 张，1,564 框，空标注 162 张；覆盖 12/12 个来源目录 |
| 标准集尺度 小/中/大 | 283 / 714 / 567 框 |
| 模型 / 默认输入 | 待交付权重；输入尺寸未定 |

## 5. 皮带原版（`PiDaiYuanBan`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；8 类：0 无煤、1 有煤、2 水煤、3 托辊、4 大块煤、5 异物、6 未堆煤、7 堆煤 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/PiDaiYuanBan`；容器 `/Dataset/Hbb/PiDaiYuanBan` |
| 台账原始规模 | 166,048 张；1,071,423 框 |
| 台账类别框数 | 0: 62,406；1: 96,893；2: 181；3: 806,412；4: 53,677；5: 36,796；6: 4,899；7: 10,159 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/PiDaiYuanBan` |
| 标准集规模 | 600 张，4,351 框，空标注 0 张；覆盖 272/273 个来源目录，见标准集报告 |
| 标准集尺度 小/中/大 | 2,262 / 1,477 / 612 框 |
| 模型 / 默认输入 | YOLO11m；640 |

## 6. 护帮板（`HuBangBan`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；3 类：2 人、3 护帮板、4 滚筒 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/HuBangBan`；容器 `/Dataset/Hbb/HuBangBan` |
| 台账原始规模 | 14,121 张；48,198 框 |
| 台账类别框数 | 2: 1,525；3: 43,956；4: 2,717 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/HuBangBan` |
| 标准集规模 | 500 张，1,938 框，空标注 22 张；覆盖 38/38 个来源目录 |
| 标准集尺度 小/中/大 | 58 / 910 / 970 框 |
| 模型 / 默认输入 | 待交付权重；输入尺寸未定 |

## 7. 探水后上杆钻机（`TanShuiHouShangGan`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；3 类：0 机器、1 堵头、2 机尾 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/TanShuiHouShangGan`；容器 `/Dataset/Hbb/TanShuiHouShangGan` |
| 台账原始规模 | 704,199 张；1,471,896 框 |
| 台账类别框数 | 0: 655,326；1: 235,126；2: 581,444 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/TanShuiHouShangGan` |
| 标准集规模 | 500 张，819 框，空标注 2 张；覆盖 58/58 个来源目录 |
| 标准集尺度 小/中/大 | 23 / 483 / 313 框 |
| 模型 / 默认输入 | YOLOv9e；960 |

标准集 `manifest.json` 的采样根目录是原始业务根目录下的 `xinjian`；58/58 不是整个业务根目录下所有层级的覆盖率。

## 8. 风门开关（`FengMenKaiGuan`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；2 类：0 风门关、1 风门开 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/FengMenKaiGuan`；容器 `/Dataset/Hbb/FengMenKaiGuan` |
| 台账原始规模 | 19,409 张；18,188 框 |
| 台账类别框数 | 0: 11,270；1: 6,918 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/FengMenKaiGuan` |
| 标准集规模 | 500 张，493 框，空标注 11 张；覆盖 10/10 个来源目录 |
| 标准集尺度 小/中/大 | 0 / 36 / 457 框 |
| 模型 / 默认输入 | YOLO11m；640 |

## 9. 传感器（`ChuanGanQi`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；3 类：0 传感器、1 传感器进气口、2 传感器架子 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/ChuanGanQi`；容器 `/Dataset/Hbb/ChuanGanQi` |
| 台账原始规模 | 2,299 张；6,191 框 |
| 台账类别框数 | 0: 2,602；1: 2,072；2: 1,517 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/ChuanGanQi` |
| 标准集规模 | 400 张，1,201 框，空标注 0 张；覆盖 8/8 个来源目录 |
| 标准集尺度 小/中/大 | 377 / 536 / 288 框 |
| 模型 / 默认输入 | YOLO11m；640 |

## 10. 卡车空满载（`ChanYunXiaoMeiZhong`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | HBB 检测；5 类：0 卡车空载、1 卡车满载、3 车辆车斗遮盖、6 块煤、7 沫煤 |
| 原始数据地址 | `/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong`；容器 `/Dataset/Hbb/ChanYunXiaoMeiZhong` |
| 原始规模 | 台账 77,074 张；复核 77,076 个图片文件、77,074 个标签文件；60,396 框。差异见文末说明 |
| 台账类别框数 | 0: 28,203；1: 952；3: 19,786；6: 5,349；7: 6,106 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/ChanYunXiaoMeiZhong` |
| 标准集规模 | 500 张，387 框，空标注 131 张；覆盖 8/8 个来源目录 |
| 标准集尺度 小/中/大 | 0 / 2 / 385 框 |
| 模型 / 默认输入 | 待交付权重；输入尺寸未定 |

## 11. 探水前上杆（原始 `TS-Front_V2` / 标准集 `TS_Front_V2`）

| 维度 | 内容 |
|---|---|
| 任务 / 台账有效类别 | OBB 旋转框检测；3 类：0 无杆、1 有杆、2 机器 |
| 原始数据地址 | `/home/jykj/Dataset/Obb/TS-Front_V2`；容器 `/Dataset/Obb/TS-Front_V2` |
| 台账原始规模 | 265,179 张；528,300 框 |
| 台账类别框数 | 0: 108,087；1: 157,463；2: 262,750 |
| 标准集地址 | `/home/jykj/vision-deploy/data/benchmark/standard/TS_Front_V2` |
| 标准集规模 | 600 张，1,166 框，空标注 0 张；标准集报告记载覆盖 4/4 个来源目录 |
| 标准集尺度 小/中/大 | 0 / 64 / 1,102 框 |
| 模型 / 默认输入 | YOLOv8l-OBB；640 |

标准集 `manifest.json` 的采样根目录为原始业务根目录下的 `Data`，4/4 不代表覆盖 `ShanXi`、`YunNan` 等整个业务根目录下的所有来源。

## 统计口径与待核对事项

- 台账历史合计：**1,602,511 张、4,670,797 框**；其中类别框数按台账有效标签相加，类别定义文件可能另含占位或未出现的类别，因此有效类别数不等于 `nc`。
- 标准集报告合计：**5,700 张、17,037 框**，空标注 596 张；详见 [`dataset-standard-report.md`](./dataset-standard-report.md)。本次仅检查了标准集目录、图片和标签文件数量，未重跑全部标注统计。
- 现有 [`dataset-full-inventory.md`](./dataset-full-inventory.md) 的扫描缓存与台账在 `PiDaiYuanBan`、`TanShuiHouShangGan`、`ChanYunXiaoMeiZhong` 的原始规模上不一致，不能混用或直接视为已纠正。缓存中皮带路径误记为 `/home/jykj/Dataset/Hbb`；这里采用台账和标准集清单中的业务根路径。
- 2026-11-11 按相对目录和文件名重新扫描 11 集，汇总位于 `output/dataset/benchmark-audit-v1/<dataset_key>/summary.json`。合计 **1,602,513 个图片文件、4,670,797 个有效格式标注框**；扫描记录读取错误和异常行均为 0，类别和尺度框数均与框总数相符。框数与台账一致，图片文件比台账多 2 个。
- `ChanYunXiaoMeiZhong` 扫描得到 77,076 个图片文件、77,074 个标签文件；其中有 2 个同目录同 stem 的额外图片，复用了同名标签。按该配对规则，缺标注图片和孤立标签均为 0；这不意味着已验证这两对图片内容是否相同或标注语义是否正确。台账的 77,074 与标签数相同，但未证实其原始统计口径。
- `PiDaiYuanBan` 复核为 166,048 个图片文件、1,071,423 框；`TanShuiHouShangGan` 为 704,199 个图片文件、1,471,896 框，均与台账一致。旧的 42 集扫描缓存及文档尚未重建，不能视为已修正。
- 扫描器以一级目录为断点；缓存不检测源文件后续变更。本次结果是扫描时的快照，断点续扫期间如源数据变化需重新扫描相应分片。
- 标准集存在近邻帧与跨业务同帧泄漏风险；标准集构建脚本目前不能从零完整复现，详见标准集报告。不可仅因存在 `excluded.txt` 认定训练和评测完全隔离。

## 全量内容级校验结果（2026-11-12 完成）

> 完整报告见 [`benchmark-quality-report.md`](./benchmark-quality-report.md)，逐分片问题清单见 `output/dataset/benchmark-quality-v1/issues.csv`，机读汇总见 `output/dataset/benchmark-quality-v1/report.json`。

在文件级扫描之外，对 11 个数据集全部 **1,602,513 张图片逐张完整解码**（`PIL.Image.open().load()`，非仅读文件头），并逐行校验全部 **4,670,797 个标注框**的数值、类别 ID 与坐标合法性。结果位于 `output/dataset/benchmark-quality-v1/<dataset_key>/summary.json`，分片明细同目录。

| 数据集 | 图片 | 标注框 | 合法框 | 空标注图 | 体积 GB | 分辨率种类 |
|---|---:|---:|---:|---:|---:|---:|
| `PersonHelmet` | 309,422 | 1,370,100 | 1,370,090 | 72,064 | 160.3 | 5,412 |
| `PersonSleep` | 31,216 | 50,738 | 50,738 | 4,207 | 16.5 | 14 |
| `WorkclothesHelmet` | 5,158 | 16,716 | 16,716 | 31 | 3.8 | 5 |
| `RenChengHouChe` | 8,386 | 28,651 | 28,651 | 2,584 | 10.2 | 4 |
| `PiDaiYuanBan` | 166,048 | 1,071,423 | 1,071,422 | 4 | 93.0 | 10 |
| `HuBangBan` | 14,121 | 48,198 | 48,198 | 1,463 | 7.8 | 6 |
| `TanShuiHouShangGan` | 704,199 | 1,471,896 | 1,471,896 | 7,043 | 295.8 | 8 |
| `FengMenKaiGuan` | 19,409 | 18,188 | 18,188 | 1,440 | 9.5 | 6 |
| `ChuanGanQi` | 2,299 | 6,191 | 6,189 | 0 | 1.0 | 3 |
| `ChanYunXiaoMeiZhong` | 77,076 | 60,396 | 60,395 | 17,838 | 36.6 | 5 |
| `TS_Front_V2`（OBB） | 265,179 | 528,300 | 351,354 | 0 | 104.9 | 6 |
| **合计** | **1,602,513** | **4,670,797** | **4,493,837** | **106,674** | **739.4** | — |

**配对完整性**：11 集全部为缺标注 0、孤立标注 0。`ChanYunXiaoMeiZhong` 有 2 个同目录同 stem 的额外图片复用同名标签（`.jpg` 与 `.jpeg` 并存），故 77,076 个图片文件对应 77,074 个标签文件，与台账 77,074 张一致。

**图片可解码性**：1,602,512 张完整解码成功，全部为 `RGB` / `JPEG`。仅 `PersonHelmet/sx-jgsk/sx-jgsk-jx-jd-zhkg-dypdd-hc-2024101801/sx-jgsk-jx-jd-zhkg-dypdd-hc-2024101801-000175.jpg` 严格解码报 `truncated (28 bytes not processed)`（文件 229,376 字节，尾部缺 28 字节）；开启容错后仍可解出完整 1280×720 RGB 画面，属文件尾部轻微截断，不影响训练读取，但建议重传该图。

**标注合法性**：4,493,837 个框通过全部校验（数值有限、类别 ID 在 `nc` 范围内、坐标在 [0,1]、HBB 宽高为正、OBB 四边形面积为正）。共 14 个异常框，已逐一定位：

- 非正宽高 13 个（HBB 退化框，宽或高为 0），已定位到文件与行号（完整路径与原始内容见
  [`benchmark-quality-report.md`](./benchmark-quality-report.md) 第 5.1 节）：
  - `PersonHelmet` 10 个：`sxx-hsz/...-wlsb17780-2025041001-000425.txt` 第 2–5 行（4 个坐标完全相同的重复框）、
    `sxx-hsz/...-wlsb19565-2025042901-001000.txt` 第 1 行、`hn-csl/...-002550.txt` 第 1 行、
    `sx-xlymk/...-002550.txt` 第 1 行、`sx-ll/...-001890.txt` 第 3 行、
    `sxx-hll/...-000750.txt` 第 4 行、`xj-hamk/...-001245.txt` 第 5 行。
  - `ChuanGanQi` 2 个：`sd-jzm/...-029240.txt` 第 1 行、`sd-tk/...-007820.txt` 第 1 行。
  - `PiDaiYuanBan` 1 个：`sxhy-ekxxjzypd/sxhy-ekxxjzypd-2025042201-000725.txt` 第 29 行。
  - 这类框宽高为 0，不参与梯度计算也不会报错，但会让框数统计虚高，建议清理。
- 坐标越界 1 个：`ChanYunXiaoMeiZhong/neimengguerlintu/elt-zcspjk-2022121801/elt-zcspjk-2022121801-2380.txt` 第 1 行，`1.370703 1.504167 0.833594 0.997222`，中心点明显超出图像范围，属标注错误。
- `TS_Front_V2` 有 176,946 个框（33.5%，分布于 51.4% 的标签文件）坐标超出 [0,1]。经幅度分析：49,396 个越界 ≤0.01、89,962 个 ≤0.05、21,565 个 ≤0.1、16,023 个 >0.1，即约 **79% 的越界不超过 0.05**，是标注工具在图像边界处产生的轻微溢出（如 `1.02`、`-0.03`），并非坐标格式错误。该集使用归一化坐标（非像素），格式正确。训练时按常规裁剪到 [0,1] 即可；若要求严格，可对这 33.5% 的框做一次边界裁剪。

**分辨率**：`PersonHelmet` 因来源极杂有 5,412 种分辨率，其余数据集为 3–14 种。各集主流分辨率均为 `1280x720` 或 `1920x1080`，详见各 `summary.json` 的 `resolutions` 字段。

**标签文件换行符**：抽查发现不统一，属格式差异而非错误（Ultralytics 解析不受影响）。`PersonHelmet` 300 个样本中 211 个为 CRLF（其余为单行文件无换行符）；`TanShuiHouShangGan` 与 `TS_Front_V2` 均为 100% CRLF；`PiDaiYuanBan` 87% 为 LF。当前状态不需要为训练特意改动，如需统一便于版本对比再做转换。

**校验方法**：按一级目录分片，超过 2 万张的分片递归下钻，共 2,391 个分片；每个分片内先收集文件并按 inode 排序再读取，使机械盘磁头接近顺序访问（实测比目录顺序读取快约 50%）；结果原子写入 JSON 分片缓存，支持断点续跑。任务由 systemd 用户服务 `vision-quality-audit.service` 托管，避免终端会话信号中断。

**未覆盖范围**：本次校验的是图片可解码性、分辨率、色彩模式、标注数值/类别/坐标合法性，以及图片与标签的路径配对。**未验证**标注框的语义正确性（即框是否真的框住了对应类别目标）、框与目标的对齐精度、类别标签是否正确，也未检查近邻帧重复与训练集/测试集泄漏。这些需要人工抽检或模型评测才能确认。
