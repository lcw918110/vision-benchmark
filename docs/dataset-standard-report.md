# 11 个矿山业务标准测试集（5,700 张 / 17,037 框）来源、抽取方法与评价报告

> **归档路径**：`docs/dataset-standard-report.md`
> **数据落盘根目录**：`data/benchmark/standard/<dataset_key>/`（容器内路径：`/data/benchmark/standard/<dataset_key>/`）
> **构建脚本**：`scripts/08-build-standard-set.py`（固定随机种子 `SEED=42`）
> **本文所有数字**均为对落盘文件的逐文件复核结果（非估算），复核口径见文末 §六。

---

## 一、 测试集来源与规模评价

### 1.1 数据来源与抽样规模

11 个业务标准测试集抽取自本机原始矿山数据仓 `/home/jykj/Dataset/` 的 2 个分区：

- `Hbb/*` —— 10 个水平框业务（`LianHeCheLiang` 未纳入标准集）
- `Obb/TS-Front_V2` —— 1 个旋转框业务

这 11 个业务在台账 `docs/benchmark-full-table.md` 中登记的原始规模为 **1,602,511 张图片 / 4,670,797 个标注框**。

> **注意口径差异**：`/home/jykj/Dataset/` 物理全量（全量扫描实测）为 **1,983,866 张 / 5,801,904 框**（`Hbb` 1,488,561 + `100/Hbb` 200,647 + `101/Hbb` 29,479 + `Obb` 265,179 + `Seg` 0）。台账的 160.2 万是**11 个业务**的口径，比物理全量少约 38.1 万张，差额来自尚未纳入台账的 `100/`、`101/` 两个扩展分区（共 30 个业务）。全量 42 个业务的逐集统计见 `docs/dataset-full-inventory.md`。

按业务分层抽样后落盘于本项目 `data/benchmark/standard/`，每个业务 **400～600 张**，合计 **5,700 张图片、17,037 个真实标注框**，磁盘占用 **3,287.0 MB**（含 images + labels + 配置清单）。

### 1.2 测试集完备性与规模评价

1. **统计置信度（按实际 N 重算，勿引用旧值）**：
   95% 置信区间半宽 `Δ = 1.96 × √(p(1-p)/N)`，`p = 0.98`：

| 统计口径 | N 范围 | Δ |
|---|---|---|
| 单集，按**图片数** | 400 ~ 600 张 | **±1.12% ~ ±1.37%** |
| 单集，按**标注框数** | 387 ~ 4,351 框 | **±0.42% ~ ±1.39%** |
| 11 集合计，按图片数 | 5,700 张 | **±0.36%** |
| 11 集合计，按标注框数 | 17,037 框 | **±0.21%** |

   结论：**单集**（400~600 张）的精度误差在 **±1.1% ~ ±1.4%** 量级，足够区分工业场景中 ≥2% 的算法差距；**合计** 17,037 框才收敛到 ±0.21%。旧版本写的"单集 ±0.25%~±0.45%"是把合计值套到了单集上，已修正。

2. **矿区空间与类别覆盖率**：
   11 集合计覆盖 **774 / 818** 个一级矿区子目录（94.6%）。未覆盖的 44 个全部集中在 `PiDaiYuanBan`（273 取 272）与 `PersonHelmet`（373 取 330）——这两个业务子目录过多，受单集 600 张上限约束无法逐目录取样。

   > 旧版本称"`PersonHelmet` 覆盖全部 330 个**含有效标注图**的矿区子目录"表述有误：实测 `PersonHelmet` **373 个一级子目录全部**同时含图片与标注（仅图无标注 = 0）。330 只是抽样命中的目录数，不是"有效目录数"。
   > `PiDaiYuanBan` 同理：273 个目录全部有图有标注，"272/273 (99.6%)"是抽样覆盖率而非数据完整性。

   台账 **44 个业务标签子类** 100% 含实测样本（`Default*` 占位类除外）。

3. **跨硬件 INT8 量化校准复用**：
   昇腾 ATC、瑞芯微 RKNN-Toolkit2、TensorRT 的 PTQ 标准校准集规模通常为 `300 ~ 500` 张，本测试集单集 400~600 张，可直接复用。

---

## 二、 11 个标准测试集全量统计总表

按 `640×640` 推理尺寸折算目标像素面积：小目标 `<32²`、中目标 `32²~96²`、大目标 `≥96²`。

- **HBB**（10 集）面积 = `w × h × 640²`，`w/h` 取标注直接给出的归一化宽高。
- **OBB**（1 集）面积 = **四点多边形鞋带公式** `|½ Σ(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)| × 640²`，而不是外接矩形——旋转 45° 的设备外接矩形会虚高近一倍。

<table>
  <thead>
    <tr>
      <th>序号</th>
      <th>业务数据集 (<code>dataset_key</code>)</th>
      <th>任务类型</th>
      <th>标准集图片张数</th>
      <th>纯背景图 (空标签图)</th>
      <th>覆盖矿区 / 总矿区数</th>
      <th>磁盘体积 (MB)</th>
      <th>真实标注总框数 (Instances)</th>
      <th>小目标框数 (<code>&lt;32×32</code>)</th>
      <th>中目标框数 (<code>32²~96²</code>)</th>
      <th>大目标框数 (<code>≥96×96</code>)</th>
      <th>类别覆盖情况（类别ID: 台账标签名 框数）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>人安全帽</strong> (<code>PersonHelmet</code>)</td>
      <td rowspan="10"><code>detect</code></td>
      <td><strong>600 张</strong></td>
      <td>243 张</td>
      <td>330 / 373</td>
      <td>341.6 MB</td>
      <td><strong>1,696 框</strong></td>
      <td>684 框 (40.3%)</td>
      <td>703 框 (41.5%)</td>
      <td>309 框 (18.2%)</td>
      <td><code>2:人(888)</code>, <code>3:戴安全帽(328)</code>, <code>8:未戴安全帽(480)</code></td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>人睡觉</strong> (<code>PersonSleep</code>)</td>
      <td rowspan="3"><strong>500 张</strong></td>
      <td>23 张</td>
      <td>21 / 21 (100%)</td>
      <td>328.1 MB</td>
      <td><strong>1,222 框</strong></td>
      <td>204 框 (16.7%)</td>
      <td>722 框 (59.1%)</td>
      <td>296 框 (24.2%)</td>
      <td><code>0:人睡觉(148)</code>, <code>2:人(1074)</code></td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>人工作服</strong> (<code>WorkclothesHelmet</code>)</td>
      <td>2 张</td>
      <td>13 / 13 (100%)</td>
      <td>337.7 MB</td>
      <td><strong>2,200 框</strong></td>
      <td>1,007 框 (45.8%)</td>
      <td>933 框 (42.4%)</td>
      <td>260 框 (11.8%)</td>
      <td><code>0:穿工作服的人(225)</code>, <code>2:未穿工作服(899)</code>, <code>3:戴安全帽(369)</code>, <code>8:未戴安全帽(707)</code></td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>人乘猴车</strong> (<code>RenChengHouChe</code>)</td>
      <td>162 张</td>
      <td>12 / 12 (100%)</td>
      <td>279.9 MB</td>
      <td><strong>1,564 框</strong></td>
      <td>283 框 (18.1%)</td>
      <td>714 框 (45.7%)</td>
      <td>567 框 (36.3%)</td>
      <td><code>0:人乘猴车(111)</code>, <code>1:睡觉(43)</code>, <code>2:人(579)</code>, <code>3:戴安全帽(359)</code>, <code>4:超长异物(101)</code>, <code>5:脚蹬地(218)</code>, <code>8:未戴安全帽(92)</code>, <code>9:打电话(61)</code></td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>皮带原版</strong> (<code>PiDaiYuanBan</code>)</td>
      <td><strong>600 张</strong></td>
      <td>0 张</td>
      <td>272 / 273 (99.6%)</td>
      <td>304.7 MB</td>
      <td><strong>4,351 框</strong></td>
      <td>2,262 框 (52.0%)</td>
      <td>1,477 框 (33.9%)</td>
      <td>612 框 (14.1%)</td>
      <td><code>0:无煤(180)</code>, <code>1:有煤(382)</code>, <code>2:水煤(11)</code>, <code>3:托辊(3256)</code>, <code>4:大块煤(332)</code>, <code>5:异物(88)</code>, <code>6:未堆煤(15)</code>, <code>7:堆煤(87)</code></td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>护帮板</strong> (<code>HuBangBan</code>)</td>
      <td rowspan="3"><strong>500 张</strong></td>
      <td>22 张</td>
      <td>38 / 38 (100%)</td>
      <td>240.8 MB</td>
      <td><strong>1,938 框</strong></td>
      <td>58 框 (3.0%)</td>
      <td>910 框 (47.0%)</td>
      <td>970 框 (50.1%)</td>
      <td><code>2:人(34)</code>, <code>3:护帮板(1783)</code>, <code>4:滚筒(121)</code></td>
    </tr>
    <tr>
      <td>7</td>
      <td><strong>探水后上杆钻机</strong> (<code>TanShuiHouShangGan</code>)</td>
      <td>2 张</td>
      <td>58 / 58 (100%)</td>
      <td>200.5 MB</td>
      <td><strong>819 框</strong></td>
      <td>23 框 (2.8%)</td>
      <td>483 框 (59.0%)</td>
      <td>313 框 (38.2%)</td>
      <td><code>0:机器(418)</code>, <code>1:堵头(12)</code>, <code>2:机尾(389)</code></td>
    </tr>
    <tr>
      <td>8</td>
      <td><strong>风门开关</strong> (<code>FengMenKaiGuan</code>)</td>
      <td>11 张</td>
      <td>10 / 10 (100%)</td>
      <td>340.7 MB</td>
      <td><strong>493 框</strong></td>
      <td>0 框 (0.0%)</td>
      <td>36 框 (7.3%)</td>
      <td>457 框 (92.7%)</td>
      <td><code>0:风门关(247)</code>, <code>1:风门开(246)</code></td>
    </tr>
    <tr>
      <td>9</td>
      <td><strong>传感器</strong> (<code>ChuanGanQi</code>)</td>
      <td><strong>400 张</strong></td>
      <td>0 张</td>
      <td>8 / 8 (100%)</td>
      <td>247.2 MB</td>
      <td><strong>1,201 框</strong></td>
      <td>377 框 (31.4%)</td>
      <td>536 框 (44.6%)</td>
      <td>288 框 (24.0%)</td>
      <td><code>0:传感器(620)</code>, <code>1:传感器进气口(483)</code>, <code>2:传感器架子(98)</code></td>
    </tr>
    <tr>
      <td>10</td>
      <td><strong>卡车空满载</strong> (<code>ChanYunXiaoMeiZhong</code>)</td>
      <td><strong>500 张</strong></td>
      <td>131 张</td>
      <td>8 / 8 (100%)</td>
      <td>347.5 MB</td>
      <td><strong>387 框</strong></td>
      <td>0 框 (0.0%)</td>
      <td>2 框 (0.5%)</td>
      <td>385 框 (99.5%)</td>
      <td><code>0:卡车空载(188)</code>, <code>1:卡车满载(22)</code>, <code>3:车辆车斗遮盖(63)</code>, <code>6:块煤(32)</code>, <code>7:沫煤(82)</code></td>
    </tr>
    <tr>
      <td>11</td>
      <td><strong>探水前上杆</strong> (<code>TS_Front_V2</code>)</td>
      <td><code>obb</code></td>
      <td><strong>600 张</strong></td>
      <td>0 张</td>
      <td>4 / 4 (100%)</td>
      <td>318.3 MB</td>
      <td><strong>1,166 框</strong></td>
      <td>0 框 (0.0%)</td>
      <td>64 框 (5.5%)</td>
      <td>1,102 框 (94.5%)</td>
      <td><code>0:无杆(230)</code>, <code>1:有杆(372)</code>, <code>2:机器(564)</code></td>
    </tr>
    <tr>
      <td><strong>合计</strong></td>
      <td><strong>11 个业务标准测试集</strong></td>
      <td><strong>10 HBB + 1 OBB</strong></td>
      <td><strong>5,700 张</strong></td>
      <td><strong>596 张</strong></td>
      <td><strong>774 / 818</strong></td>
      <td><strong>3,287.0 MB</strong></td>
      <td><strong>17,037 框</strong></td>
      <td><strong>4,898 框 (28.7%)</strong></td>
      <td><strong>6,580 框 (38.6%)</strong></td>
      <td><strong>5,559 框 (32.6%)</strong></td>
      <td><strong>台账 44 个业务标签 100% 覆盖</strong></td>
    </tr>
  </tbody>
</table>

> 校验：4,898 + 6,580 + 5,559 = 17,037；28.7% + 38.6% + 32.6% ≈ 100%。逐行相加与合计行一致。

---

## 三、 标准测试集目录结构规范

每个业务子目录（`data/benchmark/standard/<dataset_key>/`）均采用统一的自包含结构：

```text
data/benchmark/standard/<dataset_key>/
├── images/          # 抽出的原始高清测试图片（400～600 张）
├── labels/          # 对应的 YOLO 格式标注文件（坐标已规范化校验至 [0.0, 1.0]）
├── data.yaml        # 容器内可直接挂载评测的数据集配置（含权威 names 映射）
├── manifest.json    # 抽样溯源清单（总张数、总字节数、逐矿区抽取张数、seed、源根目录）
└── excluded.txt     # 抽中图片的原始 /home/jykj/Dataset 绝对路径清单（供后续重训时排除，防止数据泄漏）
```

`data.yaml` 固定写成 `train: images` / `val: images`，语义是**纯评测目录**，不提供训练入口。

---

## 四、 标准测试集抽取做法（`scripts/08-build-standard-set.py`）

> **重要**：当前入库的脚本**只是"补齐器"，不是"从零构建器"**。它读取 `images/` 已有张数，`need = target - 已有`，够数就直接返回。仓库里**已经不存在**第一阶段的 `val.txt` 白名单抽样与 `even_pick` 等距取样代码（入库前被裁剪）。因此**删掉 `images/` 后重跑无法复现当前盘上的这一份数据**。

### 4.1 两阶段设计的历史由来（为什么要两轮）

1. **第一阶段（已不在代码中）——基于 `val.txt` 白名单的跨矿区等距抽样**：
   只从验证集白名单取图，构造上保证"标准集 ∉ 训练集"。
2. **第二阶段（当前仅存这一段）——注水补齐**：
   验证集只占 1/10，且 `PersonHelmet` 单矿区配额仅 2 张，第一阶段在很多长尾类上取不到样本。第二阶段的使命是把每集补到 400~600 张、并保证 44 类全覆盖。

这就是两轮的根本原因：**第一轮保"零泄漏"，第二轮保"覆盖率与规模"，两者不可兼得**。第二轮必然引入训练批次图片，所以必须由 `excluded.txt` 兜底——这正是 §4.4 存在的理由。

#### 4.1.1 两轮的真实约束（三条硬证据）

**(1) 长尾类稀疏到只能"点名取"，随机抽必然漏。**

`TanShuiHouShangGan` 的 `1:堵头` 类全库仅 **12 个框**，标准集里 12 框 = **100% 全取**，且只分布在 2 个批次目录。这类样本靠"val 白名单随机抽"的命中概率趋近于 0，只能由 `SUPPLEMENT_PLANS` 硬编码目录路径定向取。

**(2) 矿区数差异导致单矿区配额被压到 2~3 张。**

| 数据集 | 一级矿区数 | 目标张数 | `per_mine_quota` |
|---|---|---|---|
| `PersonHelmet` | 373 | 600 | **2 张** |
| `PiDaiYuanBan` | 272 | 600 | **3 张** |
| `RenChengHouChe` | 12 | 500 | 100 张 |
| `TS_Front_V2` | 4 | 600 | 150 张 |

`PersonHelmet` 要在 373 个矿区间均匀铺 600 张，单矿区只能给 2 张。若只从 val（占 1/10）取，配额 2 张极易取不满——实测最终仍有 **43 个矿区一张都没取到**（`per_mine` 中值为 0 的条目数）。

**(3) 两轮的分工与代价。**

```text
第一轮：val 白名单 + 跨矿区等距   →  构造上保证「标准集 ∉ 训练集」
第二轮：多矿区注水轮询补齐        →  必然引入训练批次图片
                                   泄漏防线从「构造保证」降级为「清单兜底」
```

`excluded.txt` 存在的唯一理由就是第二轮的这次降级——它把"结构上不可能泄漏"换成了"靠清单人工排除"。

#### 4.1.2 第一轮规模：历史口径不可复核，勿引用

> **旧版本写"第一阶段共采到 2,194 张（38.5%）"，该数字无法复核，已从本文删除。**

用 `manifest.json` 的 `per_mine_quota` 反推第一轮理论上限（`Σ min(每矿区实际张数, quota)`）：

| 数据集 | quota | 矿区 | 第一轮上限 | 实际 | 第二轮补 |
|---|---|---|---|---|---|
| `ChanYunXiaoMeiZhong` | 63 | 8 | 370 | 500 | 130 |
| `ChuanGanQi` | 50 | 8 | 307 | 400 | 93 |
| `FengMenKaiGuan` | 50 | 10 | 403 | 500 | 97 |
| `HuBangBan` | 40 | 38 | 459 | 500 | 41 |
| `PersonHelmet` | 2 | 373 | 486 | 600 | 114 |
| `PersonSleep` | 24 | 21 | 398 | 500 | 102 |
| `PiDaiYuanBan` | 3 | 272 | 523 | 600 | 77 |
| `RenChengHouChe` | 100 | 12 | 439 | 500 | 61 |
| `TS_Front_V2` | 150 | 4 | 600 | 600 | 0 |
| `TanShuiHouShangGan` | 11 | 58 | 434 | 500 | 66 |
| `WorkclothesHelmet` | 39 | 13 | 392 | 500 | 108 |
| **合计** | | | **4,811** | **5,700** | **889** |

**4,811 与旧值 2,194 相差 2.6 倍。** 第一阶段代码已不在仓库，两种口径都无法复现。本文只保留可复核的 `4,811 / 5,700 / 889` 这一组，并把"第一轮实际采到多少"标为**不可复核**。若需精确数字，必须把第一阶段代码补回后重跑。

> 另注：`per_mine` 合计比 `images` 少 **108 张**（5,592 vs 5,700），差额与 `manifest.json` 的 `class_stratified_supplement` 字段（12/12/12/30/30/12）完全吻合——这 108 张是长尾补抽样，未计入 `per_mine`。`TS_Front_V2` 的第二轮补量为 0，说明它 600 张全部由第一轮配额（150×4）铺满。

### 4.2 当前代码的唯一路径：多矿区注水轮询（`top_up_dataset`）

1. 遍历业务根目录下全部一级矿区子目录，`fast_collect_from_mine` 对每个矿区用 `os.scandir` 最多下钻 30 层子目录、每层只扫前 600 项，收集"图片 + 同名 `.txt`"配对。
2. 剔除已在 `excluded.txt` 或 images 目录中已存在的文件。
3. 用 `SEED = 42` 对每个矿区的候选池打乱。
4. **按矿区轮询（Round-Robin）**：每轮从每个有效矿区各取 1 张，直到补满目标张数（`PersonHelmet` / `PiDaiYuanBan` / `TS_Front_V2` 各 600，`ChuanGanQi` 400，其余 7 集各 500）。
5. 同名文件冲突时加 `<矿区>__` 前缀重命名。

目标张数在 `BUSINESSES` 字典中硬编码。

### 4.3 标注合法性校验与越界坐标裁剪（`check_and_clamp_label`）

逐文件校验：

- 列数：HBB 必须 5 列、OBB 必须 9 列，否则**整图丢弃**；
- 类别 ID 必须在 `[0, nc)` 内，`nc` 取自业务根目录的 `*.names`；
- HBB 要求 `w > 0 且 h > 0`；
- 坐标逐位 `clamp` 到 `[0.0, 1.0]`，容忍 `-0.0001` / `1.0002` 这类边缘溢出（否则 Ultralytics 会静默丢图）。

全部 11 集 `skipped_bad_labels = 0`，即**没有任何一张图因标注非法被丢弃**。

> `SUPPLEMENT_PLANS`（第 66 行）在入库版本中**已无任何引用**，属于死代码。`manifest.json` 里的 `class_stratified_supplement: 12/30` 是旧版脚本写入后保留的历史字段，**不能作为"当前脚本会做长尾保底"的证据**。

### 4.4 后续重训防数据泄漏规范

任何新训练或微调任务，在构建数据集前必须执行：

```bash
# 以单业务为例：从候选池里剔除标准集原始路径
grep -vxF -f data/benchmark/standard/PersonHelmet/excluded.txt candidate_train_list.txt \
  > train_list_clean.txt
```

`excluded.txt` 是**绝对路径**，因此 grep 时须与候选清单的路径格式一致（同为宿主机路径 `/home/jykj/Dataset/...`，不要改写成别的前缀）。

> **范围**：本项目只做算法评测，不负责训练。本节只说明标准集该怎么被排除，**不改训练工程、不提供训练脚本**。训练侧是否真的用了这份清单，不在本项目范围内。

---

## 五、 除"规范"之外的工程防线（已实测生效）

| # | 防线 | 位置 | 实测状态 |
|---|---|---|---|
| 1 | 绝对路径排除清单 | 11 份 `excluded.txt` | 5,700 条，与 `images/` **严格 1:1**，指向失效的路径 **0 条** |
| 2 | 溯源清单 | `manifest.json` | `source_root` / `names_file` / `seed=42` / `per_mine` 全量可回溯 |
| 3 | 评测集**只读**挂载 | `docker-compose.yml` | `./data/benchmark:/data/benchmark:ro`，容器内**物理无法写入**标准集 |
| 4 | 目录树隔离 | 标准集在本项目内，不在 `/home/jykj/Dataset` | 直接 `find /home/jykj/Dataset` 的训练流水线**扫不到**标准集 |
| 5 | `data.yaml` 语义约束 | 每集 `train: images / val: images` | 明示 eval-only，不提供训练入口 |
| 6 | 排除清单入库 | `.gitignore` | 忽略 `images/`、`labels/`，但提交 `excluded.txt` + `manifest.json`，排除清单不随图丢失 |
| 7 | 跨集隔离 | 11 集两两比对 | 路径交集 **0**；`basename` 冲突 **1 个**（`hn-bysmk-...-017720.jpg` 同时出现在 `PersonHelmet` 与 `WorkclothesHelmet`），非 0，见 §5.1(2) |

### 5.1 已知缺口（必须收口，不是"已解决"）

**(1) 近邻帧泄漏——当前防线完全不覆盖。**
5,700 张标准图 **100%** 在同一批次目录内存在 ±30 帧内的兄弟帧（平均 6 张）。同一段矿井视频里相邻几秒的画面与测试帧几乎同分布，只按 `excluded.txt` 排除无法阻断。

三种排除粒度的实测代价：

| 数据集 | 全量图片 | 仅路径级 | ±30 帧 | 整个批次目录 |
|---|---|---|---|---|
| TanShuiHouShangGan | 704,029 | 500 (0.1%) | 1,254 (0.2%) | 72,279 (10.3%) |
| PersonHelmet | 309,422 | 600 (0.2%) | 831 (0.3%) | 29,114 (9.4%) |
| TS_Front_V2 | 265,179 | 600 (0.2%) | 2,099 (0.8%) | 23,120 (8.7%) |
| PiDaiYuanBan | 166,048 | 600 (0.4%) | 2,252 (1.4%) | 165,715 (99.8%) |
| ChanYunXiaoMeiZhong | 74,468 | 500 (0.7%) | 1,559 (2.1%) | 14,462 (19.4%) |
| PersonSleep | 31,216 | 500 (1.6%) | 1,419 (4.5%) | 9,480 (30.4%) |
| FengMenKaiGuan | 19,409 | 500 (2.6%) | 1,482 (7.6%) | 4,077 (21.0%) |
| HuBangBan | 14,121 | 500 (3.5%) | 1,272 (9.0%) | 14,121 (**100%**) |
| RenChengHouChe | 8,386 | 500 (6.0%) | 1,867 (22.3%) | 8,386 (**100%**) |
| WorkclothesHelmet | 5,158 | 500 (9.7%) | 695 (13.5%) | 2,401 (46.5%) |
| ChuanGanQi | 2,299 | 400 (17.4%) | 511 (22.2%) | 1,648 (71.7%) |

**建议**：全量 ≥7 万的 5 个业务（TanShuiHouShangGan / PersonHelmet / TS_Front_V2 / PiDaiYuanBan / ChanYunXiaoMeiZhong）改用 **±30 帧排除**，代价 0.2%~2.0%；其余小业务维持路径级，因为升到批次目录级会清零（HuBangBan / RenChengHouChe 为 100%）。

**(2) 跨任务同帧——按业务独立取排除清单时会漏，且已实测发生。**

`Hbb/WorkclothesHelmet` 与 `Hbb/PersonHelmet` 是同一批矿井视频的两套标注。实测 `WorkclothesHelmet` 标准集 500 张中 **384 张（76.8%）** 在 `PersonHelmet` 目录下存在同名同帧文件。

**这不是理论风险，泄漏已经真实发生。** 用训练侧实际 baseline 清单比对：

| 比对项 | 张数 |
|---|---|
| `PersonHelmet` baseline train 清单 | 27,134 |
| `PersonHelmet` baseline val 清单 | 7,495 |
| `WorkclothesHelmet` 标准集 | 500 |
| **∩ `PersonHelmet` train** | **37** |
| **∩ `PersonHelmet` val** | **19** |
| **合计已泄漏** | **56 张（占该标准集 11.2%）** |

→ 若重训 `PersonHelmet`，只排除 `PersonHelmet/excluded.txt` 会把这 56 帧（潜在 384 帧）放进训练集，随后在 `WorkclothesHelmet` 标准集上评测即构成泄漏。
**建议**（给以后重训时参考，本项目不执行）：排除时按 **`basename` 跨全部 11 个业务的 `excluded.txt` 取并集**，而不是按单业务各取各的。

**(3) 脚本无法从零重建。**
当前 `08-build-standard-set.py` 仅为补齐器，`SUPPLEMENT_PLANS` 为死代码。若需可复现构建，必须把第一阶段代码（`val.txt` 白名单 + `even_pick`）与长尾保底逻辑一并补回，或改写本文为"仅描述现状"。

**(4) ~~`scripts/09-dataset-full-inventory.py` 的 OBB 尺度计算有 bug~~（已修复，本条关闭）。**
旧版第 93 行 `xs = coords[2::2]` 会丢弃一半顶点。当前脚本已改为四顶点鞋带公式（`scripts/09-dataset-full-inventory.py` 第 92~104 行），并在注释中显式标注 `coords[2::2]` 勿用。`TS_Front_V2` 尺度分布已按鞋带公式重算。

---

## 六、 数字复核口径（可复现）

| 指标 | 复核方式 |
|---|---|
| 图片张数 | `len(os.listdir(<set>/images))` |
| 背景图 | `labels/*.txt` 中 `os.path.getsize == 0` 的文件数 |
| 标注框数 | 逐行解析 `labels/*.txt`，跳过空行 |
| HBB 面积 | `(w × 640) × (h × 640)`，`w/h` 取第 3、4 个字段 |
| OBB 面积 | 四顶点鞋带公式 `|½Σ(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)| × 640²` |
| 磁盘体积 | `sum(os.path.getsize)` 遍历整个 `<set>/`（含 labels、配置清单） |
| 矿区覆盖 | `manifest.json` 的 `covered_mine_count` / `mine_count` |
| 类别分布 | 逐行 `int(float(parts[0]))` 计数，标签名映射自台账 `docs/benchmark-full-table.md` |
| excluded 完整性 | 与 `images/` 张数比对 + `os.path.exists` 全量校验 |
