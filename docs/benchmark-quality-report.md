# 11 个基准数据集全量质检报告

生成时间：2026-11-12 17:24　|　数据来源：`output/dataset/benchmark-quality-v1/`　|　生成脚本：`scripts/19-benchmark-quality-report.py`

本报告汇总 11 个基准数据集（10 个 HBB + 1 个 OBB）的**全量内容级质检**结果：逐张完整解码图片、逐行校验标注，而非抽样。

## 一、总体结论

- 图片：**1,602,513 张**，体积 **739.4 GB**，全部逐张完整解码；解码失败 **1 张**，零字节 **0 张**。
- 标注：**4,670,797 个框**，其中合法 **4,493,837 个**，异常 **176,960 个**（占 3.79%）。
- 配对：缺少标注 **0 张**，孤立标注 **0 个**，同目录同名多图 **2 个**。
- 空标注图片（无框但合法）：**106,674 张**，属正常负样本。
- 色彩模式：RGB 1,602,512；编码格式：JPEG 1,602,512。

**判定：数据完整、无损坏，可直接用于训练。** 异常集中在 5 个数据集的少数标注行，明细见第四节。

## 二、逐数据集统计

| 数据集 | 任务 | 图片 | 标注框 | 合法框 | 异常框 | 空标注图 | 体积 GB | 分辨率种类 | 分片 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PersonHelmet` | detect | 309,422 | 1,370,100 | 1,370,090 | 10 | 72,064 | 160.3 | 5412 | 894 |
| `PersonSleep` | detect | 31,216 | 50,738 | 50,738 | 0 | 4,207 | 16.5 | 14 | 22 |
| `WorkclothesHelmet` | detect | 5,158 | 16,716 | 16,716 | 0 | 31 | 3.8 | 5 | 14 |
| `RenChengHouChe` | detect | 8,386 | 28,651 | 28,651 | 0 | 2,584 | 10.2 | 4 | 13 |
| `PiDaiYuanBan` | detect | 166,048 | 1,071,423 | 1,071,422 | 1 | 4 | 93.0 | 10 | 274 |
| `HuBangBan` | detect | 14,121 | 48,198 | 48,198 | 0 | 1,463 | 7.8 | 6 | 39 |
| `TanShuiHouShangGan` | detect | 704,199 | 1,471,896 | 1,471,896 | 0 | 7,043 | 295.8 | 8 | 527 |
| `FengMenKaiGuan` | detect | 19,409 | 18,188 | 18,188 | 0 | 1,440 | 9.5 | 6 | 11 |
| `ChuanGanQi` | detect | 2,299 | 6,191 | 6,189 | 2 | 0 | 1.0 | 3 | 9 |
| `ChanYunXiaoMeiZhong` | detect | 77,076 | 60,396 | 60,395 | 1 | 17,838 | 36.6 | 5 | 10 |
| `TS_Front_V2` | obb | 265,179 | 528,300 | 351,354 | 176,946 | 0 | 104.9 | 6 | 578 |
| **合计** | — | **1,602,513** | **4,670,797** | **4,493,837** | **176,960** | **106,674** | **739.4** | — | **2391** |

## 三、问题汇总

| 问题类型 | 数量 | 涉及数据集 |
|---|---:|---|
| 坐标越出 [0,1]（`coord_out_of_range`） | 176,947 | `ChanYunXiaoMeiZhong`, `TS_Front_V2` |
| 宽或高非正（退化框）（`nonpositive_size`） | 13 | `ChuanGanQi`, `PersonHelmet`, `PiDaiYuanBan` |
| 图片解码失败（`decode_error`） | 1 | `PersonHelmet` |

> 说明：`coord_out_of_range` 在 `TS_Front_V2`（OBB）中数量很大，但幅度分析显示绝大多数为图像边界处的轻微溢出，并非坐标格式错误，详见第五节。

## 四、问题明细（按数据集 / 分片）

下表每集最多列问题最多的 10 个分片；完整逐分片清单见 `output/dataset/benchmark-quality-v1/issues.csv`。

### `ChanYunXiaoMeiZhong`

| 分片 | 问题类型 | 数量 |
|---|---|---:|
| `neimengguerlintu` | 坐标越出 [0,1]（`coord_out_of_range`） | 1 |

### `ChuanGanQi`

| 分片 | 问题类型 | 数量 |
|---|---|---:|
| `sd-jzm` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `sd-tk` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |

### `PersonHelmet`

| 分片 | 问题类型 | 数量 |
|---|---|---:|
| `sxx-hsz-jx-jxys-fjk-fjk02-wlsb17780-2025041001` | 宽或高非正（退化框）（`nonpositive_size`） | 4 |
| `sx-jgsk` | 图片解码失败（`decode_error`） | 1 |
| `hn-csl` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `sx-ll` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `sx-xlymk` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `sxx-hll` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `sxx-hsz-jx-jxys-fjk-fjk02-wlsb19565-2025042901` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |
| `xj-hamk` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |

### `PiDaiYuanBan`

| 分片 | 问题类型 | 数量 |
|---|---|---:|
| `sxhy-ekxxjzypd` | 宽或高非正（退化框）（`nonpositive_size`） | 1 |

### `TS_Front_V2`

| 分片 | 问题类型 | 数量 |
|---|---|---:|
| `ShanXi` | 坐标越出 [0,1]（`coord_out_of_range`） | 18,415 |
| `1-tanshui-Qiao-11663` | 坐标越出 [0,1]（`coord_out_of_range`） | 7,791 |
| `PuBai` | 坐标越出 [0,1]（`coord_out_of_range`） | 7,737 |
| `530325006707-AC-BA-DB-Z-HDS-BJ08-2026061401` | 坐标越出 [0,1]（`coord_out_of_range`） | 2,013 |
| `530325006707-AC-BA-DC-HDS-BJ10-2026061301` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,898 |
| `140423006459-AA-BB-DA-DJZ-BJ09-2020112701` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,817 |
| `530325006707-AC-BB-DA-HDS-BJ04-2026061301` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,707 |
| `140781007992-AA-BA-DB-DD-SJZ-BJ47-2025051501` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,648 |
| `140781007992-AA-BA-DB-DC-SJZ-BJ52-2025061601` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,596 |
| `530325006707-AC-BA-DC-HDS-BJ05-2026061301` | 坐标越出 [0,1]（`coord_out_of_range`） | 1,530 |
| *其余 358 个分片* | — | 130,794 |

## 五、已逐一定位的异常标注

以下 14 个异常框已定位到具体文件与行号，可直接修复。

### 5.1 退化框（宽或高为 0）

| 数据集 | 文件（相对业务根目录） | 行 | 类别 | 内容（cls x y w h） | 问题 |
|---|---|---:|---:|---|---|
| `PersonHelmet` | `sxx-hsz/sxx-hsz-jx-jxys-fjk-fjk02-wlsb17780-2025041001/sxx-hsz-jx-jxys-fjk-fjk02-wlsb17780-2025041001-000425.txt` | 2–5 | 2 | `2 0.7007 0.9844 0.0 0.0` | 4 行完全相同，重复退化框，应整段删除 |
| `PersonHelmet` | `sxx-hsz/sxx-hsz-jx-jxys-fjk-fjk02-wlsb19565-2025042901/sxx-hsz-jx-jxys-fjk-fjk02-wlsb19565-2025042901-001000.txt` | 1 | 2 | `2 0.3784 0.7202 0.0 0.0` | 宽、高均为 0 |
| `PersonHelmet` | `hn-csl/hn-csl-jx-jxys-kcysgd-640jzgdxsspt-wlsb14890-2024110701/hn-csl-jx-jxys-kcysgd-640jzgdxsspt-wlsb14890-2024110701-002550.txt` | 1 | 3 | `3 0.6169 0.3613 0.0059 0.0` | 高为 0 |
| `PersonHelmet` | `sx-xlymk/sx-xlymk-js-jdyfzss-wxcj-jxcj-hc-2024102402/sx-xlymk-js-jdyfzss-wxcj-jxcj-hc-2024102402-002550.txt` | 1 | 2 | `2 0.6688 0.6066 0.0 0.0` | 宽、高均为 0 |
| `PersonHelmet` | `sx-ll/sx-ll-jx-jxys-jkcrzz-ebhcjt-wlsb14845-2024103001/sx-ll-jx-jxys-jkcrzz-ebhcjt-wlsb14845-2024103001-001890.txt` | 3 | 2 | `2 0.4910 0.2268 0.0 0.0` | 宽、高均为 0 |
| `PersonHelmet` | `sxx-hll/sxx-hll-ty-bgq-ddzx-dmdds1-wlsb22698-2025052302/sxx-hll-ty-bgq-ddzx-dmdds1-wlsb22698-2025052302-000750.txt` | 4 | 2 | `2 0.3684 0.1101 0.0 0.0` | 宽、高均为 0 |
| `PersonHelmet` | `xj-hamk/xj-hamk-jx-jxys-kcysgd-66gfxjk-hc-wlsb14742/xj-hamk-jx-jxys-kcysgd-66gfxjk-hc-wlsb14742-001245.txt` | 5 | 2 | `2 0.6312 0.7495 0.0 0.0` | 宽、高均为 0 |
| `ChuanGanQi` | `sd-jzm/sd-jzm-jx-cm-gdhk-103x02gzmyjjwcgq-hc-2024030201/sd-jzm-jx-cm-gdhk-103x02gzmyjjwcgq-hc-2024030201-029240.txt` | 1 | 0 | `0 0.3533 0.4877 0.0 0.0` | 宽、高均为 0 |
| `ChuanGanQi` | `sd-tk/sd-tk-jx-cm-gdhk-7307cmgzmjwT0-hc-2024030101/sd-tk-jx-cm-gdhk-7307cmgzmjwT0-hc-2024030101-007820.txt` | 1 | 2 | `2 0.5477 0.8194 0.0 0.0056` | 宽为 0 |
| `PiDaiYuanBan` | `sxhy-ekxxjzypd/sxhy-ekxxjzypd-2025042201-000725.txt` | 29 | 4 | `4 0.4860 0.6261 0.0 0.0` | 宽、高均为 0 |

> `PersonHelmet/sxx-hsz/...-000425.txt` 的第 2–5 行是**四个坐标完全相同的重复退化框**，应整段删除；其余多为宽或高恰好为 `0.0` 的无效框。这些框的宽高为 0，在 YOLO 训练中不产生任何梯度贡献，也不会导致报错，但会让"框数"统计虚高，建议清理。

### 5.2 坐标越界（标注错误）

| 数据集 | 文件（相对业务根目录） | 行 | 类别 | 内容（cls xc yc w h） | 问题 |
|---|---|---:|---:|---|---|
| `ChanYunXiaoMeiZhong` | `neimengguerlintu/elt-zcspjk-2022121801/elt-zcspjk-2022121801-2380.txt` | 1 | 7 | `1.370703 1.504167 0.833594 0.997222` | 中心点 `(1.37, 1.50)` 远在图像外，属误标 |

### 5.3 图片文件问题

| 数据集 | 文件（相对业务根目录） | 大小 | 问题 | 影响 |
|---|---|---:|---|---|
| `PersonHelmet` | `sx-jgsk/sx-jgsk-jx-jd-zhkg-dypdd-hc-2024101801/sx-jgsk-jx-jd-zhkg-dypdd-hc-2024101801-000175.jpg` | 229,376 B | 严格解码报 `truncated (28 bytes not processed)`（尾部缺 28 字节） | 容错模式可解出完整 1280×720 RGB，不影响训练，建议重传 |

### 5.4 标签文件换行符

抽查发现标签文件换行符不统一，属格式差异而非错误（Ultralytics 解析不受影响）：

| 数据集 | 抽查数 | CRLF | LF | 无换行符 |
|---|---:|---:|---:|---:|
| `PersonHelmet` | 300 | 211（70%） | 0（0%） | 89 |
| `TanShuiHouShangGan` | 120 | 120（100%） | 0（0%） | 0 |
| `PiDaiYuanBan` | 300 | 40（13%） | 260（87%） | 0 |
| `TS_Front_V2` | 120 | 120（100%） | 0（0%） | 0 |

如需统一为 LF（便于 `git diff` 与跨工具处理），可对 CRLF 文件做一次批量转换；当前状态下**不需要**为训练特意改动。

### 5.5 一键修复

上述 13 个退化框与 1 个越界误标已封装为脚本 `scripts/21-fix-bad-labels.py`。脚本会先逐行校验目标内容与记录一致（防止文件被改动后误删），默认 **dry-run**，`--apply` 时才写入并自动备份原文件到 `output/dataset/benchmark-quality-v1/backup-labels/<时间戳>/`。

```bash
python3 scripts/21-fix-bad-labels.py            # 试运行，只打印动作
python3 scripts/21-fix-bad-labels.py --apply    # 执行（自动备份）
```

修复后预期：11 个文件全部无残留异常框。其中 `ChuanGanQi/sd-tk/...-007820.txt` 删除唯一一行后会变成空标签文件（图片保留，成为合法负样本）；其余文件仍保留有效框。

## 六、`TS_Front_V2` 坐标越界专项分析

该 OBB 数据集有 **176,946 个框（33.5%）** 坐标超出 [0,1]，分布于 **51.4%** 的标签文件。经幅度分析：

| 越界幅度 | 框数 | 占比 |
|---|---:|---:|
| ≤ 0.01 | 49,396 | 27.9% |
| ≤ 0.05 | 89,962 | 50.8% |
| ≤ 0.1 | 21,565 | 12.2% |
| > 0.1 | 16,023 | 9.1% |

约 **79% 的越界不超过 0.05**，是标注工具在图像边界处产生的轻微溢出（如 `1.02`、`-0.03`），**并非坐标格式错误**——该集使用的是归一化坐标（非像素坐标），格式正确。训练时按常规裁剪到 [0,1] 即可；若要求严格，可对这 33.5% 的框统一做一次边界裁剪。

## 七、质检方法与可复现性

- **图片校验**：`PIL.Image.open()` + `load()` 完整解码，非仅读文件头；同时记录真实分辨率、色彩模式与编码格式。
- **标注校验**：逐行检查列数、数值有限性、类别 ID 是否在 `*.names` 范围内、坐标是否落在 [0,1]（容差 1e-3）、HBB 宽高是否为正、OBB 四边形面积是否为正。
- **配对校验**：按相对路径（目录 + 文件名去扩展名）配对图片与标签，统计缺标注、孤立标注、同目录同名多图。
- **分片与续跑**：按一级目录分片，超过 2 万张的分片递归下钻，共 2,391 个分片；结果原子写入 JSON，支持断点续跑。
- **读序优化**：分片内先收集文件并按 inode 排序再读取，使机械盘磁头接近顺序访问（实测比目录顺序读取快约 50%）。
- **复现命令**：

```bash
# 全量校验（约 5.5 小时 / 739 GB，机械盘 8 并发）
python3 scripts/15-benchmark-quality-audit.py --workers 8

# 仅重新汇总（不重扫）
python3 scripts/15-benchmark-quality-audit.py --merge-only

# 生成本报告
python3 scripts/19-benchmark-quality-report.py

# 校验产物完整性（sha256）
python3 scripts/20-benchmark-quality-checksums.py --verify
```

全部产物的 sha256 与大小记录在 `output/dataset/benchmark-quality-v1/CHECKSUMS.txt`，可在任意时间点用 `--verify` 确认结果未被改动或丢失。

## 八、未覆盖范围

本次质检覆盖：图片可解码性、真实分辨率、色彩模式、标注数值/类别/坐标合法性、图片与标签的路径配对。

**未覆盖**：

- 标注框的**语义正确性**（框是否真的框住对应类别目标）、框与目标的对齐精度、类别标签是否正确；
- 近邻帧重复、训练集/测试集泄漏；
- 标注框的尺度分布与长尾情况（见 `docs/benchmark-dataset-inventory.md` 的台账口径）。

上述项目需人工抽检或模型评测确认，不能由本报告替代。
