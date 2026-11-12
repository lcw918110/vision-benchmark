#!/usr/bin/env python3
"""全量原始数据集统计脚本 —— 扫描 /home/jykj/Dataset 下全部业务数据集，
产出结构化 JSON 与 Markdown 统计清单（docs/dataset-full-inventory.md）。

统计口径：
  1. 逐文件遍历，图片与同名 .txt 标注按 stem 配对，统计缺标注图 / 孤立标注 / 纯背景图。
  2. 标注框按 YOLO 归一化坐标解析：HBB 5 列、OBB 9 列；列数异常计入 malformed。
  3. 目标尺度沿用 docs/dataset-standard-report.md 的口径：按 640×640 推理尺寸折算
     归一化面积（小 <32²、中 32²~96²、大 ≥96²），便于与标准测试集报告横向对照。
  4. 分辨率分布采用抽样方式（每集最多 SAMPLE 张，仅读 JPEG/PNG 文件头），避免打开全部 200 万张。
  5. 磁盘体积按文件实际 st_size 累加，不调用 du。

用法：
  python3 scripts/09-dataset-full-inventory.py            # 全量扫描并生成文档
  python3 scripts/09-dataset-full-inventory.py --quick    # 跳过分辨率抽样（更快）
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

SRC_ROOT = Path("/home/jykj/Dataset")
ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "output/dataset/inventory.json"
OUT_MD = ROOT / "docs/dataset-full-inventory.md"
CACHE_DIR = ROOT / "output/dataset/.cache"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLE = 400
FAST_SAMPLE = 300
SEED = 42

# (分区标识, 分区根目录, 任务类型, 说明)
PARTITIONS = [
    ("Hbb", SRC_ROOT / "Hbb", "detect", "水平框主线业务（11 个，已纳入标准测试集）"),
    ("100/Hbb", SRC_ROOT / "100" / "Hbb", "detect", "水平框扩展业务（20 个）"),
    ("101/Hbb", SRC_ROOT / "101" / "Hbb", "detect", "水平框增量业务（10 个）"),
    ("Obb", SRC_ROOT / "Obb", "obb", "旋转框业务（1 个）"),
]

SCALE_SMALL = 32.0 * 32.0
SCALE_MED = 96.0 * 96.0
REF_SIZE = 640.0


def read_names(biz_dir):
    """读取业务根目录下的 *.names 类别定义。"""
    for p in sorted(biz_dir.glob("*.names")):
        try:
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
            names = [ln for ln in lines if ln]
            if names:
                return names, p.name
        except Exception:
            continue
    return [], None


def parse_label(path, task):
    """解析一个 YOLO 标注文件，返回 (框数, 类别计数, 尺度计数, 异常行数)。"""
    boxes = 0
    cls_count = {}
    scale = {"small": 0, "medium": 0, "large": 0}
    bad = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                need = 9 if task == "obb" else 5
                if len(parts) != need:
                    bad += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad += 1
                    continue
                boxes += 1
                cls_count[cid] = cls_count.get(cid, 0) + 1
                if task == "obb":
                    # 9 列 = 类别 + 4 个顶点 (x1 y1 x2 y2 x3 y3 x4 y4)，不含 cx cy。
                    # 面积用鞋带公式算真实多边形面积；外接矩形会把 45° 旋转的目标
                    # 面积虚高（实测 TS_Front_V2 大目标占比 98.0% vs 真实 94.5%）。
                    # 注意 coords[2::2] 只取到后半段顶点，会丢掉一半顶点，勿用。
                    pts = [(coords[0], coords[1]), (coords[2], coords[3]),
                           (coords[4], coords[5]), (coords[6], coords[7])]
                    double_area = 0.0
                    for i in range(4):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % 4]
                        double_area += x1 * y2 - x2 * y1
                    # 归一到 [0,1]² 的多边形面积，再折算到 REF_SIZE² 像素
                    area = abs(double_area) / 2.0 * REF_SIZE * REF_SIZE
                else:
                    w, h = coords[2], coords[3]
                    area = (max(0.0, w) * REF_SIZE) * (max(0.0, h) * REF_SIZE)
                if area < SCALE_SMALL:
                    scale["small"] += 1
                elif area < SCALE_MED:
                    scale["medium"] += 1
                else:
                    scale["large"] += 1
    except Exception:
        bad += 1
    return boxes, cls_count, scale, bad


def sample_resolutions(images, limit, rng):
    """抽样读取图片分辨率（仅读文件头）。"""
    if Image is None or not images:
        return {}, 0
    picks = images if len(images) <= limit else rng.sample(images, limit)
    counter = {}
    ok = 0
    for p in picks:
        try:
            with Image.open(p) as im:
                key = f"{im.width}×{im.height}"
            counter[key] = counter.get(key, 0) + 1
            ok += 1
        except Exception:
            continue
    return counter, ok


def scan_business(partition, biz_name, biz_dir, task, do_sample, rng, fast=False):
    """扫描单个业务数据集。

    用 os.scandir + entry.stat(follow_symlinks=False) 单次遍历完成
    文件计数、体积累加、图片/标注配对与标注解析，避免 os.walk + os.path.getsize
    对每个文件重复做路径解析（实测快约 11 倍）。

    fast=True 时仅抽样解析标注文件（每集最多 FAST_SAMPLE 个），
    框数按样本比例外推，用于机械盘冷读受限时快速产出规模统计。
    """
    t0 = time.time()
    names, names_file = read_names(biz_dir)

    img_stems = set()
    lbl_stems = set()
    label_paths = []
    total_bytes = 0
    mines = set()
    sessions = set()
    file_count = 0

    stack = [str(biz_dir)]
    while stack:
        cur = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except OSError:
            continue
        rel = os.path.relpath(cur, biz_dir)
        parts = [] if rel == "." else rel.split(os.sep)
        if len(parts) >= 1:
            mines.add(parts[0])
        if len(parts) >= 2:
            sessions.add(os.sep.join(parts[:2]))

        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                    continue
                total_bytes += e.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            file_count += 1
            name = e.name
            dot = name.rfind(".")
            ext = name[dot:].lower() if dot > 0 else ""
            if ext in IMG_EXT:
                img_stems.add(name[:dot])
            elif ext == ".txt":
                lbl_stems.add(name[:dot])
                label_paths.append(e.path)

    total_boxes = 0
    cls_count = {}
    scale = {"small": 0, "medium": 0, "large": 0}
    malformed = 0
    empty_labels = 0

    if fast and len(label_paths) > FAST_SAMPLE:
        # 抽样解析：随机取 FAST_SAMPLE 个标注，按比例外推到全量
        picks = rng.sample(label_paths, FAST_SAMPLE)
        s_boxes = s_empty = s_bad = 0
        s_cls, s_scale = {}, {"small": 0, "medium": 0, "large": 0}
        for lp in picks:
            b, c, s, bad = parse_label(lp, task)
            s_bad += bad
            if b == 0 and bad == 0:
                s_empty += 1
            s_boxes += b
            for k, v in c.items():
                s_cls[k] = s_cls.get(k, 0) + v
            for k in s_scale:
                s_scale[k] += s[k]
        ratio = len(label_paths) / len(picks)
        total_boxes = int(round(s_boxes * ratio))
        empty_labels = int(round(s_empty * ratio))
        malformed = int(round(s_bad * ratio))
        cls_count = {k: int(round(v * ratio)) for k, v in s_cls.items()}
        scale = {k: int(round(v * ratio)) for k, v in s_scale.items()}
        estimated = True
    else:
        for lp in label_paths:
            b, c, s, bad = parse_label(lp, task)
            malformed += bad
            if b == 0 and bad == 0:
                empty_labels += 1
            total_boxes += b
            for k, v in c.items():
                cls_count[k] = cls_count.get(k, 0) + v
            for k in scale:
                scale[k] += s[k]
        estimated = False

    missing_labels = len(img_stems - lbl_stems)
    orphan_labels = len(lbl_stems - img_stems)

    # 数据完整性状态判定
    if len(img_stems) == 0 and len(lbl_stems) == 0:
        status = "empty"
    elif len(img_stems) == 0:
        status = "labels_only"
    elif len(lbl_stems) == 0:
        status = "images_only"
    elif missing_labels or orphan_labels:
        status = "partial"
    else:
        status = "complete"

    resolutions, sampled = ({}, 0)
    if do_sample:
        # 仅在需要抽样时把图片路径展开，避免为 200 万张图常驻内存
        img_paths = []
        stack = [str(biz_dir)]
        while stack:
            cur = stack.pop()
            try:
                entries = list(os.scandir(cur))
            except OSError:
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif os.path.splitext(e.name)[1].lower() in IMG_EXT:
                        img_paths.append(e.path)
                except OSError:
                    continue
        resolutions, sampled = sample_resolutions(img_paths, SAMPLE, rng)
    elapsed = time.time() - t0
    print(f"  [done] {partition}/{biz_name}: {len(img_stems)} imgs, {total_boxes} boxes, "
          f"{len(mines)} mines, {elapsed:.1f}s", flush=True)

    return {
        "partition": partition,
        "dataset_key": biz_name,
        "path": str(biz_dir),
        "task": task,
        "names_file": names_file,
        "classes": names,
        "nc": len(names),
        "images": len(img_stems),
        "labels": len(lbl_stems),
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "status": status,
        "malformed_lines": malformed,
        "boxes": total_boxes,
        "class_counts": {str(k): v for k, v in sorted(cls_count.items())},
        "scale": scale,
        "mines": len(mines),
        "sessions": len(sessions),
        "files": file_count,
        "bytes": total_bytes,
        "resolutions": resolutions,
        "sampled": sampled,
        "estimated": estimated,
        "elapsed_s": round(elapsed, 1),
    }


def fmt_int(n):
    return f"{n:,}"


def fmt_mb(b):
    return f"{b / 1024 / 1024:,.1f} MB"


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


def build_markdown(results, quick):
    total = {
        "datasets": len(results),
        "images": sum(r["images"] for r in results),
        "labels": sum(r["labels"] for r in results),
        "boxes": sum(r["boxes"] for r in results),
        "empty": sum(r["empty_labels"] for r in results),
        "missing": sum(r["missing_labels"] for r in results),
        "orphan": sum(r["orphan_labels"] for r in results),
        "malformed": sum(r["malformed_lines"] for r in results),
        "bytes": sum(r["bytes"] for r in results),
        "mines": sum(r["mines"] for r in results),
        "sessions": sum(r["sessions"] for r in results),
    }

    L = []
    A = L.append

    # 完成度判定：把「已扫描业务」与「磁盘上实际存在的业务」对比。
    # 历史问题：分片扫描未完成时，脚本仍以「全量」为标题重写整篇文档，
    # 导致 19% 的进度看起来像 100%（曾出现只扫 8/42 却写「全量」）。
    scanned_keys = {(r["partition"], r["dataset_key"]) for r in results}
    expected = []
    for part, pdir, ptask, pdesc in PARTITIONS:
        if not pdir.exists():
            continue
        try:
            for e in os.scandir(pdir):
                if e.is_dir(follow_symlinks=False):
                    expected.append((part, e.name))
        except OSError:
            continue
    missing = sorted(set(expected) - scanned_keys)
    n_done, n_total = len(scanned_keys), len(expected)
    complete = (n_total > 0 and not missing)

    if complete:
        A("# 全量原始数据集统计清单（`/home/jykj/Dataset/`）")
    else:
        A(f"# 原始数据集统计清单（`/home/jykj/Dataset/`）—— **部分完成 {n_done}/{n_total}**")
    A("")
    A("> **归档路径**：`docs/dataset-full-inventory.md`  ")
    A("> **数据落盘根目录**：`/home/jykj/Dataset/`  ")
    A("> **生成脚本**：`scripts/09-dataset-full-inventory.py`（结构化产物：`output/dataset/inventory.json`）  ")
    A(f"> **统计时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    A(f"> **扫描方式**：逐文件遍历（非抽样），标注框全量解析；分辨率列为抽样统计"
      f"{'（本次已跳过）' if quick else f'（每集最多 {SAMPLE} 张）'}。")
    A("")

    if not complete:
        A("> ## ⚠️ 本文件尚未覆盖全量数据")
        A(">")
        A(f"> 磁盘上共 **{n_total}** 个业务目录，本次仅完成 **{n_done}** 个"
          f"（{n_done / n_total * 100:.1f}%），**缺 {len(missing)} 个**。")
        A("> 下文所有「总数」均为**已扫描部分**的合计，**不等于** `/home/jykj/Dataset/` 的物理全量。")
        A(">")
        A("> **未纳入的业务**：")
        A(">")
        for part, name in missing:
            A(f"> - `{part}/{name}`")
        A(">")
        A("> 补全方式：`bash scripts/12-dataset-shard-daemon.sh start`，"
          "全部分片完成后本文件会自动重写为「全量」口径。")
        A("")

    A("---")
    A("")
    A("## 一、 总体规模")
    A("")
    A(f"> 口径：以下合计仅统计**已完成的 {n_done} 个业务**"
      f"{'' if complete else f'（共 {n_total} 个）'}。")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 业务数据集总数 | **{total['datasets']} 个**{'（已扫描）' if not complete else ''} |")
    A(f"| 图片总数 | **{fmt_int(total['images'])} 张** |")
    A(f"| 标注文件总数 | **{fmt_int(total['labels'])} 个** |")
    A(f"| 标注框总数 | **{fmt_int(total['boxes'])} 框** |")
    A(f"| 纯背景图（空标注） | {fmt_int(total['empty'])} 张 |")
    A(f"| 缺标注图片 | {fmt_int(total['missing'])} 张 |")
    A(f"| 孤立标注（无对应图） | {fmt_int(total['orphan'])} 个 |")
    A(f"| 异常标注行 | {fmt_int(total['malformed'])} 行 |")
    A(f"| 矿区 / 一级子目录 | {fmt_int(total['mines'])} 个 |")
    A(f"| 点位 / 二级批次目录 | {fmt_int(total['sessions'])} 个 |")
    A(f"| 磁盘占用 | **{fmt_mb(total['bytes'])}** |")
    A("")
    A("### 1.1 分区构成")
    A("")
    A("| 分区 | 业务数 | 图片数 | 标注框数 | 磁盘占用 | 说明 |")
    A("|---|---|---|---|---|---|")
    for part, pdir, ptask, pdesc in PARTITIONS:
        sub = [r for r in results if r["partition"] == part]
        if not sub:
            continue
        A(f"| `{part}` | {len(sub)} | {fmt_int(sum(r['images'] for r in sub))} | "
          f"{fmt_int(sum(r['boxes'] for r in sub))} | {fmt_mb(sum(r['bytes'] for r in sub))} | {pdesc} |")
    seg = SRC_ROOT / "Seg"
    if seg.exists() and not any(seg.iterdir()):
        A("| `Seg` | 0 | 0 | 0 | 0 B | 分割业务空目录，暂无数据 |")
    A("")
    A("---")
    A("")
    A("## 二、 全量业务数据集总览")
    A("")
    A("按图片规模降序排列。`类别数` 取自业务根目录下的 `*.names`；`框/图` 为平均每图标注框数；"
      "`状态` 为数据完整性判定（`complete` 完整 / `partial` 部分缺失 / `labels_only` 仅标注 / "
      "`images_only` 仅图片 / `empty` 空目录）。")
    A("")
    A("| # | 分区 | 业务数据集 | 任务 | 状态 | 类别数 | 图片数 | 标注框数 | 框/图 | 空标注 | 矿区 | 磁盘占用 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(results, key=lambda x: -x["images"]), 1):
        per = f"{r['boxes'] / r['images']:.2f}" if r["images"] else "—"
        st = r.get("status", "complete")
        st_disp = {"complete": "✅ 完整", "partial": "⚠️ 部分缺失",
                   "labels_only": "⚠️ 仅标注", "images_only": "⚠️ 仅图片",
                   "empty": "❌ 空目录"}.get(st, st)
        est = "≈" if r.get("estimated") else ""
        A(f"| {i} | `{r['partition']}` | <code>{r['dataset_key']}</code> | {r['task']} | {st_disp} | {r['nc']} | "
          f"{fmt_int(r['images'])} | {est}{fmt_int(r['boxes'])} | {per} | {est}{fmt_int(r['empty_labels'])} | "
          f"{r['mines']} | {fmt_mb(r['bytes'])} |")
    A("")
    A("---")
    A("")
    A("## 三、 各数据集详细统计")
    A("")
    A("每个数据集一张表，包含规模、标注完整性、类别分布、目标尺度分布"
      f"{'、分辨率抽样分布' if not quick else ''}与矿区分布。")
    A("")

    for idx, r in enumerate(sorted(results, key=lambda x: (-x["images"], x["dataset_key"])), 1):
        A(f"### 3.{idx} {r['dataset_key']}（`{r['partition']}`）")
        A("")
        A(f"**路径**：`{r['path']}`  ")
        A(f"**任务类型**：`{r['task']}`　**类别数**：{r['nc']}"
          + (f"　**类别定义文件**：`{r['names_file']}`" if r["names_file"] else "　**类别定义文件**：未提供"))
        st = r.get("status", "complete")
        if st != "complete":
            warn = {"empty": "目录仅存在空骨架，无任何图片与标注文件，不可用于训练或评测",
                    "labels_only": "仅有标注文件而缺少对应图片，无法用于训练或评测",
                    "images_only": "仅有图片而缺少标注文件，需重新标注",
                    "partial": "存在缺标注图片或孤立标注文件，需核查"}.get(st, "")
            A(f"　**数据状态**：⚠️ `{st}` —— {warn}")
        A("")
        A("| 统计项 | 数值 | 统计项 | 数值 |")
        A("|---|---|---|---|")
        A(f"| 图片数 | **{fmt_int(r['images'])}** | 标注文件数 | {fmt_int(r['labels'])} |")
        est = "≈" if r.get("estimated") else ""
        A(f"| 标注框总数 | **{est}{fmt_int(r['boxes'])}** | 平均框/图 | "
          f"{(r['boxes'] / r['images']):.2f} |" if r["images"] else
          f"| 标注框总数 | 0 | 平均框/图 | — |")
        A(f"| 纯背景图（空标注） | {fmt_int(r['empty_labels'])} ({pct(r['empty_labels'], r['images'])}) | "
          f"缺标注图片 | {fmt_int(r['missing_labels'])} |")
        A(f"| 孤立标注文件 | {fmt_int(r['orphan_labels'])} | 异常标注行 | {fmt_int(r['malformed_lines'])} |")
        A(f"| 矿区 / 一级子目录 | {r['mines']} | 点位 / 二级批次目录 | {r['sessions']} |")
        A(f"| 文件总数 | {fmt_int(r['files'])} | 磁盘占用 | **{fmt_mb(r['bytes'])}** |")
        A("")

        if r["nc"]:
            A("**类别分布**")
            A("")
            A("| 类别 ID | 类别名 | 标注框数 | 占比 |")
            A("|---|---|---|---|")
            for cid in range(r["nc"]):
                cnt = r["class_counts"].get(str(cid), 0)
                nm = r["classes"][cid] if cid < len(r["classes"]) else "—"
                A(f"| {cid} | <code>{nm}</code> | {est}{fmt_int(cnt)} | {pct(cnt, r['boxes'])} |")
            extra = {k: v for k, v in r["class_counts"].items() if int(k) >= r["nc"]}
            for k in sorted(extra, key=lambda x: int(x)):
                A(f"| {k} | ⚠ 超出 nc 范围 | {fmt_int(extra[k])} | {pct(extra[k], r['boxes'])} |")
            A("")

        s = r["scale"]
        A("**目标尺度分布**（按 `640×640` 推理尺寸折算面积：小 `<32²`、中 `32²~96²`、大 `≥96²`）")
        A("")
        A("| 尺度 | 标注框数 | 占比 |")
        A("|---|---|---|")
        A(f"| 小目标 | {est}{fmt_int(s['small'])} | {pct(s['small'], r['boxes'])} |")
        A(f"| 中目标 | {est}{fmt_int(s['medium'])} | {pct(s['medium'], r['boxes'])} |")
        A(f"| 大目标 | {est}{fmt_int(s['large'])} | {pct(s['large'], r['boxes'])} |")
        A("")

        if not quick and r["resolutions"]:
            A(f"**分辨率分布**（抽样 {r['sampled']} 张）")
            A("")
            A("| 分辨率 | 张数 | 抽样占比 |")
            A("|---|---|---|")
            for res, cnt in sorted(r["resolutions"].items(), key=lambda x: -x[1])[:8]:
                A(f"| {res} | {cnt} | {pct(cnt, r['sampled'])} |")
            if len(r["resolutions"]) > 8:
                A(f"| 其余 {len(r['resolutions']) - 8} 种 | "
                  f"{sum(v for _, v in sorted(r['resolutions'].items(), key=lambda x: -x[1])[8:])} | — |")
            A("")

        A("")

    A("---")
    A("")
    A("## 四、 数据质量与异常清单")
    A("")
    abnormal = [r for r in results if r.get("status", "complete") != "complete"
                or r["missing_labels"] or r["orphan_labels"] or r["malformed_lines"]]
    if abnormal:
        A("以下数据集存在数据完整性问题，纳入训练/评测前需先修复：")
        A("")
        A("| 分区 | 业务数据集 | 状态 | 图片数 | 标注数 | 缺标注 | 孤立标注 | 异常行 | 问题说明 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(abnormal, key=lambda x: (x.get("status", ""), -x["images"])):
            st = r.get("status", "complete")
            desc = {"empty": "空目录骨架，无数据",
                    "labels_only": "仅有标注无图片",
                    "images_only": "仅有图片无标注",
                    "partial": "图文配对不完整"}.get(st, "标注行格式异常")
            A(f"| `{r['partition']}` | <code>{r['dataset_key']}</code> | `{st}` | "
              f"{fmt_int(r['images'])} | {fmt_int(r['labels'])} | {fmt_int(r['missing_labels'])} | "
              f"{fmt_int(r['orphan_labels'])} | {fmt_int(r['malformed_lines'])} | {desc} |")
    else:
        A("本次扫描范围内未发现数据完整性问题。")
    A("")

    # 类别空置统计：names 中定义但全量为 0 框的类别
    dead = []
    for r in results:
        if not r["nc"] or not r["boxes"]:
            continue
        for cid in range(r["nc"]):
            if r["class_counts"].get(str(cid), 0) == 0:
                nm = r["classes"][cid] if cid < len(r["classes"]) else "—"
                dead.append((r["partition"], r["dataset_key"], cid, nm))
    if dead:
        A("### 4.1 类别定义空置（`names` 中声明但全量 0 框）")
        A("")
        A("这些类别在 `*.names` 中声明，但全量数据中没有任何标注框，训练时对应输出通道为死类：")
        A("")
        A("| 分区 | 业务数据集 | 类别 ID | 类别名 |")
        A("|---|---|---|---|")
        for part, key, cid, nm in dead:
            A(f"| `{part}` | <code>{key}</code> | {cid} | <code>{nm}</code> |")
        A("")

    # 类别长尾：占比 < 0.5% 的类别
    tail = []
    for r in results:
        if not r["boxes"]:
            continue
        for cid_s, cnt in r["class_counts"].items():
            if cnt and cnt / r["boxes"] < 0.005:
                cid = int(cid_s)
                nm = r["classes"][cid] if cid < len(r["classes"]) else "—"
                tail.append((r["partition"], r["dataset_key"], cid, nm, cnt, cnt / r["boxes"]))
    if tail:
        A("### 4.2 长尾类别（全量占比 < 0.5%）")
        A("")
        A("| 分区 | 业务数据集 | 类别 ID | 类别名 | 标注框数 | 占比 |")
        A("|---|---|---|---|---|---|")
        for part, key, cid, nm, cnt, ratio in sorted(tail, key=lambda x: x[5]):
            A(f"| `{part}` | <code>{key}</code> | {cid} | <code>{nm}</code> | {fmt_int(cnt)} | {ratio * 100:.2f}% |")
        A("")

    A("---")
    A("")
    A("## 五、 统计口径说明")
    A("")
    A("1. **配对规则**：图片与其同名 `.txt` 按文件名主干（stem）配对；同一目录下多格式同名图按各自 stem 归一。")
    A("2. **纯背景图**：存在标注文件但文件为空（0 行有效框），计为空标注 / 纯背景样本。")
    A("3. **缺标注图片**：图片存在但无同名 `.txt`，此类样本无法直接用于监督训练。")
    A("4. **孤立标注**：`.txt` 存在但无同名图片，多为历史清理残留。")
    A("5. **异常标注行**：列数不等于 5（HBB）或 9（OBB），或类别 ID / 坐标无法解析的行。")
    A("6. **目标尺度**：沿用 [`docs/dataset-standard-report.md`](./dataset-standard-report.md) 的口径，"
      "以归一化宽高乘 `640` 折算像素面积后分档，便于与 5,700 张标准测试集报告横向对照；"
      "该口径对非方形输入（如 `640×1152`）为近似值。")
    A("7. **OBB 外接框**：旋转框取 4 顶点坐标的轴对齐外接矩形计算宽高。")
    A("8. **磁盘占用**：按文件 `st_size` 累加，不含文件系统块对齐开销，故略小于 `du` 结果。")
    A("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过分辨率抽样")
    ap.add_argument("--only", default=None, help="只扫描指定分区，如 Hbb")
    ap.add_argument("--only-biz", default=None, help="只扫描指定业务，如 PersonHelmet")
    ap.add_argument("--resume", action="store_true", help="复用 .cache 中已完成的业务结果")
    ap.add_argument("--md-only", action="store_true", help="仅由缓存重建 Markdown，不扫描")
    ap.add_argument("--report", action="store_true", help="只打印进度汇总，不写文档")
    ap.add_argument("--force", action="store_true", help="忽略缓存，强制重新扫描")
    ap.add_argument("--fast", action="store_true",
                    help=f"标注抽样解析（每集最多 {FAST_SAMPLE} 个）并按比例外推，用于冷盘受限时快速统计")
    args = ap.parse_args()

    if not SRC_ROOT.exists():
        sys.exit(f"数据集根目录不存在: {SRC_ROOT}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def cache_path(part, biz):
        safe = f"{part.replace('/', '_')}__{biz}.json"
        return CACHE_DIR / safe

    def load_cache(part, biz):
        p = cache_path(part, biz)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    if args.md_only:
        results = []
        for part, _, _, _ in PARTITIONS:
            for p in sorted(CACHE_DIR.glob(f"{part.replace('/', '_')}__*.json")):
                try:
                    results.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
        if not results:
            sys.exit("缓存为空，请先运行扫描")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        OUT_MD.write_text(build_markdown(results, args.quick), encoding="utf-8")
        print(f"[write] {OUT_MD} ({OUT_MD.stat().st_size / 1024:.1f} KB), {len(results)} datasets")
        return

    rng = random.Random(SEED)
    results = []
    t_all = time.time()
    total_jobs = 0
    done_jobs = 0

    for part, pdir, ptask, _ in PARTITIONS:
        if args.only and part != args.only:
            continue
        if not pdir.exists():
            print(f"[skip] {part}: 目录不存在", flush=True)
            continue
        biz_dirs = sorted([p for p in pdir.iterdir() if p.is_dir()])
        if args.only_biz:
            biz_dirs = [p for p in biz_dirs if p.name == args.only_biz]
        total_jobs += len(biz_dirs)

    for part, pdir, ptask, _ in PARTITIONS:
        if args.only and part != args.only:
            continue
        if not pdir.exists():
            continue
        biz_dirs = sorted([p for p in pdir.iterdir() if p.is_dir()])
        if args.only_biz:
            biz_dirs = [p for p in biz_dirs if p.name == args.only_biz]
        print(f"[scan] {part}: {len(biz_dirs)} 个业务目录", flush=True)
        for bd in biz_dirs:
            done_jobs += 1
            tag = f"{part}/{bd.name}"
            if args.resume and not args.force:
                cached = load_cache(part, bd.name)
                if cached is not None:
                    results.append(cached)
                    print(f"  [{done_jobs}/{total_jobs}] {tag}: cached "
                          f"({cached['images']} imgs, {cached['boxes']} boxes)", flush=True)
                    continue
            print(f"  [{done_jobs}/{total_jobs}] scanning {tag} ...", flush=True)
            r = scan_business(part, bd.name, bd, ptask, not args.quick, rng, fast=args.fast)
            cache_path(part, bd.name).write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(r)

    if args.report:
        for r in results:
            print(f"{r['partition']}/{r['dataset_key']}: {r['images']} imgs, {r['boxes']} boxes", flush=True)
        return

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {OUT_JSON} ({OUT_JSON.stat().st_size / 1024:.1f} KB)", flush=True)

    OUT_MD.write_text(build_markdown(results, args.quick), encoding="utf-8")
    print(f"[write] {OUT_MD} ({OUT_MD.stat().st_size / 1024:.1f} KB)", flush=True)
    print(f"[done] {len(results)} datasets, {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
