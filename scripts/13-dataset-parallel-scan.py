#!/usr/bin/env python3
"""全量数据集统计 —— 并行分片扫描器。

场景：宿主机机械盘 + 训练任务争抢 IO，单进程串行扫描大量时间耗在等待 IO 上。
本脚本用进程池并行扫描同层分片，把 IO 等待重叠起来（CPU 密集度低，48 核可轻松承载）。

分片粒度：业务根目录下一级子目录；若单分片文件数超过 BIG_THRESHOLD，
则自动下钻一层再切分，避免单个任务过大被长期占用。

用法：
  python3 scripts/13-dataset-parallel-scan.py --biz Hbb/PiDaiYuanBan --workers 6
  python3 scripts/13-dataset-parallel-scan.py --biz Hbb/TanShuiHouShangGan --workers 6
  python3 scripts/13-dataset-parallel-scan.py --merge Hbb/PiDaiYuanBan
  python3 scripts/13-dataset-parallel-scan.py --finish   # 合并全部并重建文档
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.machinery import SourceFileLoader
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent
_m = SourceFileLoader("inv", str(SCRIPT / "09-dataset-full-inventory.py")).load_module()

CACHE_DIR = _m.CACHE_DIR
PARTS_ROOT = _m.ROOT / "output/dataset/.parts"
BIG_THRESHOLD = 40000  # 单分片文件数超过此值时下钻一层


def count_files(d):
    n = 0
    stack = [str(d)]
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
                else:
                    n += 1
            except OSError:
                continue
    return n


def build_parts(biz_dir):
    """生成分片清单 [(part_name, part_path)]，过大分片自动下钻一层。"""
    parts = []
    for e in sorted(os.scandir(biz_dir), key=lambda x: x.name):
        if not e.is_dir(follow_symlinks=False):
            continue
        subs = sorted([s for s in os.scandir(e.path) if s.is_dir(follow_symlinks=False)],
                      key=lambda x: x.name)
        if not subs:
            parts.append((e.name, Path(e.path)))
            continue
        total = 0
        for s in subs:
            total += count_files(s.path)
        if total <= BIG_THRESHOLD:
            parts.append((e.name, Path(e.path)))
        else:
            for s in subs:
                parts.append((f"{e.name}/{s.name}", Path(s.path)))
    return parts


def scan_one(args):
    biz, part_name, part_path, task = args
    f = PARTS_ROOT / f"{biz.replace('/', '_')}__{(part_name.replace('/', '_').strip() or '__blank__')}.json"
    if f.exists():
        return part_name, "cached", None
    t = time.time()
    try:
        r = _m.scan_business(biz.split("/")[0], part_name, Path(part_path), task,
                             False, random.Random(42), fast=False)
    except Exception as ex:
        return part_name, f"error: {ex}", None
    f.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return part_name, "ok", (r["images"], r["boxes"], time.time() - t)


def merge(biz, task):
    key = biz.replace("/", "_")
    files = sorted(PARTS_ROOT.glob(f"{key}__*.json"))
    if not files:
        print(f"[merge] {biz}: 无分片结果")
        return None
    merged = {
        "partition": biz.split("/")[0], "dataset_key": biz.split("/", 1)[1],
        "path": str(_m.SRC_ROOT / biz), "task": task,
        "names_file": None, "classes": [], "nc": 0,
        "images": 0, "labels": 0, "empty_labels": 0, "missing_labels": 0,
        "orphan_labels": 0, "malformed_lines": 0, "boxes": 0,
        "class_counts": {}, "scale": {"small": 0, "medium": 0, "large": 0},
        "mines": 0, "sessions": 0, "files": 0, "bytes": 0,
        "resolutions": {}, "sampled": 0, "elapsed_s": 0.0, "estimated": False,
    }
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not merged["names_file"] and r.get("names_file"):
            merged["names_file"] = r["names_file"]
            merged["classes"] = r.get("classes", [])
            merged["nc"] = r.get("nc", 0)
        for k in ("images", "labels", "empty_labels", "missing_labels", "orphan_labels",
                  "malformed_lines", "boxes", "mines", "sessions", "files", "bytes", "sampled"):
            merged[k] += r.get(k, 0)
        for k, v in r.get("class_counts", {}).items():
            merged["class_counts"][k] = merged["class_counts"].get(k, 0) + v
        for k in merged["scale"]:
            merged["scale"][k] += r.get("scale", {}).get(k, 0)
        merged["elapsed_s"] += r.get("elapsed_s", 0.0)

    if merged["images"] == 0 and merged["labels"] == 0:
        merged["status"] = "empty"
    elif merged["images"] == 0:
        merged["status"] = "labels_only"
    elif merged["labels"] == 0:
        merged["status"] = "images_only"
    elif merged["missing_labels"] or merged["orphan_labels"]:
        merged["status"] = "partial"
    else:
        merged["status"] = "complete"
    merged["class_counts"] = {k: v for k, v in sorted(merged["class_counts"].items(), key=lambda x: int(x[0]))}
    merged["elapsed_s"] = round(merged["elapsed_s"], 1)

    out = CACHE_DIR / f"{key}.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] {biz}: {merged['images']} imgs, {merged['boxes']} boxes, "
          f"{merged['mines']} mines, status={merged['status']} ({len(files)} parts)")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biz", default=None, help="分区/业务，如 Hbb/PiDaiYuanBan")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--merge", default=None, help="只合并指定业务")
    ap.add_argument("--finish", action="store_true", help="合并全部并重建文档")
    ap.add_argument("--list", action="store_true", help="只列出分片")
    args = ap.parse_args()

    PARTS_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.finish:
        for f in sorted(CACHE_DIR.glob("*.json")):
            pass
        print("[finish] 重建文档")
        subprocess.run([sys.executable, str(SCRIPT / "09-dataset-full-inventory.py"),
                        "--md-only", "--quick"], check=False)
        return

    if args.merge:
        part = args.merge.split("/")[0]
        task = next(t for p, _, t, _ in _m.PARTITIONS if p == part)
        merge(args.merge, task)
        return

    if not args.biz:
        sys.exit("需指定 --biz 或 --merge/--finish")

    part, name = args.biz.split("/", 1)
    task = next(t for p, _, t, _ in _m.PARTITIONS if p == part)
    biz_dir = _m.SRC_ROOT / args.biz
    if not biz_dir.exists():
        sys.exit(f"目录不存在: {biz_dir}")

    parts = build_parts(biz_dir)
    if args.list:
        for n, p in parts:
            print(f"  {n}  ({count_files(p)} files)")
        print(f"共 {len(parts)} 个分片")
        return

    todo = [(args.biz, n, str(p), task) for n, p in parts]
    print(f"[plan] {args.biz}: {len(parts)} 个分片, {args.workers} 并发", flush=True)

    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(scan_one, item): item[1] for item in todo}
        for fut in as_completed(futures):
            done += 1
            try:
                pname, status, info = fut.result()
            except Exception as ex2:
                print(f"[{done}/{len(todo)}] !! {futures[fut]}: {ex2}", flush=True)
                continue
            if status == "ok" and info:
                imgs, boxes, el = info
                print(f"[{done}/{len(todo)}] {pname}: {imgs} imgs, {boxes} boxes ({el:.1f}s)", flush=True)
            elif status == "cached":
                print(f"[{done}/{len(todo)}] {pname}: cached", flush=True)
            else:
                print(f"[{done}/{len(todo)}] {pname}: {status}", flush=True)

    print(f"[scan] 完成 {done} 个分片，用时 {time.time() - t0:.1f}s", flush=True)
    merge(args.biz, task)


if __name__ == "__main__":
    main()
