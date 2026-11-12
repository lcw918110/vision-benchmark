#!/usr/bin/env python3
"""全量数据集统计 —— 分片扫描器（应对机械盘 + 训练任务 IO 争抢）。

背景：宿主机为机械盘（ROTA=1），训练任务长期占满 IO，单进程一次性扫描
超大业务（如 PiDaiYuanBan 16.6 万标注 / 约 2300 万行）会被饿死在中途。
本脚本把每个业务按「一级子目录」切成独立分片，逐片扫描并落盘，
任意时刻中断都只丢失当前分片，已完成分片永久保留。

分片结果写入 output/dataset/.shards/<分区>__<业务>/<分片>.json，
全部完成后由 --merge 合并为 output/dataset/.cache/<分区>__<业务>.json，
供 09-dataset-full-inventory.py --md-only 生成文档。

用法：
  python3 scripts/11-dataset-shard-scan.py --status              # 查看分片进度
  python3 scripts/11-dataset-shard-scan.py --biz Hbb/PiDaiYuanBan --next  # 扫下一个未完成分片
  python3 scripts/11-dataset-shard-scan.py --biz Hbb/PiDaiYuanBan --merge # 合并该业务
  python3 scripts/11-dataset-shard-scan.py --all --next          # 推进任意一个待办分片
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

_mod = SourceFileLoader(
    "inv", str(Path(__file__).resolve().parent / "09-dataset-full-inventory.py")
).load_module()

SRC_ROOT = _mod.SRC_ROOT
ROOT = _mod.ROOT
CACHE_DIR = _mod.CACHE_DIR
SHARD_DIR = ROOT / "output/dataset/.shards"
PARTITIONS = _mod.PARTITIONS
IMG_EXT = _mod.IMG_EXT


def shard_root(part, biz):
    return SHARD_DIR / f"{part.replace('/', '_')}__{biz}"


def shard_file(root, shard):
    """分片文件名：去掉首尾空白并替换路径分隔符，避免与目录名不一致。"""
    safe = shard.strip().replace("/", "_").replace(os.sep, "_")
    if not safe:
        safe = "__blank__"
    return root / f"{safe}.json"


def list_shards(biz_dir):
    """按一级子目录切分；根目录直属文件归入 __root__ 分片。"""
    shards = []
    root_files = False
    try:
        for e in os.scandir(biz_dir):
            if e.is_dir(follow_symlinks=False):
                shards.append(e.name)
            else:
                root_files = True
    except OSError:
        pass
    if root_files:
        shards.append("__root__")
    return sorted(shards)


def scan_shard(part, biz, biz_dir, shard, task, do_sample, rng):
    """扫描单个分片（复用主扫描器的统计逻辑）。"""
    if shard == "__root__":
        # 根目录直属文件：构造仅含文件的临时视图
        return _scan_root_files(part, biz, biz_dir, task, do_sample, rng)
    sub = biz_dir / shard
    r = _mod.scan_business(part, f"{biz}/{shard}", sub, task, do_sample, rng)
    r["shard"] = shard
    return r


def _scan_root_files(part, biz, biz_dir, task, do_sample, rng):
    """统计业务根目录下直属的图片/标注文件。"""
    img_stems, lbl_stems, label_paths = set(), set(), []
    total_bytes = 0
    for e in os.scandir(biz_dir):
        try:
            if e.is_dir(follow_symlinks=False):
                continue
            total_bytes += e.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        dot = e.name.rfind(".")
        ext = e.name[dot:].lower() if dot > 0 else ""
        if ext in IMG_EXT:
            img_stems.add(e.name[:dot])
        elif ext == ".txt":
            lbl_stems.add(e.name[:dot])
            label_paths.append(e.path)

    total_boxes = 0
    cls_count, scale = {}, {"small": 0, "medium": 0, "large": 0}
    malformed = empty = 0
    for lp in label_paths:
        b, c, s, bad = _mod.parse_label(lp, task)
        malformed += bad
        if b == 0 and bad == 0:
            empty += 1
        total_boxes += b
        for k, v in c.items():
            cls_count[k] = cls_count.get(k, 0) + v
        for k in scale:
            scale[k] += s[k]

    names, names_file = _mod.read_names(biz_dir)
    return {
        "partition": part, "dataset_key": f"{biz}/__root__", "path": str(biz_dir),
        "task": task, "names_file": names_file, "classes": names, "nc": len(names),
        "images": len(img_stems), "labels": len(lbl_stems), "empty_labels": empty,
        "missing_labels": len(img_stems - lbl_stems), "orphan_labels": len(lbl_stems - img_stems),
        "status": "complete", "malformed_lines": malformed, "boxes": total_boxes,
        "class_counts": {str(k): v for k, v in sorted(cls_count.items())},
        "scale": scale, "mines": 0, "sessions": 0,
        "files": len(img_stems) + len(lbl_stems), "bytes": total_bytes,
        "resolutions": {}, "sampled": 0, "elapsed_s": 0.0, "shard": "__root__",
    }


def merge_shards(part, biz, task):
    """把分片结果合并为单个业务结果。"""
    root = shard_root(part, biz)
    files = sorted(root.glob("*.json")) if root.exists() else []
    if not files:
        return None
    merged = None
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if merged is None:
            merged = {
                "partition": part, "dataset_key": biz, "path": r["path"].rsplit("/", 1)[0],
                "task": task, "names_file": r.get("names_file"), "classes": r.get("classes", []),
                "nc": r.get("nc", 0), "images": 0, "labels": 0, "empty_labels": 0,
                "missing_labels": 0, "orphan_labels": 0, "malformed_lines": 0, "boxes": 0,
                "class_counts": {}, "scale": {"small": 0, "medium": 0, "large": 0},
                "mines": 0, "sessions": 0, "files": 0, "bytes": 0,
                "resolutions": {}, "sampled": 0, "elapsed_s": 0.0,
            }
            merged["path"] = str(Path(r["path"]).parent) if r.get("shard") else r["path"]
        for k in ("images", "labels", "empty_labels", "missing_labels", "orphan_labels",
                  "malformed_lines", "boxes", "mines", "sessions", "files", "bytes", "sampled"):
            merged[k] += r.get(k, 0)
        for k, v in r.get("class_counts", {}).items():
            merged["class_counts"][k] = merged["class_counts"].get(k, 0) + v
        for k in merged["scale"]:
            merged["scale"][k] += r.get("scale", {}).get(k, 0)
        for k, v in r.get("resolutions", {}).items():
            merged["resolutions"][k] = merged["resolutions"].get(k, 0) + v
        merged["elapsed_s"] += r.get("elapsed_s", 0.0)

    # 分片各自配对，跨分片不会互相补足，故完整性按分片结论聚合
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
    return merged


def all_businesses():
    out = []
    for part, pdir, task, _ in PARTITIONS:
        if not pdir.exists():
            continue
        for bd in sorted(p for p in pdir.iterdir() if p.is_dir()):
            out.append((part, bd.name, bd, task))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--biz", default=None, help="分区/业务，如 Hbb/PiDaiYuanBan")
    ap.add_argument("--all", action="store_true", help="遍历全部业务")
    ap.add_argument("--next", action="store_true", help="只扫描一个待办分片")
    ap.add_argument("--merge", action="store_true", help="合并指定业务分片")
    ap.add_argument("--merge-all", action="store_true", help="合并全部已完成业务")
    args = ap.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        total_shard = done_shard = 0
        for part, biz, bd, task in all_businesses():
            shards = list_shards(bd)
            root = shard_root(part, biz)
            done = {f.stem for f in root.glob("*.json")} if root.exists() else set()
            want = {shard_file(root, s).stem for s in shards}
            total_shard += len(want)
            done_shard += len(done & want)
            merged = (CACHE_DIR / f"{part.replace('/', '_')}__{biz}.json").exists()
            flag = "✅" if merged else ("🔶" if done else "⬜")
            print(f"{flag} {part}/{biz}: {len(done & want)}/{len(want)} 分片")
        print(f"\n分片总计: {done_shard}/{total_shard}")
        return

    if args.merge:
        part, biz = args.biz.split("/", 1)
        task = next(t for p, _, t, _ in PARTITIONS if p == part)
        m = merge_shards(part, biz, task)
        if not m:
            sys.exit(f"无分片结果: {args.biz}")
        out = CACHE_DIR / f"{part.replace('/', '_')}__{biz}.json"
        out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[merge] {args.biz}: {m['images']} imgs, {m['boxes']} boxes -> {out.name}")
        return

    if args.merge_all:
        n = 0
        for part, biz, bd, task in all_businesses():
            root = shard_root(part, biz)
            if not root.exists():
                continue
            want = {shard_file(root, s).stem for s in list_shards(bd)}
            done = {f.stem for f in root.glob("*.json")}
            if want and want <= done:
                m = merge_shards(part, biz, task)
                if m:
                    out = CACHE_DIR / f"{part.replace('/', '_')}__{biz}.json"
                    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[merge] {part}/{biz}: {m['images']} imgs, {m['boxes']} boxes")
                    n += 1
        print(f"[merge-all] {n} datasets")
        return

    targets = []
    if args.biz:
        part, biz = args.biz.split("/", 1)
        task = next(t for p, _, t, _ in PARTITIONS if p == part)
        targets = [(part, biz, SRC_ROOT / part / biz, task)]
    elif args.all:
        targets = all_businesses()
    else:
        sys.exit("需指定 --biz 或 --all")

    rng = __import__("random").Random(_mod.SEED)
    for part, biz, bd, task in targets:
        if not bd.exists():
            continue
        root = shard_root(part, biz)
        root.mkdir(parents=True, exist_ok=True)
        shards = list_shards(bd)
        done = {f.stem for f in root.glob("*.json")}
        pending = [s for s in shards if shard_file(root, s).stem not in done]
        if not pending:
            print(f"[skip] {part}/{biz}: 全部分片已完成 ({len(shards)})")
            continue
        for shard in pending:
            print(f"[scan] {part}/{biz}/{shard} ...", flush=True)
            try:
                r = scan_shard(part, biz, bd, shard, task, False, rng)
            except Exception as ex:
                print(f"  !! {shard} 失败: {ex}", flush=True)
                continue
            shard_file(root, shard).write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [ok] {shard}: {r['images']} imgs, {r['boxes']} boxes", flush=True)
            if args.next:
                return


if __name__ == "__main__":
    main()
