#!/usr/bin/env python3
"""11 个基准数据集：全量图片解码校验 + 标注合法性校验 + 真实分辨率统计。

与 14 号脚本的分工：
  14 号只做「文件级」统计（按相对路径配对、框数、类别/尺度分布），不打开图片。
  本脚本补做「内容级」校验：
    1. 逐张完整解码（Image.load）验证图片未损坏，并统计真实分辨率与色彩模式；
    2. 逐行校验标注：列数、数值有限性、类别 ID 范围、坐标是否落在 [0,1]、宽高是否为正；
    3. 记录零字节文件、缺标注、孤立标注、同目录同名多图。

断点续扫：以「业务根目录 + 一级子目录」为分片，分片结果原子落盘，
  已存在且版本一致的分片直接复用；源数据变更需 --force 重扫。

用法：
  python3 scripts/15-benchmark-quality-audit.py --workers 6
  python3 scripts/15-benchmark-quality-audit.py --dataset ChuanGanQi --workers 4
  python3 scripts/15-benchmark-quality-audit.py --merge-only
"""
import argparse
import collections
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'configs'))
from registry import DATASETS

BASE = Path('/home/jykj/Dataset')
OUT = ROOT / 'output/dataset/benchmark-quality-v1'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
VERSION = 1
COORD_TOL = 1e-3
MAX_ERROR_SAMPLES = 50
BIG_SHARD_IMAGES = 20000


def count_images(directory):
    """快速统计目录下图片数量（仅目录项，不打开文件）。"""
    n = 0
    stack = [Path(directory)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif Path(entry.name).suffix.lower() in IMG_EXT:
                            n += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return n


def read_nc(root):
    """从业务根目录的 *.names 读取类别数；读不到返回 None（不做类别范围校验）。"""
    for path in sorted(root.glob('*.names')):
        try:
            lines = [ln.strip() for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines()]
        except OSError:
            return None, None
        names = [ln for ln in lines if ln]
        if names:
            return len(names), str(path)
    return None, None


def check_label(path, task, nc):
    """返回 (框数, 有效框数, 空文件, 异常行数, 异常分类, 类别计数)。"""
    stats = collections.Counter()
    cls = collections.Counter()
    boxes = valid = 0
    need = 9 if task == 'obb' else 5
    try:
        with path.open(encoding='utf-8', errors='strict') as fh:
            for line in fh:
                fields = line.split()
                if not fields:
                    continue
                boxes += 1
                if len(fields) != need:
                    stats['bad_columns'] += 1
                    continue
                try:
                    cid_raw = float(fields[0])
                    coords = [float(x) for x in fields[1:]]
                except ValueError:
                    stats['bad_number'] += 1
                    continue
                if not math.isfinite(cid_raw) or not cid_raw.is_integer():
                    stats['bad_class_id'] += 1
                    continue
                cid = int(cid_raw)
                if nc is not None and not (0 <= cid < nc):
                    stats['class_out_of_range'] += 1
                    continue
                if not all(math.isfinite(x) for x in coords):
                    stats['bad_number'] += 1
                    continue
                if any(x < -COORD_TOL or x > 1 + COORD_TOL for x in coords):
                    stats['coord_out_of_range'] += 1
                    continue
                if task == 'obb':
                    xs, ys = coords[0::2], coords[1::2]
                    area = abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))) / 2
                    if area <= 0:
                        stats['degenerate_polygon'] += 1
                        continue
                else:
                    if coords[2] <= 0 or coords[3] <= 0:
                        stats['nonpositive_size'] += 1
                        continue
                valid += 1
                cls[cid] += 1
    except (OSError, UnicodeError):
        stats['unreadable_label'] += 1
        return boxes, valid, False, stats, cls
    return boxes, valid, boxes == 0, stats, cls


def scan_shard(root, part, root_only, task, nc):
    """扫描一个分片：图片解码 + 标注校验。

    先收集全部文件并按 inode 排序再读取，使机械盘磁头接近顺序访问
    （实测比目录顺序读取快约 50%）。
    """
    images = collections.Counter()
    labels = set()
    resolutions = collections.Counter()
    modes = collections.Counter()
    formats = collections.Counter()
    label_stats = collections.Counter()
    class_counts = collections.Counter()
    errors = []
    files = 0
    size = 0
    boxes = valid_boxes = empty_labels = 0

    collected = []
    stack = [part]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError as exc:
            errors.append(f'SCANDIR {current}: {exc}')
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not root_only:
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(f'STAT {entry.path}: {exc}')
                continue
            collected.append((st.st_ino, entry.path, st.st_size))

    collected.sort()
    for _, raw_path, fsize in collected:
        files += 1
        size += fsize
        path = Path(raw_path)
        key = str(path.relative_to(part).with_suffix(''))
        ext = path.suffix.lower()
        if ext in IMG_EXT:
            images[key] += 1
            if fsize == 0:
                label_stats['zero_byte_image'] += 1
                if len(errors) < MAX_ERROR_SAMPLES:
                    errors.append(f'ZERO_BYTE {path}')
                continue
            try:
                from PIL import Image
                with Image.open(path) as im:
                    im.load()
                    resolutions[f'{im.width}x{im.height}'] += 1
                    modes[im.mode] += 1
                    formats[im.format or 'unknown'] += 1
            except Exception as exc:
                label_stats['decode_error'] += 1
                if len(errors) < MAX_ERROR_SAMPLES:
                    errors.append(f'DECODE {path}: {type(exc).__name__}: {exc}')
        elif ext == '.txt':
            labels.add(key)
            b, v, empty, st, cls = check_label(path, task, nc)
            boxes += b
            valid_boxes += v
            empty_labels += int(empty)
            label_stats.update(st)
            class_counts.update(cls)

    missing = sum(v for k, v in images.items() if k not in labels)
    orphan = len(labels - images.keys())
    duplicates = sum(v - 1 for v in images.values())
    return {
        'part': '__ROOT_FILES__' if root_only else part.name,
        'images': sum(images.values()),
        'labels': len(labels),
        'missing_labels': missing,
        'orphan_labels': orphan,
        'duplicate_image_stems': duplicates,
        'boxes': boxes,
        'valid_boxes': valid_boxes,
        'empty_labels': empty_labels,
        'label_issues': dict(label_stats),
        'class_counts': {str(k): v for k, v in sorted(class_counts.items())},
        'resolutions': dict(resolutions),
        'modes': dict(modes),
        'formats': dict(formats),
        'files': files,
        'bytes': size,
        'errors': errors,
    }


def shard_job(args):
    root, part, root_only, task, nc, dest, force = args
    dest = Path(dest)
    if dest.exists() and not force:
        try:
            cached = json.loads(dest.read_text(encoding='utf-8'))
            if cached.get('version') == VERSION:
                return cached['part'], 'cached', cached
        except (OSError, ValueError):
            pass
    result = scan_shard(Path(root), Path(part), root_only, task, nc)
    result['version'] = VERSION
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, dest)
    return result['part'], 'ok', result


def merge(key, target, task, root):
    shard_files = sorted(target.glob('*.json'))
    shards = []
    for f in shard_files:
        if f.name == 'summary.json':
            continue
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if d.get('version') == VERSION:
            shards.append(d)
    if not shards:
        return None
    sums = ('images', 'labels', 'missing_labels', 'orphan_labels', 'duplicate_image_stems',
            'boxes', 'valid_boxes', 'empty_labels', 'files', 'bytes')
    totals = {k: sum(s[k] for s in shards) for k in sums}
    for field in ('label_issues', 'class_counts', 'resolutions', 'modes', 'formats'):
        counter = collections.Counter()
        for s in shards:
            counter.update(s.get(field, {}))
        totals[field] = dict(counter)
    totals.update({
        'dataset_key': key,
        'task': task,
        'raw_root': str(root),
        'shards': len(shards),
        'errors': [e for s in shards for e in s.get('errors', [])][:200],
        'version': VERSION,
    })
    tmp = target / 'summary.tmp'
    tmp.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, target / 'summary.json')
    return totals


def split_big(root, part, name, depth=0):
    """递归下钻超大分片，返回 [(part, root_only, name)]。"""
    if depth >= 6:
        return [(part, False, name)]
    subs = []
    try:
        with os.scandir(part) as it:
            subs = sorted((Path(e.path) for e in it if e.is_dir(follow_symlinks=False)), key=str)
    except OSError:
        return [(part, False, name)]
    if not subs:
        return [(part, False, name)]
    total = sum(count_images(sub) for sub in subs)
    if total <= BIG_SHARD_IMAGES:
        return [(part, False, name)]
    out = []
    for sub in subs:
        out.extend(split_big(root, sub, f'{name}/{sub.name}', depth + 1))
    return out


def plan(selected, force):
    jobs = []
    for ds in DATASETS:
        key = ds['dataset_key']
        if selected and key not in selected:
            continue
        raw = 'Obb/TS-Front_V2' if ds['task'] == 'obb' else 'Hbb/' + key
        root = BASE / raw
        target = OUT / key
        target.mkdir(parents=True, exist_ok=True)
        nc, _ = read_nc(root)
        try:
            with os.scandir(root) as it:
                parts = sorted((Path(e.path) for e in it if e.is_dir(follow_symlinks=False)), key=str)
        except OSError as exc:
            print(f'ERROR {key}: {exc}', flush=True)
            continue
        entries = [(root, True)] + [(p, False) for p in parts]
        expanded = []
        for part, root_only in entries:
            if root_only:
                expanded.append((part, True, '__ROOT_FILES__'))
                continue
            # 超大分片逐级下钻，直到每个分片不超过阈值或已无可下钻目录
            expanded.extend(split_big(root, part, part.name))
        for part, root_only, name in expanded:
            dest = target / (name.replace('/', '_').replace(os.sep, '_') + '.json')
            jobs.append((str(root), str(part), root_only, ds['task'], nc, str(dest), force))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dataset', action='append', default=None, help='只处理指定 dataset_key，可重复')
    ap.add_argument('--force', action='store_true', help='忽略已缓存分片')
    ap.add_argument('--merge-only', action='store_true', help='只合并已有分片')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        jobs = plan(set(args.dataset) if args.dataset else None, args.force)
        print(f'分片任务 {len(jobs)} 个，并发 {args.workers}', flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(shard_job, j): j for j in jobs}
            for fut in as_completed(futures):
                name, state, result = fut.result()
                done += 1
                print(f'[{done}/{len(jobs)}] {result.get("part", name)} {state} '
                      f'imgs={result["images"]} boxes={result["boxes"]} '
                      f'decode_err={result["label_issues"].get("decode_error", 0)}', flush=True)

    print('--- 汇总 ---', flush=True)
    missing_shards = 0
    for ds in DATASETS:
        key = ds['dataset_key']
        if args.dataset and key not in set(args.dataset):
            continue
        raw = 'Obb/TS-Front_V2' if ds['task'] == 'obb' else 'Hbb/' + key
        target = OUT / key
        totals = merge(key, target, ds['task'], BASE / raw)
        # 缺口检查：计划中的分片是否都已产出，避免汇总静默缺数据
        planned = {Path(j[5]).stem for j in plan({key}, False)}
        actual = {f.stem for f in target.glob('*.json') if f.name != 'summary.json'}
        gap = sorted(planned - actual)
        missing_shards += len(gap)
        if totals:
            print(f'{key}: imgs={totals["images"]} boxes={totals["boxes"]}/{totals["valid_boxes"]} '
                  f'decode_err={totals["label_issues"].get("decode_error", 0)} '
                  f'issues={totals["label_issues"]}', flush=True)
        print(f'{key}: 分片 {len(actual)}/{len(planned)}' +
              (f' 缺 {len(gap)} 个: {gap[:5]}' if gap else ' 齐全'), flush=True)
    print(f'GAP_SHARDS={missing_shards}', flush=True)


if __name__ == '__main__':
    main()
