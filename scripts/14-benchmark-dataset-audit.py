#!/usr/bin/env python3
"""11 个基准数据集：按相对路径配对、逐标注解析、按一级目录断点续扫。"""
import collections
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'configs'))
from registry import DATASETS

BASE = Path('/home/jykj/Dataset')
OUT = ROOT / 'output/dataset/benchmark-audit-v1'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def scan(part, task, root_only=False):
    images, labels = collections.Counter(), set()
    counts, scale = collections.Counter(), collections.Counter()
    errors, boxes, empty, bad, files, size = [], 0, 0, 0, 0, 0
    stack = [part]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError as exc:
            errors.append(f'{current}: {exc}')
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not root_only:
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                files += 1
                size += entry.stat(follow_symlinks=False).st_size
                path = Path(entry.path)
                key = str(path.relative_to(part).with_suffix(''))
                ext = path.suffix.lower()
                if ext in EXTS:
                    images[key] += 1
                elif ext == '.txt':
                    labels.add(key)
                    n = 0
                    with path.open(encoding='utf-8', errors='strict') as f:
                        for line in f:
                            fields = line.split()
                            if not fields:
                                continue
                            try:
                                if len(fields) != (9 if task == 'obb' else 5):
                                    raise ValueError('column count')
                                cid_float = float(fields[0])
                                coords = [float(x) for x in fields[1:]]
                                if not math.isfinite(cid_float) or not cid_float.is_integer() or not all(map(math.isfinite, coords)):
                                    raise ValueError('invalid value')
                                cid = int(cid_float)
                                if task == 'obb':
                                    pts = list(zip(coords[::2], coords[1::2]))
                                    area = abs(sum(pts[i][0]*pts[(i+1)%4][1] - pts[(i+1)%4][0]*pts[i][1] for i in range(4))) / 2 * 640**2
                                else:
                                    area = max(0, coords[2]) * max(0, coords[3]) * 640**2
                                counts[cid] += 1
                                scale['small' if area < 32**2 else 'medium' if area < 96**2 else 'large'] += 1
                                boxes += 1
                                n += 1
                            except (ValueError, OverflowError):
                                bad += 1
                    if not n:
                        empty += 1
            except (OSError, UnicodeError) as exc:
                errors.append(f'{entry.path}: {exc}')
    return {'images': sum(images.values()), 'labels': len(labels), 'missing_labels': sum(v for k, v in images.items() if k not in labels),
            'orphan_labels': len(labels - images.keys()), 'duplicate_image_stems': sum(v - 1 for v in images.values()),
            'boxes': boxes, 'empty_or_no_valid_labels': empty, 'malformed_lines': bad,
            'class_counts': dict(counts), 'scale': dict(scale), 'files': files, 'bytes': size, 'errors': errors}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    for ds in DATASETS:
        key = ds['dataset_key']
        raw = 'Obb/TS-Front_V2' if ds['task'] == 'obb' else 'Hbb/' + key
        root = BASE / raw
        target = OUT / key
        target.mkdir(exist_ok=True)
        try:
            with os.scandir(root) as it:
                parts = sorted((Path(e.path) for e in it if e.is_dir(follow_symlinks=False)), key=str)
        except OSError as exc:
            print(f'ERROR {key}: {exc}', flush=True)
            failed = True
            continue
        results = []
        parts = [(root, True)] + [(part, False) for part in parts]
        for part, root_only in parts:
            part_name = '__ROOT_FILES__' if root_only else part.name
            ident = hashlib.sha256(os.fsencode(part_name)).hexdigest()
            dest = target / (ident + '.json')
            if dest.exists():
                result = json.loads(dest.read_text(encoding='utf-8'))
                if result.get('errors'):
                    result = scan(part, ds['task'], root_only)
            else:
                result = scan(part, ds['task'], root_only)
            result['part'] = part_name
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, dest)
            results.append(result)
            print(f'{key} {len(results)}/{len(parts)} {part.name}: {result["images"]} images {len(result["errors"])} errors', flush=True)
        totals = {k: sum(r[k] for r in results) for k in ('images', 'labels', 'missing_labels', 'orphan_labels', 'duplicate_image_stems', 'boxes', 'empty_or_no_valid_labels', 'malformed_lines', 'files', 'bytes')}
        for k in ('class_counts', 'scale'):
            counter = collections.Counter()
            for r in results:
                counter.update(r[k])
            totals[k] = dict(counter)
        totals.update({'dataset_key': key, 'raw_root': str(root), 'task': ds['task'], 'parts': len(parts), 'errors': [e for r in results for e in r['errors']], 'note': '包含业务根目录直属文件及各一级子目录'})
        failed |= bool(totals['errors']) or not parts
        tmp = target / 'summary.tmp'
        tmp.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, target / 'summary.json')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
