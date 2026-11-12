#!/usr/bin/env python3
"""生成质检产物校验清单，便于长期验证结果未被篡改或丢失。

输出 output/dataset/benchmark-quality-v1/CHECKSUMS.txt：
  每个 summary.json / report.json / issues.csv 的 sha256 与大小。

用法：
  python3 scripts/20-benchmark-quality-checksums.py           # 生成
  python3 scripts/20-benchmark-quality-checksums.py --verify  # 校验
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/dataset/benchmark-quality-v1'
MANIFEST = OUT / 'CHECKSUMS.txt'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def targets():
    files = sorted(OUT.glob('*/summary.json'))
    files.append(OUT / 'report.json')
    files.append(OUT / 'issues.csv')
    return [f for f in files if f.exists()]


def build():
    lines = ['# 11 个基准数据集全量质检产物校验清单',
             '# 格式: sha256  字节数  相对路径',
             '# 校验: python3 scripts/20-benchmark-quality-checksums.py --verify',
             '']
    total = 0
    for f in targets():
        size = f.stat().st_size
        total += size
        lines.append(f'{sha256(f)}  {size:>9}  {f.relative_to(OUT)}')
    lines.append('')
    lines.append(f'# 文件数 {len(targets())}，合计 {total:,} 字节')
    return '\n'.join(lines) + '\n'


def verify():
    if not MANIFEST.exists():
        print(f'缺少清单 {MANIFEST}，先运行不带 --verify 的命令生成', file=sys.stderr)
        return 1
    ok = bad = missing = 0
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        digest, size, rel = parts
        path = OUT / rel
        if not path.exists():
            print(f'缺失  {rel}')
            missing += 1
            continue
        actual = sha256(path)
        if actual == digest and str(path.stat().st_size) == size:
            ok += 1
        else:
            print(f'不一致 {rel}  (期望 {digest[:12]} / 实际 {actual[:12]})')
            bad += 1
    print(f'校验结果：一致 {ok}，不一致 {bad}，缺失 {missing}')
    return 0 if bad == 0 and missing == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    if args.verify:
        return verify()
    text = build()
    MANIFEST.write_text(text, encoding='utf-8')
    print(f'已写入 {MANIFEST.relative_to(ROOT)}')
    print(f'共 {len(targets())} 个文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())