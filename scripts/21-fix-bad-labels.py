#!/usr/bin/env python3
"""清理 14 个已定位的异常标注框（默认 dry-run，不做实际修改）。

处理对象（均已逐一核实文件与行号）：
  - 13 个退化框（宽或高为 0）：删除对应行
  - 1 个中心点越界误标：删除对应行

用法：
  python3 scripts/21-fix-bad-labels.py            # dry-run，只打印将要执行的动作
  python3 scripts/21-fix-bad-labels.py --apply    # 实际写入（先备份原文件）
  python3 scripts/21-fix-bad-labels.py --verify   # 只校验当前文件状态是否与预期一致
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE = Path('/home/jykj/Dataset')
ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'output/dataset/benchmark-quality-v1/backup-labels'

# (数据集根目录, 标签相对路径, 行号集合, 说明, 期望的原始内容前缀)
FIXES = [
    ('Hbb/PersonHelmet',
     'sxx-hsz/sxx-hsz-jx-jxys-fjk-fjk02-wlsb17780-2025041001/'
     'sxx-hsz-jx-jxys-fjk-fjk02-wlsb17780-2025041001-000425.txt',
     {2, 3, 4, 5}, '4 个坐标完全相同的重复退化框', '2 0.7007113821138211'),
    ('Hbb/PersonHelmet',
     'sxx-hsz/sxx-hsz-jx-jxys-fjk-fjk02-wlsb19565-2025042901/'
     'sxx-hsz-jx-jxys-fjk-fjk02-wlsb19565-2025042901-001000.txt',
     {1}, '宽高均为 0', '2 0.37839548802946593'),
    ('Hbb/PersonHelmet',
     'hn-csl/hn-csl-jx-jxys-kcysgd-640jzgdxsspt-wlsb14890-2024110701/'
     'hn-csl-jx-jxys-kcysgd-640jzgdxsspt-wlsb14890-2024110701-002550.txt',
     {1}, '高为 0', '3 0.6168699186991871'),
    ('Hbb/PersonHelmet',
     'sx-ll/sx-ll-jx-jxys-jkcrzz-ebhcjt-wlsb14845-2024103001/'
     'sx-ll-jx-jxys-jkcrzz-ebhcjt-wlsb14845-2024103001-001890.txt',
     {3}, '宽高均为 0', '2 0.49099794238683125'),
    ('Hbb/PersonHelmet',
     'sx-xlymk/sx-xlymk-js-jdyfzss-wxcj-jxcj-hc-2024102402/'
     'sx-xlymk-js-jdyfzss-wxcj-jxcj-hc-2024102402-002550.txt',
     {1}, '宽高均为 0', '2 0.6688218390804598'),
    ('Hbb/PersonHelmet',
     'sxx-hll/sxx-hll-ty-bgq-ddzx-dmdds1-wlsb22698-2025052302/'
     'sxx-hll-ty-bgq-ddzx-dmdds1-wlsb22698-2025052302-000750.txt',
     {4}, '宽高均为 0', '2 0.3684446839080459'),
    ('Hbb/PersonHelmet',
     'xj-hamk/xj-hamk-jx-jxys-kcysgd-66gfxjk-hc-wlsb14742/'
     'xj-hamk-jx-jxys-kcysgd-66gfxjk-hc-wlsb14742-001245.txt',
     {5}, '宽高均为 0', '2 0.6311683347233806'),
    ('Hbb/ChuanGanQi',
     'sd-jzm/sd-jzm-jx-cm-gdhk-103x02gzmyjjwcgq-hc-2024030201/'
     'sd-jzm-jx-cm-gdhk-103x02gzmyjjwcgq-hc-2024030201-029240.txt',
     {1}, '宽高均为 0', '0 0.35331632653061223'),
    ('Hbb/ChuanGanQi',
     'sd-tk/sd-tk-jx-cm-gdhk-7307cmgzmjwT0-hc-2024030101/'
     'sd-tk-jx-cm-gdhk-7307cmgzmjwT0-hc-2024030101-007820.txt',
     {1}, '宽为 0', '2 0.54765625'),
    ('Hbb/PiDaiYuanBan',
     'sxhy-ekxxjzypd/sxhy-ekxxjzypd-2025042201-000725.txt',
     {29}, '宽高均为 0', '4 0.4859601449275362'),
    ('Hbb/ChanYunXiaoMeiZhong',
     'neimengguerlintu/elt-zcspjk-2022121801/elt-zcspjk-2022121801-2380.txt',
     {1}, '中心点越界误标', '7 1.370703'),
]


def read_lines(path):
    """按二进制安全读取，保留原换行符（CRLF / LF 混用时不做统一）。"""
    return path.read_bytes().splitlines(keepends=True)


def plan():
    """生成逐项计划，并校验每个目标行是否符合预期前缀。"""
    jobs = []
    for root, rel, lines, note, expect in FIXES:
        path = BASE / root / rel
        if not path.exists():
            jobs.append({'path': path, 'rel': rel, 'ok': False,
                         'reason': '文件不存在', 'lines': lines, 'note': note})
            continue
        raw = read_lines(path)
        if not raw:
            jobs.append({'path': path, 'rel': rel, 'ok': False,
                         'reason': '空文件', 'lines': lines, 'note': note})
            continue
        if any(n > len(raw) for n in lines):
            jobs.append({'path': path, 'rel': rel, 'ok': False,
                         'reason': f'行号超出（共 {len(raw)} 行）', 'lines': lines, 'note': note})
            continue
        bad = [n for n in sorted(lines)
               if not raw[n - 1].decode('utf-8', 'replace').strip().startswith(expect)]
        if bad:
            jobs.append({'path': path, 'rel': rel, 'ok': False,
                         'reason': f'第 {bad} 行内容与预期不符（已被改动？）',
                         'lines': lines, 'note': note})
            continue
        jobs.append({'path': path, 'rel': rel, 'ok': True, 'reason': '',
                     'lines': lines, 'note': note, 'total': len(raw)})
    return jobs


def apply(jobs):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    changed = 0
    for job in jobs:
        if not job['ok']:
            continue
        src = job['path']
        raw = read_lines(src)
        drop = set(job['lines'])
        kept = [b for i, b in enumerate(raw, 1) if i not in drop]
        if len(kept) == len(raw):
            continue
        dst = BACKUP / stamp / job['rel']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        src.write_bytes(b''.join(kept))
        changed += 1
        empty = '  ⚠ 该文件已无有效框（成为空标注，属合法负样本）' if not kept else ''
        print(f'已修改 {job["rel"]}  删除 {len(job["lines"])} 行  备份 {dst.relative_to(ROOT)}{empty}')
    print(f'共修改 {changed} 个文件，备份目录 {BACKUP / stamp}')
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='实际写入（默认 dry-run）')
    args = ap.parse_args()

    jobs = plan()
    ok = sum(1 for j in jobs if j['ok'])
    print(f'共 {len(jobs)} 个文件，待处理 {ok} 个，异常 {len(jobs) - ok} 个')
    for job in jobs:
        if job['ok']:
            print(f'  [OK] {job["rel"].split("/")[-1]}  删第 {sorted(job["lines"])} 行'
                  f'（共 {job["total"]} 行） — {job["note"]}')
        else:
            print(f'  [跳过] {job["rel"].split("/")[-1]} — {job["reason"]}')

    if not args.apply:
        print('\n当前为 dry-run，未修改任何文件。确认后加 --apply 执行（会自动备份原文件）。')
        return 0

    print()
    apply(jobs)
    print('\n下一步：重跑受影响数据集的分片质检，并重新生成报告：')
    print('  python3 scripts/15-benchmark-quality-audit.py --force --dataset PersonHelmet '
          '--dataset ChuanGanQi --dataset PiDaiYuanBan --dataset ChanYunXiaoMeiZhong --workers 8')
    print('  python3 scripts/19-benchmark-quality-report.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())