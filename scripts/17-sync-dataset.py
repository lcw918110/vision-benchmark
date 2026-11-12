#!/usr/bin/env python3
"""把 /home/jykj/Dataset/{Hbb,Obb,Seg} 传到 192.168.198.102 的相同路径。

为什么用 tar 流而不是 rsync：
  * 源数据约 351 万个小文件、758 GiB。rsync 首次要逐文件建索引并做多轮
    size/mtime 比对，实测几十秒只发出几 KB，绝大多数时间耗在本地寻道。
  * 目标机是机械盘，顺序写约 177 MB/s；实测 `tar|ssh|tar` 流可以跑到
    千兆网口线速（约 116 MB/s），比逐文件传输快一个数量级。
  * 传输阶段不做压缩：图片本身已压缩，压缩只会浪费 CPU 并拖慢。

正确性保障：
  * 块清单在运行时展开成具体目录，覆盖关系没有歧义；额外单列一块
    "散落文件"，收集所有不在任何块目录内的文件（如 labels.names）。
  * 用 `tar --null -T <清单文件>` 读显式文件清单，路径含空格也不会被拆。
  * 源端只读；目标端只写入 --dest-root。
  * 每块传完立即比对"文件数 + 总字节"，不一致则标记失败，可单独重跑。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KEY = '/home/jykj/.ssh/id_ed25519_102'
HOST = 'jykj@192.168.198.102'
SRC_ROOT = Path('/home/jykj/Dataset')
SSH_OPTS = ['-i', KEY, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', '-o', 'Compression=no']

# 业务级大块 + 需要单独切开的目录。XunLian 用 XUNLIAN_SLICE 明确挑选，
# 剩下的子目录由 'XunLian-rest' 展开，不会漏也不会重。
BLOCKS = [
    'Hbb/ChanYunXiaoMeiZhong', 'Hbb/ChuanGanQi', 'Hbb/FengMenKaiGuan', 'Hbb/HuBangBan',
    'Hbb/LianHeCheLiang', 'Hbb/PersonHelmet', 'Hbb/PersonSleep', 'Hbb/PiDaiYuanBan',
    'Hbb/RenChengHouChe', 'Hbb/WorkclothesHelmet',
    'Hbb/TanShuiHouShangGan/XunLian/00078000004',
    'Hbb/TanShuiHouShangGan/XunLian/001001005',
    'Hbb/TanShuiHouShangGan/XunLian/001020004',
    'Hbb/TanShuiHouShangGan/XunLian/001030004',
    'Hbb/TanShuiHouShangGan/XunLian/001030005',
    'Hbb/TanShuiHouShangGan/XunLian/001030007',
    'Hbb/TanShuiHouShangGan/XunLian/001030010',
    'Hbb/TanShuiHouShangGan/XunLian/001040002',
    'Hbb/TanShuiHouShangGan/XunLian/001040004',
    'Hbb/TanShuiHouShangGan/XunLian/001050002',
    'Hbb/TanShuiHouShangGan/XunLian/004000006',
    'Hbb/TanShuiHouShangGan/XunLian/005000007',
    'Hbb/TanShuiHouShangGan/XunLian/006010013',
    'Hbb/TanShuiHouShangGan/XunLian/006048102',
    'Hbb/TanShuiHouShangGan/XunLian/006070021',
    'Hbb/TanShuiHouShangGan/XunLian/006110001',
    'Hbb/TanShuiHouShangGan/XunLian/012000030',
    'Hbb/TanShuiHouShangGan/XunLian/012020001',
    'Hbb/TanShuiHouShangGan/XunLian/012020013',
    'Hbb/TanShuiHouShangGan/XunLian/012020016',
    'Hbb/TanShuiHouShangGan/XunLian-rest',
    'Hbb/TanShuiHouShangGan/xinjian',
    'Hbb/TanShuiHouShangGan/xuebing',
    'Obb/TS-Front_V2/Data',
    'Obb/TS-Front_V2/ShanXi',
    'Obb/TS-Front_V2/YunNan',
    # 不在任何块目录内的零散文件（各层 labels.names 等），单独一块。
    'LOOSE:',
]

XUNLIAN = 'Hbb/TanShuiHouShangGan/XunLian'


def ssh(cmd, **kw):
    return subprocess.run(['ssh'] + SSH_OPTS + [HOST, cmd], **kw)


def expand_blocks():
    """把 '-rest' 展开为具体子目录，其余原样返回。"""
    blocks = []
    explicit = {b for b in BLOCKS if b.startswith(XUNLIAN + '/') and not b.endswith('-rest')}
    for b in BLOCKS:
        if b.endswith('-rest'):
            base = SRC_ROOT / XUNLIAN
            rest = sorted(p.name for p in base.iterdir()
                          if p.is_dir() and f'{XUNLIAN}/{p.name}' not in explicit)
            blocks.extend(f'{XUNLIAN}/{name}' for name in rest)
        else:
            blocks.append(b)
    return blocks


def loose_files(blocks):
    """块目录之外的所有普通文件。

    目录若等于某个块 -> 整棵剪枝；若是某块的祖先 -> 继续下钻；其余 -> 全收。
    LOOSE: 是虚块，不参与覆盖判定。
    """
    covered = {b for b in blocks if not b.startswith('LOOSE:')}

    def status(rel):
        if rel in covered:
            return 'full'
        if any(c.startswith(rel + '/') for c in covered):
            return 'ancestor'
        return 'none'

    found = []
    for base in ('Hbb', 'Obb', 'Seg'):
        top = SRC_ROOT / base
        if not top.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            rel = str(Path(dirpath).relative_to(SRC_ROOT))
            if status(rel) == 'full':
                dirnames[:] = []   # 整棵已由某个块负责，不再下钻
                continue
            for name in filenames:
                found.append(f'{rel}/{name}')
    return sorted(found)


def local_stat(rel, loose=None):
    """返回 (文件数, 总字节)。"""
    if rel.startswith('LOOSE:'):
        n = total = 0
        for f in loose or []:
            p = SRC_ROOT / f
            if p.is_file():
                n += 1
                total += p.stat().st_size
        return n, total
    base = SRC_ROOT / rel
    if not base.exists():
        raise SystemExit(f'源路径不存在: {base}')
    if base.is_file():
        return 1, base.stat().st_size
    n = total = 0
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and not os.path.islink(p):
                n += 1
                total += os.path.getsize(p)
    return n, total


def remote_stat(rel, dest_root, loose=None):
    if rel == 'LOOSE:':
        n = total = 0
        for f in loose or []:
            out = ssh(f"stat -c %s '{dest_root}/{f}' 2>/dev/null", capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                n += 1
                total += int(out.stdout.strip())
        return n, total
    cmd = (f"cd {dest_root} && find '{rel}' -type f -printf '%s\\n' 2>/dev/null "
           f"| awk '{{n++; s+=$1}} END {{print n+0, s+0}}'")
    out = ssh(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return 0, 0
    parts = out.stdout.split()
    return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 0)


def transfer(rel, dest_root, loose=None):
    if rel == 'LOOSE:':
        return transfer_loose(dest_root, loose)
    parent = str(Path(rel).parent)
    name = Path(rel).name
    cmd = (f"tar -C {SRC_ROOT}/{parent} -cf - {name} | "
           f"ssh {' '.join(SSH_OPTS)} {HOST} 'mkdir -p {dest_root}/{parent} && "
           f"tar -C {dest_root}/{parent} -xf -'")
    return subprocess.run(['bash', '-o', 'pipefail', '-c', cmd]).returncode


def transfer_with_retry(rel, dest_root, loose=None, attempts=3):
    """传输一块，失败自动重试。

    大块要传几十分钟，期间偶发 SSH 断连（Connection reset by peer）很常见，
    单次失败不该让整块作废。tar 流不支持断点续传，重试会从头传该块，
    但配合前面的"已一致则跳过"判定，已完成的块不会重复付出代价。
    """
    for i in range(1, attempts + 1):
        rc = transfer(rel, dest_root, loose)
        if rc == 0:
            return 0
        if i < attempts:
            print(f'  第 {i} 次失败（exit {rc}），10 秒后重试', flush=True)
            time.sleep(10)
    return rc


def transfer_loose(dest_root, loose):
    print(f'  零散文件 {len(loose)} 个', flush=True)
    if not loose:
        return 0
    # 清单写文件再用 --null -T 读取，避免超长参数/环境变量限制。
    listing = Path('/tmp/dataset-loose.lst')
    listing.write_bytes('\0'.join(loose).encode())
    cmd = (f"tar --null -T {listing} -cf - -C {SRC_ROOT} | "
           f"ssh {' '.join(SSH_OPTS)} {HOST} 'mkdir -p {dest_root} && tar -C {dest_root} -xf -'")
    return subprocess.run(['bash', '-o', 'pipefail', '-c', cmd]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest-root', default='/home/jykj/Dataset')
    ap.add_argument('--only', default='', help='只处理匹配该子串的块')
    ap.add_argument('--verify-only', action='store_true', help='只校验，不传输')
    ap.add_argument('--skip-verify', action='store_true')
    ap.add_argument('--force', action='store_true', help='即使远端已一致也重传')
    ap.add_argument('--list', action='store_true', help='只打印块清单与体量')
    ap.add_argument('--report', default='/home/jykj/vision-deploy/output/transfer/report.json')
    args = ap.parse_args()

    blocks = expand_blocks()
    loose = loose_files(blocks)
    if args.only:
        blocks = [b for b in blocks if args.only in b]
    report_path = Path(args.report)

    if args.list:
        grand_n = grand_b = 0
        for b in blocks:
            n, s = local_stat(b, loose)
            grand_n += n
            grand_b += s
            print(f'{b}: {s / 1024 ** 3:.2f} GiB {n} files')
        print(f'合计 {grand_n} 文件 {grand_b / 1024 ** 3:.2f} GiB，{len(blocks)} 块')
        return 0

    ssh(f'mkdir -p {args.dest_root}/Hbb {args.dest_root}/Obb {args.dest_root}/Seg')
    report = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'blocks': [], 'failures': []}
    started = time.time()
    failed = set()

    for idx, rel in enumerate(blocks, 1):
        print(f'\n===== [{idx}/{len(blocks)}] {rel} =====', flush=True)
        elapsed = None

        # 先校验：远端已与本端一致就跳过，中断重跑不会重传已完成的块。
        # 代价是一次目标端遍历（只读目录项+stat），远小于重传 759 GiB。
        if not args.verify_only and not args.force:
            ln, lb = local_stat(rel, loose)
            rn, rb = remote_stat(rel, args.dest_root, loose)
            if ln == rn and lb == rb and ln > 0:
                print(f'  已一致（{ln} 文件 {lb} 字节），跳过', flush=True)
                report['blocks'].append({'path': rel, 'status': 'ok', 'elapsed_s': 0,
                                         'skipped': True})
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
                continue

        if not args.verify_only:
            t0 = time.time()
            rc = transfer_with_retry(rel, args.dest_root, loose)
            elapsed = round(time.time() - t0, 1)
            print(f'  transfer exit={rc} elapsed={elapsed}s', flush=True)
            if rc != 0:
                failed.add(rel)
                report['failures'].append({'path': rel, 'reason': f'transfer exit {rc}'})

        if not args.skip_verify and rel not in failed:
            ln, lb = local_stat(rel, loose)
            rn, rb = remote_stat(rel, args.dest_root, loose)
            ok = (ln == rn and lb == rb)
            print(f'  verify files {ln}/{rn} bytes {lb}/{rb} -> '
                  f'{"OK" if ok else "MISMATCH"}', flush=True)
            if not ok:
                failed.add(rel)
                report['failures'].append(
                    {'path': rel, 'reason': f'count mismatch files {ln}/{rn} bytes {lb}/{rb}'})

        report['blocks'].append({'path': rel, 'status': 'failed' if rel in failed else 'ok',
                                 'elapsed_s': elapsed})
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    report['elapsed_s'] = round(time.time() - started, 1)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n完成 {len(blocks) - len(failed)}/{len(blocks)} 块，"
          f"耗时 {report['elapsed_s']}s，报告 {report_path}")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())