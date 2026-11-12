#!/usr/bin/env python3
"""生成同步到另一台机器的绝对路径清单（docs/sync-to-another-machine.md）。

只列**本项目**（vision-deploy）内不在 git 仓库、但有价值的内容。
本机其他目录下的数据不属于本项目，一律不在此列。
"""
import pathlib

ITEMS = [
    # (绝对路径, 类别, 内容说明, 备注)
    ('/home/jykj/vision-deploy/data/benchmark/standard', 'must',
     '11 个标准测试集的图片与标签（5,700 张图 / 17,037 框）',
     '仓库只存 data.yaml / manifest.json / excluded.txt，图片标签必须单独拷'),

    ('/home/jykj/vision-deploy/models/yolov8/custom', 'must',
     '9 个业务模型训练成果（含 2 个超 100MB 的权重）',
     'args.yaml 与模型结构 yaml 已入库，.pt 本体需单独拷'),

    ('/home/jykj/vision-deploy/data/benchmark/videos', 'optional',
     '测试视频（1 个文件，234 MB）',
     '评测视频用例，按需'),

    ('/home/jykj/vision-deploy/output/dataset/benchmark-quality-v1', 'optional',
     '质检逐分片结果（2,405 个 json，1.4 MB）',
     '可由 scripts/15 重跑重建，但需数小时磁盘 I/O；建议拷'),

    ('/home/jykj/vision-deploy/output/dataset/benchmark-audit-v1', 'optional',
     '文件级审计逐目录结果（784 个 json）',
     '可由 scripts/14 重建'),

    ('/home/jykj/vision-deploy/output/benchmark/arena', 'optional',
     '评测结果台账（JSON / Markdown）',
     '部分已入库，未入库的为可视化产物'),

    ('/home/jykj/.ssh', 'secret',
     'SSH 私钥 id_ed25519_102、config、known_hosts',
     '含 GitHub 账号级私钥。也可在新机器重新生成密钥再添加到 GitHub，'
     '无需搬运'),

    ('/home/jykj/.local/share/mihomo', 'secret',
     '代理配置 config.yaml（含订阅凭据）',
     '敏感，勿传公网'),

    ('/home/jykj/.config/systemd/user', 'config',
     '两个 systemd 用户服务定义',
     'vision-mihomo.service、vision-quality-audit.service'),

    ('/home/jykj/.local/bin/mihomo', 'config',
     '代理二进制（30.8 MB）',
     '可重新下载'),

    ('/home/jykj/.local/bin/gh', 'config',
     'GitHub CLI 二进制（39.2 MB）',
     '可重新安装'),
]

LABEL = {'must': '一、必须同步', 'optional': '二、按需同步',
         'secret': '三、敏感（单独保管，勿传公网）', 'config': '四、环境重建（可重新安装）'}


def du(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0, 0
    if p.is_file():
        return p.stat().st_size, 1
    size = count = 0
    for f in p.rglob('*'):
        try:
            if f.is_file():
                size += f.stat().st_size
                count += 1
        except OSError:
            pass
    return size, count


def main():
    groups = {}
    for path, kind, desc, note in ITEMS:
        size, count = du(path)
        groups.setdefault(kind, []).append((path, size, count, desc, note))

    head = pathlib.Path('/home/jykj/vision-deploy').resolve()
    try:
        import subprocess
        rev = subprocess.run(['git', '-C', str(head), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
        nfiles = len(subprocess.run(['git', '-C', str(head), 'ls-files'],
                                    capture_output=True, text=True).stdout.splitlines())
    except Exception:
        rev, nfiles = '?', 0

    lines = [
        '# 同步到另一台机器的清单',
        '',
        '本清单只列**本项目（vision-deploy）**内**不在 git 仓库、但有价值**的内容，',
        '全部使用绝对路径。其他项目的数据不在本项目范围内，未列入。',
        '',
        f'仓库已跟踪的 {nfiles} 个文件（代码、文档、配置、类别定义、训练参数、',
        '质检结论）**不需要**出现在这里，另一台机器直接克隆即可：',
        '',
        '```bash',
        'git clone git@github.com:lcw918110/vision-benchmark.git',
        '```',
        '',
        '> 仓库地址：https://github.com/lcw918110/vision-benchmark',
        '> 克隆后可用 `git log -1 --oneline` 查看当时的提交',
        '',
        '**9 个权重的路径与 sha256 清单已入库**（`output/dataset/weights-manifest.json`），',
        '拷贝完成后可用它校验完整性，命令见第五节。',
        '',
    ]

    total_must = sum(s for _, s, _, _, _ in groups.get('must', []))
    total_opt = sum(s for _, s, _, _, _ in groups.get('optional', []))
    lines += [
        f'**必须同步合计 {total_must / 1073741824:.2f} GB，按需 {total_opt / 1073741824:.2f} GB。**',
        '',
    ]

    for kind in ('must', 'optional', 'secret', 'config'):
        if kind not in groups:
            continue
        lines += [f'## {LABEL[kind]}', '']
        if kind in ('must', 'optional'):
            lines += ['| 绝对路径 | 体积 | 文件数 | 内容 | 备注 |',
                      '|---|---:|---:|---|---|']
            for path, size, count, desc, note in groups[kind]:
                sz = f'{size / 1048576:.1f} MB' if size < 1073741824 else f'{size / 1073741824:.2f} GB'
                lines.append(f'| `{path}` | {sz} | {count} | {desc} | {note} |')
        else:
            lines += ['| 绝对路径 | 体积 | 内容 | 备注 |', '|---|---:|---|---|']
            for path, size, count, desc, note in groups[kind]:
                sz = f'{size / 1024:.1f} KB' if size < 1048576 else f'{size / 1048576:.1f} MB'
                lines.append(f'| `{path}` | {sz} | {desc} | {note} |')
        lines.append('')

    lines += [
        '## 五、同步命令',
        '',
        '用 rsync 保留权限与时间戳，`-a` 归档模式，`--info=progress2` 显示总进度：',
        '',
        '```bash',
        '# 必须同步的两项（约 3.6 GB）',
        'rsync -a --info=progress2 \\',
        '  /home/jykj/vision-deploy/data/benchmark/standard \\',
        '  /home/jykj/vision-deploy/models/yolov8/custom \\',
        '  <目标机器>:/home/jykj/backup/',
        '```',
        '',
        '同步后校验权重完整性（在目标机器上，仓库已克隆的前提下）：',
        '',
        '```bash',
        'cd vision-benchmark',
        'python3 - <<\'EOF\'',
        'import json, hashlib, pathlib',
        'm = json.load(open("output/dataset/weights-manifest.json"))',
        'bad = 0',
        'for r in m["files"]:',
        '    p = pathlib.Path(r["path"])',
        '    if not p.exists():',
        '        print("缺失", p); bad += 1; continue',
        '    h = hashlib.sha256()',
        '    with p.open("rb") as f:',
        '        for c in iter(lambda: f.read(4 << 20), b""): h.update(c)',
        '    if h.hexdigest() != r["sha256"]:',
        '        print("不一致", p); bad += 1',
        'print(len(m["files"]), "个权重，异常", bad, "个")',
        'EOF',
        '```',
        '',
        '> 若目标机器路径不同，需相应修改 `weights-manifest.json` 里的 `path` 字段，',
        '> 或只比对文件名与 sha256。',
        '',
        '## 六、环境重建要点',
        '',
        '- **Docker 镜像** `vision/yolov8:8.4.99-cu12.8`（18.1 GB）未推送 registry，',
        '  可在新机器用 `docker compose build` 重建；Dockerfile 与 54 行依赖锁已在仓库',
        '- 基础镜像 `nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04` 固定了 SHA256 摘要',
        '- **GitHub SSH** 必须走 `ssh.github.com:443`（本机 DNS 把 github.com 解析为',
        '  `127.0.0.1`），配置写在 `~/.ssh/config`',
        '- 宿主机 Python 3.14 无 torch/ultralytics，评测全部在容器内运行',
        '',
        '## 七、不需要同步的内容',
        '',
        '| 路径 | 体积 | 原因 |',
        '|---|---:|---|',
        '| `/home/jykj/vision-deploy/logs` | 20 KB | 运行日志 |',
        '| `/home/jykj/vision-deploy/output/transfer` | 16 KB | 传输中间产物 |',
        '| `/home/jykj/.cursor-server` | 2.3 GB | Cursor 远程缓存，可重装 |',
        '| `/home/jykj/.cache` | 19 MB | 用户缓存 |',
        '| `/home/jykj/yolov5*` | 11 MB | ultralytics 官方仓库克隆，0 改动 |',
    ]

    out = pathlib.Path('/home/jykj/vision-deploy/docs/sync-to-another-machine.md')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'已写入 {out}（{len(lines)} 行）')
    print(f'必须 {total_must / 1073741824:.2f} GB，按需 {total_opt / 1073741824:.2f} GB')


if __name__ == '__main__':
    main()
