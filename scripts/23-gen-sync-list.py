#!/usr/bin/env python3
"""生成同步与还原文档（docs/sync-to-another-machine.md）。

只覆盖**本项目**（vision-deploy）内不在 git 仓库、但有价值的内容。
本机其他目录下的数据不属于本项目，一律不在此列。

配套列表 /home/jykj/sync-list.txt 只有三行目录，可直接喂给 rsync。
"""
import pathlib
import subprocess

ROOT = pathlib.Path('/home/jykj/vision-deploy')
OUT = ROOT / 'docs/sync-to-another-machine.md'

# 同步清单：三个目录覆盖全部非 git 内容
DIRS = [
    ('/home/jykj/vision-deploy/data/benchmark', '11 个标准测试集图片与标签、测试视频'),
    ('/home/jykj/vision-deploy/models', '8 个业务权重（.pt 本体不入库）'),
    ('/home/jykj/vision-deploy/output', '镜像包、质检分片、评测台账'),
]

# 环境文件：不在项目目录内，单独处理
ENV = [
    ('/home/jykj/.ssh', 'SSH 私钥 id_ed25519_102、config、known_hosts', '敏感'),
    ('/home/jykj/.local/share/mihomo', '代理配置 config.yaml（含订阅凭据）', '敏感'),
    ('/home/jykj/.config/systemd/user', '两个 systemd 用户服务', '可重建'),
    ('/home/jykj/.local/bin/mihomo', '代理二进制', '可重装'),
    ('/home/jykj/.local/bin/gh', 'GitHub CLI', '可重装'),
]


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


def human(b):
    if b < 1024:
        return f'{b} B'
    if b < 1048576:
        return f'{b / 1024:.0f} KB'
    if b < 1073741824:
        return f'{b / 1048576:.1f} MB'
    return f'{b / 1073741824:.2f} GB'


def main():
    rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    ntracked = len(subprocess.run(['git', '-C', str(ROOT), 'ls-files'],
                                  capture_output=True, text=True).stdout.splitlines())

    rows = [(p, du(p), d) for p, d in DIRS]
    total = sum(s for _, (s, _), _ in rows)
    total_files = sum(c for _, (_, c), _ in rows)

    L = []
    A = L.append
    A('# 同步到另一台机器')
    A('')
    A('本文件记录两件事：**旧机器要传什么**、**新机器怎么还原**。')
    A('只覆盖本项目（vision-deploy）内不在 git 仓库、但有价值的内容。')
    A('')
    A('---')
    A('')
    A('## 一、同步清单')
    A('')
    A(f'仓库已跟踪 {ntracked} 个文件（代码、文档、配置、`data.yaml`、`args.yaml`、')
    A('质检结论、9 个权重的 sha256 清单），新机器 clone 即可，不在下面的清单里。')
    A('')
    A('需要传的只有三个目录：')
    A('')
    A('| 绝对路径 | 体积 | 文件数 | 内容 |')
    A('|---|---:|---:|---|')
    for p, (s, c), d in rows:
        A(f'| `{p}` | {human(s)} | {c} | {d} |')
    A(f'| **合计** | **{human(total)}** | **{total_files}** | |')
    A('')
    A('列表文件 `/home/jykj/sync-list.txt` 就是这三行，可直接喂给 rsync：')
    A('')
    A('```')
    for p, _, _ in rows:
        A(p)
    A('```')
    A('')
    A('> 已核实：项目内除这三个目录外，没有任何既未被 git 跟踪、也未被 ignore')
    A('> 覆盖的文件。所以这三行就是全部缺口。')
    A('')
    A('### 传输')
    A('')
    A('```bash')
    A('rsync -a --info=progress2 \\')
    A('  --files-from=/home/jykj/sync-list.txt / \\')
    A('  <目标机器>:/home/jykj/backup/')
    A('```')
    A('')
    A('`-a` 保留权限与时间戳，`--info=progress2` 显示总进度。断点续传可加')
    A('`--partial --append-verify`；大文件多可加 `--compress` 但镜像包已经是')
    A('zstd 压缩过的，再加压缩反而更慢。')
    A('')
    A('---')
    A('')
    A('## 二、新机器还原')
    A('')
    A('按顺序执行，四步。假设目标路径也是 `/home/jykj/vision-deploy`。')
    A('')
    A('### 1. 克隆代码')
    A('')
    A('```bash')
    A('git clone git@github.com:lcw918110/vision-benchmark.git ~/vision-deploy')
    A('cd ~/vision-deploy')
    A('git log -1 --oneline   # 确认拿到的是预期的提交')
    A('```')
    A('')
    A('> GitHub SSH 走 `ssh.github.com:443`，需要在 `~/.ssh/config` 里配好，')
    A('> 详见第四节。')
    A('')
    A('### 2. 放回数据、权重、产物')
    A('')
    A('```bash')
    A('rsync -a /home/jykj/backup/vision-deploy/ ~/vision-deploy/')
    A('```')
    A('')
    A('这一步把 `data/benchmark`、`models`、`output` 三个目录覆盖回项目。')
    A('git 跟踪的文件与备份中的同名文件内容一致，覆盖不会造成差异。')
    A('')
    A('### 3. 恢复 Docker 镜像')
    A('')
    A('```bash')
    A('cd ~/vision-deploy/output/transfer/docker')
    A('zstd -d -c cuda-12.8.1-runtime-ubuntu22.04.tar.zst | docker load')
    A('zstd -d -c vision-yolov8-8.4.99-cu12.8.tar.zst | docker load')
    A('```')
    A('')
    A('两个包共 8.3 GB，压缩前 18.1 + 5.6 GB。恢复后验证：')
    A('')
    A('```bash')
    A('docker images | grep -E "vision/yolov8|cuda"')
    A('docker compose run --rm yolov8   # 打印 torch / ultralytics 版本与 CUDA 可用性')
    A('```')
    A('')
    A('**如果新机器能联网，可以跳过第 3 步**，直接重建（省 8.3 GB 传输）：')
    A('')
    A('```bash')
    A('cd ~/vision-deploy && docker compose build')
    A('```')
    A('')
    A('Dockerfile 固定了基础镜像 SHA256 摘要、依赖锁 47 行，重建结果与导出的')
    A('镜像等价。需要能访问 NGC（约 5.6 GB）与腾讯云 PyPI。')
    A('')
    A('### 4. 校验权重完整性')
    A('')
    A('```bash')
    A('cd ~/vision-deploy')
    A('python3 - <<\'EOF\'')
    A('import json, hashlib, pathlib')
    A('m = json.load(open("output/dataset/weights-manifest.json"))')
    A('bad = 0')
    A('for r in m["files"]:')
    A('    p = pathlib.Path(r["path"])')
    A('    if not p.exists():')
    A('        print("缺失", p); bad += 1; continue')
    A('    h = hashlib.sha256()')
    A('    with p.open("rb") as f:')
    A('        for c in iter(lambda: f.read(4 << 20), b""): h.update(c)')
    A('    if h.hexdigest() != r["sha256"]:')
    A('        print("不一致", p); bad += 1')
    A('print(len(m["files"]), "个权重，异常", bad, "个")')
    A('EOF')
    A('```')
    A('')
    A('> `weights-manifest.json` 里存的是旧机器的绝对路径。若新机器路径不同，')
    A('> 需改 `path` 字段，或只比对文件名与 sha256。')
    A('')
    A('---')
    A('')
    A('## 三、环境文件（不在项目目录，需单独处理）')
    A('')
    A('| 绝对路径 | 体积 | 内容 | 处理 |')
    A('|---|---:|---|---|')
    for p, desc, act in ENV:
        s, _ = du(p)
        A(f'| `{p}` | {human(s)} | {desc} | {act} |')
    A('')
    A('`~/.ssh` 与 mihomo 配置含凭据，不要在公网传输。更稳的做法是**不搬私钥**，')
    A('在新机器重新生成后添加到 GitHub：')
    A('')
    A('```bash')
    A('ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_102 -N ""')
    A('cat ~/.ssh/id_ed25519_102.pub   # 添加到 GitHub → Settings → SSH keys')
    A('```')
    A('')
    A('---')
    A('')
    A('## 四、环境重建要点')
    A('')
    A('- **GitHub SSH** 必须走 `ssh.github.com:443`。本机 DNS 把 `github.com`')
    A('  解析到 `127.0.0.1`，直连会失败。`~/.ssh/config` 写法：')
    A('  ```')
    A('  Host github.com')
    A('    HostName ssh.github.com')
    A('    Port 443')
    A('    User git')
    A('    IdentityFile ~/.ssh/id_ed25519_102')
    A('  ```')
    A('- **宿主机 Python 3.14 没有 torch/ultralytics**，所有推理与评测都在容器内。')
    A('  宿主机脚本只用标准库。')
    A('- **`docker save` 基础镜像必须用 digest 名**：')
    A('  `nvcr.io/nvidia/cuda@sha256:4a801ef9...`。用 tag 名会报 `No such image`')
    A('  （第一次导出踩过）。')
    A('- **`labels.cache`**（1.1 MB）由 ultralytics 首次运行自动生成，可一并传也')
    A('  可删掉。')
    A('- **GPU 直通**：`docker-compose.yml` 用 `runtime: nvidia`，新机器需装')
    A('  NVIDIA Container Toolkit。')
    A('')
    A('---')
    A('')
    A('## 五、不需要同步的内容')
    A('')
    A('| 路径 | 体积 | 原因 |')
    A('|---|---:|---|')
    A('| `/home/jykj/yolov5*` | 11 MB | ultralytics 官方仓库 4 个 tag 克隆（v6.0/v6.1/v6.2/v7.0），0 改动，`git clone --branch v6.0 ...` 即可恢复 |')
    A('| `/home/jykj/.cursor-server` | 2.3 GB | Cursor 远程缓存，重装自动生成 |')
    A('| `/home/jykj/.cache` | 19 MB | 用户缓存 |')
    A('| `vision-deploy/logs` | 20 KB | 运行日志 |')
    A('| `vision-deploy/**/__pycache__` | — | Python 字节码 |')

    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'已写入 {OUT}（{len(L)} 行）')
    print(f'同步合计 {human(total)} / {total_files} 个文件')


if __name__ == '__main__':
    main()
