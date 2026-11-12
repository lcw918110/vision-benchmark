# 同步到另一台机器

本文件记录两件事：**旧机器要传什么**、**新机器怎么还原**。
只覆盖本项目（vision-deploy）内不在 git 仓库、但有价值的内容。

---

## 一、同步清单

仓库已跟踪 113 个文件（代码、文档、配置、`data.yaml`、`args.yaml`、
质检结论、9 个权重的 sha256 清单），新机器 clone 即可，不在下面的清单里。

需要传的只有三个目录：

| 绝对路径 | 体积 | 文件数 | 内容 |
|---|---:|---:|---|
| `/home/jykj/vision-deploy/data/benchmark` | 3.29 GB | 11442 | 11 个标准测试集图片与标签、测试视频 |
| `/home/jykj/vision-deploy/models` | 527.5 MB | 21 | 8 个业务权重（.pt 本体不入库） |
| `/home/jykj/vision-deploy/output` | 8.16 GB | 3285 | 镜像包、质检分片、评测台账 |
| **合计** | **11.97 GB** | **14748** | |

列表文件 `/home/jykj/sync-list.txt` 就是这三行，可直接喂给 rsync：

```
/home/jykj/vision-deploy/data/benchmark
/home/jykj/vision-deploy/models
/home/jykj/vision-deploy/output
```

> 已核实：项目内除这三个目录外，没有任何既未被 git 跟踪、也未被 ignore
> 覆盖的文件。所以这三行就是全部缺口。

### 传输

```bash
rsync -a --info=progress2 \
  --files-from=/home/jykj/sync-list.txt / \
  <目标机器>:/home/jykj/backup/
```

`-a` 保留权限与时间戳，`--info=progress2` 显示总进度。断点续传可加
`--partial --append-verify`；大文件多可加 `--compress` 但镜像包已经是
zstd 压缩过的，再加压缩反而更慢。

---

## 二、新机器还原

按顺序执行，四步。假设目标路径也是 `/home/jykj/vision-deploy`。

### 1. 克隆代码

```bash
git clone git@github.com:lcw918110/vision-benchmark.git ~/vision-deploy
cd ~/vision-deploy
git log -1 --oneline   # 确认拿到的是预期的提交
```

> GitHub SSH 走 `ssh.github.com:443`，需要在 `~/.ssh/config` 里配好，
> 详见第四节。

### 2. 放回数据、权重、产物

```bash
rsync -a /home/jykj/backup/vision-deploy/ ~/vision-deploy/
```

这一步把 `data/benchmark`、`models`、`output` 三个目录覆盖回项目。
git 跟踪的文件与备份中的同名文件内容一致，覆盖不会造成差异。

### 3. 恢复 Docker 镜像

```bash
cd ~/vision-deploy/output/transfer/docker
zstd -d -c cuda-12.8.1-runtime-ubuntu22.04.tar.zst | docker load
zstd -d -c vision-yolov8-8.4.99-cu12.8.tar.zst | docker load
```

两个包共 8.3 GB，压缩前 18.1 + 5.6 GB。恢复后验证：

```bash
docker images | grep -E "vision/yolov8|cuda"
docker compose run --rm yolov8   # 打印 torch / ultralytics 版本与 CUDA 可用性
```

**如果新机器能联网，可以跳过第 3 步**，直接重建（省 8.3 GB 传输）：

```bash
cd ~/vision-deploy && docker compose build
```

Dockerfile 固定了基础镜像 SHA256 摘要、依赖锁 47 行，重建结果与导出的
镜像等价。需要能访问 NGC（约 5.6 GB）与腾讯云 PyPI。

### 4. 校验权重完整性

```bash
cd ~/vision-deploy
python3 - <<'EOF'
import json, hashlib, pathlib
m = json.load(open("output/dataset/weights-manifest.json"))
bad = 0
for r in m["files"]:
    p = pathlib.Path(r["path"])
    if not p.exists():
        print("缺失", p); bad += 1; continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""): h.update(c)
    if h.hexdigest() != r["sha256"]:
        print("不一致", p); bad += 1
print(len(m["files"]), "个权重，异常", bad, "个")
EOF
```

> `weights-manifest.json` 里存的是旧机器的绝对路径。若新机器路径不同，
> 需改 `path` 字段，或只比对文件名与 sha256。

---

## 三、环境文件（不在项目目录，需单独处理）

| 绝对路径 | 体积 | 内容 | 处理 |
|---|---:|---|---|
| `/home/jykj/.ssh` | 6 KB | SSH 私钥 id_ed25519_102、config、known_hosts | 敏感 |
| `/home/jykj/.local/share/mihomo` | 25 KB | 代理配置 config.yaml（含订阅凭据） | 敏感 |
| `/home/jykj/.config/systemd/user` | 2 KB | 两个 systemd 用户服务 | 可重建 |
| `/home/jykj/.local/bin/mihomo` | 30.8 MB | 代理二进制 | 可重装 |
| `/home/jykj/.local/bin/gh` | 39.2 MB | GitHub CLI | 可重装 |

`~/.ssh` 与 mihomo 配置含凭据，不要在公网传输。更稳的做法是**不搬私钥**，
在新机器重新生成后添加到 GitHub：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_102 -N ""
cat ~/.ssh/id_ed25519_102.pub   # 添加到 GitHub → Settings → SSH keys
```

---

## 四、环境重建要点

- **GitHub SSH** 必须走 `ssh.github.com:443`。本机 DNS 把 `github.com`
  解析到 `127.0.0.1`，直连会失败。`~/.ssh/config` 写法：
  ```
  Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519_102
  ```
- **宿主机 Python 3.14 没有 torch/ultralytics**，所有推理与评测都在容器内。
  宿主机脚本只用标准库。
- **`docker save` 基础镜像必须用 digest 名**：
  `nvcr.io/nvidia/cuda@sha256:4a801ef9...`。用 tag 名会报 `No such image`
  （第一次导出踩过）。
- **`labels.cache`**（1.1 MB）由 ultralytics 首次运行自动生成，可一并传也
  可删掉。
- **GPU 直通**：`docker-compose.yml` 用 `runtime: nvidia`，新机器需装
  NVIDIA Container Toolkit。

---

## 五、不需要同步的内容

| 路径 | 体积 | 原因 |
|---|---:|---|
| `/home/jykj/yolov5*` | 11 MB | ultralytics 官方仓库 4 个 tag 克隆（v6.0/v6.1/v6.2/v7.0），0 改动，`git clone --branch v6.0 ...` 即可恢复 |
| `/home/jykj/.cursor-server` | 2.3 GB | Cursor 远程缓存，重装自动生成 |
| `/home/jykj/.cache` | 19 MB | 用户缓存 |
| `vision-deploy/logs` | 20 KB | 运行日志 |
| `vision-deploy/**/__pycache__` | — | Python 字节码 |
