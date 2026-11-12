# 重装前备份清单

本文件说明**本项目（vision-deploy）**的内容如何划分：哪些已由 git 保护、
哪些需要另存。其他项目的数据不在本项目范围内。

> 需要同步到另一台机器的**绝对路径清单**见
> [`sync-to-another-machine.md`](./sync-to-another-machine.md)。
> 哪些内容**有意不入库**、为什么，见 [`repo-boundary.md`](./repo-boundary.md)。

## 一、已由 git 保护（无需另存）

仓库已推送到 https://github.com/lcw918110/vision-benchmark ，另一台机器直接克隆：

```bash
git clone git@github.com:lcw918110/vision-benchmark.git
```

包含：全部代码与脚本、文档、Dockerfile 与依赖锁、11 个标准集的
`data.yaml` / `manifest.json` / `excluded.txt`、9 个模型的 `args.yaml` 与
结构 yaml、质检报告与结论（`report.json`、`issues.csv`、`CHECKSUMS.txt`、
11 个 `summary.json`）、9 个权重的 sha256 清单。

## 二、需要另存（本项目内、不在仓库）

| 路径 | 体积 | 内容 |
|---|---:|---|
| `data/benchmark/standard/*/images`、`labels` | 3.06 GB | 11 个标准测试集图片与标签 |
| `models/yolov8/custom/**/*.pt` | 521 MB | 9 个业务模型权重 |
| `data/benchmark/videos` | 234 MB | 测试视频 |
| `output/dataset/benchmark-quality-v1`（分片） | 1.4 MB | 质检逐分片结果 |
| `output/dataset/benchmark-audit-v1` | 0.2 MB | 文件级审计结果 |

后两项可由脚本重建，但需要数小时磁盘 I/O，建议一并拷贝。

## 三、可重建，无需备份

| 路径 | 体积 | 重建方式 |
|---|---:|---|
| `data/benchmark/standard/*/labels.cache` | 1.1 MB | ultralytics 首次训练自动生成 |
| `output/dataset/*.log`、`logs/` | 0.7 MB | 运行日志 |
| `output/transfer` | 16 KB | 传输中间产物 |

Docker 镜像 `vision/yolov8:8.4.99-cu12.8`（18.1 GB）未推送 registry，
可在新机器用 `docker compose build` 重建。

## 四、环境配置（本机，不在仓库）

| 路径 | 内容 |
|---|---|
| `~/.ssh/id_ed25519_102` | GitHub 账号级私钥，无口令保护 |
| `~/.ssh/config` | SSH 走 `ssh.github.com:443`（本机 DNS 把 github.com 解析为 `127.0.0.1`） |
| `~/.local/share/mihomo/config.yaml` | 代理配置，含订阅凭据 |
| `~/.config/systemd/user/*.service` | 两个 systemd 用户服务 |

私钥与代理配置属敏感信息，单独保管，勿传公网。也可在新机器重新生成
SSH 密钥并添加到 GitHub，无需搬运私钥。

## 五、环境重建要点

- 评测在 Docker 内运行，基础镜像 `nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04`
  固定了 SHA256 摘要，依赖锁定于 `docker/yolov8/requirements.lock.txt`（54 行）
- 宿主机 Python 3.14 无 torch/ultralytics，所有推理都在容器内
- 质检脚本 `scripts/15` 支持断点续跑，分片缓存使已完成的目录无需重扫
