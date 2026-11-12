# 重装前备份清单

本文件列出**不在 git 版本控制内**的本地内容。仓库代码、文档、脚本、
质检报告与全部类别定义已推送到 GitHub，无需额外备份。

清单合计约 **8.18 GB**（含可重建项）。

> 哪些内容**有意不入库**、为什么，见 [`repo-boundary.md`](./repo-boundary.md)。
> 本文件解决「重装前要拷什么」，那份文件解决「什么不该进 git」。

## 一、必须备份

| 路径 | 体积 | 内容 |
|---|---:|---|
| `vision-deploy/data/benchmark/standard` | 3134.8 MB | 11 个标准测试集图片+标签（5,700 张） |
| `ZJH/PiDaiYuanBan/luanming-7004` | 2148.2 MB | 皮带队原版 3502 图 + 3502 标签 |
| `vision-deploy/models/yolov8/custom` | 521.3 MB | 8 个训练好的 .pt 权重 |

## 二、按需备份

| 路径 | 体积 | 内容 |
|---|---:|---|
| `vision-deploy/data/benchmark/videos` | 234.0 MB | 测试视频 |
| `ZJH/PiDaiYuanBan/batch_8_in_testdataset_same` | 58.3 MB | 与测试集重叠样本图片 |
| `ZJH/PiDaiYuanBan/PiDaiYuanBan-Org` | 2.9 MB | 原始标签 groups |

## 三、可重建或可忽略

| 路径 | 体积 | 处理方式 |
|---|---:|---|
| `.cursor-server` | 2253.6 MB | 可重装，无需备份 |
| `.cache` | 16.7 MB | 可重建 |
| `vision-deploy/models/yolov8/detect` | 6.2 MB | 可重新下载 |
| `vision-deploy/output/dataset/benchmark-quality-v1` | 1.4 MB | 可重建（15 号脚本） |
| `vision-deploy/output/dataset/quality-audit.log` | 0.7 MB | 日志 |
| `vision-deploy/output/dataset/benchmark-audit-v1` | 0.2 MB | 可重建（14 号脚本） |
| `vision-deploy/output/transfer` | 0.0 MB | 临时 |
| `vision-deploy/logs` | 0.0 MB | 日志 |

## 四、敏感凭据（单独保管，勿传公网）

- `~/.local/share/mihomo`（24.8 KB）— 代理配置（含订阅凭据）
- `~/.ssh/id_ed25519_102`（0.4 KB）— SSH 私钥

> `~/.ssh/id_ed25519_102` 已添加到 GitHub 账号，换机恢复后即可继续推送；
> 也可重新生成密钥并重新添加，无需搬运私钥。

## 五、超过 GitHub 单文件 100 MB 限制的权重

如需用 Git LFS 而非移动硬盘备份，以下两个文件需走 LFS：

- `models/yolov8/custom/person/yolo11x_960_best.pt` — 109.1 MB
- `models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt` — 111.8 MB

## 六、环境重建要点

- 评测在 Docker 内运行（`vision/yolo-unified:8.4.99-cu12.8`），宿主机 Python 3.14 无 torch/ultralytics
- 依赖锁定于 `docker/yolov8/requirements.lock.txt`
- 两个 systemd 用户服务定义位于 `~/.config/systemd/user/`，内容见 `docs/`
- GitHub SSH 走 `ssh.github.com:443`（本机 DNS 把 github.com 解析为 127.0.0.1），配置在 `~/.ssh/config`
