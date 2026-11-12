# 重装前备份清单

本文件列出**不在 git 版本控制内**的本地内容。仓库代码、文档、脚本、
质检报告与全部类别定义已推送到 GitHub，无需额外备份。

清单合计约 **10.76 GB**（含可重建项），其中**必须备份 8.24 GB**。

> 哪些内容**有意不入库**、为什么，见 [`repo-boundary.md`](./repo-boundary.md)。
> 本文件解决「重装前要拷什么」，那份文件解决「什么不该进 git」。

## 一、必须备份

| 路径 | 体积 | 内容 |
|---|---:|---|
| `vision-deploy/data/benchmark/standard` | 3134.8 MB | 11 个标准测试集图片+标签（5,700 张） |
| `ZJH/PiDaiYuanBan/luanming-7004` | 2148.2 MB | 皮带队原版 3502 图 + 3502 标签 |
| **`/mnt/bak/CoreYoloU_runs`** | **2534.4 MB** | **11 个业务数据集的训练产物，82 个权重** |
| `vision-deploy/models/yolov8/custom` | 521.3 MB | 8 个业务模型训练成果 |

### 模型权重全量清单（91 个 / 3.00 GB）

权重分散在两个位置，**互不重复**（已用 sha256 逐文件比对，仅 3 组是同目录内
`best.pt` 与 `last.pt` 相同，属正常）。明细见
`output/dataset/weights-manifest.json`（含每个文件的 sha256，可用于校验备份完整性）。

| 分组 | 权重数 | 体积 |
|---|---:|---:|
| `/mnt/bak/CoreYoloU_runs/_keep_best` | 19 | 790.1 MB |
| `/mnt/bak/CoreYoloU_runs/full` | 24 | 649.1 MB |
| `models/yolov8/custom` | 9 | 527.5 MB |
| `/mnt/bak/CoreYoloU_runs/unified_100k` | 9 | 441.8 MB |
| `/mnt/bak/CoreYoloU_runs/s640` | 8 | 155.4 MB |
| `/mnt/bak/CoreYoloU_runs/mix` | 8 | 155.4 MB |
| `/mnt/bak/CoreYoloU_runs/unified_50k` | 2 | 115.1 MB |
| `/mnt/bak/CoreYoloU_runs/unified_20k` | 4 | 77.6 MB |
| `/mnt/bak/CoreYoloU_runs/ft320` | 4 | 77.5 MB |
| `/mnt/bak/CoreYoloU_runs/full_s640` | 2 | 38.8 MB |
| `/mnt/bak/CoreYoloU_runs/unified_50k_strat` | 2 | 38.8 MB |

> `/mnt/bak` 是**独立挂载盘**（1.5T，已用 2.6G）。重装系统时若只格式化系统盘，
> 数据可能还在；但若整盘重装或换机，必须单独拷贝。这批权重是 11 个数据集
> （`full/` 下按数据集分目录）与统一模型（`unified_*`）的训练成果，无法从
> 仓库重建，只能重新训练。

校验备份是否完整：

```bash
cd /home/jykj/vision-deploy
python3 - <<'EOF'
import json, hashlib, pathlib
m = json.load(open('output/dataset/weights-manifest.json'))
bad = 0
for r in m['files']:
    p = pathlib.Path(r['path'])
    if not p.exists():
        print('缺失', p); bad += 1; continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(4<<20), b''): h.update(c)
    if h.hexdigest() != r['sha256']:
        print('不一致', p); bad += 1
print(f'{len(m["files"])} 个权重，异常 {bad} 个')
EOF
```

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
