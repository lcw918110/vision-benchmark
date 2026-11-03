#!/usr/bin/env bash
# 宿主机基础软件安装：Docker Engine + NVIDIA Container Toolkit
# 用法：sudo bash scripts/00-install-host-docker.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "请用 sudo 运行此脚本" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> 1/5 安装 Docker、Compose 与 NVIDIA Container Toolkit"
apt-get update
apt-get install -y docker.io docker-compose-v2 nvidia-container-toolkit

echo "==> 2/5 用官方工具注册 NVIDIA 运行时"
# 由 nvidia-ctk 负责写入，避免手工改坏已有配置
nvidia-ctk runtime configure --runtime=docker

echo "==> 3/5 追加日志与存储配置"
mkdir -p /etc/docker
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
cfg = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
cfg["log-driver"] = "json-file"
cfg["log-opts"] = {"max-size": "50m", "max-file": "5"}
json.dump(cfg, open(p, "w"), indent=2)
print(json.dumps(cfg, indent=2))
PY

echo "==> 4/5 启用并重启 Docker"
systemctl enable --now docker
systemctl restart docker

echo "==> 5/5 将当前用户加入 docker 组"
TARGET_USER="${SUDO_USER:-}"
if [[ -n "$TARGET_USER" && "$TARGET_USER" != "root" ]]; then
  usermod -aG docker "$TARGET_USER"
  echo "已加入 docker 组：$TARGET_USER（需重新登录或 newgrp docker 生效）"
fi

docker --version
docker compose version
nvidia-ctk --version
