#!/usr/bin/env bash
# 验证容器内可访问两张 RTX 4090
# 用法：bash scripts/01-verify-gpu-docker.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# NGC 源在本机可访问（Docker Hub 在本机被阻断，故不使用 docker.io 镜像）
IMAGE="nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04"

echo "==> 拉取基础镜像：$IMAGE"
docker pull "$IMAGE"

echo "==> 容器内查看 GPU"
docker run --rm --gpus all "$IMAGE" nvidia-smi

echo
echo "==> 完成。以上应显示 2 张 NVIDIA GeForce RTX 4090"
