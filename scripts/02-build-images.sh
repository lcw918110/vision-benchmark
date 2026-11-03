#!/usr/bin/env bash
# 构建 YOLOv8 镜像并检查指定 GPU；例如 YOLO_GPU=1 bash scripts/02-build-images.sh
set -euo pipefail
cd "$(dirname "$0")/.."
image=vision/yolov8:8.4.99-cu12.8
gpu=${YOLO_GPU:-0}

docker build -t "$image" docker/yolov8
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$gpu" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "$image" \
  python3 -c 'import torch, ultralytics; print("torch", torch.__version__, "CUDA", torch.version.cuda, "ultralytics", ultralytics.__version__); assert torch.cuda.is_available(), "CUDA 不可用"; print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); x = torch.arange(1024, device="cuda"); assert (x * 2).sum().item() == 1047552; print("GPU 计算通过")'
