# YOLOv8 / YOLO11 / YOLOv9 服务镜像（ultralytics 官方包）
# 基础镜像来自 NGC（本机可访问），内含 CUDA 12.8 运行库，固定 SHA256 摘要确保可复现
FROM nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

# 国内 PyPI 镜像，避免外网波动
ENV PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

# 固定核心依赖版本
RUN python3 -m pip install --no-cache-dir \
        torch==2.9.0 \
        torchvision==0.24.0 \
        numpy==2.2.6 \
        opencv-python==4.11.0.86 \
        ultralytics==8.4.99

# 业务权重（.pt 内含自定义模块）加载时需要 dill；显式固化进镜像避免运行时联网安装
RUN python3 -m pip install --no-cache-dir dill==0.4.1

WORKDIR /app

# 锁定全部 47 个直接与传递依赖版本（--no-deps 禁止漂移）并校验依赖树完整性
COPY requirements.lock.txt /app/requirements.lock.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements.lock.txt \
    && python3 -m pip check

# 权重不打包进镜像，运行时从宿主机挂载到 /models
ENV YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["python3", "-c", "import torch, ultralytics; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); print('ultralytics', ultralytics.__version__)"]
