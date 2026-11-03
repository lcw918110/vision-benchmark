#!/usr/bin/env bash
# ==============================================================================
# 矿山视觉算法横向评测与部署验证平台 — 统一启动管理入口 (Unified CLI Entrypoint)
# 统一纳管五大架构 (YOLOv8m / YOLOv8l-OBB / YOLOv9e / YOLO11m / YOLO11x) 共 8 个业务权重
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export YOLO_GPU="${YOLO_GPU:-0}"
SERVICE="${YOLO_SERVICE:-yolov8}"

usage() {
  cat <<'EOF'
用法: bash scripts/run.sh <子命令> [参数...]

核心子命令（统一管理全部 5 种算法架构与 11 个标准测试集）：
  list                         列出注册表 (configs/registry.py) 中纳管的全部数据集、权重与架构状态
  all                          一键运行全量评测擂台 (架构矩阵 + 同卷PK + 差异墙 + C++ JIT + 多路并发流)
  matrix                       运行 8 模型全维度横向对比矩阵 (FP32 / FP16 / 分尺度召回 / FAR / P99)
  backend                      运行跨推理执行后端对比 (PyTorch Eager vs TorchScript C++ JIT 静态图)
  stream                       运行 1/2/4/8/16 路 1080p 视频流并发阶梯压测
  accel                        运行端到端 FastInfer 多线程批处理加速对比 (T列 vs U列)
  verify                       运行 FP32 vs FP16 逐框 IoU 零漏检一致性校验
  admit <dataset_key> [其它参数] 单模型一键准入或增量重测 (如: bash scripts/run.sh admit TS_Front_V2)
  sync-table                   将最新评测 JSON 与注册表自动同步刷新至 docs/benchmark-full-table.md
  build-dataset [dataset_key]  从 /home/jykj/Dataset 跨矿区分层构建/校验 5,700 张完备标准测试集
  shell                        进入统一容器交互式终端 (支持 YOLOv8 / YOLOv9 / YOLO11 全系列)

环境变量:
  YOLO_GPU=0                   指定使用的 GPU 卡号 (默认: 0)
  YOLO_SERVICE=yolov8          指定 compose 服务名 (可选: yolov8 | yolov9 | yolo11 | yolo-unified)
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  list)
    python3 - <<'PY'
import sys; sys.path.insert(0, "configs")
from registry import DATASETS
print(f"{'序号':<4} | {'业务数据集':<12} | {'标识(Key)':<19} | {'算法架构':<11} | {'分辨率':<6} | {'标准集':<8} | {'权重路径'}")
print("-" * 98)
for i, d in enumerate(DATASETS, 1):
    w = d["weights_host"] or "待交付权重"
    arch = d["arch"] or "待训练"
    imgsz = str(d["imgsz"] or "—")
    n_img = f"{d['num_images']}张"
    print(f"{i:<4} | {d['name']:<12} | {d['dataset_key']:<19} | {arch:<11} | {imgsz:<6} | {n_img:<8} | {w}")
PY
    ;;
  all)
    docker compose run --rm "$SERVICE" python3 /configs/algo_arena.py --mode all "$@"
    python3 scripts/05-sync-benchmark-table.py
    ;;
  matrix)
    docker compose run --rm "$SERVICE" python3 /configs/algo_arena.py --mode arch_matrix "$@"
    python3 scripts/05-sync-benchmark-table.py
    ;;
  backend)
    docker compose run --rm "$SERVICE" python3 /configs/algo_arena.py --mode backend_arena "$@"
    python3 scripts/05-sync-benchmark-table.py
    ;;
  stream)
    docker compose run --rm "$SERVICE" python3 /configs/algo_arena.py --mode stream_arena "$@"
    ;;
  accel)
    docker compose run --rm "$SERVICE" python3 /configs/accel_compare.py --limit 400 --reps 3 --warmup 2 "$@"
    ;;
  verify)
    docker compose run --rm "$SERVICE" python3 /configs/verify_fast.py --all --n 100 "$@"
    ;;
  admit)
    if [[ $# -lt 1 ]]; then
      echo "错误: 请指定 dataset_key，例如: bash scripts/run.sh admit PersonSleep" >&2
      exit 1
    fi
    ds="$1"; shift
    docker compose run --rm "$SERVICE" python3 /configs/algo_arena.py --mode admit --dataset "$ds" "$@"
    python3 scripts/05-sync-benchmark-table.py
    ;;
  sync-table|sync-excel)
    python3 scripts/05-sync-benchmark-table.py "$@"
    ;;
  build-dataset)
    python3 scripts/08-build-standard-set.py "$@"
    ;;
  shell)
    docker compose run --rm "$SERVICE" bash "$@"
    ;;
  help|-h|--help|*)
    usage
    ;;
esac
