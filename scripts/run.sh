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
  list                              列出注册表 (configs/registry.py) 中纳管的全部数据集、权重与架构状态
  infer <dataset_key|权重> <图片路径> [conf]  对单张图片或图片目录执行推理并输出带框可视化图与JSON明细
  all                               一键运行全量评测擂台 (架构矩阵 + 同卷PK + 差异墙 + C++ JIT + 多路并发流)
  matrix                            运行 8 模型全维度横向对比矩阵 (FP32 / FP16 / 分尺度召回 / FAR / P99)
  backend                           运行跨推理执行后端对比 (PyTorch Eager vs TorchScript C++ JIT 静态图)
  stream                            运行 1/2/4/8/16 路 1080p 视频流并发阶梯压测
  accel                             运行端到端 FastInfer 多线程批处理加速对比 (T列 vs U列)
  verify                            运行 FP32 vs FP16 逐框 IoU 零漏检一致性校验
  admit <dataset_key> [其它参数]     单模型一键准入或增量重测 (如: bash scripts/run.sh admit TS_Front_V2)
  benchmark <run|compare> [参数]    可追溯同卷评测及严格可比性检查（新协议）
  sync-table                        将最新评测 JSON 与注册表自动同步刷新至 docs/benchmark-full-table.md
  build-dataset [dataset_key]       从 /home/jykj/Dataset 跨矿区分层构建/校验 5,700 张完备标准测试集
  shell                             进入统一容器交互式终端 (支持 YOLOv8 / YOLOv9 / YOLO11 全系列)

单图/目录推理示例:
  # 方式 1: 直接填业务标识或中文名 (自动匹配对应权重与最佳分辨率 imgsz=640/960)
  bash scripts/run.sh infer PersonHelmet data/benchmark/standard/PersonHelmet/images
  bash scripts/run.sh infer 人睡觉 /home/jykj/Dataset/Hbb/PersonSleep/.../xxx.jpg 0.35

  # 方式 2: 直接填任意 .pt 权重路径与任意宿主机图片路径
  bash scripts/run.sh infer models/yolov8/custom/belt/yolov11m_belt.pt /path/to/test.jpg

环境变量:
  YOLO_GPU=0                        指定使用的 GPU 卡号 (默认: 0)
  YOLO_SERVICE=yolov8               指定 compose 服务名 (可选: yolov8 | yolov9 | yolo11 | yolo-unified)
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
  infer|predict)
    if [[ $# -lt 2 ]]; then
      echo "用法: bash scripts/run.sh infer <dataset_key|业务名|.pt权重路径> <图片或目录路径> [conf门限，默认0.25]" >&2
      echo "示例: bash scripts/run.sh infer PersonHelmet data/benchmark/standard/PersonHelmet/images" >&2
      exit 1
    fi
    target_model="$1"
    target_source="$2"
    conf_thres="${3:-0.25}"

    if [[ ! -e "$target_source" ]]; then
      echo "错误: 找不到输入图片或目录: $target_source" >&2
      exit 1
    fi
    abs_source="$(realpath "$target_source")"
    if [[ -d "$abs_source" ]]; then
      mount_dir="$abs_source"
      container_source="/infer_input"
    else
      mount_dir="$(dirname "$abs_source")"
      container_source="/infer_input/$(basename "$abs_source")"
    fi

    docker compose run --rm -v "${mount_dir}:/infer_input:ro" "$SERVICE" python3 - "$target_model" "$container_source" "$conf_thres" <<'PY'
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/configs")
from registry import DATASETS
from ultralytics import YOLO

model_arg, src_path, conf = sys.argv[1], sys.argv[2], float(sys.argv[3])

# 1. 解析模型权重与默认推理分辨率
weights_c = None
imgsz = None
ds_label = Path(model_arg).stem
for d in DATASETS:
    if model_arg in (d["dataset_key"], d["name"]):
        if not d["weights_container"]:
            sys.exit(f"错误: 数据集 {model_arg} 尚未配置可用权重")
        weights_c = d["weights_container"]
        imgsz = d["imgsz"]
        ds_label = d["dataset_key"]
        break

if weights_c is None:
    if model_arg.startswith("models/yolov8/"):
        weights_c = "/models/" + model_arg[len("models/yolov8/"):]
    elif model_arg.startswith("/models/"):
        weights_c = model_arg
    else:
        sys.exit(f"错误: 未识别的 dataset_key 或权重路径: {model_arg}")

model = YOLO(weights_c)
if imgsz is None:
    imgsz = int(model.ckpt.get("train_args", {}).get("imgsz", 640)) if getattr(model, "ckpt", None) else 640

out_dir = Path("/output/infer") / ds_label
out_dir.mkdir(parents=True, exist_ok=True)

# 若输入是目录且未限制张数，默认最多推理前 20 张避免阻塞，单张图片则直接推理
src_p = Path(src_path)
if src_p.is_dir():
    imgs = sorted([str(p) for p in src_p.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])[:20]
else:
    imgs = [str(src_p)]

print(f"[Infer] 权重: {weights_c} | 分辨率: {imgsz} | conf={conf} | 待推理图片数: {len(imgs)}")
t0 = time.perf_counter()
results = model.predict(source=imgs[0] if len(imgs) == 1 else imgs, imgsz=imgsz, conf=conf, iou=0.5, save=True, project="/output/infer", name=ds_label, exist_ok=True, verbose=False)
elapsed = (time.perf_counter() - t0) * 1000.0

summary = []
for r in results:
    fname = Path(r.path).name
    items = []
    if getattr(r, "obb", None) is not None and r.obb is not None and len(r.obb):
        for cls_id, score, xywhr in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xywhr.tolist()):
            items.append({"class_id": int(cls_id), "class_name": r.names[int(cls_id)], "conf": round(score, 4), "xywhr": [round(x, 1) for x in xywhr]})
    elif getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes):
        for cls_id, score, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            items.append({"class_id": int(cls_id), "class_name": r.names[int(cls_id)], "conf": round(score, 4), "xyxy": [round(x, 1) for x in xyxy]})
    summary.append({"image": fname, "num_detections": len(items), "detections": items})
    print(f"  - {fname}: 检出 {len(items)} 个目标 -> " + (", ".join(f"{x['class_name']}({x['conf']:.2f})" for x in items) if items else "无目标"))

json_out = out_dir / "predictions.json"
json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
for p in [Path("/output/infer"), out_dir, *out_dir.rglob("*")]:
    try:
        os.chmod(p, 0o777 if p.is_dir() else 0o666)
    except OSError:
        pass
print(f"[Infer 完成] 总耗时: {elapsed:.1f} ms | 画框结果目录: output/infer/{ds_label}/ | 明细JSON: output/infer/{ds_label}/predictions.json")
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
  benchmark)
    docker compose run --rm "$SERVICE" python3 /configs/benchmark_v1.py "$@"
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
