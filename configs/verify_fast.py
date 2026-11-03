#!/usr/bin/env python3
"""推理精度逐框一致性校验工具（支持单模型与 --all 全量业务模型批处理）。

校验三个核心问题：
  1. fast_fp32_vs_ref_fp32 : fast_infer (FP32) 与官方 ultralytics.predict (FP32) 逐框对比
  2. fast_fp16_vs_ref_fp16 : fast_infer (FP16) 与官方 ultralytics.predict (FP16) 逐框对比
  3. fp16_vs_fp32          : 同一模型在 FP16 与 FP32 下的逐框对比（评估半精度上线风险）

指标定义：
  - ref_boxes / cmp_boxes : 两侧检出总框数
  - count_mismatch        : 检出框数量不一致的图片张数
  - matched_boxes         : 同类别且 IoU >= 0.5 匹配成功的框对数
  - unmatched             : 基准侧未能匹配上的框数（掉框/错类）
  - min_iou / mean_iou    : 匹配框对的最小 IoU 与平均 IoU
  - max_conf_delta        : 匹配框对的最大置信度绝对偏差

用法（容器内）：
  # 跑全部 6 个检测业务模型（标准测试集）
  python3 /configs/verify_fast.py --all --n 100

  # 单模型校验
  python3 /configs/verify_fast.py --weights <w.pt> --source <图片目录> --imgsz 640 --n 50 [--half]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_infer import letterbox, unletterbox  # noqa: E402
from registry import get_active_models  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

# 从唯一注册表动态加载全部水平框检测模型 (排除 OBB)
MODELS = [
    (
        f"{m['name']} ({m['arch']})",
        m["weights_container"],
        m["dataset_key"],
        m["imgsz"],
    )
    for m in get_active_models(task="detect")
]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load(p, imgsz):
    """返回 (letterbox 后数组, 原始 shape, letterbox 后 shape)"""
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return None, None, None
    lb = letterbox(im, imgsz)
    return lb, im.shape[:2], lb.shape[:2]


def collect_paths(source, n):
    paths = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(source, "**", e), recursive=True))
    return sorted(paths)[:n]


def run_ref(weights, paths, imgsz, batch, conf, iou_t, half):
    """官方 ultralytics.predict 路径，按原图尺寸分桶以避免混批触发 auto=False 正方形填充。"""
    m = YOLO(weights)
    buckets = {}
    for gi, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        buckets.setdefault(im.shape[:2], []).append((gi, p))

    out = [None] * len(paths)
    for _shp, items in buckets.items():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            b_paths = [c[1] for c in chunk]
            b_idxs = [c[0] for c in chunk]
            ref = m.predict(source=b_paths, imgsz=imgsz, batch=batch, conf=conf,
                            iou=iou_t, half=half, device=0, verbose=False)
            for gi, r in zip(b_idxs, ref):
                b = r.boxes
                out[gi] = (
                    b.xyxy.cpu().numpy(),
                    b.conf.cpu().numpy(),
                    b.cls.cpu().numpy().astype(int),
                )
    del m
    torch.cuda.empty_cache()
    return out


def run_fast(weights, paths, imgsz, batch, conf, iou_t, half):
    """优化路径（与 fast_infer 相同 fuse + 预处理 + NMS + 坐标还原），返回同结构列表。"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = YOLO(weights).model
    if hasattr(net, "fuse"):
        net = net.fuse(verbose=False)
    net = net.eval().to(device)
    if half:
        net.half()
    dtype = torch.float16 if half else torch.float32

    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as ex:
        loaded = list(ex.map(lambda p: load(p, imgsz), paths))

    # 按 letterbox 后尺寸分桶组批（与 fast_infer.py 完全一致，兼容跨矿区不同宽高比图片）
    buckets = {}
    for gi, (a, s, ls) in enumerate(loaded):
        if a is not None:
            buckets.setdefault(ls, []).append((gi, a, s))

    for _ls, items in buckets.items():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            arrs = [c[1] for c in chunk]
            shapes = [c[2] for c in chunk]
            idxs = [c[0] for c in chunk]
            arr = np.ascontiguousarray(np.stack(arrs)[..., ::-1].transpose(0, 3, 1, 2))
            t = torch.from_numpy(arr).to(device, dtype=dtype).div_(255.0)
            with torch.no_grad():
                o = net(t)
            pred = o[0] if isinstance(o, (tuple, list)) else o
            det = non_max_suppression(pred, conf, iou_t, max_det=300)
            for d, shp, gi in zip(det, shapes, idxs):
                if len(d):
                    d = d.clone()
                    d[:, :4] = unletterbox(d[:, :4], shp, imgsz)
                results[gi] = (
                    d[:, :4].cpu().numpy(),
                    d[:, 4].cpu().numpy(),
                    d[:, 5].cpu().numpy().astype(int),
                )
    torch.cuda.synchronize()
    del net
    torch.cuda.empty_cache()
    return results


def compare_box_sets(ref_list, cmp_list, half_flag=False):
    stats = {
        "n_images": len(ref_list),
        "count_mismatch": 0,
        "min_iou": 1.0,
        "mean_iou": 1.0,
        "max_conf_delta": 0.0,
        "compared_boxes": 0,
        "unmatched": 0,
        "ref_boxes": 0,
        "fast_boxes": 0,
    }
    ious = []
    for (ba, ca, la), (bb, cb, lb) in zip(ref_list, cmp_list):
        stats["ref_boxes"] += int(len(ba))
        stats["fast_boxes"] += int(len(bb))
        if len(ba) != len(bb):
            stats["count_mismatch"] += 1
        used = set()
        for j in range(len(ba)):
            best, best_iou, best_k = -1, 0.0, -1
            for k in range(len(bb)):
                if k in used or lb[k] != la[j]:
                    continue
                v = iou(ba[j], bb[k])
                if v > best_iou:
                    best, best_iou, best_k = k, v, k
            if best_k < 0 or best_iou < 0.5:
                stats["unmatched"] += 1
                continue
            used.add(int(best_k))
            ious.append(float(best_iou))
            stats["min_iou"] = min(float(stats["min_iou"]), float(best_iou))
            stats["max_conf_delta"] = max(float(stats["max_conf_delta"]),
                                          abs(float(ca[j]) - float(cb[best])))
            stats["compared_boxes"] += 1

    stats["min_iou"] = round(float(stats["min_iou"]), 5) if ious else 1.0
    stats["mean_iou"] = round(float(sum(ious) / len(ious)), 5) if ious else 1.0
    stats["max_conf_delta"] = round(float(stats["max_conf_delta"]), 6)
    stats["half"] = half_flag
    stats["all_identical"] = (
        stats["count_mismatch"] == 0
        and stats["unmatched"] == 0
        and stats["min_iou"] > 0.99
        and stats["max_conf_delta"] < 1e-3
    )
    return stats


def verify_one_worker(name, weights, dataset, imgsz, n, batch, conf, iou_t, out_dir, q):
    source = f"/data/benchmark/standard/{dataset}/images"
    paths = collect_paths(source, n)
    if not paths:
        q.put({"model": name, "error": f"未找到图片: {source}"})
        return

    print(f"\n=== 校验 {name} | {len(paths)} 张 | imgsz={imgsz} ===", flush=True)
    ref_fp32 = run_ref(weights, paths, imgsz, batch, conf, iou_t, half=False)
    fast_fp32 = run_fast(weights, paths, imgsz, batch, conf, iou_t, half=False)
    ref_fp16 = run_ref(weights, paths, imgsz, batch, conf, iou_t, half=True)
    fast_fp16 = run_fast(weights, paths, imgsz, batch, conf, iou_t, half=True)

    s_fp32 = compare_box_sets(ref_fp32, fast_fp32, half_flag=False)
    s_fp16 = compare_box_sets(ref_fp16, fast_fp16, half_flag=True)
    s_drift = compare_box_sets(ref_fp32, fast_fp16, half_flag=True)

    stem = os.path.splitext(os.path.basename(weights))[0]
    os.makedirs(out_dir, exist_ok=True)
    for tag, obj in [("fp32", s_fp32), ("half", s_fp16), ("fp16_vs_fp32", s_drift)]:
        with open(os.path.join(out_dir, f"verify_{stem}_{tag}.json"), "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"  [FP32 优化 vs 官方] 框 {s_fp32['ref_boxes']}/{s_fp32['fast_boxes']} | "
          f"失配 {s_fp32['unmatched']} | min_IoU={s_fp32['min_iou']} | "
          f"max_dConf={s_fp32['max_conf_delta']}", flush=True)
    print(f"  [FP16 优化 vs 官方] 框 {s_fp16['ref_boxes']}/{s_fp16['fast_boxes']} | "
          f"失配 {s_fp16['unmatched']} | min_IoU={s_fp16['min_iou']} | "
          f"max_dConf={s_fp16['max_conf_delta']}", flush=True)
    print(f"  [FP16 vs FP32 漂移] 框 {s_drift['ref_boxes']}/{s_drift['fast_boxes']} | "
          f"框数差图 {s_drift['count_mismatch']} | 失配 {s_drift['unmatched']} | "
          f"min_IoU={s_drift['min_iou']} (mean={s_drift['mean_iou']}) | "
          f"max_dConf={s_drift['max_conf_delta']}", flush=True)

    q.put({
        "model": name,
        "dataset": dataset,
        "weights": weights,
        "imgsz": imgsz,
        "n_images": len(paths),
        "fast_fp32_vs_ref_fp32": s_fp32,
        "fast_fp16_vs_ref_fp16": s_fp16,
        "fp16_vs_fp32": s_drift,
    })


def run_all(args):
    out_dir = "/output/benchmark/verify"
    os.makedirs(out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    rows = []
    picks = [s.strip() for s in args.models.split(",") if s.strip()]

    for name, weights, dataset, imgsz in MODELS:
        if picks and not any(p in name for p in picks):
            continue
        if not os.path.exists(weights):
            print(f"[跳过] {name}: 权重不存在 {weights}", flush=True)
            continue
        q = ctx.Queue()
        proc = ctx.Process(
            target=verify_one_worker,
            args=(name, weights, dataset, imgsz, args.n, args.batch,
                  args.conf, args.iou, out_dir, q),
        )
        proc.start()
        proc.join()
        if not q.empty():
            rows.append(q.get())
        else:
            rows.append({"model": name, "error": f"子进程异常退出 (exit={proc.exitcode})"})

    json_path = os.path.join(out_dir, "all_verify.json")
    md_path = os.path.join(out_dir, "all_verify.md")
    if picks and os.path.exists(json_path):
        try:
            old_rows = {r["model"]: r for r in json.load(open(json_path))}
            for r in rows:
                old_rows[r["model"]] = r
            rows = [old_rows[m[0]] for m in MODELS if m[0] in old_rows]
        except Exception:
            pass
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 模型 | 样本数 | imgsz | FP32 同精度 min_IoU / max_Δconf | FP16 同精度 min_IoU / max_Δconf | FP16 vs FP32 框数 (基准/半精度) | FP16 vs FP32 框数差图 / 未匹配框 | FP16 vs FP32 min_IoU (mean_IoU) | FP16 vs FP32 max_Δconf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | - | - | {r['error']} | - | - | - | - | - |")
            continue
        f32 = r["fast_fp32_vs_ref_fp32"]
        f16 = r["fast_fp16_vs_ref_fp16"]
        dr = r["fp16_vs_fp32"]
        lines.append(
            f"| {r['model']} | {r['n_images']} | {r['imgsz']} | "
            f"{f32['min_iou']:.5f} / {f32['max_conf_delta']:.5f} | "
            f"{f16['min_iou']:.5f} / {f16['max_conf_delta']:.5f} | "
            f"{dr['ref_boxes']} / {dr['fast_boxes']} | "
            f"{dr['count_mismatch']} / {dr['unmatched']} | "
            f"{dr['min_iou']:.5f} ({dr['mean_iou']:.5f}) | "
            f"{dr['max_conf_delta']:.5f} |"
        )
    md_text = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"\n汇总已保存:\n  {json_path}\n  {md_path}\n")
    print(md_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="单模型权重路径（省略或传 --all 则跑全量模型）")
    ap.add_argument("--source", help="单模型图片目录")
    ap.add_argument("--all", action="store_true", help="对全部 6 个业务模型执行完整逐框校验")
    ap.add_argument("--models", default="", help="--all 模式下按名称关键字过滤，逗号分隔")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    if args.all or not args.weights:
        run_all(args)
        return

    paths = collect_paths(args.source, args.n)
    if not paths:
        raise SystemExit("未找到图片")

    ref = run_ref(args.weights, paths, args.imgsz, args.batch, args.conf, args.iou, args.half)
    fast = run_fast(args.weights, paths, args.imgsz, args.batch, args.conf, args.iou, args.half)
    stats = compare_box_sets(ref, fast, half_flag=args.half)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    out = "/output/benchmark/verify"
    os.makedirs(out, exist_ok=True)
    tag = "half" if args.half else "fp32"
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    p = os.path.join(out, f"verify_{stem}_{tag}.json")
    with open(p, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"结果: {p}")
    sys.exit(0 if stats["all_identical"] else 1)


if __name__ == "__main__":
    main()
