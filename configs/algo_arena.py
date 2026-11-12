#!/usr/bin/env python3
"""视觉算法横向对比评测擂台 (Algorithm Arena v2)。

面向「多算法测试与横向对比」的统一评测引擎，支持五大模式：
  1. arch_matrix (全架构横向矩阵 + 矿山工程选型深度诊断)：
     覆盖全部 8 个业务模型（含 OBB），一次性测出：
     - 复杂度：层数、参数量 (M)、GFLOPs (@640 与 @实际分辨率)、体积 (MB)
     - 纯 GPU 前向算力与尾延迟：FP32/FP16 × b1/b16 的 FPS、Mean/P95/P99 延迟 (ms)、显存峰值
     - 原生 FP32 vs FP16 精度与速度对比（覆盖 HBB 与 OBB）：mAP50、mAP50-95、ΔmAP50、单类 AP50
     - 矿山工程选型细粒度指标：小/中/大目标分尺度召回率 (Small/Med/Large Recall)、背景图误报率与总误报率
  2. head_to_head (同数据集跨算法同卷正交 PK)：
     在 WorkclothesHelmet 与 PersonHelmet 上执行 YOLOv8m vs YOLO11x × 640 vs 960 正交实验。
  3. diff_wall (多算法 Bad Case / 差异样本并排可视化对比墙)：
     自动筛出同卷 PK 中算法 A 与算法 B 检出差异最大的 Top 样本，生成
     「Ground Truth vs YOLOv8m vs YOLO11x」1×3 并排可视化对比图（output/benchmark/arena/diff_wall/）。
  4. backend_arena (跨推理后端对比)：
     对比代表性算法在 PyTorch Eager (FP32/FP16) vs TorchScript C++ 静态图 (FP32/FP16) vs FastInfer 流水线
     下的延迟 (Mean/P99)、吞吐 (FPS) 与逐框精度一致性。
  5. custom / admit (自定义多算法对比与新模型一键准入)：
     支持任意候选模型列表同卷对比或新权重准入测试。
"""
import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DATASETS, get_active_models  # noqa: E402

ALL_ARCH_MODELS = [
    (
        m["arch"],
        m["name"],
        f"{m['name']} ({m['arch']})",
        m["weights_container"],
        m["dataset_key"],
        m["imgsz"],
        m["task"],
    )
    for m in get_active_models()
]


def _worker_result(process, queue, timeout=7200):
    """子进程异常或超时时停止等待，避免自动化任务无限挂起。"""
    import queue as queue_module
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join()
            raise TimeoutError(f"评测子进程超过 {timeout} 秒")
        try:
            result = queue.get(timeout=min(1.0, remaining))
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
                raise RuntimeError("评测子进程返回结果后未退出")
            if process.exitcode != 0:
                raise RuntimeError(f"评测子进程失败: exit={process.exitcode}")
            return result
        except queue_module.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(f"评测子进程未返回结果: exit={process.exitcode}")


def _shm_mb():
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize / 1024 / 1024
    except OSError:
        return 0


def _box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _parse_gt_boxes(txt_path, img_w, img_h, task="detect", allowed_classes=None):
    """解析 YOLO HBB 或 OBB 标注文件，统一返回外接像素框 [(cls, [x1,y1,x2,y2], norm_area)]。"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            vals = [float(x) for x in parts[1:]]
            if task == "obb" and len(vals) >= 8:
                xs = [vals[i] * img_w for i in range(0, 8, 2)]
                ys = [vals[i] * img_h for i in range(1, 8, 2)]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                norm_area = ((x2 - x1) / max(1.0, img_w)) * ((y2 - y1) / max(1.0, img_h))
            elif len(vals) >= 4:
                cx, cy, bw, bh = vals[:4]
                x1 = (cx - bw / 2.0) * img_w
                y1 = (cy - bh / 2.0) * img_h
                x2 = (cx + bw / 2.0) * img_w
                y2 = (cy + bh / 2.0) * img_h
                norm_area = bw * bh
            else:
                continue
            boxes.append((cid, [x1, y1, x2, y2], norm_area))
    return boxes


def _eval_fine_grained_diagnostics(model, ds_name, imgsz, task, conf=0.25, iou_thres=0.5):
    """计算工程选型细粒度指标：小/中/大目标分尺度召回率 + 背景误报与全集误报率。"""
    import cv2

    img_dir = f"/data/benchmark/standard/{ds_name}/images"
    lbl_dir = f"/data/benchmark/standard/{ds_name}/labels"
    img_paths = sorted(
        p for p in glob.glob(os.path.join(img_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not img_paths:
        return {}

    # 面积阈值（按折算到 640x640 分辨率下的像素面积：<32^2 为小目标，32^2~96^2 为中目标，>96^2 为大目标）
    small_thr = (32.0 * 32.0) / (640.0 * 640.0)
    med_thr = (96.0 * 96.0) / (640.0 * 640.0)

    scale_gt = {"small": 0, "medium": 0, "large": 0}
    scale_tp = {"small": 0, "medium": 0, "large": 0}
    bg_images = 0
    bg_fp_images = 0
    bg_fp_boxes = 0
    total_tp = 0
    total_fp = 0
    total_gt = 0

    # 分批推理避免显存累积（注意：传入 list 时必须显式指定 batch=8，否则 ultralytics 会把整个 list 当作单次 batch！）
    chunk_size = 16
    for i in range(0, len(img_paths), chunk_size):
        chunk = img_paths[i:i + chunk_size]
        preds = model.predict(
            chunk, imgsz=imgsz, batch=8, conf=conf, iou=0.7, device=0, verbose=False, save=False
        )
        for img_p, res in zip(chunk, preds):
            h, w = res.orig_shape
            stem = os.path.splitext(os.path.basename(img_p))[0]
            txt_p = os.path.join(lbl_dir, stem + ".txt")
            gts = _parse_gt_boxes(txt_p, w, h, task=task)

            # 提取预测框 [(cid, [x1,y1,x2,y2], conf)]
            p_boxes = []
            if task == "obb" and getattr(res, "obb", None) is not None and len(res.obb) > 0:
                xyxy = res.obb.xyxy.cpu().numpy()
                clss = res.obb.cls.cpu().numpy()
                confs = res.obb.conf.cpu().numpy()
                for c, b, cf in zip(clss, xyxy, confs):
                    p_boxes.append((int(c), b.tolist(), float(cf)))
            elif getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for c, b, cf in zip(clss, xyxy, confs):
                    p_boxes.append((int(c), b.tolist(), float(cf)))

            p_boxes.sort(key=lambda x: x[2], reverse=True)

            if len(gts) == 0:
                bg_images += 1
                if len(p_boxes) > 0:
                    bg_fp_images += 1
                    bg_fp_boxes += len(p_boxes)
                total_fp += len(p_boxes)
                continue

            total_gt += len(gts)
            gt_matched = [False] * len(gts)
            gt_scales = []
            for _, _, n_area in gts:
                if n_area < small_thr:
                    sc = "small"
                elif n_area < med_thr:
                    sc = "medium"
                else:
                    sc = "large"
                gt_scales.append(sc)
                scale_gt[sc] += 1

            for pcid, pbox, _ in p_boxes:
                best_iou = 0.0
                best_j = -1
                for j, (gcid, gbox, _) in enumerate(gts):
                    if gt_matched[j] or gcid != pcid:
                        continue
                    ov = _box_iou_xyxy(pbox, gbox)
                    if ov > best_iou:
                        best_iou = ov
                        best_j = j
                if best_iou >= iou_thres and best_j >= 0:
                    gt_matched[best_j] = True
                    scale_tp[gt_scales[best_j]] += 1
                    total_tp += 1
                else:
                    total_fp += 1

    def _safe_rec(tp, gt):
        return round(tp / gt, 4) if gt > 0 else None

    return {
        "scale_recall": {
            "small": _safe_rec(scale_tp["small"], scale_gt["small"]),
            "medium": _safe_rec(scale_tp["medium"], scale_gt["medium"]),
            "large": _safe_rec(scale_tp["large"], scale_gt["large"]),
            "counts": scale_gt,
        },
        "false_positives": {
            "bg_images": bg_images,
            "bg_fp_images": bg_fp_images,
            "bg_fp_boxes": bg_fp_boxes,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_gt": total_gt,
            "false_alarm_ratio": round(total_fp / max(1, total_tp + total_fp), 4),
        },
    }


def _arch_worker(spec, out_q):
    arch, ds_cn_name, label, weights, ds_name, imgsz, task = spec
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    file_mb = round(os.path.getsize(weights) / (1024 * 1024), 1)
    y = YOLO(weights)
    net = copy.deepcopy(y.model).to("cuda:0").eval()
    if hasattr(net, "fuse"):
        net = net.fuse(verbose=False)

    params_m = round(get_num_params(net) / 1e6, 2)
    flops_640 = round(float(get_flops(net, imgsz=640)), 1)
    flops_actual = round(flops_640 * ((imgsz / 640.0) ** 2), 1)
    n_layers = len(list(net.model)) if hasattr(net, "model") else 0

    # 1. 纯 GPU 网络前向基准 (含单帧 Mean / P95 / P99 尾延迟分位数)
    gpu_bench = {}
    for half_flag, ptag in [(False, "fp32"), (True, "fp16")]:
        m_net = copy.deepcopy(net).to("cuda:0").eval()
        if half_flag:
            m_net = m_net.half()
        dtype = torch.float16 if half_flag else torch.float32
        for bs in (1, 16):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            x = torch.zeros((bs, 3, imgsz, imgsz), dtype=dtype, device="cuda:0")
            with torch.inference_mode():
                for _ in range(8):
                    _ = m_net(x)
                torch.cuda.synchronize()
                iters = 40 if bs == 1 else 15
                lat_list = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = m_net(x)
                    torch.cuda.synchronize()
                    lat_list.append((time.perf_counter() - t0) * 1000.0 / bs)
            arr = np.array(lat_list, dtype=np.float64)
            mean_ms = round(float(arr.mean()), 3)
            p95_ms = round(float(np.percentile(arr, 95)), 3)
            p99_ms = round(float(np.percentile(arr, 99)), 3)
            fps = round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0
            vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
            gpu_bench[f"{ptag}_b{bs}"] = {
                "ms_per_img": mean_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "fps": fps,
                "vram_mb": vram_mb,
            }
        del m_net
        torch.cuda.empty_cache()
    del net
    torch.cuda.empty_cache()

    # 2. 原生标准集 FP32 vs FP16 精度对比（覆盖 HBB 与 OBB 全部 8 个模型）
    data_yaml = f"/data/benchmark/standard/{ds_name}/data.yaml"
    val_cmp = {}
    for half_flag, ptag in [(False, "fp32"), (True, "fp16")]:
        m_val = YOLO(weights)
        metrics = m_val.val(
            data=data_yaml, imgsz=imgsz, batch=16, half=half_flag,
            device=0, workers=0, plots=False, save=False, verbose=False,
        )
        box = metrics.box
        per_cls = {}
        for idx, cid in enumerate(list(getattr(box, "ap_class_index", []))):
            per_cls[str(int(cid))] = {
                "ap50": round(float(box.ap50[idx]), 4),
                "ap": round(float(box.ap[idx]), 4),
                "p": round(float(box.p[idx]), 4),
                "r": round(float(box.r[idx]), 4),
            }
        spd = getattr(metrics, "speed", {})
        tot_ms = sum(float(spd.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess"))
        val_cmp[ptag] = {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "val_fps": round(1000.0 / tot_ms, 1) if tot_ms > 0 else 0.0,
            "per_class": per_cls,
        }
        del m_val
        torch.cuda.empty_cache()

    val_cmp["fp16_mAP50_delta"] = round(val_cmp["fp16"]["mAP50"] - val_cmp["fp32"]["mAP50"], 4)
    val_cmp["fp16_mAP50_95_delta"] = round(val_cmp["fp16"]["mAP50_95"] - val_cmp["fp32"]["mAP50_95"], 4)

    # 3. 工程选型细粒度指标（小/中/大目标召回率 + 误报率）
    m_diag = YOLO(weights)
    diag = _eval_fine_grained_diagnostics(m_diag, ds_name, imgsz, task)
    del m_diag
    torch.cuda.empty_cache()

    # 清除 val 生成的临时 .cache
    cache_f = f"/data/benchmark/standard/{ds_name}/labels.cache"
    if os.path.exists(cache_f):
        try:
            os.remove(cache_f)
        except OSError:
            pass

    res = {
        "arch": arch,
        "dataset_cn": ds_cn_name,
        "label": label,
        "dataset": ds_name,
        "task": task,
        "imgsz": imgsz,
        "file_mb": file_mb,
        "layers": n_layers,
        "params_m": params_m,
        "gflops_640": flops_640,
        "gflops_actual": flops_actual,
        "pure_gpu": gpu_bench,
        "val_fp32_vs_fp16": val_cmp,
        "diagnostics": diag,
    }
    sr = diag.get("scale_recall", {})
    fp_info = diag.get("false_positives", {})
    print(
        f"[Arch] {arch:<11} | {label:<20} | {params_m:>5.2f}M | {flops_actual:>6.1f} GFLOPs@{imgsz} | "
        f"b16 FP16={gpu_bench['fp16_b16']['fps']:>5.1f} FPS (b1 P99={gpu_bench['fp16_b1']['p99_ms']:.1f}ms) | "
        f"mAP50(FP32/16)={val_cmp['fp32']['mAP50']:.4f}/{val_cmp['fp16']['mAP50']:.4f} | "
        f"Rec(S/M/L)={sr.get('small')}/{sr.get('medium')}/{sr.get('large')} | "
        f"FAR={fp_info.get('false_alarm_ratio')}",
        flush=True,
    )
    out_q.put(res)


def _h2h_worker(job, out_q):
    """在指定数据集与类别子集上评测单个 (算法, 分辨率) 组合。"""
    ds_name, data_yaml, arch_label, weights, imgsz, classes, conf, iou = job
    from ultralytics import YOLO

    workers = 4 if _shm_mb() >= 512 else 0
    model = YOLO(weights)
    t0 = time.perf_counter()
    metrics = model.val(
        data=data_yaml, imgsz=imgsz, batch=16, conf=conf, iou=iou,
        classes=classes, device=0, workers=workers, plots=False, save=False, verbose=False,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    box = metrics.box

    per_cls = {}
    ap_cls_idx = list(getattr(box, "ap_class_index", []))
    all_ap50 = getattr(box, "ap50", [])
    names_map = getattr(metrics, "names", {})
    for i, cid in enumerate(ap_cls_idx):
        cname = names_map.get(int(cid), str(cid))
        per_cls[cname] = round(float(all_ap50[i]), 4)

    speed = getattr(metrics, "speed", {})
    inf_ms = round(float(speed.get("inference", 0.0)), 2)
    total_ms = round(sum(float(speed.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess")), 2)

    res = {
        "dataset": ds_name,
        "candidate": arch_label,
        "weights": weights,
        "imgsz": imgsz,
        "classes": classes,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_AP50": per_cls,
        "infer_ms": inf_ms,
        "e2e_ms": total_ms,
        "e2e_fps": round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0,
        "elapsed_s": elapsed,
    }
    print(
        f"[H2H {ds_name}] {arch_label} @{imgsz} -> mAP50={res['mAP50']:.4f}, "
        f"mAP50-95={res['mAP50_95']:.4f}, P={res['precision']:.4f}, R={res['recall']:.4f}, "
        f"per_cls={per_cls}, infer={inf_ms}ms",
        flush=True,
    )
    out_q.put(res)


def _diff_wall_worker(out_dir, out_q):
    """在 PersonHelmet 与 WorkclothesHelmet 上自动筛出 YOLOv8m vs YOLO11x 检出差异最大的样本，生成 1x3 并排对比墙。"""
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    diff_dir = os.path.join(out_dir, "diff_wall")
    os.makedirs(diff_dir, exist_ok=True)

    w_v8m = "/models/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt"
    w_v11x = "/models/custom/person/yolo11x_960_best.pt"
    shared_cls = {2: "Person", 3: "Helmet", 8: "NoHelmet(Head)"}
    cls_colors = {
        2: (50, 205, 50),    # Person: Green
        3: (255, 180, 0),    # Helmet: Cyan/Blue-Orange
        8: (0, 60, 255),     # Head (NoHelmet): Red
    }

    m_v8 = YOLO(w_v8m)
    m_11 = YOLO(w_v11x)

    candidates_records = []
    for ds_name in ["PersonHelmet", "WorkclothesHelmet"]:
        img_dir = f"/data/benchmark/standard/{ds_name}/images"
        lbl_dir = f"/data/benchmark/standard/{ds_name}/labels"
        img_paths = sorted(
            p for p in glob.glob(os.path.join(img_dir, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for p in img_paths:
            im = cv2.imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            stem = os.path.splitext(os.path.basename(p))[0]
            gts = _parse_gt_boxes(os.path.join(lbl_dir, stem + ".txt"), w, h, allowed_classes=set(shared_cls.keys()))
            if not gts:
                continue

            r8 = m_v8.predict(p, imgsz=640, conf=0.25, iou=0.7, classes=[2, 3, 8], device=0, half=True, verbose=False)[0]
            r11 = m_11.predict(p, imgsz=960, conf=0.25, iou=0.7, classes=[2, 3, 8], device=0, half=True, verbose=False)[0]

            def _match_eval(res_obj):
                pboxes = []
                if res_obj.boxes is not None and len(res_obj.boxes) > 0:
                    for c, b, cf in zip(
                        res_obj.boxes.cls.cpu().numpy(),
                        res_obj.boxes.xyxy.cpu().numpy(),
                        res_obj.boxes.conf.cpu().numpy(),
                    ):
                        pboxes.append((int(c), b.tolist(), float(cf)))
                pboxes.sort(key=lambda x: x[2], reverse=True)
                matched_gt = [False] * len(gts)
                tp, fp = 0, 0
                box_status = []  # (cid, box, conf, is_tp)
                for cid, pb, cf in pboxes:
                    best_ov, best_j = 0.0, -1
                    for j, (gcid, gb, _) in enumerate(gts):
                        if not matched_gt[j] and gcid == cid:
                            ov = _box_iou_xyxy(pb, gb)
                            if ov > best_ov:
                                best_ov, best_j = ov, j
                    if best_ov >= 0.5 and best_j >= 0:
                        matched_gt[best_j] = True
                        tp += 1
                        box_status.append((cid, pb, cf, True))
                    else:
                        fp += 1
                        box_status.append((cid, pb, cf, False))
                fn = len(gts) - tp
                missed_gts = [gts[j] for j in range(len(gts)) if not matched_gt[j]]
                return tp, fp, fn, box_status, missed_gts

            tp8, fp8, fn8, boxes8, miss8 = _match_eval(r8)
            tp11, fp11, fn11, boxes11, miss11 = _match_eval(r11)

            # 差异评分：YOLO11x 比 YOLOv8m 少漏检或少误报的幅度
            diff_score = (tp11 - tp8) * 2 + (fp8 - fp11)
            if diff_score > 0 and len(gts) >= 2:
                candidates_records.append({
                    "ds": ds_name,
                    "img_path": p,
                    "stem": stem,
                    "gt_count": len(gts),
                    "gts": gts,
                    "v8": (tp8, fp8, fn8, boxes8, miss8),
                    "v11": (tp11, fp11, fn11, boxes11, miss11),
                    "diff_score": diff_score,
                })

    candidates_records.sort(key=lambda x: (x["diff_score"], x["gt_count"]), reverse=True)
    selected = candidates_records[:6]

    def _render_panel(base_img, title, subtitle, boxes_draw, missed_draw=None, is_gt=False):
        target_w, target_h = 640, 400
        h0, w0 = base_img.shape[:2]
        scale = min(target_w / w0, target_h / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(base_img, (nw, nh))
        canvas = np.full((target_h + 68, target_w, 3), 30, dtype=np.uint8)
        ox = (target_w - nw) // 2
        oy = 68 + (target_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized

        # 顶部标题栏
        cv2.rectangle(canvas, (0, 0), (target_w, 64), (45, 45, 45), -1)
        cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 235, 255), 1, cv2.LINE_AA)

        if is_gt:
            for cid, b, _ in boxes_draw:
                x1, y1 = int(b[0] * scale) + ox, int(b[1] * scale) + oy
                x2, y2 = int(b[2] * scale) + ox, int(b[3] * scale) + oy
                col = cls_colors.get(cid, (0, 255, 0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
                cv2.putText(canvas, shared_cls.get(cid, str(cid)), (x1, max(78, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        else:
            # 先画漏检框 (红色醒目框标注 MISS)
            for cid, b, _ in (missed_draw or []):
                x1, y1 = int(b[0] * scale) + ox, int(b[1] * scale) + oy
                x2, y2 = int(b[2] * scale) + ox, int(b[3] * scale) + oy
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(canvas, f"MISS:{shared_cls.get(cid, cid)}", (x1, max(78, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
            # 再画预测框
            for cid, b, cf, is_tp in boxes_draw:
                x1, y1 = int(b[0] * scale) + ox, int(b[1] * scale) + oy
                x2, y2 = int(b[2] * scale) + ox, int(b[3] * scale) + oy
                col = cls_colors.get(cid, (0, 255, 0)) if is_tp else (0, 140, 255)
                tag = f"{shared_cls.get(cid, cid)} {cf:.2f}" if is_tp else f"FP:{shared_cls.get(cid, cid)} {cf:.2f}"
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
                cv2.putText(canvas, tag, (x1, min(target_h + 60, y2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return canvas

    saved_items = []
    for idx, rec in enumerate(selected, 1):
        im = cv2.imread(rec["img_path"])
        tp8, fp8, fn8, boxes8, miss8 = rec["v8"]
        tp11, fp11, fn11, boxes11, miss11 = rec["v11"]

        p_gt = _render_panel(
            im,
            f"[GT] {rec['ds']} #{idx}",
            f"Ground Truth: {rec['gt_count']} objects (Person/Helmet/Head)",
            rec["gts"],
            is_gt=True,
        )
        p_v8 = _render_panel(
            im,
            "YOLOv8m (20260806-99 @640)",
            f"TP={tp8} | MISS(FN)={fn8} | FalseAlarm(FP)={fp8}",
            boxes8,
            missed_draw=miss8,
        )
        p_11 = _render_panel(
            im,
            "YOLO11x (yolo11x_960_best @960)",
            f"TP={tp11} | MISS(FN)={fn11} | FalseAlarm(FP)={fp11}",
            boxes11,
            missed_draw=miss11,
        )
        sep = np.full((p_gt.shape[0], 6, 3), 200, dtype=np.uint8)
        merged = np.hstack([p_gt, sep, p_v8, sep, p_11])
        out_fn = f"diff_case_{idx:02d}_{rec['ds']}_{rec['stem'][:28]}.jpg"
        out_path = os.path.join(diff_dir, out_fn)
        cv2.imwrite(out_path, merged, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved_items.append({
            "case_id": idx,
            "dataset": rec["ds"],
            "image": rec["stem"],
            "gt_boxes": rec["gt_count"],
            "yolov8m_tp_fn_fp": [tp8, fn8, fp8],
            "yolo11x_tp_fn_fp": [tp11, fn11, fp11],
            "diff_image_path": out_path,
        })
        print(f"[DiffWall] Case #{idx}: {out_fn} | GT={rec['gt_count']} | "
              f"YOLOv8m(TP={tp8},FN={fn8},FP={fp8}) vs YOLO11x(TP={tp11},FN={fn11},FP={fp11})", flush=True)

    with open(os.path.join(diff_dir, "diff_wall_summary.json"), "w") as f:
        json.dump(saved_items, f, ensure_ascii=False, indent=2)
    out_q.put(saved_items)


def _backend_worker(spec, out_q):
    """评测单个算法在 PyTorch Eager (FP32/FP16) vs TorchScript C++ JIT (FP32/FP16) 下的延迟、吞吐与输出一致性。"""
    arch, label, weights, imgsz = spec
    import copy
    import numpy as np
    import torch
    from ultralytics import YOLO

    y = YOLO(weights)
    net_fp32 = copy.deepcopy(y.model).to("cuda:0").eval()
    if hasattr(net_fp32, "fuse"):
        net_fp32 = net_fp32.fuse(verbose=False)
    net_fp16 = copy.deepcopy(net_fp32).half().eval()

    x32_b1 = torch.rand((1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda:0")
    x16_b1 = x32_b1.half()
    x16_b16 = torch.rand((16, 3, imgsz, imgsz), dtype=torch.float16, device="cuda:0")

    with torch.inference_mode():
        ref_out = net_fp32(x32_b1)
        ref_pred = ref_out[0] if isinstance(ref_out, (tuple, list)) else ref_out

        # 构建 TorchScript JIT Trace 模型 (消除 Python 解释器逐层调用开销，特别是 280 层 YOLOv9e)
        class _PredOnlyWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                r = self.m(x)
                return r[0] if isinstance(r, (tuple, list)) else r

        ts_fp32 = torch.jit.trace(_PredOnlyWrapper(net_fp32), x32_b1, check_trace=False)
        ts_fp32 = torch.jit.optimize_for_inference(ts_fp32)
        ts_fp16 = torch.jit.trace(_PredOnlyWrapper(net_fp16), x16_b1, check_trace=False)
        ts_fp16 = torch.jit.optimize_for_inference(ts_fp16)

    def _bench_fn(fn, inp, bs, iters=40):
        with torch.inference_mode():
            for _ in range(10):
                out = fn(inp)
            torch.cuda.synchronize()
            lats = []
            for _ in range(iters):
                t0 = time.perf_counter()
                out = fn(inp)
                torch.cuda.synchronize()
                lats.append((time.perf_counter() - t0) * 1000.0 / bs)
        arr = np.array(lats)
        mean_ms = round(float(arr.mean()), 3)
        p99_ms = round(float(np.percentile(arr, 99)), 3)
        fps = round(1000.0 / mean_ms, 1)
        pred_t = out[0] if isinstance(out, (tuple, list)) else out
        return mean_ms, p99_ms, fps, pred_t

    eager32_ms, eager32_p99, eager32_fps, _ = _bench_fn(net_fp32, x32_b1, 1)
    eager16_ms, eager16_p99, eager16_fps, p_e16 = _bench_fn(net_fp16, x16_b1, 1)
    eager16_b16_ms, _, eager16_b16_fps, _ = _bench_fn(net_fp16, x16_b16, 16, iters=20)

    ts32_ms, ts32_p99, ts32_fps, p_ts32 = _bench_fn(ts_fp32, x32_b1, 1)
    ts16_ms, ts16_p99, ts16_fps, p_ts16 = _bench_fn(ts_fp16, x16_b1, 1)
    ts16_b16_ms, _, ts16_b16_fps, _ = _bench_fn(ts_fp16, x16_b16, 16, iters=20)

    diff_ts32 = float((ref_pred.float() - p_ts32.float()).abs().max().item())
    diff_ts16 = float((p_e16.float() - p_ts16.float()).abs().max().item())

    row = {
        "arch": arch,
        "label": label,
        "imgsz": imgsz,
        "eager_fp32_b1": {"mean_ms": eager32_ms, "p99_ms": eager32_p99, "fps": eager32_fps},
        "ts_jit_fp32_b1": {"mean_ms": ts32_ms, "p99_ms": ts32_p99, "fps": ts32_fps, "max_diff": round(diff_ts32, 6)},
        "eager_fp16_b1": {"mean_ms": eager16_ms, "p99_ms": eager16_p99, "fps": eager16_fps},
        "ts_jit_fp16_b1": {"mean_ms": ts16_ms, "p99_ms": ts16_p99, "fps": ts16_fps, "max_diff": round(diff_ts16, 6)},
        "eager_fp16_b16_fps": eager16_b16_fps,
        "ts_jit_fp16_b16_fps": ts16_b16_fps,
        "b1_fp16_jit_speedup": round(ts16_fps / max(0.1, eager16_fps), 2),
    }
    print(
        f"[Backend] {arch:<11} (@{imgsz}) | b1 FP16 Eager={eager16_fps:>5.1f} FPS -> TorchScript C++={ts16_fps:>5.1f} FPS "
        f"({row['b1_fp16_jit_speedup']}x) | b16 FP16 Eager={eager16_b16_fps:>5.1f} -> JIT={ts16_b16_fps:>5.1f} FPS",
        flush=True,
    )
    out_q.put(row)


def run_arch_matrix(out_dir):
    ctx = mp.get_context("spawn")
    rows = []
    for spec in ALL_ARCH_MODELS:
        q = ctx.Queue()
        p = ctx.Process(target=_arch_worker, args=(spec, q))
        p.start()
        r = _worker_result(p, q)
        rows.append(r)

    reg_by_key = {m["dataset_key"]: m for m in get_active_models()}
    ac_path = "/output/benchmark/accel/all_fast.json"
    ac_map = {}
    if os.path.exists(ac_path):
        for item in json.load(open(ac_path)):
            ac_map[item["dataset"]] = item

    for r in rows:
        ds = r["dataset"]
        reg_m = reg_by_key.get(ds, {})
        r["official_fps_str"] = reg_m.get("official_fps", f"{r['val_fp32_vs_fp16']['fp32']['val_fps']} FPS")
        r["fast_fps_str"] = reg_m.get("fast_fps", f"{r['val_fp32_vs_fp16']['fp16']['val_fps']} FPS")
        if ds in ac_map:
            ac = ac_map[ds]
            r["e2e_accel"] = {
                "baseline_b1_fps": ac.get("baseline", {}).get("fps"),
                "best_fps": ac.get("best_fps"),
                "best_tag": ac.get("best_tag"),
                "speedup": ac.get("speedup"),
                "b1_opt_fps": ac.get("batch1_fps"),
            }

    json_path = os.path.join(out_dir, "arch_matrix.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 算法架构 | 业务模型 | 分辨率 | 参数量(M) / GFLOPs | 纯GPU b16 FP16 FPS | 单帧 b1 FP16 Mean / P99(ms) | 标准集 mAP50 (FP32 / FP16) | 分尺度召回率 (小 / 中 / 大) | 误报率 FAR (背景FP框) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        g = r["pure_gpu"]
        vc = r["val_fp32_vs_fp16"]
        sr = r["diagnostics"]["scale_recall"]
        fp_d = r["diagnostics"]["false_positives"]
        s_str = f"{sr['small']:.3f}" if sr["small"] is not None else "—"
        m_str = f"{sr['medium']:.3f}" if sr["medium"] is not None else "—"
        l_str = f"{sr['large']:.3f}" if sr["large"] is not None else "—"
        lines.append(
            f"| **{r['arch']}** | {r['label']} | {r['imgsz']} | {r['params_m']:.2f}M / **{r['gflops_actual']:.1f}** | "
            f"**{g['fp16_b16']['fps']:.1f}** | {g['fp16_b1']['ms_per_img']:.2f} / {g['fp16_b1']['p99_ms']:.2f} | "
            f"{vc['fp32']['mAP50']:.4f} / **{vc['fp16']['mAP50']:.4f}** | "
            f"{s_str} / {m_str} / {l_str} | "
            f"{fp_d['false_alarm_ratio']:.4f} ({fp_d['bg_fp_boxes']}框/{fp_d['bg_images']}背景图) |"
        )
    md_path = os.path.join(out_dir, "arch_matrix.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 全量算法架构横向对比与工程诊断矩阵 ===")
    print("\n".join(lines))
    return rows


def run_head_to_head(out_dir, conf=0.25, iou=0.7):
    ctx = mp.get_context("spawn")
    datasets = [
        ("WorkclothesHelmet (150张, 6矿区)", "/data/benchmark/standard/WorkclothesHelmet/data.yaml"),
        ("PersonHelmet (309张, 156矿区)", "/data/benchmark/standard/PersonHelmet/data.yaml"),
    ]
    candidates = [
        ("YOLOv8m (20260806-99)", "/models/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt", 640),
        ("YOLOv8m (20260806-99)", "/models/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt", 960),
        ("YOLO11x (yolo11x_960_best)", "/models/custom/person/yolo11x_960_best.pt", 640),
        ("YOLO11x (yolo11x_960_best)", "/models/custom/person/yolo11x_960_best.pt", 960),
    ]
    shared_classes = [2, 3, 8]

    rows = []
    for ds_label, yaml_path in datasets:
        for cand_label, weights, imgsz in candidates:
            q = ctx.Queue()
            job = (ds_label, yaml_path, cand_label, weights, imgsz, shared_classes, conf, iou)
            p = ctx.Process(target=_h2h_worker, args=(job, q))
            p.start()
            r = _worker_result(p, q)
            rows.append(r)

    json_path = os.path.join(out_dir, "head_to_head_person_helmet.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 标准测试集 | 算法模型 | 推理分辨率 | 3类均值 mAP50 / mAP50-95 | Precision / Recall | Person AP50 | HeadWithHelmet AP50 | Head (未戴帽) AP50 | 单图前向耗时 (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pc = r["per_class_AP50"]
        lines.append(
            f"| {r['dataset']} | **{r['candidate']}** | {r['imgsz']} | "
            f"**{r['mAP50']:.4f}** / {r['mAP50_95']:.4f} | "
            f"{r['precision']:.4f} / {r['recall']:.4f} | "
            f"{pc.get('Person', 0.0):.4f} | {pc.get('HeadWithHelmet', 0.0):.4f} | "
            f"{pc.get('Head', 0.0):.4f} | {r['infer_ms']:.2f} |"
        )
    md_path = os.path.join(out_dir, "head_to_head_person_helmet.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 同数据集跨算法同卷对比 (YOLOv8m vs YOLO11x × 640 vs 960) ===")
    print("\n".join(lines))
    return rows


def run_diff_wall(out_dir):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_diff_wall_worker, args=(out_dir, q))
    p.start()
    items = _worker_result(p, q)
    return items


def run_backend_arena(out_dir):
    ctx = mp.get_context("spawn")
    specs = [
        ("YOLO11m", "风门开关 (YOLO11m)", "/models/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt", 640),
        ("YOLOv8m", "人睡觉 (YOLOv8m)", "/models/custom/20261030-99人睡觉yolov8m/20261030-99.pt", 640),
        ("YOLOv8l-OBB", "探水前上杆 (YOLOv8l-OBB)", "/models/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt", 640),
        ("YOLOv9e", "探水后上杆 (YOLOv9e)", "/models/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt", 960),
        ("YOLO11x", "人安全帽 (YOLO11x)", "/models/custom/person/yolo11x_960_best.pt", 960),
    ]
    rows = []
    for sp in specs:
        q = ctx.Queue()
        p = ctx.Process(target=_backend_worker, args=(sp, q))
        p.start()
        r = _worker_result(p, q)
        rows.append(r)

    with open(os.path.join(out_dir, "backend_arena.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 算法架构 | 代表模型 (@分辨率) | PyTorch Eager b1 FP16 (Mean/P99 ms, FPS) | TorchScript C++ JIT b1 FP16 (Mean/P99 ms, FPS) | 单帧 C++ JIT 加速比 | b16 FP16 Eager vs JIT FPS | 张量最大绝对偏差 max_diff |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e16 = r["eager_fp16_b1"]
        t16 = r["ts_jit_fp16_b1"]
        lines.append(
            f"| **{r['arch']}** | {r['label']} (@{r['imgsz']}) | "
            f"{e16['mean_ms']:.2f} / {e16['p99_ms']:.2f} ms (**{e16['fps']:.1f}** FPS) | "
            f"{t16['mean_ms']:.2f} / {t16['p99_ms']:.2f} ms (**{t16['fps']:.1f}** FPS) | "
            f"**{r['b1_fp16_jit_speedup']}x** | "
            f"{r['eager_fp16_b16_fps']:.1f} / **{r['ts_jit_fp16_b16_fps']:.1f}** | "
            f"`{t16['max_diff']}` |"
        )
    with open(os.path.join(out_dir, "backend_arena.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 跨推理执行后端对比 (PyTorch Eager vs TorchScript C++ JIT) ===")
    print("\n".join(lines))
    return rows


def run_custom_arena(args, out_dir):
    ctx = mp.get_context("spawn")
    classes = [int(x) for x in args.classes.split(",") if x.strip()] if args.classes else None
    rows = []
    for item in args.candidates.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            w, sz = item.rsplit(":", 1)
            imgsz = int(sz)
        else:
            w, imgsz = item, 640
        label = os.path.splitext(os.path.basename(w))[0]
        q = ctx.Queue()
        job = (args.data, args.data, label, w, imgsz, classes, args.conf, args.iou)
        p = ctx.Process(target=_h2h_worker, args=(job, q))
        p.start()
        r = _worker_result(p, q)
        rows.append(r)

    out_json = os.path.join(out_dir, "custom_arena.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _save_arch_matrix_files(rows, out_dir):
    json_path = os.path.join(out_dir, "arch_matrix.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 算法架构 | 业务模型 | 分辨率 | 参数量(M) / GFLOPs | 纯GPU b16 FP16 FPS | 单帧 b1 FP16 Mean / P99(ms) | 标准集 mAP50 (FP32 / FP16) | 分尺度召回率 (小 / 中 / 大) | 误报率 FAR (背景FP框) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        g = r["pure_gpu"]
        vc = r["val_fp32_vs_fp16"]
        sr = r["diagnostics"]["scale_recall"]
        fp_d = r["diagnostics"]["false_positives"]
        s_str = f"{sr['small']:.3f}" if sr["small"] is not None else "—"
        m_str = f"{sr['medium']:.3f}" if sr["medium"] is not None else "—"
        l_str = f"{sr['large']:.3f}" if sr["large"] is not None else "—"
        lines.append(
            f"| **{r['arch']}** | {r['label']} | {r['imgsz']} | {r['params_m']:.2f}M / **{r['gflops_actual']:.1f}** | "
            f"**{g['fp16_b16']['fps']:.1f}** | {g['fp16_b1']['ms_per_img']:.2f} / {g['fp16_b1']['p99_ms']:.2f} | "
            f"{vc['fp32']['mAP50']:.4f} / **{vc['fp16']['mAP50']:.4f}** | "
            f"{s_str} / {m_str} / {l_str} | "
            f"{fp_d['false_alarm_ratio']:.4f} ({fp_d['bg_fp_boxes']}框/{fp_d['bg_images']}背景图) |"
        )
    md_path = os.path.join(out_dir, "arch_matrix.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 全量算法架构横向对比与工程诊断矩阵 ===")
    print("\n".join(lines))


def _stream_worker(spec, out_q):
    """评测单个算法在 1/2/4/8/16 路 1080p 视频流并发解码+推理下的吞吐、单路FPS、P99延迟与显存曲线。"""
    arch, label, weights, imgsz, video_path, n_frames = spec
    from concurrent.futures import ThreadPoolExecutor
    import copy
    import cv2
    import numpy as np
    import torch
    from fast_infer import letterbox, unletterbox
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression

    device = torch.device("cuda:0")
    y = YOLO(weights)
    net = copy.deepcopy(y.model).to(device).eval()
    if hasattr(net, "fuse"):
        net = net.fuse(verbose=False)
    net = net.half()

    # 预热 GPU
    dummy = torch.zeros((4, 3, imgsz, imgsz), dtype=torch.float16, device=device)
    with torch.inference_mode():
        for _ in range(10):
            _ = net(dummy)
    torch.cuda.synchronize()

    def _decode_one_frame(cap):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        lb = letterbox(frame, imgsz)
        return lb, frame.shape[:2]

    ladder_results = []
    for n_streams in (1, 2, 4, 8, 16):
        # 1) 测定多路并发解码 + 动态微批聚合 (Micro-Batch = n_streams)
        caps = [cv2.VideoCapture(video_path) for _ in range(n_streams)]
        for idx, c in enumerate(caps):
            # 错开各路起始帧以模拟真实不同步监控流
            c.set(cv2.CAP_PROP_POS_FRAMES, (idx * 97) % 1200)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        step_lats = []

        with ThreadPoolExecutor(max_workers=n_streams) as pool:
            # 预热 5 步
            for _ in range(5):
                decoded = list(pool.map(_decode_one_frame, caps))
                arrs = [d[0] for d in decoded]
                arr = np.ascontiguousarray(np.stack(arrs)[..., ::-1].transpose(0, 3, 1, 2))
                t = torch.from_numpy(arr).pin_memory().to(device, dtype=torch.float16, non_blocking=True).div_(255.0)
                with torch.inference_mode():
                    o = net(t)
                    pred = o[0] if isinstance(o, (tuple, list)) else o
                    _ = non_max_suppression(pred, 0.25, 0.7, max_det=300)
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            for _ in range(n_frames):
                t0 = time.perf_counter()
                decoded = list(pool.map(_decode_one_frame, caps))
                arrs = [d[0] for d in decoded]
                shps = [d[1] for d in decoded]
                arr = np.ascontiguousarray(np.stack(arrs)[..., ::-1].transpose(0, 3, 1, 2))
                t = torch.from_numpy(arr).pin_memory().to(device, dtype=torch.float16, non_blocking=True).div_(255.0)
                with torch.inference_mode():
                    o = net(t)
                    pred = o[0] if isinstance(o, (tuple, list)) else o
                    dets = non_max_suppression(pred, 0.25, 0.7, max_det=300)
                    for d, shp in zip(dets, shps):
                        if len(d):
                            _ = unletterbox(d[:, :4], shp, imgsz)
                torch.cuda.synchronize()
                step_lats.append((time.perf_counter() - t0) * 1000.0)
            elapsed = time.perf_counter() - t_start

        for c in caps:
            c.release()

        arr_lat = np.array(step_lats)
        total_frames = n_streams * n_frames
        total_fps = round(total_frames / elapsed, 1)
        per_stream_fps = round(total_fps / n_streams, 1)
        mean_step_ms = round(float(arr_lat.mean()), 2)
        p99_step_ms = round(float(np.percentile(arr_lat, 99)), 2)
        vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

        # 2) 测定未聚合逐路串行提交 (per-stream b1) 的单路等效 FPS 作为对比
        caps_b1 = [cv2.VideoCapture(video_path) for _ in range(n_streams)]
        steps_b1 = max(20, n_frames // 2)
        with ThreadPoolExecutor(max_workers=n_streams) as pool:
            t0_b1 = time.perf_counter()
            for _ in range(steps_b1):
                decoded = list(pool.map(_decode_one_frame, caps_b1))
                with torch.inference_mode():
                    for lb, shp in decoded:
                        arr1 = np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1))[None]
                        t1 = torch.from_numpy(arr1).to(device, dtype=torch.float16).div_(255.0)
                        o1 = net(t1)
                        p1 = o1[0] if isinstance(o1, (tuple, list)) else o1
                        _ = non_max_suppression(p1, 0.25, 0.7, max_det=300)
                torch.cuda.synchronize()
            el_b1 = time.perf_counter() - t0_b1
        for c in caps_b1:
            c.release()
        b1_total_fps = round((n_streams * steps_b1) / el_b1, 1)
        b1_per_stream_fps = round(b1_total_fps / n_streams, 1)

        rec = {
            "streams": n_streams,
            "microbatch_per_stream_fps": per_stream_fps,
            "microbatch_total_fps": total_fps,
            "unbatched_per_stream_fps": b1_per_stream_fps,
            "unbatched_total_fps": b1_total_fps,
            "step_mean_ms": mean_step_ms,
            "step_p99_ms": p99_step_ms,
            "vram_mb": vram_mb,
            "realtime_25fps_ok": per_stream_fps >= 25.0,
        }
        ladder_results.append(rec)
        print(
            f"[Stream] {arch:<8} (@{imgsz}) | {n_streams:>2}路并发 | "
            f"微批聚合: 单路={per_stream_fps:>5.1f} FPS (总={total_fps:>5.1f} FPS, P99={p99_step_ms:>5.1f}ms) | "
            f"逐路b1: 单路={b1_per_stream_fps:>5.1f} FPS | 显存={vram_mb:>5.1f}MB | "
            f"25fps实时={'✅达标' if rec['realtime_25fps_ok'] else '❌超载'}",
            flush=True,
        )

    max_rt_streams = max(
        [r["streams"] for r in ladder_results if r["realtime_25fps_ok"]], default=0
    )
    out_q.put({
        "arch": arch,
        "label": label,
        "imgsz": imgsz,
        "max_realtime_25fps_streams": max_rt_streams,
        "ladder": ladder_results,
    })


def run_stream_arena(out_dir):
    ctx = mp.get_context("spawn")
    video_path = "/data/benchmark/videos/personsleep_1080p.mp4"
    specs = [
        ("YOLOv8m", "人睡觉 (YOLOv8m)", "/models/custom/20261030-99人睡觉yolov8m/20261030-99.pt", 640, video_path, 80),
        ("YOLO11m", "风门开关 (YOLO11m)", "/models/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt", 640, video_path, 80),
        ("YOLO11x", "人安全帽 (YOLO11x)", "/models/custom/person/yolo11x_960_best.pt", 960, video_path, 60),
    ]
    rows = []
    for sp in specs:
        q = ctx.Queue()
        p = ctx.Process(target=_stream_worker, args=(sp, q))
        p.start()
        r = _worker_result(p, q)
        rows.append(r)

    with open(os.path.join(out_dir, "stream_arena.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| 算法架构 (@分辨率) | 并发路数 (1080p) | 动态微批聚合 单路FPS (总吞吐FPS) | 逐路独立b1 单路FPS (总吞吐FPS) | 微批单步 Mean / P99 延迟(ms) | 显存峰值 (MB) | 25FPS 实时达标 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for ld in r["ladder"]:
            ok_str = "✅ 达标" if ld["realtime_25fps_ok"] else "❌ 超载 (<25fps)"
            lines.append(
                f"| **{r['arch']}** (@{r['imgsz']}) | **{ld['streams']} 路** | "
                f"**{ld['microbatch_per_stream_fps']:.1f}** ({ld['microbatch_total_fps']:.1f}) | "
                f"{ld['unbatched_per_stream_fps']:.1f} ({ld['unbatched_total_fps']:.1f}) | "
                f"{ld['step_mean_ms']:.1f} / {ld['step_p99_ms']:.1f} ms | "
                f"{ld['vram_mb']:.1f} MB | {ok_str} |"
            )
    with open(os.path.join(out_dir, "stream_arena.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== 多算法 × 1/2/4/8/16路 1080p 视频流并发阶梯压测结果 ===")
    print("\n".join(lines))
    return rows


def run_admit_model(args, out_dir):
    """单模型一键准入或增量重测：评测指定数据集/新权重并自动合并更新 arch_matrix.json 与 arch_matrix.md。"""
    target_ds = None
    for d in DATASETS:
        if args.dataset in (d["dataset_key"], d["name"]):
            target_ds = d
            break
    if target_ds is None:
        raise SystemExit(f"未在 registry.py 中找到数据集: {args.dataset}")

    weights = args.weights or target_ds["weights_container"]
    arch = args.arch or target_ds["arch"] or "CustomYOLO"
    imgsz = args.imgsz or target_ds["imgsz"] or 640
    task = args.task or target_ds["task"] or "detect"
    ds_key = target_ds["dataset_key"]
    ds_cn = target_ds["name"]
    label = f"{ds_cn} ({arch})"

    if not weights or not os.path.exists(weights):
        raise SystemExit(f"准入模型权重不存在: {weights}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    spec = (arch, ds_cn, label, weights, ds_key, imgsz, task)
    p = ctx.Process(target=_arch_worker, args=(spec, q))
    p.start()
    r = _worker_result(p, q)

    # 不将旧模型的历史吞吐量套用到新候选；验证速度不能冒充 FastInfer 实测。
    r["official_fps_str"] = f"{r['val_fp32_vs_fp16']['fp32']['val_fps']} FPS (val)"
    r["fast_fps_str"] = "—"

    json_path = os.path.join(out_dir, "arch_matrix.json")
    rows = []
    if os.path.exists(json_path):
        rows = json.load(open(json_path))
    replaced = False
    for idx, item in enumerate(rows):
        if item.get("dataset") == ds_key:
            rows[idx] = r
            replaced = True
            break
    if not replaced:
        rows.append(r)

    _save_arch_matrix_files(rows, out_dir)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "arch_matrix", "head_to_head", "diff_wall", "backend_arena", "stream_arena", "custom", "admit"],
    )
    ap.add_argument("--dataset", default="", help="admit 模式下的业务数据集名称或 dataset_key")
    ap.add_argument("--weights", default="", help="admit 模式下的新模型容器内权重路径 (可选)")
    ap.add_argument("--arch", default="", help="admit 模式下的算法架构名称 (可选)")
    ap.add_argument("--imgsz", type=int, default=0, help="admit 模式下的推理分辨率 (可选)")
    ap.add_argument("--task", default="", help="admit 模式下的任务类型 detect/obb (可选)")
    ap.add_argument("--data", default="", help="custom 模式下的 data.yaml 路径")
    ap.add_argument("--candidates", default="", help="custom 模式下的权重列表: w1.pt:640,w2.pt:960")
    ap.add_argument("--classes", default="", help="可选类别 ID 过滤，逗号分隔，如 2,3,8")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default="/output/benchmark/arena")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.mode == "custom":
        run_custom_arena(args, args.out)
        return
    if args.mode == "admit":
        run_admit_model(args, args.out)
        return
    if args.mode in ("all", "arch_matrix"):
        run_arch_matrix(args.out)
    if args.mode in ("all", "head_to_head"):
        run_head_to_head(args.out, args.conf, args.iou)
    if args.mode in ("all", "diff_wall"):
        run_diff_wall(args.out)
    if args.mode in ("all", "backend_arena"):
        run_backend_arena(args.out)
    if args.mode in ("all", "stream_arena"):
        run_stream_arena(args.out)


if __name__ == "__main__":
    main()
