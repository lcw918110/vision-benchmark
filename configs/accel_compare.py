#!/usr/bin/env python3
"""业务模型加速对比：官方 ultralytics 路径 vs fast_infer 优化路径。

与旧版 bench_all_fast.py 的区别（旧版 3 次运行均空转数小时后被杀）：
  1. 全部配置在同一容器内跑完，不再为每个配置启动一次容器（省掉 35 次冷启动）
  2. 基线与优化路径使用**同一份** N 张样本，保证可比
  3. 基线不落盘可视化图（旧版 save=True 会给 2 万张验证集生成标注图，几十 GB）
  4. 基线计时同样采用 warmup + repeat，与优化路径口径一致
  5. 每个模型在**独立子进程**内评测：同进程连跑多个模型时，CUDA 上下文
     与显存碎片会让第二个模型 predict 直接 OOM（实测 5.86 GiB 申请失败）

口径：
  - 基线：官方 model.predict 路径（stream=True 避免 Results 常驻显存），batch=1，FP32
  - 优化：fast_infer（多线程解码 + pinned memory + 整批 H2D）
          FP16/FP32 × batch 16/8/1
  - 加速比 = 优化最佳 FPS / 基线 FPS
  - 视频流场景关心单路延迟，额外记录 batch=1 的最佳配置

用法（容器内）：
  python3 /configs/accel_compare.py [--models 关键字1,关键字2] [--limit 400] [--reps 3]
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import get_active_models  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

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


def _link_dir(paths):
    """把一批图片软链到临时目录，迫使 ultralytics 走 LoadImagesAndVideos。

    为什么必须这么做：ultralytics/data/build.py 的 check_source 对 list 与
    目录分派到不同加载器——

        elif isinstance(source, (list, tuple)):
            source = autocast_list(source)   # 预转 PIL/np
            from_img = True                  # -> LoadPilAndNumpy（单线程）
        目录 -> LoadImagesAndVideos（多线程）

    实测同一批 144 张、batch=1：目录源 26.2 FPS，list 源 18.9 FPS，差 39%。
    旧版拿优化路径（内部 list）去比基线目录源、或反过来，batch=1 的加速比
    完全不可信。统一走本函数生成目录，保证两条路径输入形式一致。
    """
    d = tempfile.mkdtemp(prefix="accel_")
    for i, p in enumerate(paths):
        link = os.path.join(d, f"{i:06d}{os.path.splitext(p)[1]}")
        try:
            os.symlink(p, link)
        except OSError:
            shutil.copy2(p, link)
    return d


def load_val_images(dataset, limit):
    """取基准样本，优先用标准测试集，回退到 val.txt。

    旧实现直接取 val.txt 前 limit 张：val.txt 按路径字典序排列，
    前 400 张高度集中在 1-2 个批次（实测 PersonSleep / PiDaiYuanBan
    的前 400 张全部来自单个批次），算出的吞吐无法代表跨场景表现。
    标准测试集由 scripts/08-build-standard-set.py 跨矿区分层抽样生成，
    取样顺序已按矿区间隔打散。
    """
    std_dir = f"/data/benchmark/standard/{dataset}/images"
    if os.path.isdir(std_dir):
        paths = sorted(os.path.join(std_dir, f) for f in os.listdir(std_dir)
                       if f.lower().endswith(IMG_EXT))
        if paths:
            return paths[:limit], "standard"

    lst = f"/data/benchmark/datasets/{dataset}/val.txt"
    if not os.path.exists(lst):
        return [], "none"
    paths = [ln.strip() for ln in open(lst)
             if ln.strip().lower().endswith(IMG_EXT)]
    return paths[:limit], "val.txt"


def bench_one_model(name, weights, dataset, imgsz, limit, warmup, reps, out_q):
    """在子进程内评测单个模型：官方基线 + fast_infer 六种组合。"""
    import gc

    import torch

    sys.path.insert(0, "/configs")
    from fast_infer import FastInfer
    from ultralytics import YOLO

    chunk = 50  # 官方路径每块图数，避免 Results 累积吃满显存

    def log(msg):
        print(msg, flush=True)

    paths, sample_src = load_val_images(dataset, limit)
    if not paths:
        out_q.put({"model": name, "error": f"样本缺失: {dataset}"})
        return

    log(f"\n=== {name} | 样本 {len(paths)} 张 ({sample_src}) | {dataset} ===")
    row = {"model": name, "dataset": dataset, "weights": weights,
           "sample_images": len(paths), "sample_source": sample_src,
           "imgsz": imgsz, "reps": reps, "chunk": chunk}

    # ---- 基线：官方路径，batch=1，FP32 ----
    try:
        model = YOLO(weights)

        def run():
            # 必须分块 + 显式回收：ultralytics 的 stream generator 在单次调用内
            # 仍会持有历史 Results（实测 400 张累积 20 GiB 后 OOM）。
            #
            # 关键：基线必须喂**目录**而非 list。ultralytics/data/build.py 的
            # check_source 对 list 走 autocast_list -> LoadPilAndNumpy（单线程），
            # 对目录走 LoadImagesAndVideos（多线程）。batch=1 下两者实测
            # 18.9 vs 26.2 FPS（差 39%）。旧版传 list 把基线压低，
            # 虚高了优化路径的加速比。这里用软链接目录保证走多线程加载器，
            # 且样本集与优化路径完全一致。
            for i in range(0, len(paths), chunk):
                src_dir = _link_dir(paths[i:i + chunk])
                try:
                    for _ in model.predict(source=src_dir, imgsz=imgsz,
                                           batch=1, half=False, device=0,
                                           stream=True, verbose=False):
                        pass
                finally:
                    shutil.rmtree(src_dir, ignore_errors=True)
                gc.collect()
                torch.cuda.empty_cache()

        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        lats = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
        dt = sum(lats) / len(lats)
        n = len(paths)
        base = {"images": n, "batch": 1, "half": False,
                "total_seconds": round(dt, 3),
                "ms_per_img": round(dt / n * 1000, 3),
                "fps": round(n / dt, 1)}
        del model
        row["baseline"] = base
        log(f"  官方基线 b1 FP32 : {base['fps']:>7.1f} FPS "
            f"({base['ms_per_img']:>6.2f} ms/img)")
    except Exception as e:
        log(f"  官方基线失败: {type(e).__name__}: {str(e)[:100]}")
        out_q.put({**row, "error": f"baseline {type(e).__name__}"})
        return

    # ---- 优化：fast_infer 各组合 ----
    best = None
    combos = []
    for half in (True, False):
        for bs in (16, 8, 1):
            tag = f"{'FP16' if half else 'FP32'}/b{bs}"
            eng = None
            dets = None
            try:
                # workers 交给 FastInfer 自选：batch=1 时它会自动降到 1，
                # 因为单图下线程池提交/同步开销大于并行解码收益。
                eng = FastInfer(weights, imgsz, bs, half)
                dets, t = eng.infer_paths(paths, warmup, reps)
            except Exception as e:
                log(f"  优化 {tag:<10} : 失败 {type(e).__name__}: {str(e)[:80]}")
                combos.append({"tag": tag, "error": type(e).__name__})
                continue
            finally:
                del dets, eng
                gc.collect()
                torch.cuda.empty_cache()
            combos.append({"tag": tag, "batch": bs, "half": half,
                           "fps": t["fps"], "ms_per_img": t["ms_per_img"]})
            log(f"  优化 {tag:<10} : {t['fps']:>7.1f} FPS "
                f"({t['ms_per_img']:>6.2f} ms/img)")
            if best is None or t["fps"] > best[1]:
                best = (tag, t["fps"], t["ms_per_img"])

    row["combos"] = combos
    if best:
        row["best_tag"], row["best_fps"], row["best_ms"] = best
        row["speedup"] = round(best[1] / base["fps"], 2)
        b1 = [c for c in combos if c.get("batch") == 1 and "fps" in c]
        if b1:
            b1best = max(b1, key=lambda c: c["fps"])
            row["batch1_best_tag"] = b1best["tag"]
            row["batch1_fps"] = b1best["fps"]
            row["batch1_ms"] = b1best["ms_per_img"]
        log(f"  >>> 最佳 {best[0]}: {best[1]:.1f} FPS = {row['speedup']}x")
    out_q.put(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="按名称关键字筛选，逗号分隔")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=0, help="默认 0 表示按模型自身规格 (640/960)")
    ap.add_argument("--out", default="/output/benchmark/accel")
    args = ap.parse_args()

    picks = [s.strip() for s in args.models.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    rows = []

    ctx = mp.get_context("spawn")
    for name, weights, dataset, def_imgsz in MODELS:
        if picks and not any(p in name for p in picks):
            continue
        if not os.path.exists(weights):
            print(f"[跳过] {name}: 权重缺失 {weights}", flush=True)
            continue

        imgsz = args.imgsz if args.imgsz > 0 else def_imgsz
        q = ctx.Queue()
        proc = ctx.Process(target=bench_one_model,
                           args=(name, weights, dataset, imgsz,
                                 args.limit, args.warmup, args.reps, q))
        proc.start()
        row = q.get()
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
        if row.get("error"):
            print(f"[跳过] {name}: {row['error']}", flush=True)
        rows.append(row)

    p = os.path.join(args.out, "all_fast.json")
    if picks and os.path.exists(p):
        try:
            old_rows = {r["model"]: r for r in json.load(open(p))}
            for r in rows:
                old_rows[r["model"]] = r
            rows = [old_rows[m[0]] for m in MODELS if m[0] in old_rows]
        except Exception:
            pass
    with open(p, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    md = ["| 模型 | 样本 | 官方基线 FPS (b1/FP32) | 优化最佳 FPS | 加速比 "
          "| 最佳配置 | b1 优化 FPS |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "best_fps" not in r:
            md.append(f"| {r['model']} | — | — | — | — | 失败 | — |")
            continue
        md.append(f"| {r['model']} | {r['sample_images']} | "
                  f"{r['baseline']['fps']:.1f} | {r['best_fps']:.1f} | "
                  f"**{r['speedup']}x** | {r['best_tag']} | "
                  f"{r.get('batch1_fps', 0):.1f} |")
    md_path = os.path.join(args.out, "all_fast.md")
    open(md_path, "w").write("\n".join(md) + "\n")
    print(f"\n汇总: {p}\n      {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
