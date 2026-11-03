#!/usr/bin/env python3
"""全自动同步脚本：读取 configs/registry.py 与 output/benchmark/arena/*.json 最新实测结果，
一键刷新 docs/benchmark-full-table.md（A~Z 共 26 列 × 45 行完整算法评测与工程选型台账）。

台账唯一产物为 docs/benchmark-full-table.md（A~I 列为原始业务数据，J~Z 列由本脚本刷新）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
from registry import get_dataset_map  # noqa: E402

ARCH_JSON = ROOT / "output/benchmark/arena/arch_matrix.json"
BACKEND_JSON = ROOT / "output/benchmark/arena/backend_arena.json"
TABLE_MD = ROOT / "docs/benchmark-full-table.md"

# J~Z 共 17 列定义：(列字母, 表头名称, 列宽, 是否按数据集纵向合并)
NEW_COLS = [
    ("J", "本项目权重地址(宿主机)", 52, True),
    ("K", "容器内权重路径", 46, True),
    ("L", "具体算法架构", 16, True),
    ("M", "默认推理分辨率", 15, True),
    ("N", "权重体积(MB)", 14, True),
    ("O", "参数量 / 计算量(GFLOPs)", 28, True),
    ("P", "标准测试集路径(张数)", 42, True),
    ("Q", "标准集精度 mAP50/50-95 (FP16/INT8)", 38, True),
    ("R", "标准集查准率P / 召回率R", 24, True),
    ("S", "单类AP50 / AP50-95(对应D列标签)", 28, False),
    ("T", "官方基线FPS(b1/FP32)", 20, True),
    ("U", "优化最佳FPS(配置/加速比)", 28, True),
    ("V", "纯GPU前向极限FPS(FP16/b16)", 24, True),
    ("W", "实测状态与备注", 44, True),
    ("X", "分尺度召回率(小/中/大目标)", 28, True),
    ("Y", "误报率FAR(背景图误报统计)", 28, True),
    ("Z", "单帧P99延迟 / C++静态图加速", 34, True),
]

BASE_HEADERS_A_TO_I = [
    "数据集名称",
    "标注方式",
    "数据集路径",
    "标签",
    "标签Yolo类别",
    "共: 1602511 张",
    "标注对象: 4670797 个",
    "yolo版本",
    "权重",
]


def load_live_arena_metrics():
    """加载 algo_arena.py 产出的最新 arch_matrix.json 与 backend_arena.json。"""
    arch_by_ds = {}
    if ARCH_JSON.exists():
        for item in json.loads(ARCH_JSON.read_text(encoding="utf-8")):
            arch_by_ds[item["dataset"]] = item
    backend_by_arch = {}
    if BACKEND_JSON.exists():
        for item in json.loads(BACKEND_JSON.read_text(encoding="utf-8")):
            backend_by_arch[item["arch"]] = item
    return arch_by_ds, backend_by_arch


def build_row_dict(ds_entry, arch_by_ds, backend_by_arch):
    """将 registry.py 与 algo_arena 最新实测 JSON 合并映射为台账 J~Z 列字段（实测 JSON 优先）。"""
    has_weight = ds_entry["weights_host"] is not None
    dkey = ds_entry["dataset_key"]
    arch = ds_entry["arch"]
    live = arch_by_ds.get(dkey, {})
    back = backend_by_arch.get(arch, {})

    if live:
        file_mb_str = f"{live['file_mb']} MB"
        imgsz = live["imgsz"]
        if imgsz != 640:
            params_str = f"{live['params_m']:.2f}M / {live['gflops_actual']:.1f} GFLOPs@{imgsz} ({live['gflops_640']:.1f}@640)"
        else:
            params_str = f"{live['params_m']:.2f}M / {live['gflops_actual']:.1f} GFLOPs@640"

        v32 = live.get("val_fp32_vs_fp16", {}).get("fp32", {})
        v16 = live.get("val_fp32_vs_fp16", {}).get("fp16", {})
        vi8 = live.get("val_fp32_vs_fp16", {}).get("int8", {})
        if vi8 and "int8_mAP50" in vi8:
            map_str = (
                f"{v32.get('mAP50', 0):.4f} / {v32.get('mAP50_95', 0):.4f} "
                f"(FP16:{v16.get('mAP50', 0):.4f} | INT8:{vi8.get('int8_mAP50', 0):.4f})"
            )
        else:
            map_str = f"{v32.get('mAP50', 0):.4f} / {v32.get('mAP50_95', 0):.4f} (FP16:{v16.get('mAP50', 0):.4f})"
        pr_str = f"{v32.get('precision', 0):.4f} / {v32.get('recall', 0):.4f}"

        per_cls_merged = dict(ds_entry.get("per_class", {}))
        for cid, cinfo in v32.get("per_class", {}).items():
            per_cls_merged[str(cid)] = (
                f"{cinfo['ap50']:.4f} / {cinfo['ap']:.4f} (P={cinfo['p']:.4f}, R={cinfo['r']:.4f})"
            )

        sr = live.get("diagnostics", {}).get("scale_recall", {})
        fp = live.get("diagnostics", {}).get("false_positives", {})
        g_b1 = live.get("pure_gpu", {}).get("fp16_b1", {})
        g_b16 = live.get("pure_gpu", {}).get("fp16_b16", {})
        s_v = f"{sr['small']:.3f}" if sr.get("small") is not None else "无样本"
        m_v = f"{sr['medium']:.3f}" if sr.get("medium") is not None else "无样本"
        l_v = f"{sr['large']:.3f}" if sr.get("large") is not None else "无样本"
        scale_str = f"小:{s_v} / 中:{m_v} / 大:{l_v}"
        far_pct = fp.get("false_alarm_ratio", 0.0) * 100.0
        far_str = f"{far_pct:.2f}% ({fp.get('bg_fp_boxes', 0)}框/{fp.get('bg_images', 0)}背景图)"
        p99_ms = g_b1.get("p99_ms", 0.0)
        pure_fps = f"{g_b16.get('fps', 0.0):.1f} FPS"
        if back:
            ts = back.get("ts_jit_fp16_b1", {})
            sp = back.get("b1_fp16_jit_speedup", 1.0)
            p99_jit_str = f"b1 P99={p99_ms:.1f}ms | C++ JIT b1: {ts.get('fps', 0):.1f} FPS ({sp}x)"
        else:
            p99_jit_str = f"b1 P99={p99_ms:.1f}ms"
    else:
        file_mb_str = ds_entry["weights_size_mb"]
        params_str = ds_entry["params_gflops"]
        map_str = ds_entry["mAP50_95"]
        pr_str = ds_entry["p_r"]
        per_cls_merged = ds_entry.get("per_class", {})
        scale_str = "—"
        far_str = "—"
        p99_jit_str = "—"
        pure_fps = ds_entry["pure_gpu_fps"]

    return {
        "J": ds_entry["weights_host"] if has_weight else "无",
        "K": ds_entry["weights_container"] if has_weight else "无",
        "L": ds_entry["arch"] if has_weight else "待交付权重",
        "M": str(ds_entry["imgsz"]) if has_weight else "—",
        "N": file_mb_str if has_weight else "—",
        "O": params_str if has_weight else "—",
        "P": f"{ds_entry['standard_dir_host']} ({ds_entry['num_images']}张)",
        "Q": map_str if has_weight else "—",
        "R": pr_str if has_weight else "—",
        "T": ds_entry["official_fps"],
        "U": ds_entry["fast_fps"],
        "V": pure_fps if has_weight else "—",
        "W": ds_entry["note"],
        "X": scale_str if has_weight else "—",
        "Y": far_str if has_weight else "—",
        "Z": p99_jit_str if has_weight else "—",
        "per_class": per_cls_merged,
    }


def parse_existing_base_rows():
    """从 docs/benchmark-full-table.md 读取行 2..45 的原始 A~I 列（前 9 列）数据。"""
    base_rows = {}
    if not TABLE_MD.exists():
        raise FileNotFoundError(f"未找到台账 Markdown 文件: {TABLE_MD}")
    for line in TABLE_MD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("| **"):
            continue
        # 按单元格切分（注意表格内可能包含形如 `FP16:0.9778 | INT8:0.9717` 的管道符，但 A~I 前 9 列不含管道符）
        parts = [p.strip() for p in line.strip("|").split("|")]
        rnum_str = parts[0].replace("*", "").strip()
        if not rnum_str.isdigit():
            continue
        rnum = int(rnum_str)
        # parts[0] 是行号，parts[1..9] 是 A~I 共 9 列
        base_rows[rnum] = parts[1:10]
    return base_rows


def main():
    arch_by_ds, backend_by_arch = load_live_arena_metrics()
    ds_map = {
        k: build_row_dict(v, arch_by_ds, backend_by_arch)
        for k, v in get_dataset_map().items()
    }
    base_rows = parse_existing_base_rows()

    headers = BASE_HEADERS_A_TO_I + [col[1] for col in NEW_COLS]
    lines = [
        "# 矿山视觉算法横向评测整表 Markdown 台账（A～Z 共 26 列 × 45 行）\n",
        "> **说明**：本文件为项目标准归档文档 `docs/benchmark-full-table.md`，由 `scripts/05-sync-benchmark-table.py` 从 `configs/registry.py` 与 `output/benchmark/arena/*.json` 自动同步生成。配套指标与工程约定说明见 [`docs/metrics-and-conventions.md`](./metrics-and-conventions.md)，标准测试集详情见 [`docs/dataset-standard-report.md`](./dataset-standard-report.md)。\n",
        "| 行号 | " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---"] * 26) + "|",
    ]

    for rnum in range(2, 46):
        a_to_i = list(base_rows[rnum])
        ds_name = a_to_i[0]
        cls_id = a_to_i[4]
        info = ds_map.get(ds_name)
        if info:
            # I 列统一写本地权重路径，不保留任何外部下载链接
            a_to_i[8] = info["J"]
            j_to_z = []
            for col_letter, _, _, _ in NEW_COLS:
                if col_letter == "S":
                    s_val = "—" if info["J"] == "无" else info["per_class"].get(cls_id, "—")
                    j_to_z.append(s_val)
                else:
                    j_to_z.append(info.get(col_letter, "—") or "—")
        else:
            j_to_z = ["—"] * len(NEW_COLS)

        row_vals = [v if v else "—" for v in (a_to_i + j_to_z)]
        lines.append(f"| **{rnum}** | " + " | ".join(row_vals) + " |")

    TABLE_MD.parent.mkdir(exist_ok=True)
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已从 registry.py + output/benchmark/arena/*.json 自动同步 26 列 × 45 行台账至: {TABLE_MD}")


if __name__ == "__main__":
    main()
