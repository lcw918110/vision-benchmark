#!/usr/bin/env python3
"""全自动同步脚本：读取 configs/registry.py 与 output/benchmark/arena/*.json 最新实测结果，
一键刷新 docs/benchmark-full-table.md（A~Z 共 26 列 × 45 行完整算法评测与工程选型台账）。

支持在 Markdown 中使用标准 <table> + rowspan 对同一数据集共享列（A~C、F、H~R、T~Z）进行纵向单元格合并，
仅每行独立的列（行号、D标签、E标签Yolo类别、G标注对象数、S单类AP）逐行独立显示。
"""
import html
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

# 原始 44 行（行号 2..45）A~I 基础业务数据定义（保证无论表格格式如何变更均可幂等生成）
RAW_BASE_ROWS = {
    2: ["人安全帽", "目标检测", "/home/jykj/Dataset/Hbb/PersonHelmet", "人", "2", "309422", "790421", "yolo11x", "models/yolov8/custom/person/yolo11x_960_best.pt"],
    3: ["人安全帽", "目标检测", "/home/jykj/Dataset/Hbb/PersonHelmet", "戴安全帽", "3", "309422", "278960", "yolo11x", "models/yolov8/custom/person/yolo11x_960_best.pt"],
    4: ["人安全帽", "目标检测", "/home/jykj/Dataset/Hbb/PersonHelmet", "未戴安全帽", "8", "309422", "300719", "yolo11x", "models/yolov8/custom/person/yolo11x_960_best.pt"],
    5: ["人睡觉", "目标检测", "/home/jykj/Dataset/Hbb/PersonSleep", "人睡觉", "0", "31216", "6224", "yolov8m", "models/yolov8/custom/20261030-99人睡觉yolov8m/20261030-99.pt"],
    6: ["人睡觉", "目标检测", "/home/jykj/Dataset/Hbb/PersonSleep", "人", "2", "31216", "44514", "yolov8m", "models/yolov8/custom/20261030-99人睡觉yolov8m/20261030-99.pt"],
    7: ["人工作服", "目标检测", "/home/jykj/Dataset/Hbb/WorkclothesHelmet", "穿工作服的人", "0", "5158", "4730", "yolov8m", "models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt"],
    8: ["人工作服", "目标检测", "/home/jykj/Dataset/Hbb/WorkclothesHelmet", "未穿工作服", "2", "5158", "3935", "yolov8m", "models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt"],
    9: ["人工作服", "目标检测", "/home/jykj/Dataset/Hbb/WorkclothesHelmet", "戴安全帽", "3", "5158", "5944", "yolov8m", "models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt"],
    10: ["人工作服", "目标检测", "/home/jykj/Dataset/Hbb/WorkclothesHelmet", "未戴安全帽", "8", "5158", "2107", "yolov8m", "models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt"],
    11: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "人乘猴车", "0", "8386", "2467", "无", "无"],
    12: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "睡觉", "1", "8386", "828", "无", "无"],
    13: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "人", "2", "8386", "11571", "无", "无"],
    14: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "戴安全帽", "3", "8386", "8098", "无", "无"],
    15: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "超长异物", "4", "8386", "596", "无", "无"],
    16: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "脚蹬地", "5", "8386", "3832", "无", "无"],
    17: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "未戴安全帽", "8", "8386", "675", "无", "无"],
    18: ["人乘猴车", "目标检测", "/home/jykj/Dataset/Hbb/RenChengHouChe", "打电话", "9", "8386", "584", "无", "无"],
    19: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "无煤", "0", "166048", "62406", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    20: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "有煤", "1", "166048", "96893", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    21: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "水煤", "2", "166048", "181", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    22: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "托辊", "3", "166048", "806412", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    23: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "大块煤", "4", "166048", "53677", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    24: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "异物", "5", "166048", "36796", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    25: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "未堆煤", "6", "166048", "4899", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    26: ["皮带原版", "目标检测", "/home/jykj/Dataset/Hbb/PiDaiYuanBan", "堆煤", "7", "166048", "10159", "yolov11m", "models/yolov8/custom/belt/yolov11m_belt.pt"],
    27: ["护帮板", "目标检测", "/home/jykj/Dataset/Hbb/HuBangBan", "人", "2", "14121", "1525", "无", "无"],
    28: ["护帮板", "目标检测", "/home/jykj/Dataset/Hbb/HuBangBan", "护帮板", "3", "14121", "43956", "无", "无"],
    29: ["护帮板", "目标检测", "/home/jykj/Dataset/Hbb/HuBangBan", "滚筒", "4", "14121", "2717", "无", "无"],
    30: ["探水后上杆钻机", "目标检测", "/home/jykj/Dataset/Hbb/TanShuiHouShangGan", "机器", "0", "704199", "655326", "yolov9", "models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt"],
    31: ["探水后上杆钻机", "目标检测", "/home/jykj/Dataset/Hbb/TanShuiHouShangGan", "堵头", "1", "704199", "235126", "yolov9", "models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt"],
    32: ["探水后上杆钻机", "目标检测", "/home/jykj/Dataset/Hbb/TanShuiHouShangGan", "机尾", "2", "704199", "581444", "yolov9", "models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt"],
    33: ["风门开关", "目标检测", "/home/jykj/Dataset/Hbb/FengMenKaiGuan", "风门关", "0", "19409", "11270", "yolo11", "models/yolov8/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt"],
    34: ["风门开关", "目标检测", "/home/jykj/Dataset/Hbb/FengMenKaiGuan", "风门开", "1", "19409", "6918", "yolo11", "models/yolov8/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt"],
    35: ["传感器", "目标检测", "/home/jykj/Dataset/Hbb/ChuanGanQi", "传感器", "0", "2299", "2602", "yolo11", "models/yolov8/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt"],
    36: ["传感器", "目标检测", "/home/jykj/Dataset/Hbb/ChuanGanQi", "传感器进气口", "1", "2299", "2072", "yolo11", "models/yolov8/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt"],
    37: ["传感器", "目标检测", "/home/jykj/Dataset/Hbb/ChuanGanQi", "传感器架子", "2", "2299", "1517", "yolo11", "models/yolov8/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt"],
    38: ["卡车空满载", "目标检测", "/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong", "卡车空载", "0", "77074", "28203", "无", "无"],
    39: ["卡车空满载", "目标检测", "/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong", "卡车满载", "1", "77074", "952", "无", "无"],
    40: ["卡车空满载", "目标检测", "/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong", "车辆车斗遮盖", "3", "77074", "19786", "无", "无"],
    41: ["卡车空满载", "目标检测", "/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong", "块煤", "6", "77074", "5349", "无", "无"],
    42: ["卡车空满载", "目标检测", "/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong", "沫煤", "7", "77074", "6106", "无", "无"],
    43: ["探水前上杆", "目标旋转", "/home/jykj/Dataset/Obb/TS-Front_V2", "无杆", "0", "265179", "108087", "yolov8", "models/yolov8/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt"],
    44: ["探水前上杆", "目标旋转", "/home/jykj/Dataset/Obb/TS-Front_V2", "有杆", "1", "265179", "157463", "yolov8", "models/yolov8/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt"],
    45: ["探水前上杆", "目标旋转", "/home/jykj/Dataset/Obb/TS-Front_V2", "机器", "2", "265179", "262750", "yolov8", "models/yolov8/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt"],
}

# 27 个输出列（0=行号, 1..9=A..I, 10..26=J..Z）中哪些列按数据集纵向合并：
# 不合并的列索引：0(行号), 4(D:标签), 5(E:标签Yolo类别), 7(G:标注对象数), 19(S:单类AP50/AP50-95)
UNMERGED_COL_INDICES = {0, 4, 5, 7, 19}


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
    # 旧矩阵没有权重摘要，无法证明与注册表模型一致；禁止将其指标混入台账。
    # 新协议结果单独保存，待支持按明确的模型/运行 ID 选择后再接入。
    live = {}
    back = {}

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


def main():
    # 历史结果无法验证权重身份，不载入也不混合。
    arch_by_ds, backend_by_arch = {}, {}
    ds_map = {
        k: build_row_dict(v, arch_by_ds, backend_by_arch)
        for k, v in get_dataset_map().items()
    }

    headers = ["行号"] + BASE_HEADERS_A_TO_I + [col[1] for col in NEW_COLS]

    # 先构建 44 行（rnum 2..45）的完整 27 列矩阵
    all_rows = []
    for rnum in range(2, 46):
        a_to_i = list(RAW_BASE_ROWS[rnum])
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

        row_vals = [str(rnum)] + [v if v else "—" for v in (a_to_i + j_to_z)]
        all_rows.append(row_vals)

    # 按数据集名称（col 1）计算连续行的 rowspan 区间
    n_rows = len(all_rows)
    group_span = [0] * n_rows
    i = 0
    while i < n_rows:
        ds_name = all_rows[i][1]
        j = i + 1
        while j < n_rows and all_rows[j][1] == ds_name:
            j += 1
        group_span[i] = j - i
        i = j

    lines = [
        "# 矿山视觉算法横向评测整表台账（A～Z 共 26 列 × 45 行）\n",
        "> **说明**：本文件为项目标准归档文档 `docs/benchmark-full-table.md`，由 `scripts/05-sync-benchmark-table.py` 从 `configs/registry.py` 的历史数据生成；未验证权重身份的旧评测结果不会自动混入，新协议结果另行保存。同一业务数据集共享的列已自动纵向合并（`rowspan`），仅按标签独立的列（行号、标签、类别ID、标注对象数、单类AP）逐行展开。配套指标规范见 [`docs/metrics-and-conventions.md`](./metrics-and-conventions.md)，标准测试集详情见 [`docs/dataset-standard-report.md`](./dataset-standard-report.md)。\n",
        "<table>",
        "  <thead>",
        "    <tr>",
    ]
    for h in headers:
        lines.append(f"      <th>{html.escape(h)}</th>")
    lines.extend([
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ])

    for idx, row in enumerate(all_rows):
        lines.append("    <tr>")
        span = group_span[idx]
        for col_idx, val in enumerate(row):
            esc_val = f"<strong>{html.escape(val)}</strong>" if col_idx in (0, 1) else html.escape(val)
            if col_idx in UNMERGED_COL_INDICES:
                lines.append(f"      <td>{esc_val}</td>")
            else:
                if span > 1:
                    lines.append(f'      <td rowspan="{span}">{esc_val}</td>')
                elif span == 1:
                    lines.append(f"      <td>{esc_val}</td>")
                # span == 0 表示被上方单元格合并，不输出 <td>
        lines.append("    </tr>")

    lines.extend([
        "  </tbody>",
        "</table>",
    ])

    TABLE_MD.parent.mkdir(exist_ok=True)
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已从 registry.py 历史数据生成台账（未混入旧评测结果）: {TABLE_MD}")


if __name__ == "__main__":
    main()
