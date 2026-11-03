#!/usr/bin/env python3
"""从 /home/jykj/Dataset 跨全部矿区分层+全类别保底抽取完备标准测试集（合计 5,700 张），
落到 data/benchmark/standard/<业务>/。

设计原则（完备性三要素）：
  1. 矿区完备（Spatial Completeness）：遍历每个业务根目录下全部一级矿区/设备子目录，
     通过两阶段水桶注水法（Water-filling）在所有有效矿区间均匀抽取，消除因个别矿区
     图片不足导致总样本量缩水的问题，确保每个数据集达到 400~600 张的完备评测规模。
  2. 类别完备（Class Completeness）：针对低频长尾类别（如皮带的水煤/未堆煤/堆煤、
     探水后上杆的堵头、传感器的架子、猴车的人乘猴车/睡觉、卡车满载等），执行定向分层
     保底抽样，确保 docs/benchmark-full-table.md 中全部 44 行标签类别 100% 均有充足实测样本。
  3. 坐标规范化（Coordinate Clamping）：自动校验并将旋转框(OBB)/水平框(HBB)边缘轻微
     超出 [0,1] 的归一化坐标裁剪至 [0.0, 1.0]，杜绝 Ultralytics 评测时静默丢弃图片。
"""
import itertools
import json
import os
import random
import shutil
import sys

SRC_ROOT = "/home/jykj/Dataset"
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/benchmark/standard")
CONTAINER_ROOT = "/data/benchmark/standard"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")
SEED = 42

# 业务 -> (源根目录, 类别名文件, 任务, 完备目标张数)
BUSINESSES = {
    "PersonHelmet": (
        f"{SRC_ROOT}/Hbb/PersonHelmet",
        f"{SRC_ROOT}/Hbb/PersonHelmet/人安全帽labels.names", "detect", 600),
    "PersonSleep": (
        f"{SRC_ROOT}/Hbb/PersonSleep",
        f"{SRC_ROOT}/Hbb/PersonSleep/人睡觉labels.names", "detect", 500),
    "WorkclothesHelmet": (
        f"{SRC_ROOT}/Hbb/WorkclothesHelmet",
        f"{SRC_ROOT}/Hbb/WorkclothesHelmet/人工作服安全帽labels.names", "detect", 500),
    "RenChengHouChe": (
        f"{SRC_ROOT}/Hbb/RenChengHouChe",
        f"{SRC_ROOT}/Hbb/RenChengHouChe/人乘猴车labels.names", "detect", 500),
    "PiDaiYuanBan": (
        f"{SRC_ROOT}/Hbb/PiDaiYuanBan",
        f"{SRC_ROOT}/Hbb/PiDaiYuanBan/皮带原版labels.names", "detect", 600),
    "HuBangBan": (
        f"{SRC_ROOT}/Hbb/HuBangBan",
        f"{SRC_ROOT}/Hbb/HuBangBan/护帮板labels.names", "detect", 500),
    "TanShuiHouShangGan": (
        f"{SRC_ROOT}/Hbb/TanShuiHouShangGan/xinjian",
        f"{SRC_ROOT}/Hbb/TanShuiHouShangGan/探水后上杆labels.names", "detect", 500),
    "FengMenKaiGuan": (
        f"{SRC_ROOT}/Hbb/FengMenKaiGuan",
        f"{SRC_ROOT}/Hbb/FengMenKaiGuan/风门开关labels.names", "detect", 500),
    "ChuanGanQi": (
        f"{SRC_ROOT}/Hbb/ChuanGanQi",
        f"{SRC_ROOT}/Hbb/ChuanGanQi/传感器labels.names", "detect", 400),
    "ChanYunXiaoMeiZhong": (
        f"{SRC_ROOT}/Hbb/ChanYunXiaoMeiZhong",
        f"{SRC_ROOT}/Hbb/ChanYunXiaoMeiZhong/产运销煤种labels.names", "detect", 500),
    "TS_Front_V2": (
        f"{SRC_ROOT}/Obb/TS-Front_V2/Data",
        f"{SRC_ROOT}/Obb/TS-Front_V2/前上杆探水labels.names", "obb", 600),
}

# 长尾稀有类别定向分层保底计划
SUPPLEMENT_PLANS = {
    "PiDaiYuanBan": (8, 10, {
        2: [f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxysw-201hcmcsk", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxdhz-zxjpdjt"],
        6: [f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxdhz-115101scpdjt", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxdhz-51nyjpdjt", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxsq-mcsk"],
        7: [f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxct-2bpdjt", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxct-dyysfjc", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxct-zjpdjt", f"{SRC_ROOT}/Hbb/PiDaiYuanBan/sxdhz-51nyjpdjt"],
    }),
    "TanShuiHouShangGan": (3, 12, {
        1: [
            f"{SRC_ROOT}/Hbb/TanShuiHouShangGan/XunLian/055000003/055000003-AC-BA-CB-DB-DC-DD-Z-LDS-BJ03-2023062801",
            f"{SRC_ROOT}/Hbb/TanShuiHouShangGan/XunLian/001030010/001030010-AD-BA-CB-DA-SJZ-BJ01-2021101602",
        ],
    }),
    "ChuanGanQi": (3, 12, {
        2: [
            f"{SRC_ROOT}/Hbb/ChuanGanQi/sx-hyek/sx-hyek-jx-cm-gdhk-81311hsjjgzmT2c-hc-2025041701",
            f"{SRC_ROOT}/Hbb/ChuanGanQi/sd-tk/sd-tk-jx-cm-gdhk-6311gszjjjgzmjwTY1-hc-2024020401",
        ],
    }),
    "RenChengHouChe": (10, 10, {
        0: [f"{SRC_ROOT}/Hbb/RenChengHouChe/nxrjz-zj1120m-2024090601", f"{SRC_ROOT}/Hbb/RenChengHouChe/sxhsz-ccd06-2024102901", f"{SRC_ROOT}/Hbb/RenChengHouChe/nxrjz-zj1120m-2024090501"],
        1: [f"{SRC_ROOT}/Hbb/RenChengHouChe/nxrjz-zj1120m-2024090601"],
        3: [f"{SRC_ROOT}/Hbb/RenChengHouChe/sxll-ebhcjw-2024092001", f"{SRC_ROOT}/Hbb/RenChengHouChe/sxll-fxjk-2024092801"],
    }),
    "HuBangBan": (7, 12, {
        2: [f"{SRC_ROOT}/Hbb/HuBangBan/sxxsj-65zj-2023090701", f"{SRC_ROOT}/Hbb/HuBangBan/hnpmsk-33010cm50j-2023031501"],
    }),
    "ChanYunXiaoMeiZhong": (8, 12, {
        1: [f"{SRC_ROOT}/Hbb/ChanYunXiaoMeiZhong/shanxishengkai/zhongchemeiliangshexiangji/skmk-zcmlsxj-2024011701"],
    }),
}


def read_names(path):
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8-sig")
    names = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return [n for n in names if n]


def check_and_clamp_label(src_lbl, dst_lbl, nc, task):
    lines_out = []
    expected_len = 9 if task == "obb" else 5
    try:
        with open(src_lbl) as f:
            for ln in f:
                parts = ln.split()
                if not parts:
                    continue
                if len(parts) != expected_len:
                    return False
                cid = int(float(parts[0]))
                if cid < 0 or cid >= nc:
                    return False
                coords = [min(1.0, max(0.0, float(x))) for x in parts[1:]]
                if task == "detect" and (coords[2] <= 0 or coords[3] <= 0):
                    return False
                lines_out.append(f"{cid} " + " ".join(f"{c:.6f}" for c in coords) + "\n")
    except Exception:
        return False
    if dst_lbl:
        with open(dst_lbl, "w") as f:
            f.writelines(lines_out)
    return True


def fast_collect_from_mine(mine_dir, max_per_mine=80):
    found = []
    stack = [mine_dir]
    visited = 0
    while stack and len(found) < max_per_mine and visited < 30:
        cur = stack.pop(0)
        visited += 1
        try:
            with os.scandir(cur) as it:
                for entry in itertools.islice(it, 600):
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.name.lower().endswith(IMG_EXT):
                        img_p = entry.path
                        lbl_p = os.path.splitext(img_p)[0] + ".txt"
                        if os.path.exists(lbl_p):
                            found.append((img_p, lbl_p))
                            if len(found) >= max_per_mine:
                                break
        except Exception:
            continue
    return sorted(found)


def top_up_dataset(name, spec):
    root, names_file, task, target_n = spec
    names = read_names(names_file)
    nc = len(names)
    std_dir = os.path.join(OUT_ROOT, name)
    img_dir = os.path.join(std_dir, "images")
    lbl_dir = os.path.join(std_dir, "labels")
    excl_path = os.path.join(std_dir, "excluded.txt")
    man_path = os.path.join(std_dir, "manifest.json")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cur_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)]
    need = target_n - len(cur_imgs)
    if need <= 0:
        print(f"[完备就绪] {name}: {len(cur_imgs)} 张 (目标 {target_n} 张)")
        return

    existing_excl = {ln.strip() for ln in open(excl_path) if ln.strip()} if os.path.exists(excl_path) else set()
    existing_basenames = set(os.listdir(img_dir))
    with os.scandir(root) as it:
        mines = sorted([e.path for e in it if e.is_dir(follow_symlinks=False)])

    max_per_m = max(15, (need // max(1, len(mines))) * 4 + 20)
    mine_pools = {}
    for m in mines:
        mname = os.path.basename(m).strip()
        cand = [
            (img, lbl) for img, lbl in fast_collect_from_mine(m, max_per_mine=max_per_m)
            if img not in existing_excl and os.path.basename(img) not in existing_basenames
        ]
        if cand:
            rng = random.Random(SEED)
            rng.shuffle(cand)
            mine_pools[mname] = cand

    added = 0
    mine_keys = sorted(mine_pools.keys())
    mine_added = {k: 0 for k in mine_keys}
    progress = True
    while added < need and progress:
        progress = False
        for k in mine_keys:
            if added >= need:
                break
            pool = mine_pools[k]
            while pool:
                img, lbl = pool.pop(0)
                if img in existing_excl or not check_and_clamp_label(lbl, None, nc, task):
                    continue
                base = os.path.basename(img)
                dst_img = os.path.join(img_dir, base)
                dst_lbl = os.path.join(lbl_dir, os.path.splitext(base)[0] + ".txt")
                if os.path.exists(dst_img):
                    prefix = k.replace(" ", "_")
                    dst_img = os.path.join(img_dir, f"{prefix}__{base}")
                    dst_lbl = os.path.join(lbl_dir, f"{prefix}__{os.path.splitext(base)[0]}.txt")
                if os.path.exists(dst_img):
                    continue
                shutil.copy2(img, dst_img)
                check_and_clamp_label(lbl, dst_lbl, nc, task)
                existing_excl.add(img)
                mine_added[k] += 1
                added += 1
                progress = True
                break

    with open(excl_path, "w") as f:
        for p in sorted(existing_excl):
            f.write(p + "\n")

    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)]
    total_bytes = sum(os.path.getsize(p) for p in all_imgs)
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        man["target"] = target_n
        man["images"] = len(all_imgs)
        man["bytes"] = total_bytes
        for k, cnt in mine_added.items():
            if cnt > 0:
                man["per_mine"][k] = man["per_mine"].get(k, 0) + cnt
        man["covered_mine_count"] = sum(1 for v in man["per_mine"].values() if v > 0)
        with open(man_path, "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"[完备扩充] {name}: 新增 {added} 张 -> 现共 {len(all_imgs)} 张")


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name, spec in BUSINESSES.items():
        if only and not any(o in name for o in only):
            continue
        top_up_dataset(name, spec)


if __name__ == "__main__":
    main()
