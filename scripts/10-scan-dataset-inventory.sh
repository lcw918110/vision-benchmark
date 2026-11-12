#!/usr/bin/env bash
# 全量数据集统计 —— 后台守护式增量扫描
#
# 背景：宿主机为机械盘（ST4000LM024 ×2，ROTA=1），训练任务长期占满 IO
# （/proc/pressure/io full avg300 常年在 70%~85%），单进程全量扫描会被饿死。
# 本脚本以「逐业务单次扫描 + 超时重试」的方式推进，每个业务结果落盘到
# output/dataset/.cache/，因此任意时刻中断都不会丢失已完成进度。
#
# 用法：
#   bash scripts/10-scan-dataset-inventory.sh          # 后台推进全部未完成业务
#   bash scripts/10-scan-dataset-inventory.sh --status # 只看进度
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="output/dataset/.cache"
SCANNER="scripts/09-dataset-full-inventory.py"
LOG="output/dataset/scan.log"
ATTEMPTS="${ATTEMPTS:-6}"
TIMEOUT="${TIMEOUT:-1200}"

mkdir -p "$CACHE_DIR"
touch "$LOG"

if [[ "${1:-}" == "--status" ]]; then
  total=$(python3 - <<'PY'
import os
from pathlib import Path
root = Path("/home/jykj/Dataset")
parts = {"Hbb": root/"Hbb", "100/Hbb": root/"100/Hbb", "101/Hbb": root/"101/Hbb", "Obb": root/"Obb"}
n = 0
for p in parts.values():
    if p.exists():
        n += sum(1 for x in p.iterdir() if x.is_dir())
print(n)
PY
)
  done=$(ls -1 "$CACHE_DIR"/*.json 2>/dev/null | wc -l)
  echo "已完成 $done / $total 个业务数据集"
  echo "--- 待完成 ---"
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("/home/jykj/Dataset")
cache = Path("output/dataset/.cache")
parts = {"Hbb": root/"Hbb", "100/Hbb": root/"100/Hbb", "101/Hbb": root/"101/Hbb", "Obb": root/"Obb"}
for part, pdir in parts.items():
    if not pdir.exists():
        continue
    for bd in sorted(p for p in pdir.iterdir() if p.is_dir()):
        f = cache / f"{part.replace('/', '_')}__{bd.name}.json"
        if not f.exists():
            print(f"  {part}/{bd.name}")
PY
  exit 0
fi

echo "[$(date '+%F %T')] 守护扫描启动 (attempts=$ATTEMPTS timeout=${TIMEOUT}s)" | tee -a "$LOG"

for attempt in $(seq 1 "$ATTEMPTS"); do
  pending=$(python3 - <<'PY'
import os
from pathlib import Path
root = Path("/home/jykj/Dataset")
cache = Path("output/dataset/.cache")
parts = {"Hbb": root/"Hbb", "100/Hbb": root/"100/Hbb", "101/Hbb": root/"101/Hbb", "Obb": root/"Obb"}
out = []
for part, pdir in parts.items():
    if not pdir.exists():
        continue
    for bd in sorted(p for p in pdir.iterdir() if p.is_dir()):
        if not (cache / f"{part.replace('/', '_')}__{bd.name}.json").exists():
            out.append(f"{part}/{bd.name}")
print("\n".join(out))
PY
)
  [[ -z "$pending" ]] && { echo "[$(date '+%F %T')] 全部完成" | tee -a "$LOG"; break; }

  echo "[$(date '+%F %T')] 第 $attempt 轮，待扫描 $(echo "$pending" | wc -l) 个" | tee -a "$LOG"

  while IFS= read -r item; do
    [[ -z "$item" ]] && continue
    part="${item%%/*}"
    biz="${item#*/}"
    echo "[$(date '+%F %T')] -> $item" | tee -a "$LOG"
    timeout "$TIMEOUT" python3 "$SCANNER" --resume --quick --only "$part" --only-biz "$biz" >>"$LOG" 2>&1
    rc=$?
    if [[ $rc -eq 124 ]]; then
      echo "[$(date '+%F %T')] !! $item 超时(${TIMEOUT}s)，跳过待重试" | tee -a "$LOG"
    elif [[ $rc -ne 0 ]]; then
      echo "[$(date '+%F %T')] !! $item 失败 rc=$rc" | tee -a "$LOG"
    fi
  done <<< "$pending"

  sleep 30
done

echo "[$(date '+%F %T')] 重建文档" | tee -a "$LOG"
python3 "$SCANNER" --md-only --quick >>"$LOG" 2>&1
echo "[$(date '+%F %T')] 守护扫描结束" | tee -a "$LOG"
