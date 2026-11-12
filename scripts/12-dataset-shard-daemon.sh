#!/usr/bin/env bash
# 全量数据集统计 —— 单进程连续分片推进器
#
# 设计要点（针对机械盘 + 训练任务长期占满 IO 的环境）：
#   1. 单进程连续扫描，不反复重启（反复重启会造成多进程争抢 IO 雪崩）。
#   2. 每个分片独立落盘，中断只损失当前分片。
#   3. 单分片超时仅记录并跳过，由后续轮次补扫，不阻塞整体进度。
#
# 用法：
#   bash scripts/12-dataset-shard-daemon.sh start
#   bash scripts/12-dataset-shard-daemon.sh stop
#   bash scripts/12-dataset-shard-daemon.sh status
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIDFILE="output/dataset/.shard-daemon.pid"
LOG="output/dataset/shard-scan.log"
SHARDER="scripts/11-dataset-shard-scan.py"
SCANNER="scripts/09-dataset-full-inventory.py"

mkdir -p output/dataset
touch "$LOG"

run_loop() {
  local round=0
  while true; do
    round=$((round + 1))
    echo "[$(date '+%F %T')] ===== 第 $round 轮 =====" >>"$LOG"

    # 1) 先把分片已齐但未合并的业务合并掉
    python3 "$SHARDER" --merge-all >>"$LOG" 2>&1

    # 2) 逐业务推进：分片未齐的继续扫，空目录直接判定
    python3 - "$SHARDER" "$SCANNER" <<'PY' >>"$LOG" 2>&1
import os, subprocess, sys
from pathlib import Path

sharder, scanner = sys.argv[1], sys.argv[2]
root = Path("/home/jykj/Dataset")
cache = Path("output/dataset/.cache")
shards_dir = Path("output/dataset/.shards")
parts = {"Hbb": root/"Hbb", "100/Hbb": root/"100/Hbb", "101/Hbb": root/"101/Hbb", "Obb": root/"Obb"}

progress = False
for part, pdir in parts.items():
    if not pdir.exists():
        continue
    for bd in sorted(p for p in pdir.iterdir() if p.is_dir()):
        key = f"{part.replace('/', '_')}__{bd.name}"
        if (cache / f"{key}.json").exists():
            continue
        try:
            names = sorted([e.name for e in os.scandir(bd) if e.is_dir(follow_symlinks=False)])
        except OSError:
            continue
        if not names:
            print(f"[empty] {part}/{bd.name}", flush=True)
            subprocess.run([sys.executable, scanner, "--resume", "--quick",
                            "--only", part, "--only-biz", bd.name])
            progress = True
            continue
        sd = shards_dir / key
        sd.mkdir(parents=True, exist_ok=True)
        pending = [s for s in names
                   if not (sd / f"{(s.strip().replace('/', '_') or '__blank__')}.json").exists()]
        if not pending:
            subprocess.run([sys.executable, sharder, "--merge", "--biz", f"{part}/{bd.name}"])
            progress = True
            continue
        # 一次只推进一个分片，避免长任务被饿死；单分片超时则跳过，下轮补扫
        print(f"[shard] {part}/{bd.name}/{pending[0]}", flush=True)
        try:
            subprocess.run([sys.executable, sharder, "--biz", f"{part}/{bd.name}", "--next"],
                           timeout=int(os.environ.get("SHARD_TIMEOUT", "120")))
        except subprocess.TimeoutExpired:
            print(f"[timeout] {part}/{bd.name}/{pending[0]}", flush=True)
        progress = True
        break
    else:
        continue
    break

sys.exit(0 if progress else 2)
PY
    rc=$?
    if [[ $rc -eq 2 ]]; then
      echo "[$(date '+%F %T')] 全部分片已完成，重建文档" >>"$LOG"
      python3 "$SCANNER" --md-only --quick >>"$LOG" 2>&1
      echo "[$(date '+%F %T')] 结束" >>"$LOG"
      break
    fi
    sleep 3
  done
}

case "${1:-start}" in
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "已在运行 (pid $(cat "$PIDFILE"))"; exit 0
    fi
    nohup bash -c "cd '$ROOT_DIR' && bash '$ROOT_DIR/scripts/12-dataset-shard-daemon.sh' _loop" \
      >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "已启动 (pid $(cat "$PIDFILE"))，日志: $LOG"
    ;;
  _loop)
    run_loop
    ;;
  stop)
    [[ -f "$PIDFILE" ]] && kill -9 "$(cat "$PIDFILE")" 2>/dev/null
    pkill -9 -f "11-dataset-shard-scan" 2>/dev/null
    pkill -9 -f "12-dataset-shard-daemon" 2>/dev/null
    rm -f "$PIDFILE"
    echo "已停止"
    ;;
  status)
    python3 "$SHARDER" --status 2>&1 | tail -n 6
    echo "---"
    echo "已合并业务: $(ls -1 output/dataset/.cache/*.json 2>/dev/null | wc -l) / 42"
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "守护进程: 运行中 (pid $(cat "$PIDFILE"))"
    else
      echo "守护进程: 未运行"
    fi
    ;;
  *)
    echo "用法: $0 {start|stop|status}"; exit 1
    ;;
esac
