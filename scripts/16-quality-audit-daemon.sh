#!/bin/bash
# 质量校验守护：确保 15-benchmark-quality-audit.py 持续运行直到全部汇总完成。
# 分片结果原子落盘，进程重启后已完成分片直接复用。
set -u

ROOT=/home/jykj/vision-deploy
LOG="$ROOT/output/dataset/quality-audit.log"
PIDFILE="$ROOT/output/dataset/quality-audit.pid"
DONE="$ROOT/output/dataset/quality-audit.done"
WORKERS="${WORKERS:-6}"

cd "$ROOT" || exit 1
rm -f "$DONE"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running: $(cat "$PIDFILE")"
    exit 0
fi

# setsid + 忽略挂断/停止信号，避免脱离终端后被 SIGTSTP 挂起
exec setsid nohup bash -c '
    trap "" HUP TSTP TTIN TTOU
    cd "$1" || exit 1
    ionice -c 3 nice -n 19 python3 -u scripts/15-benchmark-quality-audit.py --workers "$2" >> "$3" 2>&1
    code=$?
    echo "EXIT_CODE=$code" >> "$3"
    if [ "$code" -eq 0 ]; then
        date "+%F %T" > "$4"
    fi
' _ "$ROOT" "$WORKERS" "$LOG" "$DONE" > /dev/null 2>&1 < /dev/null &

echo $! > "$PIDFILE"
echo "started: $(cat "$PIDFILE") workers=$WORKERS"
