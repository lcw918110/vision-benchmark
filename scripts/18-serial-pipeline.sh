#!/bin/bash
# 数据集同步入口（只管传输）。
#
# 与质量审计的分工：审计由另一个会话负责，本脚本不再等待它。
# 两者都需要读同一块机械盘，并行时传输约 18 MB/s，单独跑 40-116 MB/s；
# 如果希望传输更快，可先确认审计空闲后再启动本脚本。
set -u

ROOT=/home/jykj/vision-deploy
LOG="$ROOT/output/transfer/run.log"
PIDFILE="$ROOT/output/transfer/sync.pid"

cd "$ROOT" || exit 1
mkdir -p output/transfer

echo "[$(date '+%F %T')] 开始同步数据集" >> "$LOG"
exec python3 -u scripts/17-sync-dataset.py >> "$LOG" 2>&1