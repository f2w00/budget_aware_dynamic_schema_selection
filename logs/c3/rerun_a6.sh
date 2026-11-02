#!/usr/bin/env bash
# 重跑未完成的 A6 预算点（DB 连接中断导致的部分写入），直至每个 jsonl 达到 2000 条。
# 关键：llm_runner 对 --output 是追加写，必须先 rm 掉残留部分文件，否则条数会累积。
set -u
cd /root/work/papers/extraction || exit 1
PY=.venv/bin/python
IDS=data/performance/performance_ids_2000.json
OUT=benchmarks/runs/a6-a3-sweep/raw

METHODS=(budget-150 budget-300 budget-400)
MAX_TRIES=8

for B in "${METHODS[@]}"; do
  LBL="a6-$B"
  TARGET="$OUT/$LBL.jsonl"
  TRIES=0
  while :; do
    TRIES=$((TRIES+1))
    rm -f "$TARGET"   # 关键：删除残留，避免追加累积
    echo "[$(date +%T)] RETRY $LBL attempt=$TRIES" >> logs/a6_a3.log
    "$PY" llm_runner.py \
      --methods "$B" \
      --ids-file "$IDS" \
      --limit 2000 \
      --max-concurrency 16 --queue-size 16 --max-retries 2 \
      --run-label "$LBL" \
      --experiment-id "benchmark-a6-a3-$LBL-rerun-$TRIES" \
      --output "$TARGET" \
      >> logs/a6_a3.log 2>&1
    N=$(wc -l < "$TARGET" 2>/dev/null || echo 0)
    echo "[$(date +%T)] RETRY $LBL attempt=$TRIES -> $N records" >> logs/a6_a3.log
    if [ "$N" -eq 2000 ]; then
      echo "[$(date +%T)] OK    $LBL complete" >> logs/a6_a3.log
      break
    fi
    if [ "$TRIES" -ge "$MAX_TRIES" ]; then
      echo "[$(date +%T)] FAIL  $LBL after $TRIES attempts (N=$N)" >> logs/a6_a3.log
      break
    fi
  done
done
echo "[$(date +%T)] ALL A6 RERUNS DONE" >> logs/a6_a3.log