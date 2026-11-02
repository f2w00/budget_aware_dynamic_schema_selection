#!/usr/bin/env bash
# 启动 A6 预算网格 + A3 匹配基线（2000 条冻结负载，c16/q16，每次 1 repeat）
# 顺序执行，detach 运行，日志写 logs/a6_a3.log
set -u
cd /root/work/papers/extraction || exit 1
PY=.venv/bin/python
IDS=data/performance/performance_ids_2000.json
OUT=benchmarks/runs/a6-a3-sweep/raw

# A6 预算网格（budget-200 已在主实验，不重跑）
for B in 60 100 150 300 400 600 800 1200; do
  LBL="a6-budget-$B"
  echo "[$(date +%T)] START $LBL" >> logs/a6_a3.log
  "$PY" llm_runner.py \
    --methods "budget-$B" \
    --ids-file "$IDS" \
    --limit 2000 \
    --max-concurrency 16 --queue-size 16 --max-retries 2 \
    --run-label "$LBL" \
    --experiment-id "benchmark-a6-a3-$LBL" \
    --output "$OUT/$LBL.jsonl" \
    >> logs/a6_a3.log 2>&1
  echo "[$(date +%T)] DONE  $LBL rc=$?" >> logs/a6_a3.log
done

# A3 匹配基线
for M in fixed-5 fixed-10 prompt-null prompt-omit lambda-0.003; do
  LBL="a3-$M"
  echo "[$(date +%T)] START $LBL" >> logs/a6_a3.log
  "$PY" llm_runner.py \
    --methods "$M" \
    --ids-file "$IDS" \
    --limit 2000 \
    --max-concurrency 16 --queue-size 16 --max-retries 2 \
    --run-label "$LBL" \
    --experiment-id "benchmark-a6-a3-$LBL" \
    --output "$OUT/$LBL.jsonl" \
    >> logs/a6_a3.log 2>&1
  echo "[$(date +%T)] DONE  $LBL rc=$?" >> logs/a6_a3.log
done

echo "[$(date +%T)] ALL A6/A3 RUNS COMPLETE" >> logs/a6_a3.log