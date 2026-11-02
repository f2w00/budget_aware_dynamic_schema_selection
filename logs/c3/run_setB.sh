#!/usr/bin/env bash
# 运行 C3 Set B：固定 q=16，扫描 c∈{1,2,4,8,32}（c=16 已在 Set A 完成）
# 每个点独立 output-dir，避免共享 summaries.json/csv 并发写坏。
set -u
ROOT="/root/work/papers/extraction"
cd "$ROOT"
declare -A MULT=( [1]=16 [2]=8 [4]=4 [8]=2 [32]=0.5 )
for c in 1 2 4 8 32; do
  m=${MULT[$c]}
  outdir="benchmarks/runs/c3-scan-decoupled/setB-c${c}"
  echo "[$(date +%T)] c=$c q=16 (mult=$m) -> $outdir"
  .venv/bin/python benchmarks/benchmark_pipeline.py \
    --methods budget-200 \
    --ids-file data/performance/performance_ids_2000.json \
    --workload-manifest data/performance/performance_ids_2000.manifest.json \
    --phase c3-scan-fixq16 \
    --limit 500 \
    --concurrency-values "$c" \
    --queue-multipliers "$m" \
    --output-dir "$outdir" \
    --max-retries 2 --temperature 0.0
  echo "[$(date +%T)] c=$c done rc=$?"
done
echo "[$(date +%T)] ALL SET B DONE"