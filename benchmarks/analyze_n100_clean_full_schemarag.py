#!/usr/bin/env python3
"""离线汇总 clean SchemaRAG k=8 的六次冻结负载端到端实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_CRITICAL_95_DF5 = 2.571
METHOD = "external-schemarag-k8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 clean SchemaRAG k=8 实验")
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path("benchmarks/runs/n100-2000-clean-full-schemarag-k8/summaries.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/analysis/performance_n100_2000_clean_full_schemarag_k8"),
    )
    return parser.parse_args()


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.get('run_id')} 缺少数值指标 {key}")
    return float(value)


def mean_sd_ci(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("统计值不能为空")
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = T_CRITICAL_95_DF5 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"SchemaRAG k=8 必须恰有 6 个正式运行，实际为 {len(rows)}")
    expected_pairs = set(range(1, 7))
    pairs = {int(row["pair_index"]) for row in rows}
    if pairs != expected_pairs:
        raise ValueError(f"顺序编号不完整: {sorted(pairs)}")
    shared_keys = ("method", "records", "snapshot_sha256", "raw_data_sha256")
    for row in rows:
        if row.get("dry_run") or not row.get("validation_valid"):
            raise ValueError(f"{row.get('run_id')} 不是有效正式运行")
        if row.get("method") != METHOD or row.get("selector_model_dir") is not None:
            raise ValueError(f"{row.get('run_id')} 不是 clean SchemaRAG k=8")
        if numeric(row, "prediction_ms_total") != 0:
            raise ValueError(f"{row.get('run_id')} 错误调用了 M1 预测")
        if int(numeric(row, "selected_field_count_p50")) != 8:
            raise ValueError(f"{row.get('run_id')} 选择字段数中位数不是 8")
        if int(numeric(row, "selected_field_count_max")) != 8:
            raise ValueError(f"{row.get('run_id')} 选择字段数最大值不是 8")
        if numeric(row, "embedding_ms_total") <= 0:
            raise ValueError(f"{row.get('run_id')} 缺少 query embedding 遥测")
        if any(row.get(key) is None for key in shared_keys):
            raise ValueError(f"{row.get('run_id')} 缺少冻结负载元数据")


def main() -> None:
    args = parse_args()
    rows = json.loads(args.summaries.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("summaries 必须为 JSON 数组")
    validate_rows(rows)
    rows = sorted(rows, key=lambda row: int(row["pair_index"]))
    metrics = (
        "pipeline_ms",
        "effective_records_per_second",
        "total_tokens_total",
        "input_tokens_total",
        "output_tokens_total",
        "record_end_to_end_ms_p50",
        "record_end_to_end_ms_p95",
        "record_end_to_end_ms_p99",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "queue_wait_ms_p50",
        "queue_wait_ms_p95",
        "planning_ms_p50",
        "planning_ms_p95",
        "embedding_ms_total",
        "schema_embedding_ms_total",
        "retry_count_total",
        "error_records",
    )
    result = {
        "cohort": "n100-2000-clean-full-schemarag-k8",
        "method": METHOD,
        "statistical_unit": "six independent runs on frozen orders; descriptive only",
        "method_statistics": {
            key: mean_sd_ci([numeric(row, key) for row in rows]) for key in metrics
        },
        "runs": rows,
        "limitations": [
            "SchemaRAG runs were not interleaved with the prior full/budget cohort.",
            "Six runs support descriptive uncertainty intervals, not strong causal inference.",
            "The frozen workload is same-domain deployment data, not unseen generalization.",
            "No independent human Gold evaluates extraction quality in this performance cohort.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "schemarag_runs.csv", rows)
    write_json(args.output_dir / "performance_statistics.json", result)
    print(json.dumps({"runs": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
