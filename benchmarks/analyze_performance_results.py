#!/usr/bin/env python3
"""从已完成的流水线汇总结果生成论文用离线统计表。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


MAIN_SUMMARY = Path("benchmarks/runs/main-comparison-pro/summaries.json")
ENDPOINT_SUMMARIES = (
    Path("benchmarks/runs/scaling-pro/summaries.json"),
    Path("benchmarks/runs/endpoint-scaling-pro-low-full/summaries.json"),
    Path("benchmarks/runs/endpoint-scaling-pro-low-budget/summaries.json"),
    Path("benchmarks/runs/endpoint-scaling-pro-high-full/summaries.json"),
    Path("benchmarks/runs/endpoint-scaling-pro-high-budget/summaries.json"),
)

# 小样本双侧 95% t 临界值，避免将正态近似误写成置信区间。
T_CRITICAL_95 = {2: 12.706, 3: 4.303}

METRICS = (
    "pipeline_ms",
    "effective_records_per_second",
    "total_tokens_total",
    "latency_ms_p95",
    "latency_ms_p99",
    "rss_peak_bytes",
    "queue_average",
    "producer_blocked_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线汇总第二部分性能实验")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/analysis/performance"),
        help="统计表输出目录",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"汇总文件无有效运行: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"汇总文件包含非对象行: {path}")
    if not all(row.get("validation_valid") is True for row in rows):
        raise ValueError(f"汇总文件存在未通过完整性校验的运行: {path}")
    return rows


def confidence_interval(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    critical = T_CRITICAL_95.get(len(values))
    if critical is None:
        return None, None
    margin = critical * stdev(values) / math.sqrt(len(values))
    average = mean(values)
    return average - margin, average + margin


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    summary: dict[str, Any] = {
        "method": first["method"],
        "concurrency": first["concurrency"],
        "queue_size": first["queue_size"],
        "records_per_run": first["records"],
        "repeats": len(rows),
        "success_records_total": sum(row["success_records"] for row in rows),
        "error_records_total": sum(row["error_records"] for row in rows),
        "retry_count_total": sum(row["retry_count_total"] for row in rows),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        lower, upper = confidence_interval(values)
        summary[f"{metric}_mean"] = mean(values)
        summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
        summary[f"{metric}_ci95_low"] = lower
        summary[f"{metric}_ci95_high"] = upper
    return summary


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), int(row["concurrency"]), int(row["queue_size"]))
        groups[key].append(row)
    return [summarize_group(groups[key]) for key in sorted(groups)]


def compare_to_full(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config = {(row["concurrency"], row["queue_size"], row["method"]): row for row in groups}
    comparisons = []
    for row in groups:
        if row["method"] == "full":
            continue
        baseline = by_config.get((row["concurrency"], row["queue_size"], "full"))
        if baseline is None:
            continue
        comparisons.append(
            {
                "method": row["method"],
                "concurrency": row["concurrency"],
                "queue_size": row["queue_size"],
                "pipeline_time_change_pct": (
                    (row["pipeline_ms_mean"] / baseline["pipeline_ms_mean"] - 1) * 100
                ),
                "throughput_change_pct": (
                    (row["effective_records_per_second_mean"] /
                     baseline["effective_records_per_second_mean"] - 1) * 100
                ),
                "token_change_pct": (
                    (row["total_tokens_total_mean"] /
                     baseline["total_tokens_total_mean"] - 1) * 100
                ),
            }
        )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    main_rows = load_rows(MAIN_SUMMARY)
    endpoint_rows = [row for path in ENDPOINT_SUMMARIES for row in load_rows(path)]
    main_groups = group_rows(main_rows)
    endpoint_groups = group_rows(endpoint_rows)
    payload = {
        "analysis": {
            "description": "只读取已完成实验汇总；不触发 LLM 或 embedding 调用。",
            "ci_method": "two-sided t interval; available for n=2 or n=3 only",
            "source_files": [str(MAIN_SUMMARY), *map(str, ENDPOINT_SUMMARIES)],
        },
        "main_comparison": main_groups,
        "main_relative_to_full": compare_to_full(main_groups),
        "endpoint_scaling": endpoint_groups,
        "endpoint_relative_to_full": compare_to_full(endpoint_groups),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "performance_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "main_comparison.csv", main_groups)
    write_csv(args.output_dir / "main_relative_to_full.csv", payload["main_relative_to_full"])
    write_csv(args.output_dir / "endpoint_scaling.csv", endpoint_groups)
    write_csv(
        args.output_dir / "endpoint_relative_to_full.csv",
        payload["endpoint_relative_to_full"],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "main_groups": len(main_groups),
                      "endpoint_groups": len(endpoint_groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
