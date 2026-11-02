#!/usr/bin/env python3
"""离线汇总 n=100 M1 交错配对性能实验，不发起模型调用。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


INIT_LOAD_SECONDS = 0.17826
INIT_TRAIN_AND_LOAD_SECONDS = 0.91539
T_CRITICAL_95_DF2 = 4.303


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 n=100 M1 配对性能实验")
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path("benchmarks/runs/n100-2000-paired/summaries.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/analysis/performance_n100_2000_paired"),
    )
    return parser.parse_args()


def load_summaries(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"summaries 必须是 JSON 数组: {path}")
    if any(row.get("dry_run") for row in rows):
        raise ValueError("不能分析 dry-run 汇总；请提供正式 --execute 结果")
    return rows


def mean_sd_ci(values: list[float]) -> dict[str, float | int | None]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values)) if len(values) == 3 else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width if half_width is not None else None,
        "ci95_high": mean + half_width if half_width is not None else None,
    }


def metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{row.get('run_id')} 缺少数值指标 {key}")
        values.append(float(value))
    return values


def paired_delta(
    pairs: dict[int, dict[str, dict[str, Any]]],
    key: str,
    positive_when_budget_better: bool,
) -> dict[str, float | int | None]:
    values = []
    relative = []
    for seed, pair in sorted(pairs.items()):
        full = float(pair["full"][key])
        budget = float(pair["budget-200"][key])
        delta = budget - full if positive_when_budget_better else full - budget
        values.append(delta)
        relative.append(delta / full * 100)
    return {"absolute": mean_sd_ci(values), "relative_percent": mean_sd_ci(relative)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_summaries(args.summaries)
    expected = {(seed, method) for seed in (42, 43, 44) for method in ("full", "budget-200")}
    observed = {(row.get("selector_seed"), row.get("method")) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"配对运行不完整或重复: observed={sorted(observed)}")
    if not all(row.get("validation_valid") for row in rows):
        raise ValueError("存在未通过 validation 的运行")

    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["selector_seed"])
        pairs.setdefault(seed, {})[str(row["method"])] = row
    if any(set(pair) != {"full", "budget-200"} for pair in pairs.values()):
        raise ValueError("每个 seed 必须同时有 full 和 budget-200")

    methods = {method: [row for row in rows if row["method"] == method]
               for method in ("full", "budget-200")}
    metrics = (
        "pipeline_ms",
        "effective_records_per_second",
        "total_tokens_total",
        "input_tokens_total",
        "output_tokens_total",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "retry_count_total",
        "error_records",
        "queue_wait_ratio",
    )
    method_statistics = {
        method: {key: mean_sd_ci(metric(method_rows, key)) for key in metrics}
        for method, method_rows in methods.items()
    }
    paired_effects = {
        "pipeline_time_saved_ms": paired_delta(pairs, "pipeline_ms", False),
        "throughput_gain_records_per_second": paired_delta(
            pairs, "effective_records_per_second", True
        ),
        "total_tokens_saved": paired_delta(pairs, "total_tokens_total", False),
        "p95_latency_saved_ms": paired_delta(pairs, "latency_ms_p95", False),
        "p99_latency_saved_ms": paired_delta(pairs, "latency_ms_p99", False),
        "queue_wait_ratio_reduction": paired_delta(pairs, "queue_wait_ratio", False),
    }

    full_time = method_statistics["full"]["pipeline_ms"]["mean"] / 1000
    budget_time = method_statistics["budget-200"]["pipeline_ms"]["mean"] / 1000
    records = int(rows[0]["records"])
    saved_per_record = (full_time - budget_time) / records
    if saved_per_record <= 0:
        break_even = {"available": False, "reason": "budget-200 未观察到正的平均时间节省"}
    else:
        break_even = {
            "available": True,
            "records_per_run": records,
            "mean_full_seconds_per_record": full_time / records,
            "mean_budget_seconds_per_record": budget_time / records,
            "mean_saved_seconds_per_record": saved_per_record,
            "load_only_seconds": INIT_LOAD_SECONDS,
            "load_only_break_even_records": INIT_LOAD_SECONDS / saved_per_record,
            "train_and_load_seconds": INIT_TRAIN_AND_LOAD_SECONDS,
            "train_and_load_break_even_records": INIT_TRAIN_AND_LOAD_SECONDS / saved_per_record,
            "scope": "基于本 cohort 三个 n=100 交错配对运行的均值；一次性成本不计入在线指标。",
        }

    per_pair = []
    for seed, pair in sorted(pairs.items()):
        full = pair["full"]
        budget = pair["budget-200"]
        per_pair.append({
            "selector_seed": seed,
            "full_run_id": full["run_id"],
            "budget_run_id": budget["run_id"],
            "full_pipeline_ms": full["pipeline_ms"],
            "budget_pipeline_ms": budget["pipeline_ms"],
            "pipeline_time_saved_ms": full["pipeline_ms"] - budget["pipeline_ms"],
            "full_throughput": full["effective_records_per_second"],
            "budget_throughput": budget["effective_records_per_second"],
            "throughput_gain": (
                budget["effective_records_per_second"] - full["effective_records_per_second"]
            ),
            "full_total_tokens": full["total_tokens_total"],
            "budget_total_tokens": budget["total_tokens_total"],
            "total_tokens_saved": full["total_tokens_total"] - budget["total_tokens_total"],
            "full_p95_ms": full["latency_ms_p95"],
            "budget_p95_ms": budget["latency_ms_p95"],
            "full_p99_ms": full["latency_ms_p99"],
            "budget_p99_ms": budget["latency_ms_p99"],
            "full_retry_count": full["retry_count_total"],
            "budget_retry_count": budget["retry_count_total"],
            "full_error_records": full["error_records"],
            "budget_error_records": budget["error_records"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_runs.csv", per_pair)
    report = {
        "cohort": "n100-2000-paired",
        "source_summaries": str(args.summaries),
        "runs": len(rows),
        "pair_seeds": sorted(pairs),
        "method_statistics": method_statistics,
        "paired_effects": paired_effects,
        "initialization_cost_sensitivity": break_even,
        "limitations": [
            "该工作负载来自训练集与验证集，是同域部署性能负载，不是未见泛化测试。",
            "full 逻辑上不依赖字段选择器，但当前 Runner 为统一规划遥测仍加载模型并计算预测。",
            "置信区间仅基于三个交错配对运行，主要反映当前服务时段与模型 seed 的联合波动。",
        ],
    }
    write_json(args.output_dir / "performance_statistics.json", report)
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
