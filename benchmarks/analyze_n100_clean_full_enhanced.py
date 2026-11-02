#!/usr/bin/env python3
"""离线汇总 clean FullSchema 与 n=100 budget-200 的六组增强配对实验。"""

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
T_CRITICAL_95_DF5 = 2.571
METHODS = ("full", "budget-200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 n=100 clean-full 增强配对实验")
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path(
            "benchmarks/runs/n100-2000-clean-full-paired/summaries.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmarks/analysis/performance_n100_2000_clean_full_enhanced"
        ),
    )
    return parser.parse_args()


def load_summaries(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"summaries 必须是 JSON 数组: {path}")
    if any(row.get("dry_run") for row in rows):
        raise ValueError("不能分析 dry-run 汇总；请提供正式 --execute 结果")
    return rows


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


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.get('run_id')} 缺少数值指标 {key}")
    return float(value)


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    return mean_sd_ci([numeric(row, key) for row in rows])


def paired_values(
    pairs: dict[int, dict[str, dict[str, Any]]],
    key: str,
    higher_is_better: bool,
) -> tuple[list[float], list[float]]:
    absolute = []
    relative = []
    for pair in pairs.values():
        full = numeric(pair["full"], key)
        budget = numeric(pair["budget-200"], key)
        delta = budget - full if higher_is_better else full - budget
        absolute.append(delta)
        relative.append(delta / full * 100)
    return absolute, relative


def paired_summary(
    pairs: dict[int, dict[str, dict[str, Any]]],
    key: str,
    higher_is_better: bool,
) -> dict[str, dict[str, float | int]]:
    absolute, relative = paired_values(pairs, key, higher_is_better)
    return {
        "absolute": mean_sd_ci(absolute),
        "relative_percent": mean_sd_ci(relative),
    }


def validate_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    if len(rows) != 12:
        raise ValueError(f"增强实验必须恰有 12 个正式运行，实际为 {len(rows)}")
    if not all(row.get("validation_valid") for row in rows):
        raise ValueError("存在未通过 validation 的运行")

    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_index = int(row["pair_index"])
        method = str(row["method"])
        if method not in METHODS:
            raise ValueError(f"未知方法: {method}")
        pair = pairs.setdefault(pair_index, {})
        if method in pair:
            raise ValueError(f"pair {pair_index} 中方法 {method} 重复")
        pair[method] = row

    if set(pairs) != set(range(1, 7)):
        raise ValueError(f"配对编号不完整: {sorted(pairs)}")
    for pair_index, pair in pairs.items():
        if set(pair) != set(METHODS):
            raise ValueError(f"pair {pair_index} 未同时包含两个方法")
        full = pair["full"]
        budget = pair["budget-200"]
        shared = (
            "selector_seed",
            "repeat",
            "ids_file",
            "ids_sha256",
            "snapshot_sha256",
        )
        if any(full.get(key) != budget.get(key) for key in shared):
            raise ValueError(f"pair {pair_index} 的共享元数据不一致")
        if full.get("selector_model_dir") is not None:
            raise ValueError(f"pair {pair_index} 的 clean full 不应绑定模型")
        if numeric(full, "prediction_ms_total") != 0:
            raise ValueError(f"pair {pair_index} 的 clean full 发生了字段预测")
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pair_rows(pairs: dict[int, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for pair_index, pair in sorted(pairs.items()):
        full = pair["full"]
        budget = pair["budget-200"]
        full_time = numeric(full, "pipeline_ms")
        budget_time = numeric(budget, "pipeline_ms")
        full_rate = numeric(full, "effective_records_per_second")
        budget_rate = numeric(budget, "effective_records_per_second")
        full_tokens = numeric(full, "total_tokens_total")
        budget_tokens = numeric(budget, "total_tokens_total")
        result.append(
            {
                "pair_index": pair_index,
                "selector_seed": full["selector_seed"],
                "repeat": full["repeat"],
                "sequence": "->".join(full["pair_sequence"]),
                "full_run_id": full["run_id"],
                "budget_run_id": budget["run_id"],
                "full_pipeline_ms": full_time,
                "budget_pipeline_ms": budget_time,
                "pipeline_saved_ms": full_time - budget_time,
                "pipeline_saved_percent": (full_time - budget_time) / full_time * 100,
                "full_throughput": full_rate,
                "budget_throughput": budget_rate,
                "throughput_gain": budget_rate - full_rate,
                "throughput_gain_percent": (budget_rate - full_rate) / full_rate * 100,
                "full_total_tokens": full_tokens,
                "budget_total_tokens": budget_tokens,
                "tokens_saved": full_tokens - budget_tokens,
                "tokens_saved_percent": (full_tokens - budget_tokens) / full_tokens * 100,
                "full_record_e2e_p95_ms": numeric(full, "record_end_to_end_ms_p95"),
                "budget_record_e2e_p95_ms": numeric(
                    budget, "record_end_to_end_ms_p95"
                ),
                "full_request_p95_ms": numeric(full, "latency_ms_p95"),
                "budget_request_p95_ms": numeric(budget, "latency_ms_p95"),
                "full_retries": numeric(full, "retry_count_total"),
                "budget_retries": numeric(budget, "retry_count_total"),
                "full_errors": numeric(full, "error_records"),
                "budget_errors": numeric(budget, "error_records"),
            }
        )
    return result


def grouped_effects(
    per_pair: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in per_pair:
        groups.setdefault(str(row[group_key]), []).append(row)
    metrics = (
        "pipeline_saved_percent",
        "throughput_gain_percent",
        "tokens_saved_percent",
    )
    return {
        group: {
            key: mean_sd_ci([float(row[key]) for row in group_rows])
            for key in metrics
        }
        for group, group_rows in groups.items()
    }


def initialization_sensitivity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_rows = [row for row in rows if row["method"] == "full"]
    budget_rows = [row for row in rows if row["method"] == "budget-200"]
    records = int(full_rows[0]["records"])
    full_seconds = statistics.mean(numeric(row, "pipeline_ms") for row in full_rows) / 1000
    budget_seconds = (
        statistics.mean(numeric(row, "pipeline_ms") for row in budget_rows) / 1000
    )
    saved_per_record = (full_seconds - budget_seconds) / records
    if saved_per_record <= 0:
        return {
            "available": False,
            "reason": "budget-200 未观察到正的平均在线时间节省",
        }

    # 完整银标生成成本尚无同协议现场测量，必须显式留空。
    return {
        "available": True,
        "records_per_run": records,
        "mean_online_full_seconds": full_seconds,
        "mean_online_budget_seconds": budget_seconds,
        "mean_saved_seconds_per_record": saved_per_record,
        "cost_layers": {
            "online_only": {
                "one_time_seconds": 0.0,
                "break_even_records": 0.0,
            },
            "deployed_cold_load": {
                "one_time_seconds": INIT_LOAD_SECONDS,
                "break_even_records": INIT_LOAD_SECONDS / saved_per_record,
            },
            "local_train_and_load": {
                "one_time_seconds": INIT_TRAIN_AND_LOAD_SECONDS,
                "break_even_records": INIT_TRAIN_AND_LOAD_SECONDS / saved_per_record,
            },
            "full_silver_label_generation": {
                "one_time_seconds": None,
                "break_even_records": None,
                "status": "未按本协议现场测量，不生成估计数字",
            },
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_summaries(args.summaries)
    pairs = validate_rows(rows)
    per_pair = pair_rows(pairs)
    for row in per_pair:
        row["order_group"] = (
            "full_first" if row["sequence"].startswith("full") else "budget_first"
        )

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
        "retry_count_total",
        "error_records",
    )
    methods = {
        method: [row for row in rows if row["method"] == method]
        for method in METHODS
    }
    method_statistics = {
        method: {
            key: summarize_metric(method_rows, key)
            for key in metrics
        }
        for method, method_rows in methods.items()
    }
    paired_effects = {
        "pipeline_time_saved_ms": paired_summary(pairs, "pipeline_ms", False),
        "throughput_gain_records_per_second": paired_summary(
            pairs, "effective_records_per_second", True
        ),
        "total_tokens_saved": paired_summary(pairs, "total_tokens_total", False),
        "record_e2e_p95_saved_ms": paired_summary(
            pairs, "record_end_to_end_ms_p95", False
        ),
        "record_e2e_p99_saved_ms": paired_summary(
            pairs, "record_end_to_end_ms_p99", False
        ),
        "request_p95_saved_ms": paired_summary(pairs, "latency_ms_p95", False),
        "request_p99_saved_ms": paired_summary(pairs, "latency_ms_p99", False),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_runs.csv", per_pair)
    report = {
        "cohort": "n100-2000-clean-full-paired",
        "source_summaries": str(args.summaries),
        "runs": len(rows),
        "pairs": len(pairs),
        "statistical_unit": "完整的相邻方法配对运行",
        "t_critical_95_df5": T_CRITICAL_95_DF5,
        "method_statistics": method_statistics,
        "paired_effects": paired_effects,
        "order_effects_descriptive": grouped_effects(per_pair, "order_group"),
        "selector_seed_effects_descriptive": grouped_effects(
            per_pair, "selector_seed"
        ),
        "initialization_cost_sensitivity": initialization_sensitivity(rows),
        "limitations": [
            "工作负载来自训练集与验证集，是同域部署性能负载，不是未见泛化测试。",
            "六个 pair 支持运行级描述和配对区间，但不足以作强显著性推断。",
            "共享远程服务的版本、GPU 状态和外部负载不能完全冻结。",
            "性能实验没有独立人工 Gold，不能据此证明语义质量保持。",
            "完整银标生成成本未按本协议现场测量，生命周期分析不填造数字。",
        ],
    }
    write_json(args.output_dir / "performance_statistics.json", report)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "runs": len(rows), "pairs": len(pairs)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
