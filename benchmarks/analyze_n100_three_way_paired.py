#!/usr/bin/env python3
"""离线汇总 clean full、budget-200 和 SchemaRAG k=8 三方法实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = ("full", "budget-200", "external-schemarag-k8")
T_CRITICAL_95_DF5 = 2.571
EXPECTED_BLOCKS = 6
EXPECTED_RECORDS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总三方法冻结负载配对实验")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("benchmarks/runs/n100-2000-three-way-paired"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/analysis/performance_n100_2000_three_way_paired"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.get('run_id')} 缺少数值指标 {key}")
    return float(value)


def validate_and_load(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = run_dir / "summaries.json"
    cohort_path = run_dir / "cohort.json"
    if not summary_path.exists() or not cohort_path.exists():
        raise FileNotFoundError("缺少正式 cohort 的 summaries.json 或 cohort.json")
    summaries = read_json(summary_path)
    cohort = read_json(cohort_path)
    if len(summaries) != EXPECTED_BLOCKS * len(METHODS):
        raise ValueError(f"正式实验必须有18个run，实际为 {len(summaries)}")
    if any(row.get("dry_run") for row in summaries):
        raise ValueError("不能分析 dry-run 汇总")

    seen: set[str] = set()
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        run_id = str(summary.get("run_id"))
        if run_id in seen:
            raise ValueError(f"run_id重复: {run_id}")
        seen.add(run_id)
        method = str(summary.get("method"))
        if method not in METHODS:
            raise ValueError(f"未知方法: {method}")
        if not summary.get("validation_valid"):
            raise ValueError(f"未通过validation: {run_id}")
        if summary.get("records") != EXPECTED_RECORDS:
            raise ValueError(f"记录数不是2000: {run_id}")
        by_block[int(summary["block_index"])].append(summary)

    if set(by_block) != set(range(1, EXPECTED_BLOCKS + 1)):
        raise ValueError(f"block编号不完整: {sorted(by_block)}")
    for block_index, rows in by_block.items():
        if {row["method"] for row in rows} != set(METHODS):
            raise ValueError(f"block {block_index} 未包含三种方法")
        if len({row["ids_sha256"] for row in rows}) != 1:
            raise ValueError(f"block {block_index} 的输入顺序不一致")
        if len({row["snapshot_sha256"] for row in rows}) != 1:
            raise ValueError(f"block {block_index} 的snapshot不一致")

    return summaries, cohort


def load_raw_audit(
    run_dir: Path, summaries: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        run_id = str(summary["run_id"])
        rows = read_jsonl(run_dir / "raw" / f"{run_id}.jsonl")
        statuses = Counter(row.get("status") for row in rows)
        field_counts = Counter(len(row.get("selected_fields", [])) for row in rows)
        unexpected = Counter(
            field
            for row in rows
            for field in (row.get("unexpected_fields") or [])
        )
        extra_records = sum(bool(row.get("unexpected_fields")) for row in rows)
        ids = [row.get("artifact_id") for row in rows]
        audits[run_id] = {
            "run_id": run_id,
            "method": summary["method"],
            "block_index": summary["block_index"],
            "records": len(rows),
            "unique_ids": len(set(ids)),
            "status_counts": dict(statuses),
            "all_success": statuses == Counter({"success": EXPECTED_RECORDS}),
            "field_counts": dict(sorted(field_counts.items())),
            "extra_record_count": extra_records,
            "extra_record_rate": extra_records / len(rows) if rows else None,
            "unexpected_field_counts": dict(unexpected),
            "retry_count_total": sum(row.get("retry_count", 0) or 0 for row in rows),
            "record_e2e_missing": sum(
                not isinstance(row.get("record_end_to_end_ms"), (int, float))
                for row in rows
            ),
            "embedding_missing": sum(
                not isinstance(row.get("embedding_ms"), (int, float))
                for row in rows
            ),
        }
        if len(rows) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
            raise ValueError(f"raw记录数或ID唯一性错误: {run_id}")
        if not audits[run_id]["all_success"]:
            raise ValueError(f"raw存在失败记录，不能纳入主统计: {run_id}")
    return audits


def metric_summary(
    summaries: list[dict[str, Any]], metric: str, method: str
) -> dict[str, float | int]:
    values = [number(row, metric) for row in summaries if row["method"] == method]
    return mean_sd_ci(values)


def block_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_block[int(row["block_index"])][str(row["method"])] = row

    result = []
    for block_index in range(1, EXPECTED_BLOCKS + 1):
        block = by_block[block_index]
        full = block["full"]
        output: dict[str, Any] = {
            "block_index": block_index,
            "selector_seed": full["selector_seed"],
            "repeat": full["repeat"],
            "sequence": "->".join(full["sequence"]),
        }
        for method in METHODS:
            prefix = method.replace("-", "_")
            row = block[method]
            output[f"{prefix}_pipeline_ms"] = number(row, "pipeline_ms")
            output[f"{prefix}_throughput"] = number(
                row, "effective_records_per_second"
            )
            output[f"{prefix}_tokens"] = number(row, "total_tokens_total")
            output[f"{prefix}_e2e_p95_ms"] = number(row, "record_end_to_end_ms_p95")
            output[f"{prefix}_e2e_p99_ms"] = number(row, "record_end_to_end_ms_p99")
            output[f"{prefix}_request_p95_ms"] = number(row, "latency_ms_p95")
            output[f"{prefix}_embedding_ms"] = number(row, "embedding_ms_total")
        for method in METHODS[1:]:
            prefix = method.replace("-", "_")
            output[f"{prefix}_pipeline_saved_vs_full_ms"] = (
                output["full_pipeline_ms"] - output[f"{prefix}_pipeline_ms"]
            )
            output[f"{prefix}_pipeline_saved_vs_full_percent"] = (
                output[f"{prefix}_pipeline_saved_vs_full_ms"]
                / output["full_pipeline_ms"]
                * 100
            )
            output[f"{prefix}_throughput_gain_vs_full"] = (
                output[f"{prefix}_throughput"] - output["full_throughput"]
            )
            output[f"{prefix}_tokens_saved_vs_full"] = (
                output["full_tokens"] - output[f"{prefix}_tokens"]
            )
            output[f"{prefix}_tokens_saved_vs_full_percent"] = (
                output[f"{prefix}_tokens_saved_vs_full"]
                / output["full_tokens"]
                * 100
            )
        result.append(output)
    return result


def position_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[int, list[float]]] = {
        method: defaultdict(list) for method in METHODS
    }
    for row in summaries:
        grouped[row["method"]][int(row["order_position"])].append(
            number(row, "pipeline_ms")
        )
    return {
        method: {
            str(position): mean_sd_ci(values)
            for position, values in sorted(positions.items())
        }
        for method, positions in grouped.items()
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summaries, cohort = validate_and_load(args.run_dir)
    audits = load_raw_audit(args.run_dir, summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        "pipeline_ms",
        "effective_records_per_second",
        "total_tokens_total",
        "record_end_to_end_ms_p50",
        "record_end_to_end_ms_p95",
        "record_end_to_end_ms_p99",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "queue_wait_ms_p50",
        "queue_wait_ms_p95",
        "planning_ms_p50",
        "embedding_ms_total",
        "schema_embedding_ms_total",
        "llm_ms_total",
        "retry_count_total",
    )
    method_statistics = {
        method: {
            metric: metric_summary(summaries, metric, method)
            for metric in metrics
        }
        for method in METHODS
    }
    pairs = block_rows(summaries)
    pair_effects = {
        method: {
            metric: mean_sd_ci(
                [
                    (
                        row[f"{method.replace('-', '_')}_{metric}"]
                        - row[f"full_{metric}"]
                        if metric == "throughput"
                        else row[f"full_{metric}"]
                        - row[f"{method.replace('-', '_')}_{metric}"]
                    )
                    for row in pairs
                ]
            )
            for metric in (
                "pipeline_ms",
                "throughput",
                "tokens",
                "e2e_p95_ms",
                "e2e_p99_ms",
            )
        }
        for method in METHODS[1:]
    }

    audit_rows = list(audits.values())
    audit_totals = {
        "runs": len(audit_rows),
        "records": sum(row["records"] for row in audit_rows),
        "all_success": all(row["all_success"] for row in audit_rows),
        "retry_count_total": sum(row["retry_count_total"] for row in audit_rows),
        "record_e2e_missing": sum(row["record_e2e_missing"] for row in audit_rows),
        "embedding_missing": sum(row["embedding_missing"] for row in audit_rows),
        "extra_record_count_by_method": {
            method: sum(
                row["extra_record_count"]
                for row in audit_rows
                if row["method"] == method
            )
            for method in METHODS
        },
        "unexpected_field_counts_by_method": {
            method: dict(
                Counter(
                    field
                    for row in audit_rows
                    if row["method"] == method
                    for field, count in row["unexpected_field_counts"].items()
                    for _ in range(count)
                )
            )
            for method in METHODS
        },
    }
    report = {
        "protocol": cohort,
        "statistics_unit": "six complete three-method blocks; n=6 per method",
        "t_critical_95_df5": T_CRITICAL_95_DF5,
        "methods": list(METHODS),
        "method_statistics": method_statistics,
        "block_effects_vs_full": pair_effects,
        "block_rows": pairs,
        "position_pipeline_summary": position_summary(summaries),
        "raw_audit": {"runs": audit_rows, "totals": audit_totals},
        "limitations": [
            "2000条记录是同一冻结负载内的重复观测，不作为独立统计样本。",
            "六个block支持运行级描述性比较和配对区间，不构成广泛服务环境的显著性证明。",
            "SchemaRAG为外部独立重实现，额外字段按允许策略过滤但保留审计。",
            "远程模型服务、GPU和外部负载不能完全冻结。",
        ],
    }
    (args.output_dir / "performance_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "block_rows.csv", pairs)
    write_csv(args.output_dir / "raw_audit.csv", audit_rows)
    print(
        json.dumps(
            {
                "runs": len(summaries),
                "blocks": EXPECTED_BLOCKS,
                "records": audit_totals["records"],
                "all_success": audit_totals["all_success"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
