#!/usr/bin/env python3
"""以冻结的 model-assisted reference 评估 LLM Schema 对照实验。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "name", "category", "form_type", "usage", "desc", "condition",
    "dynasty", "period", "acq_type", "region", "site", "culture",
    "material", "motif", "craft", "craft_school", "related_event",
    "related_person",
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def normalize(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return "".join(text.split()) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估真实 LLM 抽取实验")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("experiments/extraction_results.jsonl")],
        help="一个或多个结果 JSONL；后面的文件可用于补并发实验失败任务",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/gold/agent_tasks/gold_main.jsonl"),
        help="包含 artifact_id 和完整 reference_result 的冻结 Gold JSONL",
    )
    parser.add_argument(
        "--require-methods",
        nargs="+",
        help="仅评估这些方法均成功的共同 artifact_id，用于配对正式比较",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/llm_experiment_metrics.json"))
    return parser.parse_args()


def load_references(path: Path) -> dict[int, dict[str, Any]]:
    """加载已补齐的轻量 Gold，避免在正式评估中回退到弱监督结果。"""
    references = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        artifact_id = row.get("artifact_id")
        reference = row.get("reference_result")
        if not isinstance(artifact_id, int) or not isinstance(reference, dict):
            raise ValueError(f"Gold 第 {line_number} 行格式无效")
        if set(reference) != set(FIELDS):
            raise ValueError(f"Gold 第 {line_number} 行字段集合不完整")
        if artifact_id in references:
            raise ValueError(f"Gold 存在重复 artifact_id: {artifact_id}")
        references[artifact_id] = reference
    if not references:
        raise ValueError(f"Gold 文件为空: {path}")
    return references


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "records": len(values),
        **{
            f"average_{key}": float(np.mean([item[key] for item in values]))
            for key in values[0]
        },
    }
    tp = sum(item["presence_tp"] for item in values)
    fp = sum(item["presence_fp"] for item in values)
    fn = sum(item["presence_fn"] for item in values)
    micro_precision = tp / (tp + fp) if tp + fp else 1.0
    micro_recall = tp / (tp + fn) if tp + fn else 1.0
    summary["presence_micro_precision"] = micro_precision
    summary["presence_micro_recall"] = micro_recall
    summary["presence_micro_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    summary["presence_macro_f1"] = summary["average_presence_f1"]
    return summary


def summarize_execution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总最终状态及所有尝试累计的实际运行成本。"""
    submitted = len(rows)
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") != "success"]
    total_tokens = sum(row.get("total_tokens") or 0 for row in rows)
    return {
        "submitted": submitted,
        "successful": len(successful),
        "final_failures": len(failed),
        "success_rate": len(successful) / submitted if submitted else 0.0,
        "failure_rate": len(failed) / submitted if submitted else 0.0,
        "retry_tasks": sum((row.get("retry_count") or 0) > 0 for row in rows),
        "total_retries": sum(row.get("retry_count") or 0 for row in rows),
        "total_attempts": sum(len(row.get("attempts") or []) or 1 for row in rows),
        "average_total_tokens_per_submitted": total_tokens / submitted if submitted else 0.0,
        "total_tokens_per_success": total_tokens / len(successful) if successful else None,
    }


def result_group(row: dict[str, Any]) -> str:
    """优先使用运行标签，兼容未设置标签的历史实验。"""
    return row.get("run_label") or row.get("method", "unknown")


def main() -> None:
    args = parse_args()
    selected_rows = {}
    invalid_rows = 0
    for input_path in args.input:
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row.get("artifact_id"), result_group(row))
                previous = selected_rows.get(key)
                if previous is None or (
                    previous.get("status") != "success"
                    and row.get("status") == "success"
                ):
                    selected_rows[key] = row
            except (json.JSONDecodeError, AttributeError):
                invalid_rows += 1
    execution_rows = list(selected_rows.values())
    rows = [row for row in execution_rows if row.get("status") == "success"]
    references = load_references(args.gold)
    if args.require_methods:
        successful_ids_by_method = {
            method: {row["artifact_id"] for row in rows if result_group(row) == method}
            for method in args.require_methods
        }
        common_ids = set.intersection(*successful_ids_by_method.values())
        rows = [
            row
            for row in rows
            if row["artifact_id"] in common_ids and result_group(row) in args.require_methods
        ]
    else:
        common_ids = None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reference = references.get(row["artifact_id"])
        if reference is None:
            continue
        selected = set(row["selected_fields"])
        result = row["result"] if isinstance(row["result"], dict) else {}
        gold_positive = {field for field in FIELDS if normalize(reference[field]) is not None}
        predicted_positive = {
            field for field in selected if normalize(result.get(field)) is not None
        }
        exact = {
            field for field in selected
            if normalize(reference[field]) is not None
            and normalize(reference[field]) == normalize(result.get(field))
        }
        tp = len(predicted_positive & gold_positive)
        fp = len(predicted_positive - gold_positive)
        fn = len(gold_positive - predicted_positive)
        precision = tp / len(predicted_positive) if predicted_positive else 1.0
        recall = tp / len(gold_positive) if gold_positive else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        record_metrics = {
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": f1,
            "presence_tp": tp,
            "presence_fp": fp,
            "presence_fn": fn,
            "reference_present_recall": recall,
            "reference_absent_fpr": (
                fp / (len(FIELDS) - len(gold_positive))
                if len(gold_positive) < len(FIELDS)
                else 0.0
            ),
            "selected_gold_coverage": len(selected & gold_positive) / len(gold_positive) if gold_positive else 1.0,
            "exact_value_recall": len(exact) / len(gold_positive) if gold_positive else 1.0,
            "input_tokens": row.get("input_tokens") or 0,
            "output_tokens": row.get("output_tokens") or 0,
            "total_tokens": row.get("total_tokens") or 0,
            "latency_ms": row.get("latency_ms") or 0,
            "queue_wait_ms": row.get("queue_wait_ms") or 0,
            "retry_count": row.get("retry_count") or 0,
            "actual_cost": row.get("actual_cost") or 0,
            "estimated_field_cost": row.get("estimated_field_cost") or 0,
        }
        groups[result_group(row)].append(record_metrics)
        # 按方法组汇总，便于 Full / Prompt-only / Adaptive 的对照报告
        method_groups[row.get("method_group", "unknown")].append(record_metrics)
    summary = {
        "invalid_rows": invalid_rows,
        "gold_records": len(references),
        "paired_records": len(common_ids) if common_ids is not None else None,
    }
    for method, values in groups.items():
        summary[method] = summarize(values)
    group_summary = {
        group: summarize(values) for group, values in method_groups.items()
    }
    execution_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in execution_rows:
        execution_groups[result_group(row)].append(row)
    execution_summary = {
        method: summarize_execution(values)
        for method, values in execution_groups.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
                {
                    "methods": summary,
                    "method_groups": group_summary,
                    "execution": execution_summary,
                },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
