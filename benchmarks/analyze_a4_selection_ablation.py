#!/usr/bin/env python3
"""A4 离线选择策略消融，不调用 LLM 或 embedding 服务。"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budget_selector import select_fields
from llm_runner import predict_fields
from training_size_ablation import load_jsonl, load_splits


INPUT = ROOT / "training_dataset.jsonl"
SPLIT_DIR = ROOT / "data/splits"
SCHEMA_PATH = ROOT / "simple2.json"
COSTS_PATH = ROOT / "schema_costs.json"
WEIGHTS_PATH = ROOT / "field_weights.json"
GOLD_IDS_PATH = ROOT / "data/gold/gold_main_ids.json"
GOLD_REFERENCE_PATH = ROOT / "data/gold/agent_tasks/gold_main.jsonl"
MODEL_ROOT = ROOT / "models/training_size_ablation/100"
OUT_DIR = ROOT / "benchmarks/analysis/a4_selection_ablation"
MODEL_SEEDS = (42, 43, 44, 45, 46)
RANDOM_SEEDS = tuple(range(100, 120))
BUDGET = int(os.environ.get("A4_BUDGET", "200"))


def load_model(path: Path) -> dict[str, Any]:
    """读取训练规模消融已保存的分离式模型产物。"""
    return {
        "vectorizer": joblib.load(path / "vectorizer.joblib"),
        "classifiers": joblib.load(path / "classifiers.joblib"),
        "constants": json.loads((path / "constants.json").read_text(encoding="utf-8")),
        "regressors": joblib.load(path / "length_regressors.joblib"),
        "length_metadata": json.loads(
            (path / "length_metadata.json").read_text(encoding="utf-8")
        ),
    }


def top_k(fields: list[str], scores: dict[str, float], required: list[str], size: int) -> list[str]:
    """保留必选字段后按分数取 Top-k，并以 schema 顺序稳定打破并列。"""
    order = {field: index for index, field in enumerate(fields)}
    ranked = sorted(
        (field for field in fields if field not in required),
        key=lambda field: (-scores[field], order[field]),
    )
    return (required + ranked)[:size]


def estimated_cost(selected: list[str], costs: dict[str, Any], prediction: dict[str, Any]) -> int:
    return sum(
        int(costs["fields"][field]["field_name_tokens"])
        + int(costs["fields"][field]["schema_value_tokens"])
        + int(prediction["predicted_output_tokens"][field])
        for field in selected
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    }


def evaluate_strategy(
    selections: list[list[str]],
    golds: list[set[str]],
    costs: list[int],
    fields: list[str],
) -> dict[str, Any]:
    coverages = [len(set(selected) & gold) / len(gold) for selected, gold in zip(selections, golds)]
    recalls = [len(set(selected) & gold) for selected, gold in zip(selections, golds)]
    total_gold = sum(len(gold) for gold in golds)
    total_hit = sum(recalls)
    frequencies = Counter(field for selected in selections for field in selected)
    return {
        "records": len(selections),
        "gold_fields": total_gold,
        "gold_nonempty_coverage": float(np.mean(coverages)),
        "micro_field_recall": total_hit / total_gold if total_gold else 0.0,
        "gold_field_miss_rate": 1.0 - total_hit / total_gold if total_gold else 0.0,
        "mean_selected_fields": float(np.mean([len(selected) for selected in selections])),
        "estimated_tokens": float(np.mean(costs)),
        "field_selection_frequency": {
            field: frequencies[field] / len(selections) for field in fields
        },
    }


def main() -> None:
    rows = load_jsonl(INPUT)
    splits = load_splits(rows, SPLIT_DIR)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = list(schema)
    costs = json.loads(COSTS_PATH.read_text(encoding="utf-8"))
    weight_config = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    required = weight_config["required_fields"]
    equal_weights = {field: 1.0 for field in fields}
    gold_ids = set(json.loads(GOLD_IDS_PATH.read_text(encoding="utf-8")))
    test_rows = [row for row in splits["test"] if row["record_id"] in gold_ids]
    if len(test_rows) != len(gold_ids):
        raise SystemExit("Gold ID 不完全属于固定测试集")
    gold_rows = load_jsonl(GOLD_REFERENCE_PATH)
    gold_by_id = {
        row["artifact_id"]: {
            field for field in fields if row["reference_result"].get(field) is not None
        }
        for row in gold_rows
    }
    if set(gold_by_id) != gold_ids:
        raise SystemExit("Gold ID 清单与独立 Gold 记录不一致")
    golds = [gold_by_id[row["record_id"]] for row in test_rows]

    runs = []
    for model_seed in MODEL_SEEDS:
        models = load_model(MODEL_ROOT / f"seed_{model_seed}")
        predictions = [predict_fields(json.loads(row["record_text"]), fields, models) for row in test_rows]

        full_selections = [
            select_fields(
                prediction,
                schema,
                costs,
                weight_config["weights"],
                required,
                BUDGET,
            )
            for prediction in predictions
        ]
        full_fields = [selection["selected_fields"] for selection in full_selections]
        full_sizes = [len(selected) for selected in full_fields]
        strategies: dict[str, tuple[list[list[str]], list[int]]] = {
            "m1_weighted_knapsack": (
                full_fields,
                [selection["estimated_cost"] for selection in full_selections],
            ),
            "m1_equal_weight_knapsack": (
                [
                    select_fields(prediction, schema, costs, equal_weights, required, BUDGET)["selected_fields"]
                    for prediction in predictions
                ],
                [],
            ),
            "m1_score_topk": (
                [
                    top_k(fields, prediction["field_scores"], required, size)
                    for prediction, size in zip(predictions, full_sizes)
                ],
                [],
            ),
        }

        # 静态频率仅由该 seed 的 n=100 训练子集统计，测试标签不会参与字段排序。
        metadata = json.loads((MODEL_ROOT / f"seed_{model_seed}" / "metadata.json").read_text(encoding="utf-8"))
        by_id = {row["record_id"]: row for row in splits["train"]}
        train_rows = [by_id[record_id] for record_id in metadata["train_ids"]]
        frequency = {
            field: sum(row["labels"][f"{field}_present"] for row in train_rows) / len(train_rows)
            for field in fields
        }
        strategies["static_frequency_topk"] = (
            [top_k(fields, frequency, required, size) for size in full_sizes],
            [],
        )

        for strategy, (selected, strategy_costs) in strategies.items():
            if not strategy_costs:
                strategy_costs = [
                    estimated_cost(choice, costs, prediction)
                    for choice, prediction in zip(selected, predictions)
                ]
            result = evaluate_strategy(selected, golds, strategy_costs, fields)
            result.update({"model_seed": model_seed, "strategy": strategy, "random_seed": None})
            runs.append(result)

        for random_seed in RANDOM_SEEDS:
            rng = np.random.default_rng(random_seed)
            selected = []
            for size in full_sizes:
                optional = [field for field in fields if field not in required]
                sampled = rng.choice(optional, size=max(0, size - len(required)), replace=False).tolist()
                selected.append(required + sampled)
            result = evaluate_strategy(
                selected,
                [*golds],
                [estimated_cost(choice, costs, prediction) for choice, prediction in zip(selected, predictions)],
                fields,
            )
            result.update({"model_seed": model_seed, "strategy": "random_topk", "random_seed": random_seed})
            runs.append(result)

    metrics = [
        "gold_nonempty_coverage",
        "micro_field_recall",
        "gold_field_miss_rate",
        "mean_selected_fields",
        "estimated_tokens",
    ]
    summary = []
    for strategy in sorted({run["strategy"] for run in runs}):
        selected = [run for run in runs if run["strategy"] == strategy]
        row: dict[str, Any] = {"strategy": strategy, "replicates": len(selected)}
        for metric in metrics:
            row.update({f"{metric}_{name}": value for name, value in summarize([run[metric] for run in selected]).items()})
        summary.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "budget": BUDGET,
            "model_size": 100,
            "model_seeds": list(MODEL_SEEDS),
            "random_seeds": list(RANDOM_SEEDS),
            "evaluation_split": "fixed test split Gold subset",
            "gold_reference": str(GOLD_REFERENCE_PATH.relative_to(ROOT)),
            "records": len(test_rows),
            "required_fields": required,
            "comparability": "All strategies retain required fields and match weighted strategy per-record field count.",
        },
        "runs": runs,
        "summary": summary,
    }
    (OUT_DIR / f"a4_selection_ablation_budget_{BUDGET}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT_DIR / f"a4_selection_ablation_budget_{BUDGET}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    labels = [row["strategy"] for row in summary]
    coverage = [row["gold_nonempty_coverage_mean"] for row in summary]
    coverage_ci = [row["gold_nonempty_coverage_ci95"] for row in summary]
    token_cost = [row["estimated_tokens_mean"] for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, coverage, yerr=coverage_ci, capsize=3)
    axes[0].set_ylabel("Gold nonempty coverage")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, token_cost)
    axes[1].set_ylabel("Estimated tokens per record")
    for axis in axes:
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"a4_selection_ablation_budget_{BUDGET}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
