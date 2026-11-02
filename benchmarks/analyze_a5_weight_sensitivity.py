#!/usr/bin/env python3
"""A5 字段权重敏感性分析，不调用 LLM 或 embedding 服务。"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
from analyze_a4_selection_ablation import load_model


INPUT = ROOT / "training_dataset.jsonl"
SPLIT_DIR = ROOT / "data/splits"
SCHEMA_PATH = ROOT / "simple2.json"
COSTS_PATH = ROOT / "schema_costs.json"
WEIGHTS_PATH = ROOT / "field_weights.json"
GOLD_IDS_PATH = ROOT / "data/gold/gold_main_ids.json"
GOLD_REFERENCE_PATH = ROOT / "data/gold/agent_tasks/gold_main.jsonl"
MODEL_ROOT = ROOT / "models/training_size_ablation/100"
OUT_DIR = ROOT / "benchmarks/analysis/a5_weight_sensitivity"
MODEL_SEEDS = (42, 43, 44, 45, 46)
PERTURBATION_SEEDS = tuple(range(200, 220))
BUDGETS = (200, 400)


def normalize(weights: dict[str, float]) -> dict[str, float]:
    mean = sum(weights.values()) / len(weights)
    return {field: value / mean for field, value in weights.items()}


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    # 小样本模型 seed 使用 t 临界值，扰动组合多时采用大样本近似。
    critical = 2.776 if len(array) == 5 else 1.984 if len(array) == 100 else 1.96
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(critical * std / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    }


def evaluate(
    selections: list[dict[str, Any]], golds: list[set[str]], fields: list[str]
) -> dict[str, Any]:
    selected = [set(item["selected_fields"]) for item in selections]
    total_gold = sum(len(gold) for gold in golds)
    hits = sum(len(choice & gold) for choice, gold in zip(selected, golds))
    frequencies = Counter(field for choice in selected for field in choice)
    return {
        "records": len(selected),
        "gold_fields": total_gold,
        "gold_nonempty_coverage": float(
            np.mean([len(choice & gold) / len(gold) for choice, gold in zip(selected, golds)])
        ),
        "micro_field_recall": hits / total_gold,
        "gold_field_miss_rate": 1.0 - hits / total_gold,
        "mean_selected_fields": float(np.mean([len(choice) for choice in selected])),
        "estimated_tokens": float(np.mean([item["estimated_cost"] for item in selections])),
        "budget_satisfied_rate": float(
            np.mean([item["estimated_cost"] <= item["budget"] for item in selections])
        ),
        "field_selection_frequency": {
            field: frequencies[field] / len(selected) for field in fields
        },
    }


def build_training_weights(
    train_rows: list[dict[str, Any]], fields: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    frequency = {
        field: sum(row["labels"][f"{field}_present"] for row in train_rows) / len(train_rows)
        for field in fields
    }
    # 反频率仅作压力测试，截断避免极罕见字段获得不成比例的权重。
    inverse = {field: 1.0 / max(frequency[field], 0.05) for field in fields}
    return normalize(frequency), normalize(inverse)


def main() -> None:
    rows = load_jsonl(INPUT)
    splits = load_splits(rows, SPLIT_DIR)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = list(schema)
    costs = json.loads(COSTS_PATH.read_text(encoding="utf-8"))
    config = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    required = config["required_fields"]
    gold_ids = set(json.loads(GOLD_IDS_PATH.read_text(encoding="utf-8")))
    test_rows = [row for row in splits["test"] if row["record_id"] in gold_ids]
    gold_rows = load_jsonl(GOLD_REFERENCE_PATH)
    gold_by_id = {
        row["artifact_id"]: {
            field for field in fields if row["reference_result"].get(field) is not None
        }
        for row in gold_rows
    }
    if len(test_rows) != len(gold_ids) or set(gold_by_id) != gold_ids:
        raise SystemExit("固定测试集 Gold 文件不完整或不一致")
    golds = [gold_by_id[row["record_id"]] for row in test_rows]
    train_by_id = {row["record_id"]: row for row in splits["train"]}
    current_weights = {field: float(config["weights"][field]) for field in fields}
    runs = []

    for model_seed in MODEL_SEEDS:
        model_path = MODEL_ROOT / f"seed_{model_seed}"
        metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        train_rows = [train_by_id[record_id] for record_id in metadata["train_ids"]]
        frequency_weights, inverse_frequency_weights = build_training_weights(train_rows, fields)
        weights_by_strategy = {
            "current_manual": (current_weights, None),
            "equal": ({field: 1.0 for field in fields}, None),
            "training_frequency": (frequency_weights, None),
            "inverse_frequency_stress": (inverse_frequency_weights, None),
        }
        for perturbation_seed in PERTURBATION_SEEDS:
            rng = np.random.default_rng(perturbation_seed)
            factors = rng.uniform(0.8, 1.2, len(fields))
            weights_by_strategy[f"current_perturbed_{perturbation_seed}"] = (
                {field: current_weights[field] * float(factor) for field, factor in zip(fields, factors)},
                perturbation_seed,
            )

        models = load_model(model_path)
        predictions = [
            predict_fields(json.loads(row["record_text"]), fields, models)
            for row in test_rows
        ]
        for budget in BUDGETS:
            for strategy, (weights, perturbation_seed) in weights_by_strategy.items():
                selections = [
                    select_fields(prediction, schema, costs, weights, required, budget)
                    for prediction in predictions
                ]
                result = evaluate(selections, golds, fields)
                result.update(
                    {
                        "budget": budget,
                        "model_seed": model_seed,
                        "strategy": strategy,
                        "perturbation_seed": perturbation_seed,
                    }
                )
                runs.append(result)

    metrics = (
        "gold_nonempty_coverage",
        "micro_field_recall",
        "gold_field_miss_rate",
        "mean_selected_fields",
        "estimated_tokens",
        "budget_satisfied_rate",
    )
    summary = []
    for budget in BUDGETS:
        for strategy in (
            "current_manual",
            "equal",
            "training_frequency",
            "inverse_frequency_stress",
            "current_perturbed",
        ):
            selected = [
                run
                for run in runs
                if run["budget"] == budget
                and (
                    run["strategy"] == strategy
                    or (strategy == "current_perturbed" and run["strategy"].startswith(strategy))
                )
            ]
            row: dict[str, Any] = {
                "budget": budget,
                "strategy": strategy,
                "replicates": len(selected),
            }
            for metric in metrics:
                values = [run[metric] for run in selected]
                row.update({f"{metric}_{key}": value for key, value in summarize(values).items()})
            summary.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "budgets": list(BUDGETS),
            "model_size": 100,
            "model_seeds": list(MODEL_SEEDS),
            "perturbation_seeds": list(PERTURBATION_SEEDS),
            "perturbation": "per-field multiplicative uniform [0.8, 1.2]",
            "frequency_weights": "positive rates in each seed's n=100 training subset, mean-normalized",
            "inverse_frequency_stress": "inverse positive rates clipped at 0.05, mean-normalized",
            "evaluation_split": "fixed test split Gold subset",
            "gold_reference": str(GOLD_REFERENCE_PATH.relative_to(ROOT)),
            "records": len(test_rows),
            "required_fields": required,
        },
        "runs": runs,
        "summary": summary,
    }
    (OUT_DIR / "a5_weight_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT_DIR / "a5_weight_sensitivity.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, budget in zip(axes, BUDGETS):
        selected = [row for row in summary if row["budget"] == budget]
        labels = [row["strategy"] for row in selected]
        values = [row["gold_nonempty_coverage_mean"] for row in selected]
        errors = [row["gold_nonempty_coverage_ci95"] for row in selected]
        axis.bar(labels, values, yerr=errors, capsize=3)
        axis.set_title(f"Budget {budget}")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Gold nonempty coverage")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "a5_weight_sensitivity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
