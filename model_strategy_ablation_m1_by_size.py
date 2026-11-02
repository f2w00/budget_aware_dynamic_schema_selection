#!/usr/bin/env python3
"""离线运行 M1 全训练规模消融，产物与既有 M1 实验完全隔离。"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib

import model_strategy_ablation as m1
import training_size_ablation as base


SIZES = [50, 100, 250, 500, 1000, 2500, 5410]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 全训练规模消融")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/model_strategy_ablation/m1_by_size"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/model_strategy_ablation/m1_by_size"))
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=base.SEEDS)
    parser.add_argument("--min-length-samples", type=int, default=10)
    return parser.parse_args()


def budget_baseline(test_rows, fields, schema, costs, weights, predictions):
    baseline = {budget: [] for budget in base.BUDGETS}
    for index in range(len(test_rows)):
        prediction = {
            "field_scores": {field: float(predictions["field_scores"][field][index]) for field in fields},
            "predicted_output_tokens": {
                field: int(predictions["predicted_output_tokens"][field][index]) for field in fields
            },
        }
        for budget in base.BUDGETS:
            selected = base.select_fields(
                prediction, schema, costs, weights["weights"],
                weights["required_fields"], budget,
            )
            baseline[budget].append(set(selected["selected_fields"]))
    return baseline


def main() -> None:
    config = parse_args()
    rows = base.load_jsonl(config.input)
    splits = base.load_splits(rows, config.split_dir)
    fields = list(json.loads(config.schema.read_text(encoding="utf-8")))
    schema = dict(json.loads(config.schema.read_text(encoding="utf-8")))
    costs = json.loads(config.costs.read_text(encoding="utf-8"))
    weights = json.loads(config.weights.read_text(encoding="utf-8"))
    gold_ids = set(json.loads(config.gold_ids.read_text(encoding="utf-8")))
    train_ids = [row["record_id"] for row in splits["train"]]
    validation_ids = [row["record_id"] for row in splits["validation"]]
    test_ids = [row["record_id"] for row in splits["test"]]
    if gold_ids & set(train_ids + validation_ids) or not gold_ids <= set(test_ids):
        raise SystemExit("Gold 划分不满足固定隔离约束")
    runs = []
    for seed in config.seeds:
        order = base.nested_order(splits["train"], fields, seed)
        _, full_metrics, full_predictions = m1.train_m1(
            splits["train"], splits["validation"], splits["test"], fields, seed,
            config.min_length_samples,
        )
        full_summary = base.summarize_metrics(full_metrics, full_predictions, splits["test"], fields)
        full_baseline = budget_baseline(
            splits["test"], fields, schema, costs, weights, full_predictions,
        )
        for size in config.sizes:
            if size > len(order):
                raise SystemExit(f"训练规模超过训练池: {size}")
            train_rows = [splits["train"][index] for index in order[:size]]
            models, metrics, predictions = m1.train_m1(
                train_rows, splits["validation"], splits["test"], fields, seed,
                config.min_length_samples,
            )
            summary = base.summarize_metrics(metrics, predictions, splits["test"], fields)
            budgets = base.evaluate_budgets(
                splits["test"], fields, predictions, schema, costs, weights,
                full_baseline, gold_ids,
            )
            metadata = {
                "method": "M1_char_word_tfidf_threshold_shrinkage",
                "size": size,
                "seed": seed,
                "train_ids": [row["record_id"] for row in train_rows],
                "train_pool_hash": base.sha_ids(train_ids),
                "validation_ids_hash": base.sha_ids(validation_ids),
                "test_ids_hash": base.sha_ids(test_ids),
                "gold_ids_hash": base.sha_ids(sorted(gold_ids)),
                "sampling": "nested_multilabel_deficit_greedy",
                "tfidf": {"char_ngram": [2, 5], "word_ngram": [1, 2],
                          "char_max_features": 100000, "word_max_features": 50000},
                "threshold": "validation F1 optimum shrunk 25% toward 0.5",
                "llm_api_called": False,
                "sklearn_version": base.sklearn.__version__,
                "python_version": platform.python_version(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            model_dir = config.model_root / str(size) / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(models, model_dir / "model.joblib")
            (model_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            report_dir = config.report_root / "runs" / str(size) / f"seed_{seed}"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "metrics.json").write_text(
                json.dumps({"metadata": metadata, "presence": metrics["fields"],
                            "length": metrics["length_fields"], "budgets": budgets},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            run = {
                "size": size, "seed": seed, **summary,
                "budget_200_gold_nonempty_coverage": budgets[0]["gold_nonempty_coverage"],
                "budget_400_gold_nonempty_coverage": budgets[1]["gold_nonempty_coverage"],
                "budget_200_estimated_cost": budgets[0]["estimated_cost"],
                "budget_400_estimated_cost": budgets[1]["estimated_cost"],
                "presence_macro_recovery": summary["presence_macro_f1"] / full_summary["presence_macro_f1"],
                "presence_micro_recovery": summary["presence_micro_f1"] / full_summary["presence_micro_f1"],
                "length_mae_recovery": full_summary["length_mae"] / summary["length_mae"],
            }
            runs.append(run)
            print(json.dumps(run, ensure_ascii=False))
    base.aggregate(runs, config.report_root)
    (config.report_root / "comparison.json").write_text(
        json.dumps({"method": "M1_by_size", "runs": runs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
