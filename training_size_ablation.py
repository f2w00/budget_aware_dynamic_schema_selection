#!/usr/bin/env python3
"""离线运行字段选择器训练样本量消融，不调用 LLM API。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from budget_selector import select_fields
from train_field_selector import choose_threshold


SIZES = [50, 100, 250, 500, 1000, 2500, 5410]
SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [200, 400]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字段选择器训练量消融")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/training_size_ablation"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/training_size_ablation"))
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--min-length-samples", type=int, default=10)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_ids(ids: list[Any]) -> str:
    payload = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_splits(rows: list[dict[str, Any]], split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_id = {row["record_id"]: row for row in rows}
    result = {}
    id_sets = {}
    for name in ("train", "validation", "test"):
        ids = json.loads((split_dir / f"{name}_ids.json").read_text(encoding="utf-8"))
        id_sets[name] = set(ids)
        missing = id_sets[name] - by_id.keys()
        if missing:
            raise SystemExit(f"{name} 划分存在未知 record_id: {next(iter(missing))}")
        result[name] = [by_id[item] for item in ids]
    if set.union(*id_sets.values()) != set(by_id) or sum(map(len, id_sets.values())) != len(by_id):
        raise SystemExit("固定划分未覆盖全部样本或存在 ID 重叠")
    return result


def nested_order(rows: list[dict[str, Any]], fields: list[str], seed: int) -> list[int]:
    """用标签缺口贪心构建嵌套前缀，无法严格分层时保持可解释记录。"""
    rng = np.random.default_rng(seed)
    labels = np.asarray(
        [[int(row["labels"][f"{field}_present"]) for field in fields] for row in rows],
        dtype=np.float64,
    )
    target = labels.mean(axis=0)
    remaining = np.arange(len(rows))
    order: list[int] = []
    counts = np.zeros(len(fields))
    while len(remaining):
        step = len(order) + 1
        deficit = target * step - counts
        scores = labels[remaining] @ deficit + rng.random(len(remaining)) * 1e-9
        selected = int(np.argmax(scores))
        index = int(remaining[selected])
        order.append(index)
        counts += labels[index]
        remaining = np.delete(remaining, selected)
    return order


def train_and_evaluate(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fields: list[str],
    seed: int,
    min_length_samples: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    texts = [row["record_text"] for row in train_rows]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2, max_features=100_000, sublinear_tf=True
    )
    train_matrix = vectorizer.fit_transform(texts)
    validation_matrix = vectorizer.transform([row["record_text"] for row in validation_rows])
    test_matrix = vectorizer.transform([row["record_text"] for row in test_rows])
    targets = {
        name: np.asarray(
            [[int(row["labels"][f"{field}_present"]) for field in fields] for row in values],
            dtype=np.int8,
        )
        for name, values in (("validation", validation_rows), ("test", test_rows))
    }
    classifiers: dict[str, Any] = {}
    constants: dict[str, float] = {}
    regressors: dict[str, Any] = {}
    length_metadata: dict[str, Any] = {}
    field_metrics = []
    length_metrics = []
    predictions = {"field_scores": {}, "predicted_output_tokens": {}}
    for index, field in enumerate(fields):
        y_train = np.asarray([int(row["labels"][f"{field}_present"]) for row in train_rows])
        y_validation = targets["validation"][:, index]
        y_test = targets["test"][:, index]
        if len(np.unique(y_train)) < 2:
            constants[field] = float(y_train[0]) if len(y_train) else 0.0
            threshold = 0.5
            test_scores = np.full(len(test_rows), constants[field])
        else:
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
            )
            model.fit(train_matrix, y_train)
            classifiers[field] = model
            threshold = choose_threshold(y_validation, model.predict_proba(validation_matrix)[:, 1])
            test_scores = model.predict_proba(test_matrix)[:, 1]
        test_pred = test_scores >= threshold
        field_metrics.append({
            "field": field,
            "positive_test": int(y_test.sum()),
            "threshold": float(threshold),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, test_scores))
            if len(np.unique(y_test)) > 1 else None,
        })
        train_lengths = np.asarray(
            [row["field_output_tokens"][field] for row in train_rows], dtype=float
        )[y_train == 1]
        validation_lengths = np.asarray(
            [row["field_output_tokens"][field] for row in validation_rows], dtype=float
        )[y_validation == 1]
        test_lengths = np.asarray(
            [row["field_output_tokens"][field] for row in test_rows], dtype=float
        )[y_test == 1]
        if len(train_lengths) < min_length_samples:
            value = float(np.median(train_lengths)) if len(train_lengths) else 0.0
            length_metadata[field] = {"type": "median_constant", "train_count": len(train_lengths), "value": value}
            validation_predictions = np.full(len(validation_lengths), value)
            test_predictions = np.full(len(test_lengths), value)
        else:
            regressor = Ridge(alpha=10.0)
            regressor.fit(train_matrix[y_train == 1], np.log1p(train_lengths))
            regressors[field] = regressor
            length_metadata[field] = {"type": "ridge_log1p", "train_count": len(train_lengths), "alpha": 10.0}
            validation_predictions = np.maximum(0, np.expm1(regressor.predict(validation_matrix[y_validation == 1])))
            test_predictions = np.maximum(0, np.expm1(regressor.predict(test_matrix[y_test == 1])))
        test_length_values = np.zeros(len(test_rows), dtype=float)
        test_length_values[y_test == 1] = test_predictions
        length_metrics.append({
            "field": field,
            "train_count": int(len(train_lengths)),
            "positive_test": int(len(test_lengths)),
            "model_type": length_metadata[field]["type"],
            "mae": float(mean_absolute_error(test_lengths, test_predictions)) if len(test_lengths) else None,
            "rmse": float(mean_squared_error(test_lengths, test_predictions) ** 0.5) if len(test_lengths) else None,
            "median_absolute_error": float(np.median(np.abs(test_lengths - test_predictions))) if len(test_lengths) else None,
            "spearman": float(spearmanr(test_lengths, test_predictions).statistic)
            if len(test_lengths) > 1 and np.ptp(test_lengths) > 0 and np.ptp(test_predictions) > 0 else None,
        })
        predictions["field_scores"][field] = test_scores
        predictions["predicted_output_tokens"][field] = test_length_values
    return (
        {"vectorizer": vectorizer, "classifiers": classifiers, "constants": constants,
         "regressors": regressors, "length_metadata": length_metadata},
        {"fields": field_metrics, "length_fields": length_metrics},
        predictions,
    )


def evaluate_budgets(
    test_rows: list[dict[str, Any]], fields: list[str], models: dict[str, Any], schema: dict[str, str],
    costs: dict[str, Any], weights: dict[str, Any], baseline: dict[int, list[set[str]]],
    gold_ids: set[str],
) -> list[dict[str, Any]]:
    output = []
    selections_by_budget = {budget: [] for budget in BUDGETS}
    for index, row in enumerate(test_rows):
        prediction = {"field_scores": {field: float(models["field_scores"][field][index]) for field in fields},
                      "predicted_output_tokens": {field: int(models["predicted_output_tokens"][field][index]) for field in fields}}
        gold = {
            field for field in fields
            if row["record_id"] in gold_ids
            and row["labels"][f"{field}_present"]
        }
        for budget in BUDGETS:
            selection = select_fields(prediction, schema, costs, weights["weights"], weights["required_fields"], budget)
            selected = set(selection["selected_fields"])
            selections_by_budget[budget].append((selected, selection, gold))
    for budget, values in selections_by_budget.items():
        nonempty = [
            (selected, item, gold)
            for selected, item, gold in values
            if gold
        ]
        coverages = [
            len(selected & gold) / len(gold)
            for selected, _, gold in nonempty
        ]
        costs_values = [item["estimated_cost"] for _, item, _ in values]
        counts = [len(selected) for selected, _, _ in values]
        all_selected = [field for selected, _, _ in values for field in selected]
        output.append({
            "budget": budget,
            "records": len(values),
            "gold_nonempty_records": len(nonempty),
            "gold_nonempty_coverage": float(np.mean(coverages)) if coverages else None,
            "gold_coverage": float(np.mean([
                len(selected & gold) / len(gold) if gold else 1.0
                for selected, _, gold in values
            ])),
            "selected_fields": float(np.mean(counts)),
            "estimated_cost": float(np.mean(costs_values)),
            "budget_satisfied_rate": float(np.mean([item["estimated_cost"] <= budget for _, item, _ in values])),
             "jaccard_vs_full": float(np.mean([
                 len(selected & baseline[budget][index]) / len(selected | baseline[budget][index])
                 if selected | baseline[budget][index] else 1.0
                 for index, (selected, _, _) in enumerate(values)
             ])),
            "field_selection_frequency": {field: all_selected.count(field) / len(values) for field in fields},
        })
    return output


def save_model(models: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["vectorizer"], path / "vectorizer.joblib")
    joblib.dump(models["classifiers"], path / "classifiers.joblib")
    joblib.dump(models["regressors"], path / "length_regressors.joblib")
    for name in ("constants", "length_metadata"):
        (path / f"{name}.json").write_text(json.dumps(models[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_metrics(
    metrics: dict[str, Any], predictions: dict[str, Any],
    test_rows: list[dict[str, Any]], fields: list[str],
) -> dict[str, float]:
    presence_macro = float(np.mean([item["f1"] for item in metrics["fields"]]))
    truth = [
        int(row["labels"][f"{field}_present"])
        for row in test_rows for field in fields
    ]
    predicted = [
        int(float(predictions["field_scores"][field][index]) >= metrics["fields"][field_index]["threshold"])
        for index in range(len(test_rows))
        for field_index, field in enumerate(fields)
    ]
    length_values = [
        item["mae"] for item in metrics["length_fields"]
        if item["mae"] is not None
    ]
    return {
        "presence_macro_f1": presence_macro,
        "presence_micro_f1": float(f1_score(truth, predicted, zero_division=0)),
        "length_mae": float(np.mean(length_values)) if length_values else 0.0,
    }


def aggregate(runs: list[dict[str, Any]], report_root: Path) -> None:
    numeric_keys = ["presence_macro_f1", "presence_micro_f1", "length_mae",
                    "budget_200_gold_nonempty_coverage", "budget_400_gold_nonempty_coverage",
                    "budget_200_estimated_cost", "budget_400_estimated_cost",
                    "presence_macro_recovery", "presence_micro_recovery", "length_mae_recovery"]
    summary = []
    for size in sorted({run["size"] for run in runs}):
        selected = [run for run in runs if run["size"] == size]
        row = {"size": size, "seeds": len(selected)}
        for key in numeric_keys:
            values = np.asarray([run[key] for run in selected], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{key}_ci95"] = float(1.96 * values.std(ddof=1) / len(values) ** 0.5) if len(values) > 1 else 0.0
        summary.append(row)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "summary.json").write_text(json.dumps({"runs": runs, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_root / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    baseline = next(row for row in summary if row["size"] == max(row["size"] for row in summary))
    thresholds = {}
    for metric in ("presence_macro_f1", "presence_micro_f1", "length_mae_recovery"):
        target = 1.0 if metric == "length_mae_recovery" else None
        values = [(row["size"], row[f"{metric}_mean"]) for row in summary]
        thresholds[metric] = {
            str(rate): next((size for size, value in values if (value >= rate if target is None else value >= rate)), None)
            for rate in (0.90, 0.95, 0.99)
        }
    (report_root / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figures = report_root / "figures"
    figures.mkdir(exist_ok=True)
    if not summary:
        return
    plots = [("presence_macro_f1", "Presence Macro-F1", "presence_macro_f1.png"),
             ("length_mae", "Length MAE", "length_mae.png"),
             ("budget_200_gold_nonempty_coverage", "Budget Gold Coverage", "budget_coverage.png"),
             ("presence_macro_recovery", "Presence Macro-F1 Recovery", "recovery.png")]
    for metric, label, filename in plots:
        values = [row[f"{metric}_mean"] for row in summary]
        errors = [row[f"{metric}_ci95"] for row in summary]
        plt.figure(figsize=(7, 4))
        plt.errorbar([row["size"] for row in summary], values, yerr=errors, marker="o", capsize=3)
        plt.xscale("log")
        plt.xticks([row["size"] for row in summary], [str(row["size"]) for row in summary])
        plt.xlabel("Training samples")
        plt.ylabel(label)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=180)
        plt.close()


def main() -> None:
    config = args()
    rows = load_jsonl(config.input)
    splits = load_splits(rows, config.split_dir)
    fields = list(json.loads(config.schema.read_text(encoding="utf-8")))
    costs = json.loads(config.costs.read_text(encoding="utf-8"))
    weights = json.loads(config.weights.read_text(encoding="utf-8"))
    train_ids = [row["record_id"] for row in splits["train"]]
    validation_ids = [row["record_id"] for row in splits["validation"]]
    test_ids = [row["record_id"] for row in splits["test"]]
    gold_ids = set(json.loads(config.gold_ids.read_text(encoding="utf-8")))
    if gold_ids & set(train_ids + validation_ids):
        raise SystemExit("Gold 与训练或验证集重叠")
    if not gold_ids <= set(test_ids):
        raise SystemExit("Gold 不是固定测试集子集")
    schema = dict(json.loads(config.schema.read_text(encoding="utf-8")))
    runs = []
    full_baselines = {}
    orders = {seed: nested_order(splits["train"], fields, seed) for seed in config.seeds}
    for seed in config.seeds:
        order = orders[seed]
        full_models, full_metrics, full_predictions = train_and_evaluate(
            splits["train"],
            splits["validation"],
            splits["test"],
            fields,
            seed,
            config.min_length_samples,
        )
        full_baselines[seed] = summarize_metrics(
            full_metrics, full_predictions, splits["test"], fields
        )
        full_baseline = {budget: [] for budget in BUDGETS}
        for index in range(len(splits["test"])):
            prediction = {
                "field_scores": {
                    field: float(full_predictions["field_scores"][field][index])
                    for field in fields
                },
                "predicted_output_tokens": {
                    field: int(full_predictions["predicted_output_tokens"][field][index])
                    for field in fields
                },
            }
            for budget in BUDGETS:
                selection = select_fields(
                    prediction,
                    schema,
                    costs,
                    weights["weights"],
                    weights["required_fields"],
                    budget,
                )
                full_baseline[budget].append(set(selection["selected_fields"]))
        for size in config.sizes:
            if size > len(order):
                raise SystemExit(f"训练规模超过训练池: {size}")
            train_rows = [splits["train"][index] for index in order[:size]]
            model_dir = config.model_root / str(size) / f"seed_{seed}"
            report_dir = config.report_root / "runs" / str(size) / f"seed_{seed}"
            if size == max(config.sizes):
                models, metrics, predictions = full_models, full_metrics, full_predictions
            else:
                models, metrics, predictions = train_and_evaluate(
                    train_rows,
                    splits["validation"],
                    splits["test"],
                    fields,
                    seed,
                    config.min_length_samples,
                )
            metric_summary = summarize_metrics(metrics, predictions, splits["test"], fields)
            budget_metrics = evaluate_budgets(
                splits["test"],
                fields,
                predictions,
                schema,
                costs,
                weights,
                full_baseline,
                gold_ids,
            )
            length_mae = metric_summary["length_mae"]
            metadata = {"size": size, "seed": seed, "train_ids": [row["record_id"] for row in train_rows], "train_pool_hash": sha_ids(train_ids), "validation_ids_hash": sha_ids(validation_ids), "test_ids_hash": sha_ids(test_ids), "gold_ids_hash": sha_ids(sorted(gold_ids)), "fields": fields, "sklearn_version": sklearn.__version__, "python_version": platform.python_version(), "trained_at": datetime.now(timezone.utc).isoformat(), "llm_api_called": False, "sampling": "nested_multilabel_deficit_greedy"}
            save_model(models, metadata, model_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "metrics.json").write_text(json.dumps({"metadata": metadata, "presence": metrics["fields"], "length": metrics["length_fields"], "budgets": budget_metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            baseline = full_baselines[seed]
            run = {"size": size, "seed": seed, **metric_summary, "budget_200_gold_nonempty_coverage": budget_metrics[0]["gold_nonempty_coverage"], "budget_400_gold_nonempty_coverage": budget_metrics[1]["gold_nonempty_coverage"], "budget_200_estimated_cost": budget_metrics[0]["estimated_cost"], "budget_400_estimated_cost": budget_metrics[1]["estimated_cost"], "presence_macro_recovery": metric_summary["presence_macro_f1"] / baseline["presence_macro_f1"] if baseline["presence_macro_f1"] else 0.0, "presence_micro_recovery": metric_summary["presence_micro_f1"] / baseline["presence_micro_f1"] if baseline["presence_micro_f1"] else 0.0, "length_mae_recovery": baseline["length_mae"] / metric_summary["length_mae"] if metric_summary["length_mae"] else 0.0}
            runs.append(run)
            print(json.dumps({"size": size, "seed": seed, "report": str(report_dir)}, ensure_ascii=False))
    aggregate(runs, config.report_root)


if __name__ == "__main__":
    main()
