#!/usr/bin/env python3
"""离线字段选择模型策略消融：字符+词级 TF-IDF，独立于正式实验目录。"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack
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
from scipy.stats import spearmanr

import training_size_ablation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字段选择模型策略消融")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/model_strategy_ablation/m1"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/model_strategy_ablation/m1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=base.SEEDS)
    parser.add_argument("--min-length-samples", type=int, default=10)
    return parser.parse_args()


def train_m1(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
             test_rows: list[dict[str, Any]], fields: list[str], seed: int,
             min_length_samples: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    texts = [row["record_text"] for row in train_rows]
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                           max_features=100_000, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                           max_features=50_000, sublinear_tf=True)
    train_matrix = hstack([char.fit_transform(texts), word.fit_transform(texts)], format="csr")
    validation_matrix = hstack([
        char.transform([row["record_text"] for row in validation_rows]),
        word.transform([row["record_text"] for row in validation_rows]),
    ], format="csr")
    test_matrix = hstack([
        char.transform([row["record_text"] for row in test_rows]),
        word.transform([row["record_text"] for row in test_rows]),
    ], format="csr")
    classifiers: dict[str, Any] = {}
    regressors: dict[str, Any] = {}
    constants: dict[str, float] = {}
    length_metadata: dict[str, Any] = {}
    presence, lengths = [], []
    predictions = {"field_scores": {}, "predicted_output_tokens": {}}
    for index, field in enumerate(fields):
        y_train = np.asarray([int(row["labels"][f"{field}_present"]) for row in train_rows])
        y_validation = np.asarray([int(row["labels"][f"{field}_present"]) for row in validation_rows])
        y_test = np.asarray([int(row["labels"][f"{field}_present"]) for row in test_rows])
        if len(np.unique(y_train)) < 2:
            constants[field] = float(y_train[0]) if len(y_train) else 0.0
            threshold, scores = 0.5, np.full(len(test_rows), constants[field])
        else:
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000,
                                            random_state=seed, solver="liblinear")
            classifier.fit(train_matrix, y_train)
            classifiers[field] = classifier
            raw_threshold = base.choose_threshold(
                y_validation, classifier.predict_proba(validation_matrix)[:, 1]
            )
            # 验证集很小时限制阈值偏移，降低长尾字段因偶然样本造成的过拟合。
            threshold = float(np.clip(0.5 + 0.75 * (raw_threshold - 0.5), 0.10, 0.90))
            scores = classifier.predict_proba(test_matrix)[:, 1]
        predicted = scores >= threshold
        presence.append({
            "field": field, "positive_test": int(y_test.sum()), "threshold": threshold,
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, scores))
            if len(np.unique(y_test)) > 1 else None,
        })
        train_lengths = np.asarray([row["field_output_tokens"][field] for row in train_rows])[y_train == 1]
        test_lengths = np.asarray([row["field_output_tokens"][field] for row in test_rows])[y_test == 1]
        if len(train_lengths) < min_length_samples:
            value = float(np.median(train_lengths)) if len(train_lengths) else 0.0
            length_metadata[field] = {"type": "median_constant", "train_count": len(train_lengths), "value": value}
            test_pred_lengths = np.full(len(test_lengths), value)
        else:
            regressor = Ridge(alpha=10.0).fit(train_matrix[y_train == 1], np.log1p(train_lengths))
            regressors[field] = regressor
            length_metadata[field] = {"type": "ridge_log1p", "train_count": len(train_lengths), "alpha": 10.0}
            test_pred_lengths = np.maximum(0, np.expm1(regressor.predict(test_matrix[y_test == 1])))
        full_lengths = np.zeros(len(test_rows), dtype=float)
        full_lengths[y_test == 1] = test_pred_lengths
        lengths.append({
            "field": field, "train_count": int(len(train_lengths)), "positive_test": int(len(test_lengths)),
            "model_type": length_metadata[field]["type"],
            "mae": float(mean_absolute_error(test_lengths, test_pred_lengths)) if len(test_lengths) else None,
            "rmse": float(mean_squared_error(test_lengths, test_pred_lengths) ** 0.5) if len(test_lengths) else None,
            "median_absolute_error": float(np.median(np.abs(test_lengths - test_pred_lengths))) if len(test_lengths) else None,
            "spearman": float(spearmanr(test_lengths, test_pred_lengths).statistic)
            if len(test_lengths) > 1 and np.ptp(test_lengths) > 0 and np.ptp(test_pred_lengths) > 0 else None,
        })
        predictions["field_scores"][field] = scores
        predictions["predicted_output_tokens"][field] = full_lengths
    models = {"char_vectorizer": char, "word_vectorizer": word, "classifiers": classifiers,
              "regressors": regressors, "constants": constants, "length_metadata": length_metadata}
    return models, {"fields": presence, "length_fields": lengths}, predictions


def save_model(models: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path / "model.joblib")
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = parse_args()
    rows = base.load_jsonl(config.input)
    splits = base.load_splits(rows, config.split_dir)
    fields = list(json.loads(config.schema.read_text(encoding="utf-8")))
    schema = dict(json.loads(config.schema.read_text(encoding="utf-8")))
    costs = json.loads(config.costs.read_text(encoding="utf-8"))
    weights = json.loads(config.weights.read_text(encoding="utf-8"))
    gold_ids = set(json.loads(config.gold_ids.read_text(encoding="utf-8")))
    if gold_ids & {row["record_id"] for row in splits["train"] + splits["validation"]}:
        raise SystemExit("Gold 与训练或验证集重叠")
    if not gold_ids <= {row["record_id"] for row in splits["test"]}:
        raise SystemExit("Gold 不是固定测试集子集")
    order = {seed: base.nested_order(splits["train"], fields, seed) for seed in config.seeds}
    all_runs = []
    for seed in config.seeds:
        # 完整模型只用于同 seed 的恢复率和预算选择一致性基线。
        _, full_metrics, full_predictions = base.train_and_evaluate(
            splits["train"], splits["validation"], splits["test"], fields, seed, config.min_length_samples
        )
        full_summary = base.summarize_metrics(full_metrics, full_predictions, splits["test"], fields)
        full_baseline = {budget: [] for budget in base.BUDGETS}
        for index in range(len(splits["test"])):
            prediction = {"field_scores": {field: float(full_predictions["field_scores"][field][index]) for field in fields},
                          "predicted_output_tokens": {field: int(full_predictions["predicted_output_tokens"][field][index]) for field in fields}}
            for budget in base.BUDGETS:
                selected = base.select_fields(prediction, schema, costs, weights["weights"], weights["required_fields"], budget)
                full_baseline[budget].append(set(selected["selected_fields"]))
        train_rows = [splits["train"][index] for index in order[seed][:100]]
        models, metrics, predictions = train_m1(train_rows, splits["validation"], splits["test"], fields, seed, config.min_length_samples)
        summary = base.summarize_metrics(metrics, predictions, splits["test"], fields)
        budgets = base.evaluate_budgets(splits["test"], fields, predictions, schema, costs, weights, full_baseline, gold_ids)
        metadata = {"method": "M1_char_word_tfidf_threshold_shrinkage", "size": 100, "seed": seed,
                    "sampling": "same_nested_multilabel_deficit_greedy_as_training_size_ablation",
                    "tfidf": {"char_ngram": [2, 5], "word_ngram": [1, 2], "char_max_features": 100000, "word_max_features": 50000},
                    "threshold": "validation F1 optimum shrunk 25% toward 0.5", "llm_api_called": False,
                    "trained_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
        report_dir = config.report_root / "runs" / f"seed_{seed}"
        report_dir.mkdir(parents=True, exist_ok=True)
        save_model(models, metadata, config.model_root / f"seed_{seed}")
        report = {"metadata": metadata, "presence": metrics["fields"], "length": metrics["length_fields"], "budgets": budgets}
        (report_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run = {"size": 100, "seed": seed, **summary,
               "budget_200_gold_nonempty_coverage": budgets[0]["gold_nonempty_coverage"],
               "budget_400_gold_nonempty_coverage": budgets[1]["gold_nonempty_coverage"],
               "budget_200_estimated_cost": budgets[0]["estimated_cost"], "budget_400_estimated_cost": budgets[1]["estimated_cost"],
               "presence_macro_recovery": summary["presence_macro_f1"] / full_summary["presence_macro_f1"],
               "presence_micro_recovery": summary["presence_micro_f1"] / full_summary["presence_micro_f1"],
               "length_mae_recovery": full_summary["length_mae"] / summary["length_mae"]}
        all_runs.append(run)
        print(json.dumps(run, ensure_ascii=False))
    base.aggregate(all_runs, config.report_root)
    (config.report_root / "comparison.json").write_text(json.dumps({"method": "M1", "runs": all_runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
