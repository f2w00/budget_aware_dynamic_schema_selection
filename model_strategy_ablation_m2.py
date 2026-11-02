#!/usr/bin/env python3
"""离线字段选择模型策略消融 M2：字符/词级概率融合，独立于 M1。"""

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
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)

import training_size_ablation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 字符/词级概率融合消融")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/model_strategy_ablation/m2"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/model_strategy_ablation/m2"))
    parser.add_argument("--sizes", type=int, nargs="+", default=base.SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=base.SEEDS)
    parser.add_argument("--min-length-samples", type=int, default=10)
    return parser.parse_args()


def matrixes(vectorizers: tuple[TfidfVectorizer, TfidfVectorizer], rows: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
    char, word = vectorizers
    texts = [row["record_text"] for row in rows]
    return char.transform(texts), word.transform(texts), hstack(
        [char.transform(texts), word.transform(texts)], format="csr"
    )


def train_m2(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
             test_rows: list[dict[str, Any]], fields: list[str], seed: int,
             min_length_samples: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    texts = [row["record_text"] for row in train_rows]
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                           max_features=100_000, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                           max_features=50_000, sublinear_tf=True)
    char_train = char.fit_transform(texts)
    word_train = word.fit_transform(texts)
    both_train = hstack([char_train, word_train], format="csr")
    char_val, word_val, both_val = matrixes((char, word), validation_rows)
    char_test, word_test, both_test = matrixes((char, word), test_rows)
    classifiers: dict[str, Any] = {}
    regressors: dict[str, Any] = {}
    constants: dict[str, float] = {}
    length_metadata: dict[str, Any] = {}
    presence, lengths = [], []
    predictions = {"field_scores": {}, "predicted_output_tokens": {}}
    for index, field in enumerate(fields):
        y_train = np.asarray([int(row["labels"][f"{field}_present"]) for row in train_rows])
        y_val = np.asarray([int(row["labels"][f"{field}_present"]) for row in validation_rows])
        y_test = np.asarray([int(row["labels"][f"{field}_present"]) for row in test_rows])
        if len(np.unique(y_train)) < 2:
            constants[field] = float(y_train[0]) if len(y_train) else 0.0
            alpha, threshold = 0.5, 0.5
            scores = np.full(len(test_rows), constants[field])
        else:
            char_model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                            random_state=seed, solver="liblinear").fit(char_train, y_train)
            word_model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                            random_state=seed, solver="liblinear").fit(word_train, y_train)
            char_val_score = char_model.predict_proba(char_val)[:, 1]
            word_val_score = word_model.predict_proba(word_val)[:, 1]
            candidates = []
            for candidate_alpha in np.linspace(0.0, 1.0, 5):
                blended = candidate_alpha * char_val_score + (1 - candidate_alpha) * word_val_score
                candidate_threshold = base.choose_threshold(y_val, blended)
                candidates.append((f1_score(y_val, blended >= candidate_threshold, zero_division=0),
                                   float(candidate_alpha), float(candidate_threshold)))
            _, alpha, raw_threshold = max(candidates, key=lambda item: (item[0], abs(item[1] - 0.5)))
            threshold = float(np.clip(0.5 + 0.75 * (raw_threshold - 0.5), 0.10, 0.90))
            classifiers[field] = {"char": char_model, "word": word_model,
                                  "alpha": alpha, "raw_threshold": raw_threshold}
            scores = alpha * char_model.predict_proba(char_test)[:, 1] + (1 - alpha) * word_model.predict_proba(word_test)[:, 1]
        predicted = scores >= threshold
        presence.append({"field": field, "positive_test": int(y_test.sum()), "threshold": threshold,
                         "precision": float(precision_score(y_test, predicted, zero_division=0)),
                         "recall": float(recall_score(y_test, predicted, zero_division=0)),
                         "f1": float(f1_score(y_test, predicted, zero_division=0)),
                         "average_precision": float(average_precision_score(y_test, scores))
                         if len(np.unique(y_test)) > 1 else None})
        train_lengths = np.asarray([row["field_output_tokens"][field] for row in train_rows])[y_train == 1]
        test_lengths = np.asarray([row["field_output_tokens"][field] for row in test_rows])[y_test == 1]
        if len(train_lengths) < min_length_samples:
            value = float(np.median(train_lengths)) if len(train_lengths) else 0.0
            length_metadata[field] = {"type": "median_constant", "train_count": len(train_lengths), "value": value}
            test_pred_lengths = np.full(len(test_lengths), value)
        else:
            regressor = Ridge(alpha=10.0).fit(both_train[y_train == 1], np.log1p(train_lengths))
            regressors[field] = regressor
            length_metadata[field] = {"type": "ridge_log1p", "train_count": len(train_lengths), "alpha": 10.0}
            test_pred_lengths = np.maximum(0, np.expm1(regressor.predict(both_test[y_test == 1])))
        full_lengths = np.zeros(len(test_rows), dtype=float)
        full_lengths[y_test == 1] = test_pred_lengths
        lengths.append({"field": field, "train_count": int(len(train_lengths)), "positive_test": int(len(test_lengths)),
                        "model_type": length_metadata[field]["type"],
                        "mae": float(mean_absolute_error(test_lengths, test_pred_lengths)) if len(test_lengths) else None,
                        "rmse": float(mean_squared_error(test_lengths, test_pred_lengths) ** 0.5) if len(test_lengths) else None,
                        "median_absolute_error": float(np.median(np.abs(test_lengths - test_pred_lengths))) if len(test_lengths) else None,
                        "spearman": float(spearmanr(test_lengths, test_pred_lengths).statistic)
                        if len(test_lengths) > 1 and np.ptp(test_lengths) > 0 and np.ptp(test_pred_lengths) > 0 else None})
        predictions["field_scores"][field] = scores
        predictions["predicted_output_tokens"][field] = full_lengths
    return ({"char_vectorizer": char, "word_vectorizer": word, "classifiers": classifiers,
             "regressors": regressors, "constants": constants, "length_metadata": length_metadata},
            {"fields": presence, "length_fields": lengths}, predictions)


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
    if gold_ids & {row["record_id"] for row in splits["train"] + splits["validation"]} or not gold_ids <= {row["record_id"] for row in splits["test"]}:
        raise SystemExit("Gold 划分不满足固定隔离约束")
    runs = []
    for seed in config.seeds:
        order = base.nested_order(splits["train"], fields, seed)
        _, full_metrics, full_predictions = base.train_and_evaluate(splits["train"], splits["validation"], splits["test"], fields, seed, config.min_length_samples)
        full_summary = base.summarize_metrics(full_metrics, full_predictions, splits["test"], fields)
        full_baseline = {budget: [] for budget in base.BUDGETS}
        for index in range(len(splits["test"])):
            prediction = {"field_scores": {field: float(full_predictions["field_scores"][field][index]) for field in fields},
                          "predicted_output_tokens": {field: int(full_predictions["predicted_output_tokens"][field][index]) for field in fields}}
            for budget in base.BUDGETS:
                selected = base.select_fields(prediction, schema, costs, weights["weights"], weights["required_fields"], budget)
                full_baseline[budget].append(set(selected["selected_fields"]))
        for size in config.sizes:
            if size > len(order):
                raise SystemExit(f"训练规模 {size} 超过训练集大小 {len(order)}")
            train_rows = [splits["train"][index] for index in order[:size]]
            models, metrics, predictions = train_m2(
                train_rows,
                splits["validation"],
                splits["test"],
                fields,
                seed,
                config.min_length_samples,
            )
            summary = base.summarize_metrics(metrics, predictions, splits["test"], fields)
            budgets = base.evaluate_budgets(
                splits["test"], fields, predictions, schema, costs, weights, full_baseline, gold_ids
            )
            metadata = {"method": "M2_char_word_probability_fusion", "size": size, "seed": seed,
                        "sampling": "same_nested_multilabel_deficit_greedy_as_training_size_ablation",
                        "alpha_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
                        "threshold": "validation F1 optimum shrunk 25% toward 0.5",
                        "llm_api_called": False, "trained_at": datetime.now(timezone.utc).isoformat(),
                        "python_version": platform.python_version()}
            model_dir = config.model_root / f"size_{size}" / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(models, model_dir / "model.joblib")
            (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report_dir = config.report_root / "runs" / f"size_{size}" / f"seed_{seed}"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "metrics.json").write_text(json.dumps({"metadata": metadata, "presence": metrics["fields"], "length": metrics["length_fields"], "budgets": budgets}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            run = {"size": size, "seed": seed, **summary,
                   "budget_200_gold_nonempty_coverage": budgets[0]["gold_nonempty_coverage"], "budget_400_gold_nonempty_coverage": budgets[1]["gold_nonempty_coverage"],
                   "budget_200_estimated_cost": budgets[0]["estimated_cost"], "budget_400_estimated_cost": budgets[1]["estimated_cost"],
                   "presence_macro_recovery": summary["presence_macro_f1"] / full_summary["presence_macro_f1"],
                   "presence_micro_recovery": summary["presence_micro_f1"] / full_summary["presence_micro_f1"], "length_mae_recovery": full_summary["length_mae"] / summary["length_mae"]}
            runs.append(run)
            print(json.dumps(run, ensure_ascii=False))
    base.aggregate(runs, config.report_root)
    (config.report_root / "comparison.json").write_text(json.dumps({"method": "M2", "runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
