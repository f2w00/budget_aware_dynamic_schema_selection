#!/usr/bin/env python3
"""训练基于字符级 TF-IDF 的多标签字段选择模型。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练字段选择模型")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length-samples", type=int, default=10)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ids(path: Path) -> set[Any]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def split_rows(rows: list[dict[str, Any]], split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_id = {row["record_id"]: row for row in rows}
    result = {}
    for name in ("train", "validation", "test"):
        ids = load_ids(split_dir / f"{name}_ids.json")
        missing = ids - by_id.keys()
        if missing:
            raise SystemExit(f"{name} 划分包含不存在的 record_id: {next(iter(missing))}")
        result[name] = [row for row in rows if row["record_id"] in ids]
    split_ids = [set(row["record_id"] for row in values) for values in result.values()]
    all_ids = set().union(*split_ids)
    if len(all_ids) != len(rows) or sum(map(len, split_ids)) != len(all_ids):
        raise SystemExit("数据划分没有覆盖全部记录，或不同划分存在 ID 重叠")
    return result


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.05, 0.5, 0.95], scores)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        predicted = scores >= threshold
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    splits = split_rows(rows, args.split_dir)
    fields = [key.removesuffix("_present") for key in rows[0]["labels"]]
    texts = {name: [row["record_text"] for row in values] for name, values in splits.items()}
    targets = {
        name: np.array(
            [[int(row["labels"][f"{field}_present"]) for field in fields] for row in values],
            dtype=np.int8,
        )
        for name, values in splits.items()
    }

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(texts["train"])
    matrices = {name: vectorizer.transform(texts[name]) for name in texts}
    classifiers: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    constant_scores: dict[str, float] = {}
    metrics: list[dict[str, Any]] = []
    length_regressors: dict[str, Any] = {}
    length_metadata: dict[str, Any] = {}
    length_metrics: list[dict[str, Any]] = []

    for index, field in enumerate(fields):
        y_train = targets["train"][:, index]
        y_validation = targets["validation"][:, index]
        y_test = targets["test"][:, index]
        if len(np.unique(y_train)) < 2:
            constant = float(y_train[0])
            constant_scores[field] = constant
            thresholds[field] = 0.5
            scores = np.full(len(y_test), constant)
            predicted = scores >= 0.5
        else:
            classifier = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=args.seed,
                solver="liblinear",
            )
            classifier.fit(train_matrix, y_train)
            classifiers[field] = classifier
            validation_scores = classifier.predict_proba(matrices["validation"])[:, 1]
            thresholds[field] = choose_threshold(y_validation, validation_scores)
            scores = classifier.predict_proba(matrices["test"])[:, 1]
            predicted = scores >= thresholds[field]

        row = {
            "field": field,
            "positive_test": int(y_test.sum()),
            "negative_test": int(len(y_test) - y_test.sum()),
            "threshold": thresholds[field],
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "average_precision": (
                average_precision_score(y_test, scores)
                if len(np.unique(y_test)) > 1
                else None
            ),
        }
        metrics.append(row)

        length_train_mask = targets["train"][:, index] == 1
        length_validation_mask = targets["validation"][:, index] == 1
        length_test_mask = targets["test"][:, index] == 1
        train_lengths = np.array(
            [row["field_output_tokens"][field] for row in splits["train"]],
            dtype=float,
        )[length_train_mask]
        validation_lengths = np.array(
            [row["field_output_tokens"][field] for row in splits["validation"]],
            dtype=float,
        )[length_validation_mask]
        test_lengths = np.array(
            [row["field_output_tokens"][field] for row in splits["test"]],
            dtype=float,
        )[length_test_mask]

        if len(train_lengths) < args.min_length_samples:
            default_length = float(np.median(train_lengths)) if len(train_lengths) else 0.0
            length_metadata[field] = {
                "type": "median_constant",
                "train_count": int(len(train_lengths)),
                "value": default_length,
            }
            validation_predictions = np.full(len(validation_lengths), default_length)
            test_predictions = np.full(len(test_lengths), default_length)
        else:
            regressor = Ridge(alpha=10.0)
            regressor.fit(train_matrix[length_train_mask], np.log1p(train_lengths))
            length_regressors[field] = regressor
            length_metadata[field] = {
                "type": "ridge_log1p",
                "train_count": int(len(train_lengths)),
                "alpha": 10.0,
            }
            validation_predictions = np.maximum(
                0.0,
                np.expm1(regressor.predict(matrices["validation"][length_validation_mask])),
            )
            test_predictions = np.maximum(
                0.0,
                np.expm1(regressor.predict(matrices["test"][length_test_mask])),
            )

        length_metrics.append(
            {
                "field": field,
                "train_count": int(len(train_lengths)),
                "validation_count": int(len(validation_lengths)),
                "test_count": int(len(test_lengths)),
                "mean_true_tokens": float(np.mean(test_lengths)) if len(test_lengths) else None,
                "median_true_tokens": float(np.median(test_lengths)) if len(test_lengths) else None,
                "mae": float(mean_absolute_error(test_lengths, test_predictions))
                if len(test_lengths)
                else None,
                "rmse": float(mean_squared_error(test_lengths, test_predictions) ** 0.5)
                if len(test_lengths)
                else None,
                "median_absolute_error": float(
                    np.median(np.abs(test_lengths - test_predictions))
                )
                if len(test_lengths)
                else None,
                "r2": float(r2_score(test_lengths, test_predictions))
                if len(test_lengths) > 1
                else None,
                "spearman": float(spearmanr(test_lengths, test_predictions).statistic)
                if len(test_lengths) > 1
                and np.ptp(test_lengths) > 0
                and np.ptp(test_predictions) > 0
                else None,
            }
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.model_dir / "vectorizer.joblib")
    joblib.dump(classifiers, args.model_dir / "classifiers.joblib")
    joblib.dump(length_regressors, args.model_dir / "length_regressors.joblib")
    (args.model_dir / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.model_dir / "constants.json").write_text(
        json.dumps(constant_scores, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.model_dir / "length_metadata.json").write_text(
        json.dumps(length_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "fields": fields,
        "tfidf": {
            "analyzer": "char",
            "ngram_range": [2, 5],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "sublinear_tf": True,
        },
        "classifier": "LogisticRegression(class_weight=balanced, solver=liblinear)",
        "input": str(args.input),
        "split_dir": str(args.split_dir),
        "sizes": {name: len(values) for name, values in splits.items()},
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {"metadata": metadata, "fields": metrics}
    (args.report_dir / "field_selector_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.report_dir / "field_output_token_metrics.json").write_text(
        json.dumps(
            {"metadata": metadata, "fields": length_metrics},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.report_dir / "field_selector_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    with (args.report_dir / "field_output_token_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=length_metrics[0].keys())
        writer.writeheader()
        writer.writerows(length_metrics)
    print(json.dumps({"model_dir": str(args.model_dir), "report_dir": str(args.report_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
