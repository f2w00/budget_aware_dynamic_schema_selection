#!/usr/bin/env python3
"""在验证集和测试集上评估无硬预算字段选择策略。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from unbudgeted_selector import select_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估无硬预算字段选择")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--schema", type=Path, default=Path("simple.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    )
    parser.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    parser.add_argument("--output", type=Path, default=Path("reports/unbudgeted_replay.json"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ids(path: Path) -> set[Any]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def load_split(rows: list[dict[str, Any]], split_dir: Path, name: str) -> list[dict[str, Any]]:
    ids = load_ids(split_dir / f"{name}_ids.json")
    return [row for row in rows if row["record_id"] in ids]


def predict(row: dict[str, Any], vectorizer: Any, classifiers: dict[str, Any], constants: dict[str, Any], regressors: dict[str, Any], length_metadata: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    matrix = vectorizer.transform([row["record_text"]])
    scores = {}
    lengths = {}
    for field in fields:
        scores[field] = float(
            constants[field]
            if field in constants
            else classifiers[field].predict_proba(matrix)[0, 1]
        )
        if length_metadata[field]["type"] == "median_constant":
            length = length_metadata[field]["value"]
        else:
            length = float(np.expm1(regressors[field].predict(matrix)[0]))
        lengths[field] = max(0, round(length))
    return {"field_scores": scores, "predicted_output_tokens": lengths}


def evaluate_split(rows: list[dict[str, Any]], lambda_value: float, models: dict[str, Any], schema: dict[str, str], costs: dict[str, Any], weights: dict[str, Any]) -> dict[str, float]:
    selected_costs = []
    selected_counts = []
    gold_counts = []
    coverages = []
    utilities = []
    for row in rows:
        prediction = predict(row, **models)
        selection = select_fields(
            prediction, schema, costs, weights, lambda_value
        )
        gold = {
            field for field in schema
            if row["labels"][f"{field}_present"]
        }
        selected = set(selection["selected_fields"])
        selected_costs.append(selection["estimated_cost"])
        selected_counts.append(len(selected))
        gold_counts.append(len(gold))
        coverages.append(len(selected & gold) / len(gold) if gold else 1.0)
        utilities.append(selection["total_utility"])
    return {
        "records": len(rows),
        "average_estimated_cost": float(np.mean(selected_costs)),
        "average_selected_fields": float(np.mean(selected_counts)),
        "average_gold_fields": float(np.mean(gold_counts)),
        "average_gold_coverage": float(np.mean(coverages)),
        "average_weighted_utility": float(np.mean(utilities)),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    costs = json.loads(args.costs.read_text(encoding="utf-8"))
    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    model_dir = args.model_dir
    models = {
        "vectorizer": joblib.load(model_dir / "vectorizer.joblib"),
        "classifiers": joblib.load(model_dir / "classifiers.joblib"),
        "constants": json.loads((model_dir / "constants.json").read_text(encoding="utf-8")),
        "regressors": joblib.load(model_dir / "length_regressors.joblib"),
        "length_metadata": json.loads((model_dir / "length_metadata.json").read_text(encoding="utf-8")),
        "fields": list(schema),
    }
    validation_rows = load_split(rows, args.split_dir, "validation")
    test_rows = load_split(rows, args.split_dir, "test")
    validation_results = [
        {
            "lambda": value,
            **evaluate_split(validation_rows, value, models, schema, costs, weights),
        }
        for value in args.lambdas
    ]
    policies = {}
    for target in args.targets:
        eligible = [item for item in validation_results if item["average_gold_coverage"] >= target]
        if eligible:
            best = min(eligible, key=lambda item: (item["average_estimated_cost"], -item["lambda"]))
            policies[f"coverage_{int(target * 100)}"] = {
                "lambda": best["lambda"],
                "validation": best,
                "target_coverage": target,
            }
    test_results = [
        {
            "lambda": value,
            **evaluate_split(test_rows, value, models, schema, costs, weights),
        }
        for value in args.lambdas
    ]
    output = {
        "input": str(args.input),
        "validation": validation_results,
        "policies": policies,
        "test": test_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy_path = args.output.parent / "unbudgeted_policy.json"
    policy_path.write_text(json.dumps({"policies": policies}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
