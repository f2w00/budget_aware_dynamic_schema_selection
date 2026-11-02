#!/usr/bin/env python3
"""在已有完整抽取结果上离线评估预算字段选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from budget_selector import select_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线回放预算字段选择")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--schema", type=Path, default=Path("simple.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--output", type=Path, default=Path("reports/budget_replay.json"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(row: dict[str, Any], vectorizer: Any, classifiers: dict[str, Any], constants: dict[str, Any], regressors: dict[str, Any], length_metadata: dict[str, Any], fields: list[str], thresholds: dict[str, float]) -> dict[str, Any]:
    matrix = vectorizer.transform([row["record_text"]])
    scores = {}
    lengths = {}
    for field in fields:
        score = constants[field] if field in constants else classifiers[field].predict_proba(matrix)[0, 1]
        scores[field] = float(score)
        if length_metadata[field]["type"] == "median_constant":
            length = length_metadata[field]["value"]
        else:
            length = float(np.expm1(regressors[field].predict(matrix)[0]))
        lengths[field] = max(0, round(length))
    return {"field_scores": scores, "predicted_output_tokens": lengths, "thresholds": thresholds}


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    costs = json.loads(args.costs.read_text(encoding="utf-8"))
    weight_config = json.loads(args.weights.read_text(encoding="utf-8"))
    model_dir = args.model_dir
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    classifiers = joblib.load(model_dir / "classifiers.joblib")
    constants = json.loads((model_dir / "constants.json").read_text(encoding="utf-8"))
    regressors = joblib.load(model_dir / "length_regressors.joblib")
    length_metadata = json.loads((model_dir / "length_metadata.json").read_text(encoding="utf-8"))
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    fields = list(schema)
    results = []
    for budget in args.budgets:
        rows_result = []
        for row in rows:
            prediction = predict(row, vectorizer, classifiers, constants, regressors, length_metadata, fields, thresholds)
            selection = select_fields(
                prediction, schema, costs, weight_config["weights"], weight_config["required_fields"], budget
            )
            gold = {field for field in fields if row["labels"][f"{field}_present"]}
            selected = set(selection["selected_fields"])
            rows_result.append({
                "record_id": row["record_id"],
                "estimated_cost": selection["estimated_cost"],
                "selected_count": len(selected),
                "gold_count": len(gold),
                "gold_coverage": len(selected & gold) / len(gold) if gold else 1.0,
                "weighted_utility": selection["total_utility"],
            })
        results.append({
            "budget": budget,
            "records": len(rows_result),
            "average_estimated_cost": float(np.mean([item["estimated_cost"] for item in rows_result])),
            "average_selected_fields": float(np.mean([item["selected_count"] for item in rows_result])),
            "average_gold_fields": float(np.mean([item["gold_count"] for item in rows_result])),
            "average_gold_coverage": float(np.mean([item["gold_coverage"] for item in rows_result])),
            "average_weighted_utility": float(np.mean([item["weighted_utility"] for item in rows_result])),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"budgets": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
