#!/usr/bin/env python3
"""在无硬预算条件下，按字段净收益选择动态 Schema。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无硬预算成本感知字段选择")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("simple.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_fields(
    prediction: dict[str, Any],
    schema: dict[str, str],
    costs: dict[str, Any],
    weight_config: dict[str, Any],
    lambda_value: float,
) -> dict[str, Any]:
    scores = prediction["field_scores"]
    lengths = prediction.get("predicted_output_tokens", {})
    weights = weight_config["weights"]
    required_fields = weight_config["required_fields"]
    details = {}
    selected_fields = []

    for field in schema:
        base_cost = costs["fields"][field]
        cost = (
            int(base_cost["field_name_tokens"])
            + int(base_cost["schema_value_tokens"])
            + int(lengths.get(field, 0))
        )
        utility = float(scores.get(field, 0.0)) * float(weights[field])
        net_gain = utility - lambda_value * cost
        selected = field in required_fields or net_gain > 0
        details[field] = {
            "score": float(scores.get(field, 0.0)),
            "weight": float(weights[field]),
            "utility": utility,
            "predicted_output_tokens": int(lengths.get(field, 0)),
            "cost": cost,
            "net_gain": net_gain,
            "selected": selected,
        }
        if selected:
            selected_fields.append(field)

    estimated_cost = sum(details[field]["cost"] for field in selected_fields)
    total_utility = sum(details[field]["utility"] for field in selected_fields)
    return {
        "lambda": lambda_value,
        "estimated_cost": estimated_cost,
        "total_utility": total_utility,
        "total_net_gain": total_utility - lambda_value * estimated_cost,
        "selected_fields": selected_fields,
        "dynamic_schema": {field: schema[field] for field in selected_fields},
        "field_details": details,
    }


def main() -> None:
    args = parse_args()
    result = select_fields(
        load_json(args.prediction),
        load_json(args.schema),
        load_json(args.costs),
        load_json(args.weights),
        args.lambda_value,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
