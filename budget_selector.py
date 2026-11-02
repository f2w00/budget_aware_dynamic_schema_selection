#!/usr/bin/env python3
"""基于字段价值权重和预算生成动态 Schema。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按预算选择动态 Schema")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("simple.json"))
    parser.add_argument("--costs", type=Path, default=Path("schema_costs.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def field_cost(field: str, prediction: dict[str, Any], costs: dict[str, Any]) -> int:
    definition = costs["fields"][field]
    output_tokens = prediction.get("predicted_output_tokens", {}).get(field, 0)
    return (
        int(definition["field_name_tokens"])
        + int(definition["schema_value_tokens"])
        + int(output_tokens)
    )


def select_fields(
    prediction: dict[str, Any],
    schema: dict[str, str],
    costs: dict[str, Any],
    weights: dict[str, float],
    required_fields: list[str],
    budget: int,
) -> dict[str, Any]:
    scores = prediction["field_scores"]
    all_fields = list(schema)
    details = {
        field: {
            "score": float(scores.get(field, 0.0)),
            "weight": float(weights[field]),
            "predicted_output_tokens": int(
                prediction.get("predicted_output_tokens", {}).get(field, 0)
            ),
            "cost": field_cost(field, prediction, costs),
        }
        for field in all_fields
    }
    required_cost = sum(details[field]["cost"] for field in required_fields)
    optional_fields = [field for field in all_fields if field not in required_fields]
    available_budget = max(0, budget - required_cost)

    # 0-1 背包：以预算 Token 为容量，最大化字段的加权效用。
    dp: list[tuple[float, tuple[str, ...]] | None] = [None] * (available_budget + 1)
    dp[0] = (0.0, ())
    for field in optional_fields:
        cost = details[field]["cost"]
        utility = details[field]["score"] * details[field]["weight"]
        if cost <= 0 or cost > available_budget:
            continue
        for capacity in range(available_budget, cost - 1, -1):
            previous = dp[capacity - cost]
            if previous is None:
                continue
            candidate = (previous[0] + utility, previous[1] + (field,))
            if dp[capacity] is None or candidate[0] > dp[capacity][0]:
                dp[capacity] = candidate

    best = max((entry for entry in dp if entry is not None), key=lambda item: item[0])
    selected_fields = list(required_fields) + list(best[1])
    selected_cost = sum(details[field]["cost"] for field in selected_fields)
    selected_utility = sum(
        details[field]["score"] * details[field]["weight"] for field in selected_fields
    )
    return {
        "budget": budget,
        "required_cost": required_cost,
        "estimated_cost": selected_cost,
        "remaining_budget": budget - selected_cost,
        "total_utility": selected_utility,
        "selected_fields": selected_fields,
        "dynamic_schema": {field: schema[field] for field in selected_fields},
        "field_details": details,
    }


def main() -> None:
    args = parse_args()
    prediction = load_json(args.prediction)
    schema = load_json(args.schema)
    costs = load_json(args.costs)
    weight_config = load_json(args.weights)
    result = select_fields(
        prediction,
        schema,
        costs,
        weight_config["weights"],
        weight_config["required_fields"],
        args.budget,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
