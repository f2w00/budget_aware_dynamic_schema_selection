#!/usr/bin/env python3
"""为真实 LLM 实验生成可审计的任务计划和预算估计。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg

from budget_selector import select_fields as select_budgeted
from llm_runner import (
    fields_cost,
    fixed_fields,
    load_dotenv,
    load_ids,
    load_models,
    method_info,
    predict_fields,
    schema_costs,
)
from unbudgeted_selector import select_fields as select_unbudgeted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 LLM 实验任务计划")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="冻结样本 ID 文件，支持 JSON 数组或包含 artifact_id 的 JSONL",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "prompt-null", "prompt-omit", "fixed-5", "budget-200", "budget-400", "lambda-0.003"],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/task_plan.jsonl"))
    return parser.parse_args()


def load_records(database_url: str, ids: list[int], limit: int) -> list[tuple[int, Any]]:
    selected_ids = ids[:limit] if limit > 0 else ids
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT artifact_id, raw_data FROM public.artifact_raw "
            "WHERE artifact_id = ANY(%s) ORDER BY artifact_id",
            (selected_ids,),
        )
        return list(cursor.fetchall())


def plan_fields(
    method: str,
    prediction: dict[str, Any],
    schema: dict[str, str],
    weights: dict[str, Any],
    costs: dict[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    if method in {"full", "prompt-null", "prompt-omit"}:
        return list(schema), None
    if method.startswith("fixed-"):
        size = int(method.split("-", 1)[1])
        return fixed_fields(schema, weights, size), None
    if method.startswith("budget-"):
        selection = select_budgeted(
            prediction, schema, costs, weights["weights"],
            weights["required_fields"], int(method.split("-", 1)[1])
        )
        return selection["selected_fields"], selection
    if method.startswith("lambda-"):
        selection = select_unbudgeted(
            prediction, schema, costs, weights, float(method.split("-", 1)[1])
        )
        return selection["selected_fields"], selection
    raise ValueError(f"未知实验方法: {method}")


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    costs = schema_costs(schema)
    models = load_models(args.model_dir)
    ids = load_ids(args.ids_file, args.split_dir, args.split)
    records = load_records(database_url, ids, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for artifact_id, raw_data in records:
            # 每条记录只做一次 ML 预测，供 C_full 与所有自适应方法复用
            prediction = predict_fields(raw_data, list(schema), models)
            # C_full：全字段定义 Token + 全字段预测输出 Token，作为 Full Schema 成本基准
            full_cost = fields_cost(list(schema), costs, prediction)
            for method in args.methods:
                selected_fields, selection = plan_fields(method, prediction, schema, weights, costs)
                method_group, setting, budget, lambda_value = method_info(method)
                # B_required：达到当前方法目标所需选中字段的预计总成本
                required_budget = (
                    selection["estimated_cost"]
                    if selection
                    else fields_cost(selected_fields, costs, prediction)
                )
                row = {
                    "task_id": f"{artifact_id}:{method}",
                    "artifact_id": artifact_id,
                    "method": method,
                    "method_group": method_group,
                    "setting": setting,
                    "budget": budget,
                    "lambda": lambda_value,
                    "selected_fields": selected_fields,
                    "full_estimated_cost": full_cost,
                    "estimated_field_cost": required_budget,
                    "required_budget": required_budget,
                    "allocated_budget": budget,
                    "budget_status": "full" if method in {"full", "prompt-null", "prompt-omit"} else "planned",
                }
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    print(json.dumps({"tasks": count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
