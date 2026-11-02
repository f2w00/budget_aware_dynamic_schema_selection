#!/usr/bin/env python3
"""训练湖南博物馆 n=100 M1，并评估 held-out 字段选择质量。"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tiktoken
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_strategy_ablation as m1
import training_size_ablation as base
from budget_selector import select_fields
from generate_training_dataset import build_sample
from llm_runner import predict_fields, schema_costs
from llm_prompt import build_prompt


COHORT_DIR = ROOT / "data/hunan_museum_300000001_300000500"
SILVER = ROOT / "experiments/hunan_museum_300000001_300000500_full_silver/formal/full.jsonl"
MODEL_DIR = ROOT / "models/hunan_museum_300000001_300000500_n100_m1/seed_42"
REPORT_DIR = ROOT / "reports/hunan_museum_300000001_300000500_n100_m1"
SPLIT_SEED = 20260917
BUDGETS = (200, 400)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_rows(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, list[int]]:
    labels = np.asarray(
        [[row["labels"][f"{field}_present"] for field in fields] for row in rows],
        dtype=np.int8,
    )
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.7,
        random_state=SPLIT_SEED,
    )
    development, test = next(first.split(np.zeros(len(rows)), labels))
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1 / 3,
        random_state=SPLIT_SEED,
    )
    train_relative, validation_relative = next(
        second.split(np.zeros(len(development)), labels[development])
    )
    return {
        "train": [int(index) for index in development[train_relative]],
        "validation": [int(index) for index in development[validation_relative]],
        "test": [int(index) for index in test],
    }


def prompt_tokens(
    template: str,
    schema: dict[str, str],
    row: dict[str, Any],
    selected_fields: list[str],
    encoder: Any,
) -> int:
    prompt, _, _ = build_prompt(template, schema, selected_fields, row["raw_data"])
    return len(encoder.encode(prompt))


def evaluate_selection(
    models: dict[str, Any],
    test_rows: list[dict[str, Any]],
    fields: list[str],
    schema: dict[str, str],
    costs: dict[str, Any],
    weights: dict[str, Any],
    template: str,
    encoder: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_tokens: list[int] = []
    records: list[dict[str, Any]] = []
    totals = {
        budget: {"truth": [], "selected": [], "coverage": [], "cost": [], "tokens": []}
        for budget in BUDGETS
    }
    for index, row in enumerate(test_rows):
        # 必须复用线上 Runner 的预测路径；不能用 held-out 真实标签决定长度成本。
        prediction = predict_fields(row["raw_data"], fields, models)
        gold = {field for field in fields if row["labels"][f"{field}_present"]}
        full_prompt_tokens = prompt_tokens(template, schema, row, fields, encoder)
        full_tokens.append(full_prompt_tokens)
        for budget in BUDGETS:
            selection = select_fields(
                prediction,
                schema,
                costs,
                weights["weights"],
                weights["required_fields"],
                budget,
            )
            selected = set(selection["selected_fields"])
            budget_prompt_tokens = prompt_tokens(
                template, schema, row, selection["selected_fields"], encoder
            )
            totals[budget]["truth"].extend(int(field in gold) for field in fields)
            totals[budget]["selected"].extend(int(field in selected) for field in fields)
            if gold:
                totals[budget]["coverage"].append(len(selected & gold) / len(gold))
            totals[budget]["cost"].append(int(selection["estimated_cost"]))
            totals[budget]["tokens"].append(budget_prompt_tokens)
            records.append(
                {
                    "record_id": row["record_id"],
                    "budget": budget,
                    "selected_fields": selection["selected_fields"],
                    "gold_present_fields": sorted(gold),
                    "estimated_cost": int(selection["estimated_cost"]),
                    "gold_coverage": len(selected & gold) / len(gold) if gold else None,
                    "full_prompt_tokens": full_prompt_tokens,
                    "budget_prompt_tokens": budget_prompt_tokens,
                    "prompt_token_reduction": full_prompt_tokens - budget_prompt_tokens,
                }
            )
    summary = {}
    for budget, values in totals.items():
        truth = values["truth"]
        selected = values["selected"]
        summary[str(budget)] = {
            "records": len(test_rows),
            "gold_nonempty_records": len(values["coverage"]),
            "selection_micro_precision": float(precision_score(truth, selected, zero_division=0)),
            "selection_micro_recall": float(recall_score(truth, selected, zero_division=0)),
            "selection_micro_f1": float(f1_score(truth, selected, zero_division=0)),
            "selection_macro_f1": float(
                f1_score(truth, selected, average="macro", zero_division=0)
            ),
            "gold_nonempty_coverage": float(np.mean(values["coverage"])),
            "estimated_cost_mean": float(np.mean(values["cost"])),
            "budget_satisfied_rate": float(
                np.mean(np.asarray(values["cost"]) <= budget)
            ),
            "selected_fields_mean": float(
                np.mean([sum(chunk) for chunk in np.asarray(selected).reshape(-1, len(fields))])
            ),
            "full_prompt_tokens_mean": float(np.mean(full_tokens)),
            "budget_prompt_tokens_mean": float(np.mean(values["tokens"])),
            "prompt_token_reduction_pct": float(
                100 * (1 - np.sum(values["tokens"]) / np.sum(full_tokens))
            ),
        }
    return summary, records


def main() -> None:
    global COHORT_DIR, SILVER, MODEL_DIR, REPORT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    parser.add_argument("--silver", type=Path, default=SILVER)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    COHORT_DIR = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    SILVER = args.silver if args.silver.is_absolute() else ROOT / args.silver
    MODEL_DIR = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    REPORT_DIR = args.report_dir if args.report_dir.is_absolute() else ROOT / args.report_dir
    snapshot = read_jsonl(COHORT_DIR / "snapshot.jsonl")
    silver = read_jsonl(SILVER)
    if len(snapshot) != 500 or len(silver) != 500:
        raise SystemExit("冻结输入和 full silver 必须各为500条")
    if any(row.get("status") != "success" or row.get("method") != "full" for row in silver):
        raise SystemExit("full silver 存在失败或非 full 记录")
    raw_by_id = {int(row["artifact_id"]): row["raw_data"] for row in snapshot}
    silver_by_id = {int(row["artifact_id"]): row for row in silver}
    if len(silver_by_id) != len(silver) or set(silver_by_id) != set(raw_by_id):
        raise SystemExit("full silver 与冻结输入 ID 集合不一致")

    schema = json.loads((ROOT / "simple2.json").read_text(encoding="utf-8"))
    fields = list(schema)
    encoder = tiktoken.get_encoding("cl100k_base")
    rows = [
        {
            **build_sample(
                int(result["artifact_id"]),
                raw_by_id[int(result["artifact_id"])],
                result["result"],
                encoder,
            ),
            "raw_data": raw_by_id[int(result["artifact_id"])],
        }
        for artifact_id in raw_by_id
        for result in [silver_by_id[artifact_id]]
    ]
    splits = split_rows(rows, fields)
    split_rows_by_name = {
        name: [rows[index] for index in indices] for name, indices in splits.items()
    }
    if {name: len(values) for name, values in split_rows_by_name.items()} != {
        "train": 100,
        "validation": 50,
        "test": 350,
    }:
        raise SystemExit("切分规模不是100/50/350")
    split_ids = {
        name: [row["record_id"] for row in values]
        for name, values in split_rows_by_name.items()
    }
    if sum(map(len, split_ids.values())) != len(set().union(*map(set, split_ids.values()))):
        raise SystemExit("切分 ID 存在重叠")

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    (COHORT_DIR / "silver_training_dataset.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    for name, ids in split_ids.items():
        write_json(COHORT_DIR / f"{name}_ids.json", ids)
    write_json(
        COHORT_DIR / "split_manifest.json",
        {
            "method": "two-stage multilabel iterative stratification",
            "seed": SPLIT_SEED,
            "sizes": {name: len(values) for name, values in split_rows_by_name.items()},
            "snapshot_sha256": sha256(COHORT_DIR / "snapshot.jsonl"),
            "silver_sha256": sha256(SILVER),
            "label_source": "same-cohort full stable-Prompt silver output",
        },
    )

    models, metrics, predictions = m1.train_m1(
        split_rows_by_name["train"],
        split_rows_by_name["validation"],
        split_rows_by_name["test"],
        fields,
        42,
        10,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_DIR / "model.joblib")
    metadata = {
        "method": "M1_char_word_tfidf_threshold_shrinkage",
        "training_size": 100,
        "validation_size": 50,
        "heldout_test_size": 350,
        "seed": 42,
        "split_seed": SPLIT_SEED,
        "label_source": "Hunan Museum full silver",
        "llm_api_called": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(MODEL_DIR / "metadata.json", metadata)

    raw_summary = base.summarize_metrics(metrics, predictions, split_rows_by_name["test"], fields)
    # 与线上 Runner 使用同一 Schema Token 成本计算，避免预算选择口径漂移。
    costs = schema_costs(schema)
    weights = json.loads((ROOT / "field_weights.json").read_text(encoding="utf-8"))
    template = (ROOT / "1_stable.prompt").read_text(encoding="utf-8")
    budget_summary, records = evaluate_selection(
        models,
        split_rows_by_name["test"],
        fields,
        schema,
        costs,
        weights,
        template,
        encoder,
    )
    report = {
        "cohort": "Hunan Museum artifact_id 300000001 through 300000500",
        "design": "500 full silver labels; 100 train / 50 validation / 350 held-out test",
        "scope": "offline selector evaluation before held-out budget LLM runs",
        "raw_m1": raw_summary,
        "budget_selection": budget_summary,
        "input_hashes": {
            "snapshot": sha256(COHORT_DIR / "snapshot.jsonl"),
            "silver": sha256(SILVER),
            "prompt": sha256(ROOT / "1_stable.prompt"),
            "schema": sha256(ROOT / "simple2.json"),
        },
    }
    write_json(REPORT_DIR / "summary.json", report)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "heldout_budget_selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(REPORT_DIR / "field_metrics.json", metrics)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
