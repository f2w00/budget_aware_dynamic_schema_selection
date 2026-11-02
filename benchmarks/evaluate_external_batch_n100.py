#!/usr/bin/env python3
"""在冻结外部批次上训练 n=100 M1，并离线评估字段与 Prompt 削减。"""

from __future__ import annotations

import csv
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
from scipy.sparse import hstack
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_strategy_ablation as m1
import training_size_ablation as base
from budget_selector import select_fields
from generate_training_dataset import build_sample
from llm_prompt import build_prompt


COHORT_DIR = ROOT / "data/external_batch_400000001"
SILVER = ROOT / "experiments/external_batch_400000001_full_silver/formal/full.jsonl"
MODEL_ROOT = ROOT / "models/external_batch_400000001_n100_m1"
REPORT_ROOT = ROOT / "reports/external_batch_400000001_n100_m1"
SEEDS = (42,)
SPLIT_SEED = 20260916
BUDGETS = (200, 400)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def multilabel_split(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, list[int]]:
    labels = np.asarray(
        [[row["labels"][f"{field}_present"] for field in fields] for row in rows],
        dtype=np.int8,
    )
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.7, random_state=SPLIT_SEED
    )
    development, test = next(first.split(np.zeros(len(rows)), labels))
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SPLIT_SEED
    )
    train_relative, validation_relative = next(
        second.split(np.zeros(len(development)), labels[development])
    )
    return {
        "train": [int(index) for index in development[train_relative]],
        "validation": [int(index) for index in development[validation_relative]],
        "test": [int(index) for index in test],
    }


def field_rates(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {
        field: float(np.mean([row["labels"][f"{field}_present"] for row in rows]))
        for field in fields
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
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


def summarize_run(
    seed: int,
    models: dict[str, Any],
    metrics: dict[str, Any],
    predictions: dict[str, Any],
    test_rows: list[dict[str, Any]],
    fields: list[str],
    schema: dict[str, str],
    costs: dict[str, Any],
    weights: dict[str, Any],
    template: str,
    encoder: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = base.summarize_metrics(metrics, predictions, test_rows, fields)
    truth = np.asarray(
        [[row["labels"][f"{field}_present"] for field in fields] for row in test_rows],
        dtype=np.int8,
    )
    predicted = np.asarray([
        [
            int(predictions["field_scores"][field][index] >= metrics["fields"][field_index]["threshold"])
            for field_index, field in enumerate(fields)
        ]
        for index in range(len(test_rows))
    ], dtype=np.int8)
    full_token_values: list[int] = []
    records: list[dict[str, Any]] = []
    # 每个预算分别累积覆盖、选择数、估计成本和 Prompt Token，供预算 Pareto 分析。
    budget_values: dict[str, dict[str, list[float]]] = {
        str(budget): {"coverage": [], "selected": [], "cost": [], "tokens": []}
        for budget in BUDGETS
    }
    for index, row in enumerate(test_rows):
        prediction = {
            "field_scores": {
                field: float(predictions["field_scores"][field][index]) for field in fields
            },
            "predicted_output_tokens": {
                field: int(predictions["predicted_output_tokens"][field][index]) for field in fields
            },
        }
        gold = {field for field in fields if row["labels"][f"{field}_present"]}
        full_prompt_tokens = prompt_tokens(template, schema, row, fields, encoder)
        full_token_values.append(full_prompt_tokens)
        for budget in BUDGETS:
            selection = select_fields(
                prediction, schema, costs, weights["weights"], weights["required_fields"], budget
            )
            selected = selection["selected_fields"]
            budget_prompt_tokens = prompt_tokens(template, schema, row, selected, encoder)
            coverage = len(set(selected) & gold) / len(gold) if gold else 1.0
            values = budget_values[str(budget)]
            values["coverage"].append(coverage)
            values["selected"].append(len(selected))
            values["cost"].append(int(selection["estimated_cost"]))
            values["tokens"].append(budget_prompt_tokens)
            records.append({
                "record_id": row["record_id"],
                "seed": seed,
                "budget": budget,
                "selected_fields": selected,
                "estimated_cost": selection["estimated_cost"],
                "gold_present_fields": sorted(gold),
                "gold_coverage": coverage,
                "full_prompt_tokens": full_prompt_tokens,
                "budget_prompt_tokens": budget_prompt_tokens,
                "prompt_token_reduction": full_prompt_tokens - budget_prompt_tokens,
            })
    full_tokens_total = float(np.sum(full_token_values))
    budgets = {}
    for budget in BUDGETS:
        values = budget_values[str(budget)]
        budgets[str(budget)] = {
            "budget": budget,
            "mean_gold_coverage": float(np.mean(values["coverage"])),
            "selected_fields_mean": float(np.mean(values["selected"])),
            "estimated_cost_mean": float(np.mean(values["cost"])),
            "budget_satisfied_rate": float(np.mean(np.asarray(values["cost"]) <= budget)),
            "full_prompt_tokens_mean": float(np.mean(full_token_values)),
            "budget_prompt_tokens_mean": float(np.mean(values["tokens"])),
            "prompt_token_reduction_mean": float(
                np.mean(np.asarray(full_token_values) - values["tokens"])
            ),
            "prompt_token_reduction_pct": float(
                100 * (1 - np.sum(values["tokens"]) / full_tokens_total)
            ),
        }
    return {
        "seed": seed,
        **summary,
        "presence_micro_f1_direct": float(f1_score(truth.ravel(), predicted.ravel(), zero_division=0)),
        "test_records": len(test_rows),
        "budgets": budgets,
    }, records


def main() -> None:
    snapshot = read_jsonl(COHORT_DIR / "snapshot.jsonl")
    silver = read_jsonl(SILVER)
    if len(snapshot) != 500 or len(silver) != 500:
        raise SystemExit("冻结输入和正式 silver 输出必须各为500条")
    if any(row.get("status") != "success" or row.get("method") != "full" for row in silver):
        raise SystemExit("silver 输出存在非 full 或失败记录")
    raw_by_id = {int(row["artifact_id"]): row["raw_data"] for row in snapshot}
    if set(raw_by_id) != {int(row["artifact_id"]) for row in silver}:
        raise SystemExit("snapshot 和 silver 的 ID 集合不一致")
    schema = json.loads((ROOT / "simple2.json").read_text(encoding="utf-8"))
    fields = list(schema)
    encoder = tiktoken.get_encoding("cl100k_base")
    template = (ROOT / "1_stable.prompt").read_text(encoding="utf-8")
    rows = [
        {
            **build_sample(int(result["artifact_id"]), raw_by_id[int(result["artifact_id"])],
                           result["result"], encoder),
            "raw_data": raw_by_id[int(result["artifact_id"])],
        }
        for result in silver
    ]
    if any(set(row["labels"]) != {f"{field}_present" for field in fields} for row in rows):
        raise SystemExit("silver 标签字段不完整")
    splits = multilabel_split(rows, fields)
    split_rows = {name: [rows[index] for index in indices] for name, indices in splits.items()}
    if {name: len(values) for name, values in split_rows.items()} != {
        "train": 100, "validation": 50, "test": 350
    }:
        raise SystemExit("切分规模不是100/50/350")
    split_ids = {name: [row["record_id"] for row in values] for name, values in split_rows.items()}
    if sum(map(len, split_ids.values())) != len(set().union(*map(set, split_ids.values()))):
        raise SystemExit("切分存在重叠")

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    (COHORT_DIR / "silver_training_dataset.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name, ids in split_ids.items():
        write_json(COHORT_DIR / f"{name}_ids.json", ids)
    split_manifest = {
        "method": "two-stage multilabel iterative stratification",
        "seed": SPLIT_SEED,
        "sizes": {name: len(values) for name, values in split_rows.items()},
        "snapshot_sha256": sha256(COHORT_DIR / "snapshot.jsonl"),
        "silver_sha256": sha256(SILVER),
        "label_source": "same-cohort full stable-Prompt silver output",
        "field_presence_rates": {name: field_rates(values, fields) for name, values in split_rows.items()},
    }
    write_json(COHORT_DIR / "split_manifest.json", split_manifest)

    costs = json.loads((ROOT / "schema_costs.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "field_weights.json").read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        models, metrics, predictions = m1.train_m1(
            split_rows["train"], split_rows["validation"], split_rows["test"], fields, seed, 10
        )
        metadata = {
            "method": "M1_char_word_tfidf_threshold_shrinkage",
            "training_size": 100,
            "validation_size": 50,
            "heldout_test_size": 350,
            "seed": seed,
            "split_seed": SPLIT_SEED,
            "label_source": "external batch full silver",
            "threshold": "validation F1 optimum shrunk 25% toward 0.5",
            "llm_api_called": False,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        model_dir = MODEL_ROOT / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, model_dir / "model.joblib")
        write_json(model_dir / "metadata.json", metadata)
        summary, records = summarize_run(
            seed, models, metrics, predictions, split_rows["test"], fields, schema,
            costs, weights, template, encoder,
        )
        summaries.append(summary)
        all_records.extend(records)
        write_json(REPORT_ROOT / "runs" / f"seed_{seed}" / "metrics.json", {
            "metadata": metadata, "presence": metrics["fields"],
            "length": metrics["length_fields"], "summary": summary,
        })

    numeric = ["presence_macro_f1", "presence_micro_f1", "length_mae"]
    aggregate = {
        key: mean_std([item[key] for item in summaries])
        for key in numeric
    }
    budget_keys = [
        "mean_gold_coverage", "selected_fields_mean", "estimated_cost_mean",
        "budget_satisfied_rate", "full_prompt_tokens_mean", "budget_prompt_tokens_mean",
        "prompt_token_reduction_mean", "prompt_token_reduction_pct",
    ]
    aggregate["budgets"] = {
        str(budget): {
            key: mean_std([item["budgets"][str(budget)][key] for item in summaries])
            for key in budget_keys
        }
        for budget in BUDGETS
    }
    report = {
        "cohort": "artifact_id > 400000001 external batch",
        "design": "500 full silver labels; 100 train / 50 validation / 350 held-out test",
        "scope": "offline field prediction and deterministic Prompt token proxy only; no new budget LLM pipeline",
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "repeat_note": "固定训练、验证和测试划分下，liblinear 拟合是确定性的；"
        "本实验报告单一冻结 n=100 hold-out 验证，不将伪重复当作独立重复。",
        "runs": summaries,
        "aggregate": aggregate,
        "input_hashes": {
            "snapshot": sha256(COHORT_DIR / "snapshot.jsonl"),
            "silver": sha256(SILVER),
            "prompt": sha256(ROOT / "1_stable.prompt"),
            "schema": sha256(ROOT / "simple2.json"),
        },
    }
    write_json(REPORT_ROOT / "summary.json", report)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with (REPORT_ROOT / "heldout_budget_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    print(json.dumps({"splits": {key: len(value) for key, value in split_rows.items()},
                      "aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
