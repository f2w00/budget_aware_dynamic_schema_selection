#!/usr/bin/env python3
"""测量 M1 训练、加载和 full 本地规划成本，不调用 LLM。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_runner
from training_size_ablation import load_jsonl, load_splits, nested_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量 M1 初始化与 full 本地规划成本")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--ids-file", type=Path, default=Path("data/performance/performance_ids_2000.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--planning-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--load-repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/analysis/initialization")
    )
    return parser.parse_args()


def ensure_isolated_output(path: Path) -> None:
    """拒绝向既有模型目录写入，防止基准训练污染线上模型。"""
    resolved = path.resolve()
    models_dir = (PROJECT_ROOT / "models").resolve()
    if resolved == models_dir or models_dir in resolved.parents:
        raise ValueError("--output-dir 不能位于 models/ 下")


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean_ms": round(float(array.mean()), 2),
        "std_ms": round(float(array.std(ddof=1)) if len(array) > 1 else 0.0, 2),
        "p50_ms": round(float(np.percentile(array, 50)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "max_ms": round(float(array.max()), 2),
    }


def train_m1(train_rows: list[dict[str, Any]], fields: list[str], seed: int) -> dict[str, Any]:
    """按 Runner 可加载的打包格式训练最小 M1，不进行验证或测试推理。"""
    texts = [row["record_text"] for row in train_rows]
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )
    matrix = hstack([char.fit_transform(texts), word.fit_transform(texts)], format="csr")
    classifiers: dict[str, Any] = {}
    regressors: dict[str, Any] = {}
    constants: dict[str, float] = {}
    length_metadata: dict[str, Any] = {}

    for field in fields:
        present = np.asarray(
            [int(row["labels"][f"{field}_present"]) for row in train_rows], dtype=int
        )
        if len(np.unique(present)) < 2:
            constants[field] = float(present[0]) if len(present) else 0.0
        else:
            classifier = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            )
            classifier.fit(matrix, present)
            classifiers[field] = classifier

        lengths = np.asarray(
            [row["field_output_tokens"][field] for row in train_rows], dtype=float
        )[present == 1]
        if len(lengths) < 10:
            length_metadata[field] = {
                "type": "median_constant",
                "train_count": int(len(lengths)),
                "value": float(np.median(lengths)) if len(lengths) else 0.0,
            }
        else:
            regressor = Ridge(alpha=10.0).fit(matrix[present == 1], np.log1p(lengths))
            regressors[field] = regressor
            length_metadata[field] = {
                "type": "ridge_log1p",
                "train_count": int(len(lengths)),
                "alpha": 10.0,
            }

    return {
        "char_vectorizer": char,
        "word_vectorizer": word,
        "classifiers": classifiers,
        "constants": constants,
        "regressors": regressors,
        "length_metadata": length_metadata,
    }


def train_isolated_models(
    train_rows: list[dict[str, Any]], fields: list[str], seeds: list[int], artifact_dir: Path
) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        target = artifact_dir / f"m1-n{len(train_rows)}-seed-{seed}"
        target.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        model = train_m1(train_rows, fields, seed)
        train_ms = (time.perf_counter() - started) * 1000
        artifact = target / "model.joblib"
        joblib.dump(model, artifact)
        metadata = {
            "purpose": "isolated_initialization_benchmark",
            "seed": seed,
            "train_records": len(train_rows),
            "train_ms": round(train_ms, 2),
            "artifact_bytes": artifact.stat().st_size,
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append({"path": str(target), **metadata})
    return results


def measure_loads(model_dir: Path, repeats: int) -> dict[str, Any]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        llm_runner.load_models(model_dir)
        values.append((time.perf_counter() - started) * 1000)
    model_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    return {"model_dir": str(model_dir), "artifact_bytes": model_bytes, **percentiles(values)}


def measure_full_planning(
    ids_file: Path,
    size: int,
    schema_path: Path,
    weights_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    llm_runner.load_dotenv(PROJECT_ROOT / ".env")
    database_url = llm_runner.os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("缺少 DATABASE_URL，无法测量与 Runner 一致的 full 规划路径")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    costs = json.loads((PROJECT_ROOT / "schema_costs.json").read_text(encoding="utf-8"))
    template = (PROJECT_ROOT / "1.prompt").read_text(encoding="utf-8")
    ids = llm_runner.load_ids(ids_file, PROJECT_ROOT / "data/splits", "test")
    models = llm_runner.load_models(model_dir)
    schema_hash = llm_runner.text_hash(json.dumps(schema, ensure_ascii=False, sort_keys=True))
    planning_values = []
    prediction_values = []
    prompt_values = []
    started = time.perf_counter()
    records = 0
    for artifact_id, raw_data in llm_runner.iter_records(database_url, ids, size):
        task = next(
            llm_runner.build_record_tasks(
                artifact_id=artifact_id,
                raw_data=raw_data,
                methods=["full"],
                completed=set(),
                schema=schema,
                weights=weights,
                costs=costs,
                models=models,
                templates={"base": template},
                method_prompt_hashes={"full": llm_runner.text_hash(template)},
                method_selector_hashes={"full": None},
                schema_hash=schema_hash,
                experiment_id="offline-initialization-benchmark",
                model="not-called",
                model_dir=model_dir,
                run_label="full-local-planning",
            )
        )
        records += 1
        planning_values.append(float(task["planning_ms"]))
        prediction_values.append(float(task["prediction_ms"]))
        prompt_values.append(float(task["prompt_build_ms"]))
    elapsed_ms = (time.perf_counter() - started) * 1000
    if records != min(size, len(ids)):
        raise RuntimeError(f"期望规划 {min(size, len(ids))} 条，实际获得 {records} 条")
    return {
        "records": records,
        "elapsed_ms": round(elapsed_ms, 2),
        "planning": percentiles(planning_values),
        "prediction": percentiles(prediction_values),
        "prompt_build": percentiles(prompt_values),
    }


def online_full_estimates(records: int) -> list[dict[str, Any]]:
    sources = [
        (
            PROJECT_ROOT / "benchmarks/runs/main-comparison-pro/summaries.json",
            "c16-q16-main",
            16,
            16,
        ),
        (
            PROJECT_ROOT / "benchmarks/runs/scaling-pro/summaries.json",
            "c1-q1-endpoint",
            1,
            1,
        ),
    ]
    estimates = []
    for path, label, concurrency, queue_size in sources:
        rows = json.loads(path.read_text(encoding="utf-8"))
        selected = [
            row
            for row in rows
            if row["method"] == "full"
            and row["concurrency"] == concurrency
            and row["queue_size"] == queue_size
            and row["validation_valid"] is True
        ]
        mean_rps = float(np.mean([row["effective_records_per_second"] for row in selected]))
        estimates.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "configuration": label,
                "runs": len(selected),
                "mean_effective_records_per_second": round(mean_rps, 4),
                "estimated_online_seconds_for_records": round(records / mean_rps, 2),
            }
        )
    return estimates


def main() -> None:
    args = parse_args()
    ensure_isolated_output(args.output_dir)
    if args.output_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖: {args.output_dir}")
    if args.train_size <= 0 or args.planning_size <= 0 or args.load_repeats <= 0:
        raise SystemExit("样本数和重复次数必须为正数")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = load_jsonl(args.input)
    splits = load_splits(rows, args.split_dir)
    fields = list(json.loads(args.schema.read_text(encoding="utf-8")))
    order = nested_order(splits["train"], fields, args.seeds[0])
    train_rows = [splits["train"][index] for index in order[: args.train_size]]
    if len(train_rows) != args.train_size:
        raise SystemExit("训练集不足指定样本数")

    artifact_dir = args.output_dir / "artifacts"
    trained = train_isolated_models(train_rows, fields, args.seeds, artifact_dir)
    deployed_load = measure_loads(args.model_dir, args.load_repeats)
    isolated_loads = [
        measure_loads(Path(item["path"]), args.load_repeats) for item in trained
    ]
    full_planning = measure_full_planning(
        args.ids_file,
        args.planning_size,
        args.schema,
        args.weights,
        args.model_dir,
    )
    output = {
        "benchmark": "offline_initialization_and_full_planning",
        "llm_calls": False,
        "training": {
            "records": args.train_size,
            "seeds": args.seeds,
            "selection": "nested_multilabel_deficit_greedy_prefix",
            "trained_models": trained,
            "train_time": percentiles([item["train_ms"] for item in trained]),
        },
        "deployed_model_load": deployed_load,
        "isolated_model_load": isolated_loads,
        "full_local_planning": full_planning,
        "online_full_estimates": online_full_estimates(args.planning_size),
        "input_hashes": {
            "training_dataset_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "performance_ids_sha256": hashlib.sha256(args.ids_file.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "initialization_cost.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "llm_calls": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
