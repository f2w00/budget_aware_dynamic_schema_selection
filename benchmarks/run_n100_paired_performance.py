#!/usr/bin/env python3
"""运行 n=100 M1 在冻结 2,000 条负载上的交错配对性能实验。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_pipeline import (
    load_ids,
    load_workload_manifest,
    parse_runner_summary,
    read_jsonl,
    summarize,
    validate,
    write_summaries,
)


PAIRS = (
    (42, ("full", "budget-200")),
    (43, ("budget-200", "full")),
    (44, ("full", "budget-200")),
)
PAIR_SEQUENCES = {seed: sequence for seed, sequence in PAIRS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 n=100 M1 配对端到端性能实验")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("data/performance/performance_ids_2000.json"),
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        default=Path("data/performance/performance_ids_2000.manifest.json"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/model_strategy_ablation/m1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/runs/n100-2000-paired"),
    )
    parser.add_argument("--runner", type=Path, default=Path("llm_runner.py"))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际调用 LLM；默认仅运行 dry-run 审计。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复正式运行：跳过已通过校验的完整 run，不覆盖既有产物。",
    )
    return parser.parse_args()


def run_id(seed: int, method: str) -> str:
    return f"n100-s{seed}-{method}-c16-q16"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_empty_run_paths(output_dir: Path, run_name: str, dry_run: bool) -> None:
    suffix = ".dry_run.jsonl" if dry_run else ".jsonl"
    paths = (
        output_dir / "raw" / f"{run_name}{suffix}",
        output_dir / "logs" / f"{run_name}{'.dry' if dry_run else ''}.stdout.log",
        output_dir / "logs" / f"{run_name}{'.dry' if dry_run else ''}.stderr.log",
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖既有运行产物: {', '.join(existing)}")


def run_one(
    args: argparse.Namespace,
    ids: list[int],
    workload_meta: dict[str, Any],
    seed: int,
    pair_order: int,
    method: str,
    dry_run: bool,
) -> dict[str, Any]:
    model_dir = args.model_root / f"seed_{seed}"
    if not (model_dir / "model.joblib").exists():
        raise FileNotFoundError(f"缺少 n=100 M1 模型: {model_dir / 'model.joblib'}")

    run_name = run_id(seed, method)
    experiment_id = f"n100-2000-paired-{run_name}"
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    validation_dir = output_dir / "validation"
    for directory in (raw_dir, log_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_empty_run_paths(output_dir, run_name, dry_run)

    output = raw_dir / f"{run_name}.jsonl"
    log_suffix = ".dry" if dry_run else ""
    stdout_path = log_dir / f"{run_name}{log_suffix}.stdout.log"
    stderr_path = log_dir / f"{run_name}{log_suffix}.stderr.log"
    command = [
        args.python_executable,
        str(args.runner),
        "--methods",
        method,
        "--ids-file",
        str(args.ids_file),
        "--limit",
        str(args.limit),
        "--max-concurrency",
        str(args.max_concurrency),
        "--queue-size",
        str(args.queue_size),
        "--max-retries",
        str(args.max_retries),
        "--model-dir",
        str(model_dir),
        "--run-label",
        run_name,
        "--experiment-id",
        experiment_id,
        "--output",
        str(output),
    ]
    if dry_run:
        command.append("--dry-run")

    environment = os.environ.copy()
    environment["LLM_TEMPERATURE"] = str(args.temperature)
    started_at = time.time()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"运行失败 {run_name}，详见 {stderr_path}")

    runner_meta = parse_runner_summary(completed.stdout)
    expected_ids = ids[: args.limit] if args.limit > 0 else ids
    result_path = output.with_suffix(".dry_run.jsonl") if dry_run else output
    rows = read_jsonl(result_path)
    validation = validate(
        rows,
        expected_ids,
        [method],
        run_name,
        runner_meta.get("submitted"),
        experiment_id,
    )
    validation["returncode"] = completed.returncode
    write_json(validation_dir / f"{run_name}{log_suffix}.json", validation)
    if not validation["valid"]:
        raise RuntimeError(f"运行校验失败 {run_name}，详见 validation 目录")

    meta = {
        "cohort": "n100-2000-paired",
        "run_id": run_name,
        "experiment_id": experiment_id,
        "method": method,
        "pair_seed": seed,
        "pair_order": pair_order,
        "pair_sequence": list(PAIR_SEQUENCES[seed]),
        "selector_training_size": 100,
        "selector_seed": seed,
        "selector_model_dir": str(model_dir),
        "max_concurrency": args.max_concurrency,
        "queue_size": args.queue_size,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "ids_file": str(args.ids_file),
        "ids_requested": len(expected_ids),
        "dry_run": dry_run,
        "started_at_unix": started_at,
        "wall_clock_ms": elapsed_ms,
        "command": command,
        "returncode": completed.returncode,
        **workload_meta,
        **runner_meta,
    }
    write_json(output_dir / f"{run_name}{log_suffix}.meta.json", meta)
    return summarize(rows, meta) | {"validation_valid": True}


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.max_concurrency < 1 or args.queue_size < 1:
        raise SystemExit("limit 不能为负，max-concurrency 和 queue-size 必须为正数")
    if args.max_concurrency != 16 or args.queue_size != 16:
        raise SystemExit("正式配对协议固定为 max-concurrency=16、queue-size=16")

    ids = load_ids(args.ids_file)
    workload_meta = load_workload_manifest(args.workload_manifest, ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry_run = not args.execute
    summary_dir = args.output_dir / "dry_run" if dry_run else args.output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    cohort_meta = {
        "cohort": "n100-2000-paired",
        "purpose": "n=100 M1 在冻结开发负载上的交错配对端到端性能验证",
        "protocol": {
            "methods_per_pair": ["full", "budget-200"],
            "pair_seeds": [seed for seed, _ in PAIRS],
            "pair_sequences": {str(seed): list(sequence) for seed, sequence in PAIRS},
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "max_retries": args.max_retries,
            "temperature": args.temperature,
            "stable_prompt": True,
            "thinking_disabled": True,
        },
        "selector_training_size": 100,
        "model_root": str(args.model_root),
        "ids_file": str(args.ids_file),
        "dry_run": dry_run,
        **workload_meta,
    }
    write_json(args.output_dir / f"cohort{'.dry' if dry_run else ''}.json", cohort_meta)

    summaries = []
    existing_summaries = {}
    if args.resume and not dry_run:
        summary_path = args.output_dir / "summaries.json"
        if summary_path.exists():
            existing_summaries = {
                (int(row["selector_seed"]), str(row["method"])): row
                for row in json.loads(summary_path.read_text(encoding="utf-8"))
            }
    for seed, methods in PAIRS:
        for pair_order, method in enumerate(methods, 1):
            run_name = run_id(seed, method)
            if args.resume and not dry_run:
                validation_path = args.output_dir / "validation" / f"{run_name}.json"
                raw_path = args.output_dir / "raw" / f"{run_name}.jsonl"
                validation = (
                    json.loads(validation_path.read_text(encoding="utf-8"))
                    if validation_path.exists()
                    else {}
                )
                existing = existing_summaries.get((seed, method))
                complete = existing is not None and validation.get("valid") and raw_path.exists()
                if complete:
                    summaries.append(existing)
                    print(json.dumps({"skipped": run_name}, ensure_ascii=False))
                    continue
                # 归档未完成或不完整的 raw 证据，允许在不覆盖原始证据的前提下重跑该 run
                if raw_path.exists():
                    partial_path = raw_path.with_suffix(".partial.jsonl")
                    if partial_path.exists():
                        raise RuntimeError(f"部分产物归档文件已存在，拒绝覆盖: {partial_path}")
                    raw_path.rename(partial_path)
                    print(json.dumps({"archived_partial": str(partial_path)}, ensure_ascii=False))
            summary = run_one(
                args,
                ids,
                workload_meta,
                seed,
                pair_order,
                method,
                dry_run,
            )
            summaries.append(summary)
            write_summaries(summary_dir, summaries)

    validation_dir = args.output_dir / "validation"
    suffix = ".dry.json" if dry_run else ".json"
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(validation_dir.glob(f"*{suffix}"))
    ]
    write_json(
        args.output_dir / f"validation{'.dry' if dry_run else ''}.json",
        {"valid": all(item["valid"] for item in validations), "runs": validations},
    )
    print(
        json.dumps(
            {
                "runs": len(summaries),
                "output_dir": str(args.output_dir),
                "dry_run": dry_run,
                "validation": all(item["valid"] for item in validations),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
