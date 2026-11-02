#!/usr/bin/env python3
"""批量运行流水线基准，并生成可复核的运行记录与汇总。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 extraction pipeline 基准")
    parser.add_argument("--methods", nargs="+", default=["full"])
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help="性能工作负载清单；用于校验冻结 ID 与记录运行元数据",
    )
    parser.add_argument(
        "--phase",
        default="benchmark",
        help="实验阶段标识，例如 concurrency-scan 或 method-comparison",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency-values", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--queue-multipliers", nargs="+", type=float, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runs"))
    parser.add_argument("--runner", type=Path, default=Path("llm_runner.py"))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_ids(path: Path) -> list[int]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"ID 文件为空: {path}")
    if content.startswith("["):
        ids = json.loads(content)
    else:
        ids = [json.loads(line)["artifact_id"] for line in content.splitlines() if line.strip()]
    if not all(isinstance(item, int) for item in ids):
        raise ValueError(f"ID 文件包含非整数 artifact_id: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"ID 文件包含重复 artifact_id: {path}")
    return ids


def ids_hash(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_workload_manifest(path: Path | None, ids: list[int]) -> dict[str, Any]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"工作负载清单必须为 JSON 对象: {path}")
    expected_count = manifest.get("selected_records")
    expected_hash = manifest.get("ids_sha256")
    if expected_count != len(ids) or expected_hash != ids_hash(ids):
        raise ValueError("工作负载清单与 --ids-file 不匹配")
    return {
        "workload_manifest": str(path),
        "workload_type": manifest.get("workload_type"),
        "workload_records": expected_count,
        "workload_ids_sha256": expected_hash,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def attempt_tokens(rows: list[dict[str, Any]], key: str) -> float:
    """失败尝试也可能消耗 Token，不能只依赖最终行的累计值。"""
    return sum(
        float(attempt[key])
        for row in rows
        for attempt in row.get("attempts", [])
        if isinstance(attempt.get(key), (int, float))
    )


def distribution(values: list[float], prefix: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        f"{prefix}_p50": percentile(values, 0.50),
        f"{prefix}_p95": percentile(values, 0.95),
        f"{prefix}_p99": percentile(values, 0.99),
        f"{prefix}_max": round(max(values), 4) if values else None,
    }
    if len(values) > 1 and statistics.mean(values):
        result[f"{prefix}_cv"] = round(statistics.stdev(values) / statistics.mean(values), 4)
    else:
        result[f"{prefix}_cv"] = 0.0 if values else None
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 解析失败 {path}:{line_number}: {error}") from error
    return rows


def summarize(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    latency = numeric(rows, "latency_ms")
    record_end_to_end = numeric(rows, "record_end_to_end_ms")
    queue_wait = numeric(rows, "queue_wait_ms")
    enqueue_blocked = numeric(rows, "enqueue_blocked_ms")
    planning = numeric(rows, "planning_ms")
    prediction = numeric(rows, "prediction_ms")
    selection = numeric(rows, "selection_ms")
    embedding = numeric(rows, "embedding_ms")
    schema_embedding = numeric(rows, "schema_embedding_ms")
    prompt_build = numeric(rows, "prompt_build_ms")
    llm = numeric(rows, "llm_ms")
    parse_validate = numeric(rows, "parse_validate_ms")
    retry_backoff = numeric(rows, "retry_backoff_ms")
    estimated = numeric(rows, "estimated_field_cost")
    actual = numeric(rows, "total_tokens")
    retries = numeric(rows, "retry_count")
    selected_field_counts = [
        float(len(row["selected_fields"]))
        for row in rows
        if isinstance(row.get("selected_fields"), list)
    ]
    success = [row for row in rows if row.get("status") == "success"]
    is_dry_run = bool(meta.get("dry_run"))
    pipeline_ms = meta.get("pipeline_ms")
    summary: dict[str, Any] = {
        **meta,
        "records": len(rows),
        "planned_records": len(rows) if is_dry_run else None,
        "success_records": None if is_dry_run else len(success),
        "error_records": None if is_dry_run else len(rows) - len(success),
        "effective_records_per_second": (
            len(success) / (float(pipeline_ms) / 1000)
            if not is_dry_run and pipeline_ms else None
        ),
        "retry_records": sum(value > 0 for value in retries),
        "retry_count_total": int(sum(retries)),
        "input_tokens_total": sum(numeric(rows, "input_tokens")),
        "output_tokens_total": sum(numeric(rows, "output_tokens")),
        "total_tokens_total": sum(actual),
        "attempt_input_tokens_total": attempt_tokens(rows, "input_tokens"),
        "attempt_output_tokens_total": attempt_tokens(rows, "output_tokens"),
        "attempt_total_tokens_total": attempt_tokens(rows, "total_tokens"),
        "planning_ms_total": sum(planning),
        "prediction_ms_total": sum(prediction),
        "selection_ms_total": sum(selection),
        "embedding_ms_total": sum(embedding),
        "schema_embedding_ms_total": sum(schema_embedding),
        "prompt_build_ms_total": sum(prompt_build),
        "llm_ms_total": sum(llm),
        "parse_validate_ms_total": sum(parse_validate),
        "retry_backoff_ms_total": sum(retry_backoff),
        "enqueue_blocked_ms_total": sum(enqueue_blocked),
        "queue_wait_ratio": (
            sum(queue_wait) / sum(latency) if latency and sum(latency) else None
        ),
        "tail_ratio_p95_p50": (
            percentile(latency, 0.95) / percentile(latency, 0.50)
            if percentile(latency, 0.50) else None
        ),
        "actual_to_estimated_tokens": (
            sum(actual) / sum(estimated) if estimated and sum(estimated) else None
        ),
    }
    summary.update(distribution(estimated, "estimated_cost"))
    summary.update(distribution(selected_field_counts, "selected_field_count"))
    summary.update(distribution(actual, "actual_tokens"))
    summary.update(distribution(latency, "latency_ms"))
    summary.update(distribution(record_end_to_end, "record_end_to_end_ms"))
    summary.update(distribution(queue_wait, "queue_wait_ms"))
    summary.update(distribution(planning, "planning_ms"))
    summary.update(distribution(embedding, "embedding_ms"))
    summary.update(distribution(schema_embedding, "schema_embedding_ms"))
    summary.update(distribution(llm, "llm_ms"))
    summary.update(distribution(parse_validate, "parse_validate_ms"))
    summary.update(distribution(retry_backoff, "retry_backoff_ms"))
    summary.update(distribution(enqueue_blocked, "enqueue_blocked_ms"))
    return summary


def validate(
    rows: list[dict[str, Any]], expected_ids: list[int], methods: list[str], run_id: str,
    submitted: int | None, experiment_id: str,
) -> dict[str, Any]:
    expected = {(artifact_id, method) for artifact_id in expected_ids for method in methods}
    observed = [(row.get("artifact_id"), row.get("method")) for row in rows]
    observed_set = set(observed)
    duplicate_keys = sorted({key for key in observed if observed.count(key) > 1}, key=str)
    missing = sorted(expected - observed_set, key=str)
    unexpected = sorted(observed_set - expected, key=str)
    bad_run_id = sorted(
        {row.get("artifact_id") for row in rows if row.get("run_label") != run_id},
        key=str,
    )
    bad_experiment_id = sorted(
        {row.get("artifact_id") for row in rows if row.get("experiment_id") != experiment_id},
        key=str,
    )
    submitted_mismatch = submitted is not None and submitted != len(expected)
    return {
        "valid": not (
            missing
            or unexpected
            or duplicate_keys
            or bad_run_id
            or bad_experiment_id
            or submitted_mismatch
        ),
        "expected_records": len(expected),
        "observed_records": len(rows),
        "submitted": submitted,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "duplicate_keys": duplicate_keys,
        "run_id_mismatch_artifacts": bad_run_id,
        "experiment_id_mismatch_artifacts": bad_experiment_id,
        "submitted_mismatch": submitted_mismatch,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows}, key=str)
        },
    }


def parse_runner_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "pipeline_ms" in value:
            return value
    return {}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_one(
    args: argparse.Namespace,
    ids: list[int],
    workload_meta: dict[str, Any],
    method: str,
    concurrency: int,
    queue_multiplier: float,
    repeat: int,
) -> dict[str, Any]:
    queue_size = max(1, round(concurrency * queue_multiplier))
    run_id = f"{slug(args.phase)}-{slug(method)}-c{concurrency}-q{queue_size}-r{repeat}"
    experiment_id = f"benchmark-{run_id}"
    raw_dir = args.output_dir / "raw"
    log_dir = args.output_dir / "logs"
    validation_dir = args.output_dir / "validation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{run_id}.jsonl"
    stdout_path = log_dir / f"{run_id}.stdout.log"
    stderr_path = log_dir / f"{run_id}.stderr.log"
    command = [
        args.python_executable, str(args.runner), "--methods", method,
        "--ids-file", str(args.ids_file), "--limit", str(args.limit),
        "--max-concurrency", str(concurrency), "--queue-size", str(queue_size),
        "--max-retries", str(args.max_retries), "--run-label", run_id,
        "--experiment-id", experiment_id, "--output", str(output),
    ]
    if args.dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["LLM_TEMPERATURE"] = str(args.temperature)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    runner_meta = parse_runner_summary(completed.stdout)
    meta = {
        "run_id": run_id,
        "method": method,
        "concurrency": concurrency,
        "queue_multiplier": queue_multiplier,
        "queue_size": queue_size,
        "repeat": repeat,
        "temperature": args.temperature,
        "limit": args.limit,
        "ids_file": str(args.ids_file),
        "ids_requested": len(ids[:args.limit] if args.limit > 0 else ids),
        "phase": args.phase,
        "dry_run": args.dry_run,
        "command": command,
        "returncode": completed.returncode,
        "wall_clock_ms": elapsed_ms,
        **workload_meta,
        **runner_meta,
    }
    result_path = output.with_suffix(".dry_run.jsonl") if args.dry_run else output
    rows = read_jsonl(result_path)
    validation = validate(rows, ids[:args.limit] if args.limit > 0 else ids,
                          [method], run_id, runner_meta.get("submitted"), experiment_id)
    validation["returncode"] = completed.returncode
    (validation_dir / f"{run_id}.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    meta_path = args.output_dir / f"{run_id}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"运行失败 {run_id}，详见 {stderr_path}")
    return summarize(rows, meta) | {"validation_valid": validation["valid"]}


def write_summaries(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    (output_dir / "summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not summaries:
        return
    fieldnames = sorted({key for row in summaries for key in row})
    with (output_dir / "summaries.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.concurrency_values):
        raise SystemExit("repeats 和 concurrency-values 必须为正数")
    ids = load_ids(args.ids_file)
    workload_meta = load_workload_manifest(args.workload_manifest, ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in args.methods:
        for concurrency in args.concurrency_values:
            for queue_multiplier in args.queue_multipliers:
                for repeat in range(1, args.repeats + 1):
                    summaries.append(run_one(
                        args, ids, workload_meta, method, concurrency, queue_multiplier, repeat
                    ))
                    write_summaries(args.output_dir, summaries)

    validation_files = sorted((args.output_dir / "validation").glob("*.json"))
    validations = [json.loads(path.read_text(encoding="utf-8")) for path in validation_files]
    (args.output_dir / "validation.json").write_text(
        json.dumps({"valid": all(item["valid"] for item in validations),
                    "runs": validations}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(summaries), "output_dir": str(args.output_dir),
                      "validation": all(item["valid"] for item in validations)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
