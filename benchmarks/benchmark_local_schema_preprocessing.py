#!/usr/bin/env python3
"""测量本地动态 Schema 预处理的规模与多进程扩展，不调用 LLM。"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budget_selector import select_fields
from llm_prompt import build_prompt, load_prompt_assets
from llm_runner import load_models, predict_fields, schema_costs


DEFAULT_COHORT = ROOT / "data/hunan_museum_300000001_300000500_without_url"
DEFAULT_MODEL = ROOT / "models/hunan_museum_300000001_300000500_without_url_n100_m1/seed_42"

_WORKER_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=ROOT / "1_stable.prompt")
    parser.add_argument("--schema", type=Path, default=ROOT / "simple2.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "field_weights.json")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"快照为空: {path}")
    return rows


def init_worker(
    model_dir: str,
    prompt_path: str,
    schema_path: str,
    weights_path: str,
    budget: int,
) -> None:
    template, schema = load_prompt_assets(Path(prompt_path), Path(schema_path))
    config = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    _WORKER_STATE.update(
        {
            "models": load_models(Path(model_dir)),
            "template": template,
            "schema": schema,
            "costs": schema_costs(schema),
            "weights": config["weights"],
            "required_fields": config["required_fields"],
            "budget": budget,
        }
    )


def process_shard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = _WORKER_STATE
    schema = state["schema"]
    digest_xor = 0
    planning_ns = 0
    for row in rows:
        started = time.perf_counter_ns()
        prediction = predict_fields(row["raw_data"], list(schema), state["models"])
        selection = select_fields(
            prediction,
            schema,
            state["costs"],
            state["weights"],
            state["required_fields"],
            state["budget"],
        )
        prompt, _, _ = build_prompt(
            state["template"], schema, selection["selected_fields"], row["raw_data"]
        )
        planning_ns += time.perf_counter_ns() - started
        record_digest = hashlib.sha256(
            f"{row['benchmark_index']}:{selection['selected_fields']}:"
            f"{selection['estimated_cost']}:{hashlib.sha256(prompt.encode()).hexdigest()}".encode()
        )
        digest_xor ^= int.from_bytes(record_digest.digest(), byteorder="big")
    return {"records": len(rows), "planning_ns": planning_ns, "digest_xor": digest_xor}


def make_workload(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return [
        {"benchmark_index": index, "raw_data": rows[index % len(rows)]["raw_data"]}
        for index in range(size)
    ]


def shard(rows: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    return [rows[index::workers] for index in range(workers) if rows[index::workers]]


def process_tree_pids(root_pid: int) -> set[int]:
    """从 /proc 构建进程树，避免为基准引入额外运行时依赖。"""
    parent_to_children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
            parent = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
        except (FileNotFoundError, StopIteration, ValueError):
            continue
        parent_to_children.setdefault(parent, []).append(int(entry.name))
    pids = {root_pid}
    pending = [root_pid]
    while pending:
        for child in parent_to_children.get(pending.pop(), []):
            if child not in pids:
                pids.add(child)
                pending.append(child)
    return pids


def rss_bytes(root_pid: int) -> int:
    total = 0
    for pid in process_tree_pids(root_pid):
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            rss_kb = int(next(line.split()[1] for line in status.splitlines() if line.startswith("VmRSS:")))
            total += rss_kb * 1024
        except (FileNotFoundError, StopIteration, ValueError):
            continue
    return total


def run_once(
    workload: list[dict[str, Any]],
    workers: int,
    initargs: tuple[str, str, str, str, int],
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    root_pid = os.getpid()
    peak_rss = rss_bytes(root_pid)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=init_worker,
        initargs=initargs,
    ) as executor:
        futures = [executor.submit(process_shard, item) for item in shard(workload, workers)]
        while any(not future.done() for future in futures):
            peak_rss = max(peak_rss, rss_bytes(root_pid))
            time.sleep(0.01)
        results = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, rss_bytes(root_pid))
    digest_xor = 0
    for result in results:
        digest_xor ^= result["digest_xor"]
    digest = f"{digest_xor:064x}"
    return {
        "records": len(workload),
        "workers": workers,
        "elapsed_seconds": elapsed_seconds,
        "throughput_records_per_second": len(workload) / elapsed_seconds,
        "mean_planning_ms": sum(item["planning_ns"] for item in results) / len(workload) / 1_000_000,
        "peak_process_tree_rss_bytes": peak_rss,
        "result_digest": digest,
    }


def main() -> None:
    args = parse_args()
    if min(args.sizes) <= 0 or min(args.workers) <= 0 or args.repetitions <= 0:
        raise SystemExit("sizes、workers 和 repetitions 均必须为正数")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {output}")
    cohort_dir = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    records = load_records(cohort_dir / "snapshot.jsonl")
    initargs = tuple(
        str(value if value.is_absolute() else ROOT / value)
        for value in (args.model_dir, args.prompt, args.schema, args.weights)
    ) + (args.budget,)
    runs = []
    for size in args.sizes:
        workload = make_workload(records, size)
        expected_digest: str | None = None
        for workers in args.workers:
            for repetition in range(1, args.repetitions + 1):
                result = run_once(workload, workers, initargs)
                if expected_digest is None:
                    expected_digest = result["result_digest"]
                result.update(
                    {
                        "size": size,
                        "repetition": repetition,
                        "consistent_with_first_run": result["result_digest"] == expected_digest,
                    }
                )
                runs.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "description": "仅测量本地预测、预算选择和 Prompt 构造；不调用 LLM。",
                "cohort_dir": str(cohort_dir),
                "model_dir": initargs[0],
                "budget": args.budget,
                "sizes": args.sizes,
                "workers": args.workers,
                "repetitions": args.repetitions,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
