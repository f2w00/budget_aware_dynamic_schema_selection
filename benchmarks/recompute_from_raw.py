#!/usr/bin/env python3
"""从逐条 raw 运行日志重算性能指标，用于回应审稿问题 9/11/12/13/26/31。

核心目的：
- 不读取 summaries.json 中的聚合值，而是从 raw/*.jsonl 逐条重算，
  验证论文主表数字的聚合口径是否可复现。
- 显式区分两种吞吐聚合：
    A) 每次运行先算吞吐再取平均（算术平均，论文采用，问题 9 的 9.06/17.61）；
    B) 合并多次运行的总成功数 / 总流水线时间（审稿人用 2000/220.82 的算法）。
  两者对"比率型"指标不相等，需在正文明确说明。
- 拆分 Token（输入/输出）、失败/重试、首次失败/重试成功/最终失败、错误类型，
  回应问题 11/26/31。
- 给出每次运行（per-run）的逐次指标，回应问题 12/32。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# 小样本双侧 95% t 临界值（n=2 或 n=3）。
T_CRITICAL_95 = {2: 12.706, 3: 4.303}

RAW_DIR = Path("benchmarks/runs/main-comparison-pro/raw")
SUMMARY = Path("benchmarks/runs/main-comparison-pro/summaries.json")


def percentile(sorted_values: list[float], p: float) -> float:
    """线性插值分位数，与 numpy.percentile 默认方式一致。"""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    frac = rank - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def compute_per_record_tokens(record: dict[str, Any]) -> tuple[float, float, float]:
    """按记录顶层字段统计 Token（若存在顶层则用顶层，否则累加 attempts）。"""
    it = record.get("input_tokens")
    ot = record.get("output_tokens")
    tt = record.get("total_tokens")
    if tt is not None:
        return float(it or 0), float(ot or 0), float(tt or 0)
    # 回退：累加 attempts 中的 Token
    i = o = t = 0.0
    for att in record.get("attempts", []) or []:
        i += float(att.get("input_tokens") or 0)
        o += float(att.get("output_tokens") or 0)
        t += float(att.get("total_tokens") or 0)
    return i, o, t


def analyze_attempts(record: dict[str, Any]) -> dict[str, Any]:
    """解析逐条 attempts 数组，还原首次失败/重试成功/最终失败。

    约定：attempt 状态非 success 即为失败；末次 attempt 非 success 即最终失败。
    """
    attempts = record.get("attempts", []) or []
    out = {
        "n_attempts": len(attempts),
        "first_attempt_ok": bool(attempts and attempts[0].get("status") == "success"),
        "final_status": attempts[-1].get("status") if attempts else record.get("status"),
        "retry_success": False,
        "error_types": [],
        "retry_backoff_ms": 0.0,
    }
    for idx, att in enumerate(attempts):
        if att.get("status") != "success":
            if att.get("error"):
                out["error_types"].append(str(att["error"]))
        # 重试成功：首次之后存在某次 attempt 成功。
        elif idx > 0:
            out["retry_success"] = True
        out["retry_backoff_ms"] += float(att.get("retry_backoff_ms") or 0)
    return out


def load_raw(method: str, repeat: int) -> list[dict[str, Any]]:
    pattern = f"main-comparison-pro-{method}-c16-q16-r{repeat}.jsonl"
    path = RAW_DIR / pattern
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recompute_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """对单次运行逐条重算，返回逐次指标。"""
    n_total = len(rows)
    n_success = sum(1 for r in rows if r.get("status") == "success")
    n_error = n_total - n_success

    in_tok = out_tok = tot_tok = 0.0
    llm_ms = parse_ms = embed_ms = schema_ms = 0.0
    planning_ms = prompt_ms = selection_ms = prediction_ms = 0.0
    queue_wait_ms = enqueue_blocked_ms = 0.0
    retry_records = 0
    retry_count_total = 0
    error_counter: dict[str, int] = defaultdict(int)
    n_first_attempt_ok = 0
    n_retry_success = 0
    n_final_fail = 0
    # 附加成本：重试消耗的额外 Token 与时间（attempts 中首次之后的累加）。
    retry_in_tok = retry_out_tok = retry_llm_ms = 0.0

    latencies: list[float] = []

    for r in rows:
        i, o, t = compute_per_record_tokens(r)
        in_tok += i
        out_tok += o
        tot_tok += t
        llm_ms += float(r.get("llm_ms") or 0)
        parse_ms += float(r.get("parse_validate_ms") or 0)
        embed_ms += float(r.get("embedding_ms") or 0)
        schema_ms += float(r.get("schema_embedding_ms") or 0)
        planning_ms += float(r.get("planning_ms") or 0)
        prompt_ms += float(r.get("prompt_build_ms") or 0)
        selection_ms += float(r.get("selection_ms") or 0)
        prediction_ms += float(r.get("prediction_ms") or 0)
        queue_wait_ms += float(r.get("queue_wait_ms") or 0)
        enqueue_blocked_ms += float(r.get("enqueue_blocked_ms") or 0)

        lat = r.get("latency_ms")
        if lat is not None:
            latencies.append(float(lat))

        att = analyze_attempts(r)
        if att["n_attempts"] > 1:
            retry_records += 1
        retry_count_total += max(0, att["n_attempts"] - 1)
        if att["first_attempt_ok"]:
            n_first_attempt_ok += 1
        if att["retry_success"]:
            n_retry_success += 1
        if att["final_status"] != "success":
            n_final_fail += 1
        for et in att["error_types"]:
            error_counter[et] += 1
        # 累加重试（首次之后 attempt）的额外 Token 与 LLM 时间。
        for att_idx, attempt in enumerate(r.get("attempts", []) or []):
            if att_idx == 0:
                continue
            retry_in_tok += float(attempt.get("input_tokens") or 0)
            retry_out_tok += float(attempt.get("output_tokens") or 0)
            retry_llm_ms += float(attempt.get("llm_ms") or 0)

    lat_sorted = sorted(latencies)
    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_error": n_error,
        "n_retry_records": retry_records,
        "retry_count_total": retry_count_total,
        "n_first_attempt_ok": n_first_attempt_ok,
        "n_retry_success": n_retry_success,
        "n_final_fail": n_final_fail,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
        "llm_ms_total": llm_ms,
        "parse_validate_ms_total": parse_ms,
        "embedding_ms_total": embed_ms,
        "schema_embedding_ms_total": schema_ms,
        "planning_ms_total": planning_ms,
        "prompt_build_ms_total": prompt_ms,
        "selection_ms_total": selection_ms,
        "prediction_ms_total": prediction_ms,
        "queue_wait_ms_total": queue_wait_ms,
        "enqueue_blocked_ms_total": enqueue_blocked_ms,
        # 完整在线请求时长（每次请求的流水线时间，含 LLM+嵌入+规划+排队+退避等）。
        "online_request_ms_total": (
            llm_ms + parse_ms + embed_ms + schema_ms + planning_ms
            + prompt_ms + selection_ms + prediction_ms + queue_wait_ms
            + enqueue_blocked_ms
        ),
        # 重试附加成本。
        "retry_input_tokens": retry_in_tok,
        "retry_output_tokens": retry_out_tok,
        "retry_llm_ms_total": retry_llm_ms,
        "latency_p50_ms": percentile(lat_sorted, 50),
        "latency_p95_ms": percentile(lat_sorted, 95),
        "latency_p99_ms": percentile(lat_sorted, 99),
        "latency_max_ms": lat_sorted[-1] if lat_sorted else None,
        "error_types": dict(error_counter),
    }


def ci(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    crit = T_CRITICAL_95.get(len(values))
    if crit is None:
        return None, None
    margin = crit * stdev(values) / math.sqrt(len(values))
    avg = mean(values)
    return avg - margin, avg + margin


def main() -> None:
    parser = argparse.ArgumentParser(description="从逐条日志重算性能指标")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/analysis/performance"))
    args = parser.parse_args()

    methods = ["full", "budget-200", "external-schemarag-k10"]
    repeats = [1, 2, 3]

    # 载入 summaries.json 中的 pipeline_ms（流水线总时长，由 harness 计时，
    # raw 记录无逐条完成时间戳，无法纯靠 raw 重建 makespan）。
    summaries = {
        (s["method"], int(s["repeat"])): s
        for s in json.loads(SUMMARY.read_text(encoding="utf-8"))
    }

    per_run: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for method in methods:
        for rep in repeats:
            rows = load_raw(method, rep)
            run = recompute_run(rows)
            # 从 summary 取该 run 的流水线时间用于吞吐计算
            run["repeat"] = rep
            run["pipeline_ms"] = summaries[(method, rep)]["pipeline_ms"]
            per_run[method].append(run)

    # 汇总：两种吞吐口径 + 各指标 mean/sd/CI
    aggregated: list[dict[str, Any]] = []
    for method in methods:
        runs = per_run[method]
        total_success = sum(r["n_success"] for r in runs)
        total_pipe_s = sum(r["pipeline_ms"] for r in runs) / 1000.0

        thr_per_run = [r["n_success"] / (r["pipeline_ms"] / 1000.0) for r in runs]
        thr_pooled = total_success / total_pipe_s

        tok_per_run = [r["total_tokens"] for r in runs]
        in_tok_per_run = [r["input_tokens"] for r in runs]
        out_tok_per_run = [r["output_tokens"] for r in runs]
        p50 = [r["latency_p50_ms"] for r in runs]
        p95 = [r["latency_p95_ms"] for r in runs]
        p99 = [r["latency_p99_ms"] for r in runs]

        def stat(vals: list[float]) -> dict[str, Any]:
            lo, hi = ci(vals)
            return {"mean": mean(vals), "sd": stdev(vals) if len(vals) > 1 else None,
                    "ci95_low": lo, "ci95_high": hi}

        aggregated.append({
            "method": method,
            "repeats": len(runs),
            "n_total": sum(r["n_total"] for r in runs),
            "n_success": total_success,
            "n_error": sum(r["n_error"] for r in runs),
            "n_retry_records": sum(r["n_retry_records"] for r in runs),
            "retry_count_total": sum(r["retry_count_total"] for r in runs),
            "n_first_attempt_ok": sum(r["n_first_attempt_ok"] for r in runs),
            "n_retry_success": sum(r["n_retry_success"] for r in runs),
            "n_final_fail": sum(r["n_final_fail"] for r in runs),
            "pipeline_ms_per_run": [r["pipeline_ms"] for r in runs],
            # 两种吞吐口径
            "throughput_per_run": thr_per_run,
            "throughput_mean_of_per_run": mean(thr_per_run),
            "throughput_pooled": thr_pooled,
            # 指标 mean/sd/CI
            "total_tokens": stat(tok_per_run),
            "input_tokens": stat(in_tok_per_run),
            "output_tokens": stat(out_tok_per_run),
            "latency_p50": stat(p50),
            "latency_p95": stat(p95),
            "latency_p99": stat(p99),
            "latency_p99_pooled_ms": percentile(
                sorted(lat for r in runs for lat in [r["latency_p99_ms"]]), 50),
            # 错误类型合并（问题 26）
            "error_types": _merge_errors(runs),
            # 成本拆分（问题 11/26）：LLM-only 与完整在线请求
            "llm_only": {
                "tokens_per_run": [r["total_tokens"] for r in runs],
                "llm_ms_per_run": [r["llm_ms_total"] for r in runs],
            },
            "online_request": {
                "tokens_per_run": [r["total_tokens"] for r in runs],
                "ms_per_run": [r["online_request_ms_total"] for r in runs],
                "embedding_ms_per_run": [r["embedding_ms_total"] for r in runs],
                "schema_embedding_ms_per_run": [r["schema_embedding_ms_total"] for r in runs],
                "queue_wait_ms_per_run": [r["queue_wait_ms_total"] for r in runs],
                "enqueue_blocked_ms_per_run": [r["enqueue_blocked_ms_total"] for r in runs],
            },
            "retry_extra_cost": {
                "input_tokens_per_run": [r["retry_input_tokens"] for r in runs],
                "output_tokens_per_run": [r["retry_output_tokens"] for r in runs],
                "llm_ms_per_run": [r["retry_llm_ms_total"] for r in runs],
            },
        })

    payload = {
        "analysis": {
            "description": "从逐条 raw 日志离线重算；不触发 LLM/embedding。",
            "throughput_aggregation": (
                "mean_of_per_run=平均每次运行吞吐(论文采用)；"
                "pooled=合并总成功数/总流水线时间(审稿人口径)。两者对比率型指标不相等。"
            ),
            "token_scope": (
                "LLM-only tokens 来自 LLM API usage（input/output/total_tokens，已含重试累计）；"
                "embedding 服务（text-embedding）不返回 usage，embedding 的 token 数在本日志中不可得，"
                "故不将其计入 token，仅能报告完整在线请求的时长（含 embedding_ms/schema_embedding_ms）。"
            ),
            "source_dir": str(RAW_DIR),
            "pipeline_ms_source": str(SUMMARY),
        },
        "aggregated": aggregated,
        "per_run": per_run,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recompute_from_raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir / "recompute_from_raw.json")}, ensure_ascii=False))


def _merge_errors(runs: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = defaultdict(int)
    for r in runs:
        for et, cnt in r["error_types"].items():
            merged[et] += cnt
    return dict(merged)


if __name__ == "__main__":
    main()