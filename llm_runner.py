#!/usr/bin/env python3
"""并发执行 Full、Fixed 和 Adaptive Schema 的真实 LLM 抽取实验。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg
import tiktoken
from openai import AsyncOpenAI
from scipy.sparse import hstack

from budget_selector import select_fields as select_budgeted
from llm_prompt import build_prompt, load_prompt_assets
from unbudgeted_selector import select_fields as select_unbudgeted
from external_schemarag_selector import (
    select_fields as select_schemarag,
    selector_config_hash,
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行真实 LLM Schema 对照实验")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "prompt-null", "prompt-omit", "fixed-5", "budget-400", "lambda-0.003"],
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="冻结样本 ID 文件，支持 JSON 数组或包含 artifact_id 的 JSONL",
    )
    parser.add_argument(
        "--records-file",
        type=Path,
        help="冻结记录 JSONL；提供后不再从数据库读取 raw_data",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/extraction_results.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("1.prompt"))
    parser.add_argument("--prompt-null-template", type=Path, default=Path("1_prompt_null.prompt"))
    parser.add_argument("--prompt-omit-template", type=Path, default=Path("1_prompt_omit.prompt"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--weights", type=Path, default=Path("field_weights.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--queue-size", type=int, default=0)
    # 首次请求之外最多重试两次，避免瞬时协议错误直接成为最终失败。
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--allow-extra-fields",
        action="store_true",
        help="允许模型返回目标字段之外的键，并在归一化时丢弃",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def schema_costs(schema: dict[str, str]) -> dict[str, Any]:
    encoder = tiktoken.get_encoding("cl100k_base")
    return {
        "tokenizer": "cl100k_base",
        "fields": {
            field: {
                "field_name_tokens": len(encoder.encode(field)),
                "schema_value_tokens": len(encoder.encode(description)),
            }
            for field, description in schema.items()
        }
    }


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return text_hash(path.read_text(encoding="utf-8"))



# Prompt-only 基线：完整 Schema 照常输入，仅通过 Prompt 指令改变对无信息字段的处理
PROMPT_VARIANT_METHODS = {
    "prompt-null": "prompt_null",
    "prompt-omit": "prompt_omit",
}

SCHEMARAG_METHODS = {
    "external-schemarag-k8": 8,
    "external-schemarag-k10": 10,
    "external-schemarag-k15": 15,
}


def prompt_variant(method: str) -> str:
    """按实验方法映射 Prompt 模板变体，其余方法共用基础模板。"""
    return PROMPT_VARIANT_METHODS.get(method, "base")


def method_info(method: str) -> tuple[str, str, int | None, float | None]:
    if method == "full":
        return "full", "full", None, None
    if method in {"prompt-null", "prompt-omit"}:
        return "prompt_only", method, None, None
    if method.startswith("fixed-"):
        return "fixed", method, None, None
    if method.startswith("budget-"):
        return "adaptive_budget", method, int(method.split("-", 1)[1]), None
    if method.startswith("lambda-"):
        return "adaptive_unbudgeted", method, None, float(method.split("-", 1)[1])
    if method in SCHEMARAG_METHODS:
        return "external_schema_retrieval", method, None, None
    raise ValueError(f"未知实验方法: {method}")


def method_requires_prediction(method: str) -> bool:
    """仅为实际依赖字段预测的策略加载和调用选择器。"""
    return (
        method.startswith("budget-")
        or method.startswith("lambda-")
    )


def selected_cost(selected_fields: list[str], costs: dict[str, Any]) -> int:
    return sum(
        int(costs["fields"][field]["field_name_tokens"])
        + int(costs["fields"][field]["schema_value_tokens"])
        for field in selected_fields
    )


def fields_cost(fields: list[str], costs: dict[str, Any], prediction: dict[str, Any]) -> int:
    """统一字段成本口径：定义 Token + 该记录的预测输出 Token。"""
    return sum(
        int(costs["fields"][field]["field_name_tokens"])
        + int(costs["fields"][field]["schema_value_tokens"])
        + int(prediction["predicted_output_tokens"].get(field, 0))
        for field in fields
    )


def load_models(model_dir: Path) -> dict[str, Any]:
    bundled_model = model_dir / "model.joblib"
    if bundled_model.exists():
        models = joblib.load(bundled_model)
        required = {
            "char_vectorizer",
            "word_vectorizer",
            "classifiers",
            "constants",
            "regressors",
            "length_metadata",
        }
        missing = required - models.keys()
        if missing:
            raise ValueError(f"模型产物缺少必要组件: {sorted(missing)}")
        return models
    return {
        "vectorizer": joblib.load(model_dir / "vectorizer.joblib"),
        "classifiers": joblib.load(model_dir / "classifiers.joblib"),
        "constants": json.loads((model_dir / "constants.json").read_text(encoding="utf-8")),
        "regressors": joblib.load(model_dir / "length_regressors.joblib"),
        "length_metadata": json.loads((model_dir / "length_metadata.json").read_text(encoding="utf-8")),
    }


def predict_fields(raw_data: Any, fields: list[str], models: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(raw_data, ensure_ascii=False, separators=(",", ":"), default=str)
    if "vectorizer" in models:
        matrix = models["vectorizer"].transform([text])
    else:
        # M1 的分类器和长度回归器均使用字符、词级特征的拼接矩阵。
        matrix = hstack(
            [
                models["char_vectorizer"].transform([text]),
                models["word_vectorizer"].transform([text]),
            ],
            format="csr",
        )
    scores = {}
    lengths = {}
    for field in fields:
        scores[field] = float(
            models["constants"][field]
            if field in models["constants"]
            else models["classifiers"][field].predict_proba(matrix)[0, 1]
        )
        length_info = models["length_metadata"][field]
        if length_info["type"] == "median_constant":
            length = length_info["value"]
        else:
            length = float(np.expm1(models["regressors"][field].predict(matrix)[0]))
        lengths[field] = max(0, round(length))
    return {"field_scores": scores, "predicted_output_tokens": lengths}


def fixed_fields(schema: dict[str, str], weights: dict[str, Any], size: int) -> list[str]:
    required = weights["required_fields"]
    ranked = sorted(
        (field for field in schema if field not in required),
        key=lambda field: (-weights["weights"][field], list(schema).index(field)),
    )
    return (required + ranked)[:size]


def resolve_fields(
    method: str,
    raw_data: Any,
    schema: dict[str, str],
    weights: dict[str, Any],
    costs: dict[str, Any],
    models: dict[str, Any] | None,
    prediction: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    if method in {"full", "prompt-null", "prompt-omit"}:
        return list(schema), None
    if method.startswith("fixed-"):
        size = int(method.split("-", 1)[1])
        return fixed_fields(schema, weights, size), None
    if method in SCHEMARAG_METHODS:
        selection = select_schemarag(
            raw_data,
            schema,
            costs,
            # SchemaRAG 的 Top-k 只依赖 query 与字段向量相似度；不加载 M1。
            prediction or {},
            cache_path=Path("experiments/schemarag_row_embeddings.json"),
            k=SCHEMARAG_METHODS[method],
        )
        return selection["selected_fields"], selection
    # 允许传入已计算的预测结果，避免同一记录在多个方法间重复推理
    if prediction is None:
        prediction = predict_fields(raw_data, list(schema), models)
    if method.startswith("budget-"):
        budget = int(method.split("-", 1)[1])
        selection = select_budgeted(
            prediction,
            schema,
            costs,
            weights["weights"],
            weights["required_fields"],
            budget,
        )
        return selection["selected_fields"], selection
    if method.startswith("lambda-"):
        lambda_value = float(method.split("-", 1)[1])
        selection = select_unbudgeted(
            prediction,
            schema,
            costs,
            weights,
            lambda_value,
        )
        return selection["selected_fields"], selection
    raise ValueError(f"未知实验方法: {method}")


def iter_records(
    database_url: str, ids: list[int], limit: int
) -> Iterator[tuple[int, Any]]:
    """按冻结 ID 的原始顺序流式读取记录，避免全量载入任务数据。"""
    selected_ids = ids[:limit] if limit > 0 else ids
    with psycopg.connect(database_url) as connection, connection.cursor(
        name="artifact_stream"
    ) as cursor:
        cursor.execute(
            "SELECT raw.artifact_id, raw.raw_data "
            "FROM unnest(%s::bigint[]) WITH ORDINALITY "
            "AS requested(artifact_id, ordinal) "
            "JOIN public.artifact_raw AS raw "
            "ON raw.artifact_id = requested.artifact_id "
            "ORDER BY requested.ordinal",
            (selected_ids,),
        )
        yield from cursor


def iter_frozen_records(
    records_file: Path, ids: list[int], limit: int
) -> Iterator[tuple[int, Any]]:
    """按请求 ID 顺序读取冻结快照，并拒绝缺失或重复记录。"""
    records: dict[int, Any] = {}
    for line_number, line in enumerate(records_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, int) or "raw_data" not in row:
            raise ValueError(f"冻结记录格式错误: {records_file}:{line_number}")
        if artifact_id in records:
            raise ValueError(f"冻结记录包含重复 artifact_id: {artifact_id}")
        records[artifact_id] = row["raw_data"]
    selected_ids = ids[:limit] if limit > 0 else ids
    missing = [artifact_id for artifact_id in selected_ids if artifact_id not in records]
    if missing:
        raise ValueError(f"冻结记录缺少 {len(missing)} 个请求 ID")
    for artifact_id in selected_ids:
        yield artifact_id, records[artifact_id]


def load_records(database_url: str, ids: list[int], limit: int) -> list[tuple[int, Any]]:
    """保留列表接口，供需要一次性加载的小型工具调用。"""
    return list(iter_records(database_url, ids, limit))


def current_rss_bytes() -> int | None:
    """读取 Linux 进程常驻内存；非 Linux 或不可读取时不阻塞运行。"""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def build_record_tasks(
    artifact_id: int,
    raw_data: Any,
    methods: list[str],
    completed: set[tuple[int, str]],
    schema: dict[str, str],
    weights: dict[str, Any],
    costs: dict[str, Any],
    models: dict[str, Any] | None,
    templates: dict[str, str],
    method_prompt_hashes: dict[str, str],
    method_selector_hashes: dict[str, str | None],
    schema_hash: str,
    experiment_id: str,
    model: str,
    model_dir: Path | None,
    run_label: str | None,
) -> Iterator[dict[str, Any]]:
    """为单条记录构造有限任务，预测结果仅在该记录的方法间复用。"""
    prediction: dict[str, Any] | None = None
    full_estimated_cost: int | None = None
    for method in methods:
        if (artifact_id, method) in completed:
            continue
        record_started_at = time.perf_counter()
        task_prediction_ms = 0.0
        # FullSchema 基线不加载或调用选择器，避免引入方法外规划开销。
        if method_requires_prediction(method) and prediction is None:
            if models is None:
                raise ValueError(f"方法 {method} 需要字段选择模型")
            prediction_started = time.perf_counter()
            prediction = predict_fields(raw_data, list(schema), models)
            task_prediction_ms = (time.perf_counter() - prediction_started) * 1000
            full_estimated_cost = fields_cost(list(schema), costs, prediction)
        selection_started = time.perf_counter()
        selected_fields, selection = resolve_fields(
            method, raw_data, schema, weights, costs, models, prediction
        )
        selection_ms = (time.perf_counter() - selection_started) * 1000
        method_group, setting, budget, lambda_value = method_info(method)
        prompt_build_started = time.perf_counter()
        prompt, target_structure, focus_schema = build_prompt(
            templates[prompt_variant(method)], schema, selected_fields, raw_data
        )
        prompt_build_ms = (time.perf_counter() - prompt_build_started) * 1000
        embedding_ms = selection.get("query_embedding_ms", 0.0) if selection else 0.0
        schema_embedding_ms = (
            selection.get("schema_embedding_ms", 0.0) if selection else 0.0
        )
        estimated_cost = (
            None
            if method in SCHEMARAG_METHODS
            else selection["estimated_cost"]
            if selection
            else fields_cost(selected_fields, costs, prediction)
            if prediction is not None
            else None
        )
        yield {
            "artifact_id": artifact_id,
            "method": method,
            "selected_fields": selected_fields,
            "target_structure": target_structure,
            "focus_schema": focus_schema,
            "raw_data": raw_data,
            "full_estimated_cost": full_estimated_cost,
            "estimated_field_cost": estimated_cost,
            "required_budget": estimated_cost,
            "allocated_budget": budget,
            "budget_status": (
                "full" if method in {"full", "prompt-null", "prompt-omit"} else "planned"
            ),
            "experiment_id": experiment_id,
            "prompt_variant": prompt_variant(method),
            "prompt_hash": method_prompt_hashes[method],
            "schema_hash": schema_hash,
            "model_name": model,
            "run_label": run_label,
            "selector_model_dir": str(model_dir) if model_dir is not None else None,
            "setting": setting,
            "method_group": method_group,
            "budget": budget,
            "lambda": lambda_value,
            "method_metadata": selection.get("method_metadata") if selection else None,
            "selector_config_hash": method_selector_hashes[method],
            "selector_similarities": selection.get("similarities") if selection else None,
            "prediction_ms": round(task_prediction_ms, 2),
            "selection_ms": round(selection_ms, 2),
            "prompt_build_ms": round(prompt_build_ms, 2),
            "embedding_ms": round(embedding_ms, 2),
            "schema_embedding_ms": round(schema_embedding_ms, 2),
            "planning_ms": round(
                task_prediction_ms + selection_ms + prompt_build_ms, 2
            ),
            "missing_fields_expected": method == "prompt-omit",
            "prompt": prompt,
            "_record_started_at": record_started_at,
        }


def load_ids(ids_file: Path | None, split_dir: Path, split: str) -> list[int]:
    """加载冻结 ID，避免正式 Gold 实验误用整个数据划分。"""
    if ids_file is None:
        return json.loads((split_dir / f"{split}_ids.json").read_text(encoding="utf-8"))

    content = ids_file.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"ID 文件为空: {ids_file}")
    if content.startswith("["):
        ids = json.loads(content)
    else:
        ids = [json.loads(line)["artifact_id"] for line in content.splitlines() if line.strip()]
    if not all(isinstance(artifact_id, int) for artifact_id in ids):
        raise ValueError(f"ID 文件包含非整数 artifact_id: {ids_file}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"ID 文件包含重复 artifact_id: {ids_file}")
    return ids


def parse_json_response(
    content: str, allow_missing_values: bool = False
) -> tuple[dict[str, Any], bool]:
    content = content.strip()
    # 兼容 Qwen3 服务仍返回思考标签的情况，抽取思考后的 JSON 主体。
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    repaired_missing_values = False
    if allow_missing_values:
        # prompt-omit 明确允许漏出字段；仅修复已闭合键名后的无值属性，避免改写字符串内容。
        repaired = re.sub(
            r'("(?:\\.|[^"\\])*")\s*:\s*(?=[,}])', r"\1: null", content
        )
        repaired_missing_values = repaired != content
        content = repaired
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM 返回结果不是 JSON 对象")
    return parsed, repaired_missing_values


def validate_result_types(result: dict[str, Any], fields: list[str]) -> None:
    """检查字段值类型，允许常见的结构化抽取值。"""
    invalid = [
        field for field in fields
        if field in result and not (
            result[field] is None
            or isinstance(result[field], (str, int, float, bool, list, dict))
        )
    ]
    if invalid:
        raise ValueError(f"字段值类型无效: {invalid}")


def normalize_result_aliases(
    result: dict[str, Any], selected_fields: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """仅归一化可无损映射到目标字段的已知别名。"""
    normalized = dict(result)
    applied = {}
    for alias, field in {"文物名称": "name"}.items():
        if alias not in normalized or field not in selected_fields:
            continue
        if field not in normalized:
            normalized[field] = normalized[alias]
        del normalized[alias]
        applied[alias] = field
    return normalized, applied


def normalize_result(result: dict[str, Any], selected_fields: list[str]) -> dict[str, Any]:
    """补齐模型漏出的目标字段，保证结果结构统一。"""
    return {field: result.get(field) for field in selected_fields}


def existing_keys(
    path: Path,
    experiment_id: str,
    method_prompt_hashes: dict[str, str],
    method_selector_hashes: dict[str, str | None],
    schema_hash: str,
    model: str,
) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("status") == "success"
            and row.get("experiment_id") == experiment_id
            # 断点判断按方法比较各自模板哈希，避免 Prompt 变体之间误判已完成
            and row.get("prompt_hash") == method_prompt_hashes.get(row.get("method"))
            and row.get("schema_hash") == schema_hash
            and row.get("model_name") == model
            and (
                not row.get("method", "").startswith("external-schemarag-")
                or row.get("selector_config_hash")
                == method_selector_hashes.get(row.get("method"))
            )
        ):
            keys.add((row["artifact_id"], row["method"]))
    return keys


async def run_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore | None,
    write_lock: asyncio.Lock,
    output: Path,
    record: dict[str, Any],
    model: str,
    timeout: float,
    temperature: float,
    max_retries: int,
    input_price: float,
    output_price: float,
    allow_extra_fields: bool,
) -> None:
    request_started = time.perf_counter()
    queue_entered = record.get("queue_entered_at", request_started)
    queue_wait_ms = round((request_started - queue_entered) * 1000, 2)
    error = None
    attempts = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_latency_ms = 0.0
    total_llm_ms = 0.0
    total_parse_validate_ms = 0.0
    total_retry_backoff_ms = 0.0
    observed_usage = False
    observed_total_tokens = False
    total_actual_cost = 0.0
    observed_actual_cost = False
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        response = None
        content = None
        input_tokens = None
        output_tokens = None
        response_total_tokens = None
        attempt_actual_cost = None
        llm_started = None
        parse_validate_started = None
        llm_ms = 0.0
        parse_validate_ms = 0.0
        try:
            # 并发已由消费者数量控制，信号量仅作可选的额外限流
            acquire_context = (
                semaphore if semaphore is not None else contextlib.nullcontext()
            )
            async with acquire_context:
                llm_started = time.perf_counter()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": record["prompt"]}],
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": False,
                            }
                        },
                    ),
                    timeout=timeout,
                )
            llm_ms = (time.perf_counter() - llm_started) * 1000
            total_llm_ms += llm_ms
            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            response_total_tokens = getattr(usage, "total_tokens", None)
            if input_tokens is not None and output_tokens is not None:
                observed_usage = True
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            if response_total_tokens is not None:
                observed_total_tokens = True
                total_tokens += response_total_tokens
            attempt_actual_cost = None
            if input_tokens is not None and output_tokens is not None:
                attempt_actual_cost = (
                    input_tokens * input_price + output_tokens * output_price
                )
                observed_actual_cost = True
                total_actual_cost += attempt_actual_cost
            parse_validate_started = time.perf_counter()
            result, repaired_missing_values = parse_json_response(
                content, allow_missing_values=record["missing_fields_expected"]
            )
            result, normalized_aliases = normalize_result_aliases(
                result, record["selected_fields"]
            )
            validate_result_types(result, record["selected_fields"])
            unexpected = sorted(set(result) - set(record["selected_fields"]))
            missing = sorted(set(record["selected_fields"]) - set(result))
            if unexpected and not allow_extra_fields:
                raise ValueError(f"返回了目标字段之外的键: {unexpected}")
            if unexpected:
                result = {
                    key: value
                    for key, value in result.items()
                    if key in record["selected_fields"]
                }
            result = normalize_result(result, record["selected_fields"])
            parse_validate_ms = (time.perf_counter() - parse_validate_started) * 1000
            total_parse_validate_ms += parse_validate_ms
            attempt_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            total_latency_ms += attempt_latency_ms
            attempts.append({
                "attempt": attempt + 1,
                "status": "success",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response_total_tokens,
                "actual_cost": attempt_actual_cost,
                "latency_ms": attempt_latency_ms,
                "llm_ms": round(llm_ms, 2),
                "parse_validate_ms": round(parse_validate_ms, 2),
                "retry_backoff_ms": 0.0,
                "error": None,
            })
            row = {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "prompt" and not key.startswith("_")
                },
                "result": result,
                "raw_response": content,
                "input_tokens": total_input_tokens if observed_usage else None,
                "output_tokens": total_output_tokens if observed_usage else None,
                "total_tokens": total_tokens if observed_total_tokens else None,
                "actual_cost": total_actual_cost if observed_actual_cost else None,
                "input_price_per_token": input_price,
                "output_price_per_token": output_price,
                "latency_ms": round(total_latency_ms, 2),
                "llm_ms": round(total_llm_ms, 2),
                "parse_validate_ms": round(total_parse_validate_ms, 2),
                "retry_backoff_ms": round(total_retry_backoff_ms, 2),
                "queue_wait_ms": queue_wait_ms,
                "retry_count": attempt,
                "attempts": attempts,
                "status": "success",
                "unexpected_fields": unexpected,
                "missing_fields": missing,
                "normalized_field_aliases": normalized_aliases,
                "repaired_missing_json_values": repaired_missing_values,
                "error": None,
            }
            break
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            attempt_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            total_latency_ms += attempt_latency_ms
            if llm_started is not None and llm_ms == 0.0:
                llm_ms = (time.perf_counter() - llm_started) * 1000
                total_llm_ms += llm_ms
            if parse_validate_started is not None and parse_validate_ms == 0.0:
                parse_validate_ms = (
                    time.perf_counter() - parse_validate_started
                ) * 1000
                total_parse_validate_ms += parse_validate_ms
            attempts.append({
                "attempt": attempt + 1,
                "status": "error",
                "input_tokens": (
                    getattr(response.usage, "prompt_tokens", None)
                    if response is not None else None
                ),
                "output_tokens": (
                    getattr(response.usage, "completion_tokens", None)
                    if response is not None else None
                ),
                "total_tokens": (
                    getattr(response.usage, "total_tokens", None)
                    if response is not None else None
                ),
                "actual_cost": (
                    input_tokens * input_price + output_tokens * output_price
                    if response is not None
                    and input_tokens is not None
                    and output_tokens is not None else None
                ),
                "latency_ms": attempt_latency_ms,
                "llm_ms": round(llm_ms, 2),
                "parse_validate_ms": round(parse_validate_ms, 2),
                "retry_backoff_ms": 0.0,
                "error": error,
            })
            if attempt >= max_retries:
                row = {
                    **{
                        key: value
                        for key, value in record.items()
                        if key != "prompt" and not key.startswith("_")
                    },
                    "result": None,
                    "raw_response": content,
                    "input_tokens": total_input_tokens if observed_usage else None,
                    "output_tokens": total_output_tokens if observed_usage else None,
                    "total_tokens": total_tokens if observed_total_tokens else None,
                    "actual_cost": total_actual_cost if observed_actual_cost else None,
                    "input_price_per_token": input_price,
                    "output_price_per_token": output_price,
                    "latency_ms": round(total_latency_ms, 2),
                    "llm_ms": round(total_llm_ms, 2),
                    "parse_validate_ms": round(total_parse_validate_ms, 2),
                    "retry_backoff_ms": round(total_retry_backoff_ms, 2),
                    "queue_wait_ms": queue_wait_ms,
                    "retry_count": attempt,
                    "attempts": attempts,
                    "status": "error",
                    "unexpected_fields": [],
                    "missing_fields": record["selected_fields"],
                    "error": error,
                }
                break
            backoff_seconds = min(30.0, (2**attempt) + random.random())
            backoff_started = time.perf_counter()
            await asyncio.sleep(backoff_seconds)
            backoff_ms = (time.perf_counter() - backoff_started) * 1000
            total_retry_backoff_ms += backoff_ms
            attempts[-1]["retry_backoff_ms"] = round(backoff_ms, 2)
    async with write_lock:
        record_started_at = record.get("_record_started_at")
        row["record_end_to_end_ms"] = (
            round((time.perf_counter() - record_started_at) * 1000, 2)
            if isinstance(record_started_at, (int, float))
            else None
        )
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def main_async() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not args.records_file and not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    base_url = os.getenv("LLM_BASE_URL", "http://172.18.20.44:30059/v1")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "kg-assist")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    concurrency = args.max_concurrency or int(os.getenv("LLM_MAX_CONCURRENCY", "30"))
    input_price = float(os.getenv("LLM_INPUT_PRICE_PER_TOKEN", "0"))
    output_price = float(os.getenv("LLM_OUTPUT_PRICE_PER_TOKEN", "0"))
    if not args.dry_run and not api_key:
        raise SystemExit("缺少 LLM_API_KEY；请通过环境变量或 .env 提供")

    base_template, schema = load_prompt_assets(args.prompt, args.schema)
    # 按需加载 Prompt 变体模板；prompt-only 基线与 Full 输入完整 Schema，仅指令不同
    variant_paths = {
        "prompt_null": args.prompt_null_template,
        "prompt_omit": args.prompt_omit_template,
    }
    templates = {"base": base_template}
    for variant in sorted({prompt_variant(m) for m in args.methods} - {"base"}):
        variant_path = variant_paths[variant]
        if not variant_path.exists():
            raise SystemExit(f"缺少 Prompt 变体模板: {variant_path}")
        templates[variant] = variant_path.read_text(encoding="utf-8")
    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    costs = schema_costs(schema)
    needs_models = any(method_requires_prediction(method) for method in args.methods)
    models = load_models(args.model_dir) if needs_models else None
    # dry-run 只检查任务构造，不应访问远程 Embedding 服务。
    if args.dry_run and any(method in SCHEMARAG_METHODS for method in args.methods):
        os.environ["EMBEDDING_OFFLINE_STUB"] = "1"
    selector_model_dir = args.model_dir if needs_models else None
    # 每个方法记录各自模板哈希，保证断点续跑与 experiment_id 能区分 Prompt 变体
    method_prompt_hashes = {
        method: text_hash(templates[prompt_variant(method)])
        for method in args.methods
    }
    method_selector_hashes = {
        method: (
            selector_config_hash(schema, os.environ.get("EMBEDDING_MODEL", ""))
            if method in SCHEMARAG_METHODS else None
        )
        for method in args.methods
    }
    schema_hash = text_hash(json.dumps(schema, ensure_ascii=False, sort_keys=True))
    experiment_id = args.experiment_id or text_hash(
        json.dumps(
            {
                "prompt_hashes": method_prompt_hashes,
                "schema_hash": schema_hash,
                "model": model,
                "methods": args.methods,
                "split": args.split,
            },
            sort_keys=True,
        )
    )
    ids = load_ids(args.ids_file, args.split_dir, args.split)
    completed = existing_keys(
        args.output,
        experiment_id,
        method_prompt_hashes,
        method_selector_hashes,
        schema_hash,
        model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = (
        iter_frozen_records(args.records_file, ids, args.limit)
        if args.records_file
        else iter_records(database_url, ids, args.limit)
    )
    if args.dry_run:
        preview = args.output.with_suffix(".dry_run.jsonl")
        planned = 0
        with preview.open("w", encoding="utf-8") as file:
            for artifact_id, raw_data in records:
                for task in build_record_tasks(
                    artifact_id,
                    raw_data,
                    args.methods,
                    completed,
                    schema,
                    weights,
                    costs,
                    models,
                    templates,
                    method_prompt_hashes,
                    method_selector_hashes,
                    schema_hash,
                    experiment_id,
                    model,
                    selector_model_dir,
                    args.run_label,
                ):
                    preview_task = {
                        key: value for key, value in task.items() if not key.startswith("_")
                    }
                    file.write(
                        json.dumps(preview_task, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    planned += 1
        print(json.dumps({"tasks": planned, "preview": str(preview)}, ensure_ascii=False))
        return

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    write_lock = asyncio.Lock()
    queue_size = args.queue_size or max(concurrency * 2, 1)
    queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Event] | None] = asyncio.Queue(
        maxsize=queue_size
    )
    queue_peak = 0
    queue_samples = []
    producer_blocked_ms = 0.0
    consumer_idle_ms = 0.0
    submitted = 0
    rss_samples: list[int] = []
    rss_stop = asyncio.Event()

    async def sample_rss() -> None:
        while not rss_stop.is_set():
            rss = current_rss_bytes()
            if rss is not None:
                rss_samples.append(rss)
            await asyncio.sleep(0.1)

    async def consumer() -> None:
        nonlocal consumer_idle_ms, queue_peak
        while True:
            idle_started = time.perf_counter()
            task = await queue.get()
            consumer_idle_ms += (time.perf_counter() - idle_started) * 1000
            queue_samples.append(queue.qsize())
            queue_peak = max(queue_peak, queue.qsize())
            try:
                if task is None:
                    return
                record, ready = task
                await ready.wait()
                await run_request(
                    client,
                    None,
                    write_lock,
                    args.output,
                    record,
                    model,
                    timeout,
                    temperature,
                    args.max_retries,
                    input_price,
                    output_price,
                    args.allow_extra_fields,
                )
            finally:
                queue.task_done()

    pipeline_started = time.perf_counter()
    rss_sampler = asyncio.create_task(sample_rss())
    consumers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
    enqueue_started = time.perf_counter()
    for artifact_id, raw_data in records:
        for task in build_record_tasks(
            artifact_id,
            raw_data,
            args.methods,
            completed,
            schema,
            weights,
            costs,
            models,
            templates,
            method_prompt_hashes,
            method_selector_hashes,
            schema_hash,
            experiment_id,
            model,
            selector_model_dir,
            args.run_label,
        ):
            # 同时保留端到端提交等待和 Producer 的背压阻塞时长。
            task["queue_entered_at"] = time.perf_counter()
            task["enqueue_blocked_ms"] = 0.0
            ready = asyncio.Event()
            put_started = time.perf_counter()
            await queue.put((task, ready))
            enqueue_blocked_ms = (time.perf_counter() - put_started) * 1000
            producer_blocked_ms += enqueue_blocked_ms
            task["enqueue_blocked_ms"] = round(enqueue_blocked_ms, 2)
            ready.set()
            submitted += 1
            queue_samples.append(queue.qsize())
            queue_peak = max(queue_peak, queue.qsize())
    enqueue_ms = (time.perf_counter() - enqueue_started) * 1000
    # FIFO 队列：哨兵位于所有真实任务之后，消费者处理完任务后自然退出
    for _ in consumers:
        await queue.put(None)
    await asyncio.gather(*consumers)
    rss_stop.set()
    await rss_sampler
    await client.close()
    print(json.dumps({
        "submitted": submitted,
        "output": str(args.output),
        "queue_size": queue_size,
        "enqueue_ms": round(enqueue_ms, 2),
        "queue_peak": queue_peak,
        "queue_average": round(sum(queue_samples) / len(queue_samples), 4)
        if queue_samples else 0.0,
        "producer_blocked_ms": round(producer_blocked_ms, 2),
        "consumer_idle_ms": round(consumer_idle_ms, 2),
        "rss_peak_bytes": max(rss_samples) if rss_samples else None,
        "rss_average_bytes": round(sum(rss_samples) / len(rss_samples), 2)
        if rss_samples else None,
        "rss_samples": len(rss_samples),
        "pipeline_ms": round((time.perf_counter() - pipeline_started) * 1000, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main_async())
