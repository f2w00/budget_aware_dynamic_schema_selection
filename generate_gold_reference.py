#!/usr/bin/env python3
"""用强模型生成 model-assisted Gold reference，并做结构审核与轻量落盘。

最终 gold_main.jsonl 每行只保留 artifact_id、schema_version、reference_model、
reference_result；prompt hash、Token 和原始返回等审计信息保存在 drafts 目录。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_runner import (
    file_hash,
    load_dotenv,
    load_records,
    parse_json_response,
    text_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成强模型 Gold reference")
    parser.add_argument("--ids-file", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--limit", type=int, default=30, help="试标数量，0 表示全部")
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--prompt", type=Path, default=Path("gold_reference.prompt"))
    parser.add_argument("--output", type=Path, default=Path("data/gold/gold_main.jsonl"))
    parser.add_argument("--drafts-dir", type=Path, default=Path("data/gold/drafts"))
    parser.add_argument("--stats", type=Path, default=Path("reports/gold_reference_stats.json"))
    parser.add_argument("--model", default=None, help="默认读取 LLM_MODEL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--schema-version", default="simple2-v1")
    return parser.parse_args()


def build_gold_prompt(template_text: str, schema: dict[str, str], raw_data: Any) -> str:
    # 占位符用替换而非 format，避免 raw JSON 中的花括号干扰格式化
    return (
        template_text
        .replace("__SCHEMA_JSON__", json.dumps(schema, ensure_ascii=False, indent=2))
        .replace("__RAW_JSON__", json.dumps(raw_data, ensure_ascii=False, indent=2))
    )


def validate_reference(result: dict[str, Any], schema: dict[str, str]) -> list[str]:
    """结构审核：18 键完整、无额外键、值类型合法。"""
    errors = []
    missing = [field for field in schema if field not in result]
    unexpected = sorted(set(result) - set(schema))
    if missing:
        errors.append(f"缺少字段: {missing}")
    if unexpected:
        errors.append(f"多余字段: {unexpected}")
    scalars = (str, int, float, bool)
    for field, value in result.items():
        if field not in schema:
            continue
        if value is None or isinstance(value, scalars):
            if isinstance(value, str) and value.strip().lower() in {
                "null", "none", "n/a", "未知", "未详", "无"
            }:
                errors.append(f"字段必须使用 JSON null 而非空值字符串: {field}")
            continue
        if isinstance(value, list) and all(
            item is None or isinstance(item, scalars) for item in value
        ):
            continue
        errors.append(f"字段值类型无效: {field}")
    return errors


def normalize_null_sentinels(value: Any) -> Any:
    """将模型常见的空值字符串统一为 JSON null。"""
    null_strings = {"null", "none", "n/a", "未知", "未详", "无"}
    if isinstance(value, str) and value.strip().lower() in null_strings:
        return None
    if isinstance(value, dict):
        return {key: normalize_null_sentinels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_null_sentinels(item) for item in value]
    return value


def generate_reference(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    schema: dict[str, str],
) -> dict[str, Any]:
    """调用强模型并重试，直到结构审核通过或次数用尽。"""
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            result = normalize_null_sentinels(parse_json_response(content))
            errors = validate_reference(result, schema)
            usage = response.usage
            row = {
                "status": "success" if not errors else "invalid",
                "attempt": attempt,
                "reference_result": result,
                "raw_response": content,
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "error": "; ".join(errors) if errors else None,
            }
            if not errors:
                return row
        except Exception as exception:  # 网络或解析失败同样进入退避重试
            errors = [f"{type(exception).__name__}: {exception}"]
            row = {
                "status": "error",
                "attempt": attempt,
                "reference_result": None,
                "raw_response": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": errors[0],
            }
        if attempt < max_retries:
            time.sleep(min(30.0, (2**attempt) + random.random()))
    return row


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    base_url = os.getenv("LLM_BASE_URL", "http://172.18.20.44:30059/v1")
    api_key = os.getenv("LLM_API_KEY")
    model = args.model or os.getenv("LLM_MODEL", "kg-assist")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    if not api_key:
        raise SystemExit("缺少 LLM_API_KEY；请通过环境变量或 .env 提供")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    template_text = args.prompt.read_text(encoding="utf-8")
    prompt_hash = file_hash(args.prompt)
    schema_hash = text_hash(json.dumps(schema, ensure_ascii=False, sort_keys=True))

    ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
    selected_ids = ids if args.limit <= 0 else ids[: args.limit]
    records = load_records(database_url, selected_ids, len(selected_ids))

    args.drafts_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    gold_rows: dict[int, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                gold_rows[int(row["artifact_id"])] = row

    total_input = 0
    total_output = 0
    failed: list[dict[str, Any]] = []
    for artifact_id, raw_data in records:
        prompt = build_gold_prompt(template_text, schema, raw_data)
        draft_path = args.drafts_dir / f"{artifact_id}.json"
        draft: dict[str, Any] | None = None
        if draft_path.exists():
            previous = json.loads(draft_path.read_text(encoding="utf-8"))
            # 断点复用：同 Prompt、同 Schema、同模型的成功草稿不再重复调用
            if (
                previous.get("status") == "success"
                and previous.get("prompt_hash") == prompt_hash
                and previous.get("schema_hash") == schema_hash
                and previous.get("reference_model") == model
            ):
                draft = previous
        if draft is None:
            started = time.perf_counter()
            outcome = generate_reference(
                client, prompt, model, args.temperature, timeout, args.max_retries, schema
            )
            draft = {
                "artifact_id": artifact_id,
                "status": outcome["status"],
                "reference_model": model,
                "temperature": args.temperature,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "schema_version": args.schema_version,
                "reference_result": outcome["reference_result"],
                "raw_response": outcome["raw_response"],
                "input_tokens": outcome["input_tokens"],
                "output_tokens": outcome["output_tokens"],
                "attempt": outcome["attempt"],
                "error": outcome["error"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            draft_path.write_text(
                json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        total_input += draft.get("input_tokens") or 0
        total_output += draft.get("output_tokens") or 0
        if draft["status"] == "success":
            gold_rows[artifact_id] = {
                "artifact_id": artifact_id,
                "schema_version": args.schema_version,
                "reference_model": model,
                "reference_result": draft["reference_result"],
            }
        else:
            failed.append({"artifact_id": artifact_id, "status": draft["status"], "error": draft["error"]})

    with args.output.open("w", encoding="utf-8") as output:
        for artifact_id in sorted(gold_rows):
            output.write(
                json.dumps(gold_rows[artifact_id], ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    non_null = {field: 0 for field in schema}
    for row in gold_rows.values():
        for field, value in row["reference_result"].items():
            if value is not None and value != []:
                non_null[field] += 1
    stats = {
        "requested": len(selected_ids),
        "success_in_output": len(gold_rows),
        "failed_this_run": failed,
        "reference_model": model,
        "temperature": args.temperature,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "field_non_null_counts": non_null,
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: stats[key] for key in (
        "requested", "success_in_output", "reference_model", "prompt_hash",
        "total_input_tokens", "total_output_tokens",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
