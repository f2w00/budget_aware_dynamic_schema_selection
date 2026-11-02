#!/usr/bin/env python3
"""离线修复 prompt-omit 中键后缺失值导致的可恢复 JSON 解析失败。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_runner import normalize_result, normalize_result_aliases, parse_json_response, validate_result_types


INPUT = ROOT / "experiments/prompt_omit_500_formal/formal/prompt-omit.jsonl"
OUTPUT = ROOT / "experiments/prompt_omit_500_formal/formal-repaired/prompt-omit.jsonl"


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repair_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") == "success":
        return row
    if not str(row.get("error", "")).startswith("JSONDecodeError:"):
        raise ValueError(f"不可修复的非 JSON 失败: {row.get('artifact_id')} {row.get('error')}")
    result, repaired = parse_json_response(
        row.get("raw_response") or "", allow_missing_values=True
    )
    if not repaired:
        raise ValueError(f"失败响应不符合缺失值修复模式: {row.get('artifact_id')}")
    selected_fields = row.get("selected_fields") or []
    result, aliases = normalize_result_aliases(result, selected_fields)
    validate_result_types(result, selected_fields)
    unexpected = sorted(set(result) - set(selected_fields))
    if unexpected:
        raise ValueError(f"修复响应含额外字段: {row.get('artifact_id')} {unexpected}")
    missing = sorted(set(selected_fields) - set(result))
    repaired_row = dict(row)
    repaired_row.update({
        "result": normalize_result(result, selected_fields),
        "status": "success",
        "unexpected_fields": unexpected,
        "missing_fields": missing,
        "normalized_field_aliases": aliases,
        "repaired_missing_json_values": True,
        "error": None,
        "repair_source_error": row.get("error"),
        "repair_mode": "offline_missing_json_value_to_null",
    })
    return repaired_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖已有修复结果: {args.output}")

    rows = read_rows(args.input)
    repaired_rows = [repair_row(row) for row in rows]
    if len(repaired_rows) != 500 or len({row["artifact_id"] for row in repaired_rows}) != 500:
        raise SystemExit("修复结果不包含 500 条唯一记录")
    if any(row.get("status") != "success" for row in repaired_rows):
        raise SystemExit("修复后仍存在失败记录")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "records": len(repaired_rows),
        "repaired": sum(bool(row.get("repair_mode")) for row in repaired_rows),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
