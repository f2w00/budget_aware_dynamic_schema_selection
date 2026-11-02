#!/usr/bin/env python3
"""从原始响应中恢复仅因已知字段别名而失败的 LLM 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_runner import (
    normalize_result,
    normalize_result_aliases,
    parse_json_response,
    validate_result_types,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="恢复可无损归一化的字段别名失败记录")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def recover(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") == "success" or not row.get("raw_response"):
        return None
    selected_fields = row.get("selected_fields")
    if not isinstance(selected_fields, list):
        return None
    result = parse_json_response(row["raw_response"])
    result, aliases = normalize_result_aliases(result, selected_fields)
    if not aliases or set(result) - set(selected_fields):
        return None
    validate_result_types(result, selected_fields)
    recovered = dict(row)
    recovered.update(
        {
            "result": normalize_result(result, selected_fields),
            "status": "success",
            "unexpected_fields": [],
            "missing_fields": sorted(set(selected_fields) - set(result)),
            "normalized_field_aliases": aliases,
            "error": None,
            "recovered_from_raw_response": True,
        }
    )
    if recovered.get("attempts"):
        recovered["attempts"][-1] = {
            **recovered["attempts"][-1],
            "status": "success",
            "error": None,
        }
    return recovered


def main() -> None:
    args = parse_args()
    recovered_rows = []
    for path in args.input:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            recovered = recover(json.loads(line))
            if recovered is not None:
                recovered_rows.append(recovered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in recovered_rows
        ),
        encoding="utf-8",
    )
    print(json.dumps({"recovered": len(recovered_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
