#!/usr/bin/env python3
"""从 PostgreSQL 的原始记录和完整抽取结果生成训练数据集。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg
import tiktoken


FIELDS = [
    "acq_type",
    "category",
    "condition",
    "craft",
    "craft_school",
    "culture",
    "desc",
    "dynasty",
    "form_type",
    "material",
    "motif",
    "name",
    "period",
    "region",
    "related_event",
    "related_person",
    "site",
    "usage",
]


def load_dotenv(path: Path) -> None:
    """加载简单的 KEY=VALUE 配置，不覆盖已有环境变量。"""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def is_present(value: Any) -> int:
    """判断抽取字段是否有可用的非空值。"""
    if value is None:
        return 0
    if isinstance(value, str):
        return int(bool(value.strip()))
    if isinstance(value, (list, tuple, dict, set)):
        return int(bool(value))
    return 1


def serialize_raw_data(raw_data: Any) -> str:
    """用稳定、紧凑的 JSON 表示原始记录。"""
    return json.dumps(
        raw_data,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_sample(
    artifact_id: int,
    raw_data: Any,
    extracted: dict[str, Any],
    encoder: Any,
) -> dict[str, Any]:
    record_text = serialize_raw_data(raw_data)
    labels = {}
    field_output_tokens = {}

    for field in FIELDS:
        value = extracted.get(field)
        present = is_present(value)
        labels[f"{field}_present"] = present
        field_output_tokens[field] = (
            len(encoder.encode(value_to_text(value))) if present else 0
        )

    return {
        "record_id": artifact_id,
        "record_text": record_text,
        "labels": labels,
        "field_output_tokens": field_output_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 artifact_raw 和 artifact_extract 生成 JSONL 训练数据集"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_dataset.jsonl"),
        help="训练数据集输出路径",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("simple.json"),
        help="目标 Schema 文件路径",
    )
    parser.add_argument(
        "--cost-output",
        type=Path,
        default=Path("schema_costs.json"),
        help="Schema 字段 Token 成本输出路径",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="tiktoken 编码名称，默认 cl100k_base",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="数据库游标批量读取大小",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少条匹配记录，默认处理全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL，请设置环境变量或 .env 文件")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    missing_fields = [field for field in FIELDS if field not in schema]
    if missing_fields:
        raise SystemExit(f"Schema 缺少字段: {', '.join(missing_fields)}")

    encoder = tiktoken.get_encoding(args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cost_output.parent.mkdir(parents=True, exist_ok=True)

    selected_columns = ", ".join(f'e."{field}"' for field in FIELDS)
    limit_clause = "LIMIT %s" if args.limit is not None else ""
    query = f"""
        SELECT r.artifact_id, r.raw_data, {selected_columns}
        FROM public.artifact_raw AS r
        INNER JOIN public.artifact_extract AS e
            ON r.artifact_id = e.artifact_id
        ORDER BY r.artifact_id
        {limit_clause}
    """

    stats = {"matched": 0, "written": 0, "errors": 0}
    with (
        psycopg.connect(database_url) as connection,
        connection.cursor(name="training_dataset_cursor") as cursor,
        args.output.open("w", encoding="utf-8") as output,
    ):
        if args.limit is None:
            cursor.execute(query)
        else:
            cursor.execute(query, (args.limit,))

        while True:
            rows = cursor.fetchmany(args.batch_size)
            if not rows:
                break
            for row in rows:
                stats["matched"] += 1
                artifact_id, raw_data, *field_values = row
                try:
                    extracted = dict(zip(FIELDS, field_values, strict=True))
                    sample = build_sample(
                        artifact_id,
                        raw_data,
                        extracted,
                        encoder,
                    )
                    output.write(
                        json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    stats["written"] += 1
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    stats["errors"] += 1
                    print(f"跳过 artifact_id={artifact_id}: {error}")

    field_costs = {
        field: {
            "schema_value_tokens": len(encoder.encode(str(schema[field]))),
            "field_name_tokens": len(encoder.encode(field)),
        }
        for field in FIELDS
    }
    args.cost_output.write_text(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "fields": field_costs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False))
    print(f"训练数据集: {args.output}")
    print(f"Schema 成本: {args.cost_output}")


if __name__ == "__main__":
    main()
