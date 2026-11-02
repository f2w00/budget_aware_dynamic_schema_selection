#!/usr/bin/env python3
"""导出 Agent 标注所需的 raw 数据和 Schema，不调用任何 LLM。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg

from llm_runner import load_dotenv


def project_raw_data(raw_data: Any) -> Any:
    """删除导出副本中的重复顶层 ID，不修改数据库原始数据。"""
    if not isinstance(raw_data, dict):
        return raw_data
    return {key: value for key, value in raw_data.items() if key != "artifact_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 Agent Gold 标注任务")
    parser.add_argument("--ids-file", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--output", type=Path, default=Path("data/gold/agent_tasks/gold_main_raw.jsonl"))
    parser.add_argument("--schema-output", type=Path, default=Path("data/gold/agent_tasks/simple2-v1.json"))
    parser.add_argument("--metadata-output", type=Path, default=Path("data/gold/agent_tasks/task_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")

    ids = [int(item) for item in json.loads(args.ids_file.read_text(encoding="utf-8"))]
    if len(ids) != len(set(ids)):
        raise SystemExit("ID 清单包含重复 artifact_id")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT artifact_id, raw_data FROM public.artifact_raw "
            "WHERE artifact_id = ANY(%s)",
            (ids,),
        )
        records: dict[int, Any] = {int(row[0]): row[1] for row in cursor.fetchall()}
    missing = sorted(set(ids) - set(records))
    if missing:
        raise SystemExit(f"artifact_raw 缺少 {len(missing)} 条记录")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.schema_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.schema_output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output.open("w", encoding="utf-8") as output:
        for artifact_id in ids:
            output.write(json.dumps(
                {
                    "artifact_id": artifact_id,
                    "raw_data": project_raw_data(records[artifact_id]),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n")
    metadata = {
        "source_ids_file": str(args.ids_file),
        "source_table": "public.artifact_raw",
        "schema_file": str(args.schema),
        "schema_version": "simple2-v1",
        "task_count": len(ids),
        "raw_copied": True,
        "removed_raw_keys": ["artifact_id"],
        "llm_called": False,
    }
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
