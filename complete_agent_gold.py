#!/usr/bin/env python3
"""将 Agent 的稀疏字段输出补全为正式 Gold JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补全 Agent Gold 标注结果")
    parser.add_argument("--input", type=Path, help="单批 Agent 稀疏结果 JSONL")
    parser.add_argument("--source", type=Path, help="单批带 artifact_id 的 Agent 输入 raw JSONL")
    parser.add_argument("--batch-dir", type=Path, help="包含 batch_*_raw/gold.jsonl 的目录")
    parser.add_argument(
        "--repair-json",
        action="store_true",
        help="修复字段字符串中未转义的引号后再解析",
    )
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument("--schema-version", default="simple2-v1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return
    raise ValueError(f"字段 {field} 的值类型不合法")


def load_source(path: Path) -> list[int]:
    ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or isinstance(row.get("artifact_id"), bool):
            raise ValueError(f"source 第 {line_number} 行缺少合法的 artifact_id")
        if "raw_data" not in row:
            raise ValueError(f"source 第 {line_number} 行缺少 raw_data")
        ids.append(int(row["artifact_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("source 包含重复 artifact_id")
    return ids


def repair_unescaped_quotes(line: str) -> str:
    """仅修复 JSON 字符串内部、且后面不是结构分隔符的裸引号。"""
    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            repaired.append(char)
            escaped = True
            continue
        if char != '"':
            repaired.append(char)
            continue
        if not in_string:
            in_string = True
            repaired.append(char)
            continue
        next_char = next((item for item in line[index + 1:] if not item.isspace()), "")
        if next_char in {",", "}", "]", ":"}:
            in_string = False
            repaired.append(char)
        else:
            repaired.append('\\"')
    return "".join(repaired)


def load_agent_results(
    path: Path,
    expected_ids: set[int],
    fields: set[str],
    repair_json: bool = False,
) -> dict[int, dict[str, Any]]:
    """读取一批结果，按 ID 校验并建立映射，不依赖输出顺序。"""
    results: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not repair_json:
                    raise
                row = json.loads(repair_unescaped_quotes(line))
            if not isinstance(row, dict):
                raise ValueError("每行必须是 JSON 对象")
            artifact_id = row.get("artifact_id")
            if artifact_id is None or isinstance(artifact_id, bool):
                raise ValueError("缺少合法的 artifact_id")
            artifact_id = int(artifact_id)
            if artifact_id not in expected_ids:
                raise ValueError(f"未知 artifact_id: {artifact_id}")
            if artifact_id in results:
                raise ValueError(f"重复 artifact_id: {artifact_id}")
            result = {key: value for key, value in row.items() if key != "artifact_id"}
            unexpected = set(result) - fields
            if unexpected:
                raise ValueError(f"存在额外字段: {sorted(unexpected)}")
            for field, value in result.items():
                validate_value(value, field)
            results[artifact_id] = result
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise SystemExit(f"{path.name} 第 {line_number} 行校验失败: {error}") from error
    missing = expected_ids - set(results)
    if missing:
        raise SystemExit(
            f"{path.name} 缺少 {len(missing)} 个 artifact_id: {sorted(missing)[:10]}"
        )
    return results


def load_batch_pairs(batch_dir: Path) -> list[tuple[Path, Path]]:
    raw_files = sorted(batch_dir.glob("batch_*_raw.jsonl"))
    if not raw_files:
        raise SystemExit(f"未找到批次 raw 文件: {batch_dir}")
    pairs = []
    for source in raw_files:
        gold = source.with_name(source.name.replace("_raw.jsonl", "_gold.jsonl"))
        if not gold.exists():
            raise SystemExit(f"缺少对应的 Agent Gold 文件: {gold.name}")
        pairs.append((source, gold))
    return pairs


def main() -> None:
    args = parse_args()
    if bool(args.batch_dir) == bool(args.input or args.source):
        raise SystemExit("请提供 --batch-dir，或同时提供 --input 和 --source")
    if (args.input is None) != (args.source is None):
        raise SystemExit("--input 和 --source 必须同时提供")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    fields = list(schema)
    expected = set(fields)
    if args.batch_dir:
        pairs = load_batch_pairs(args.batch_dir)
    else:
        pairs = [(args.source, args.input)]
    artifact_ids: list[int] = []
    results: dict[int, dict[str, Any]] = {}
    for source, agent_output in pairs:
        batch_ids = load_source(source)
        overlap = set(batch_ids) & set(artifact_ids)
        if overlap:
            raise SystemExit(f"批次之间存在重复 artifact_id: {sorted(overlap)[:10]}")
        batch_results = load_agent_results(
            agent_output, set(batch_ids), expected, args.repair_json
        )
        artifact_ids.extend(batch_ids)
        results.update(batch_results)
    completed = [
        {
            "artifact_id": artifact_id,
            "schema_version": args.schema_version,
            "reference_result": {
                field: results[artifact_id].get(field) for field in fields
            },
        }
        for artifact_id in artifact_ids
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in completed:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"completed": len(completed), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
