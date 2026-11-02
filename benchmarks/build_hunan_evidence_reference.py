"""构建湖南博物馆独立证据参考（agent-adjudicated, evidence-grounded）。

设计约束：
- 只依据 snapshot.jsonl 的 raw_data 中**明确存在**的信息生成参考；
- 湖南快照 raw_data 仅含 文物名称/类型/年代/尺寸 等少量键，故多数 18 字段无依据；
- 无依据字段一律写 null（不省略键、不编造），符合 agent 金标注协议的轻量参考格式。

可推导字段（保守、可审计）：
- name     <- raw_data["文物名称"]
- category <- raw_data["类型"]
- dynasty  <- 从 raw_data["年代"] 中可识别的朝代 token
- period   <- raw_data["年代"] 原文
其余 14 字段恒为 null。

注意：该参考不含任何 LLM 调用，也不使用被评估模型 kg-assist-pro 的输出，
因此与 heldout_full_silver_reference.jsonl（由被测模型 full 输出派生）相互独立。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "name",
    "category",
    "form_type",
    "usage",
    "desc",
    "condition",
    "dynasty",
    "period",
    "acq_type",
    "region",
    "site",
    "culture",
    "material",
    "motif",
    "craft",
    "craft_school",
    "related_event",
    "related_person",
]

# 朝代识别 token，按从具体到宽泛排列；民国优先于其余。
DYNASTY_TOKENS = [
    "民国",
    "清",
    "明",
    "元",
    "宋",
    "唐",
    "隋",
    "晋",
    "汉",
    "秦",
    "周",
    "辽",
    "金",
    "西夏",
    "五代",
]


def dynasty_of(era: str | None) -> str | None:
    """从年代文本中抽取可识别的朝代；无法识别时返回 None。"""
    if not era:
        return None
    for token in DYNASTY_TOKENS:
        if token in era:
            return token
    return None


def raw_value(raw: dict, key: str) -> str | None:
    """读取 raw_data 字段并做去空白；空字符串视为缺失。"""
    value = raw.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_result(raw: dict) -> dict:
    """仅用原文可推导信息构造 18 字段参考。"""
    era = raw_value(raw, "年代")
    result = {field: None for field in FIELDS}
    result["name"] = raw_value(raw, "文物名称")
    result["category"] = raw_value(raw, "类型")
    result["dynasty"] = dynasty_of(era)
    result["period"] = era
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建湖南独立证据参考")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=ROOT / "data/hunan_museum_300000001_300000500_without_url",
        help="包含 snapshot.jsonl 与 test_ids.json 的目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL；默认 <cohort-dir>/agent_evidence_reference.jsonl",
    )
    parser.add_argument("--schema-version", default="simple2-v1")
    parser.add_argument("--reference-model", default="agent-evidence-reference-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_dir = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    output = args.output or (cohort_dir / "agent_evidence_reference.jsonl")
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有参考: {output}")

    records = [
        json.loads(line)
        for line in (cohort_dir / "snapshot.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    references = []
    for record in records:
        raw = record.get("raw_data") or {}
        references.append(
            {
                "artifact_id": int(record["artifact_id"]),
                "schema_version": args.schema_version,
                "reference_model": args.reference_model,
                "reference_result": build_result(raw),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for reference in references:
            handle.write(json.dumps(reference, ensure_ascii=False) + "\n")

    non_null = {field: 0 for field in FIELDS}
    for reference in references:
        for field in FIELDS:
            if reference["reference_result"].get(field) not in (None, ""):
                non_null[field] += 1
    audit = {
        "records": len(references),
        "output": str(output),
        "reference_model": args.reference_model,
        "per_field_non_null": non_null,
        "average_non_null_fields": round(
            sum(non_null.values()) / len(references), 4
        ),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
