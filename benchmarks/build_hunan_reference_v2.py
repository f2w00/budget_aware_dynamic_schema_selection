"""按 gold_reference.prompt 的标注规则构建湖南 without_url 证据参考 v2（无 LLM 调用）。

规则来源（`gold_reference.prompt`）：
1. 只依据原始记录信息标注，禁止外部知识补全（尤其 related_person/related_event/
   craft_school/site）。
2. 无可靠依据输出 JSON null。
3. 输出且仅输出全部 18 个字段键。
4. 值只能是字符串/数字/布尔/数组/null。
5. desc 用简洁中文概括原始记录中与该文物直接相关的信息。
6. 只输出 JSON。

本脚本为“机械可复现”实现：name/category/dynasty/period/desc/form_type 由规则生成；
高风险字段 related_person/site/region/related_event/culture 由人工裁决表 MANUAL_EXTRAS
显式给出（仅当字面出现在原始字段），其余字段一律 null。人工裁决表当前覆盖 pilot 30 条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_MODEL = "agent-evidence-reference-v2"
RULE_PROMPT = ROOT / "gold_reference.prompt"

# 年代 -> 朝代（严格只在原文出现对应朝代字时给出）
DYNASTY_TOKENS = ("清", "明", "宋", "唐", "汉", "民国")

# form_type 关键词优先级（先匹配者胜）
FORM_TYPE_KEYWORDS = (
    ("摩崖", "摩崖"),
    ("画稿", "画稿"),
    ("墓志", "墓志"),
    ("碑", "碑刻"),
    ("刻石", "刻石"),
    ("法帖", "法帖"),
    ("帖", "法帖"),
    ("联", "对联"),
    ("立轴", "立轴"),
    ("横披", "横披"),
    ("条幅", "条幅"),
    ("阙", "阙"),
    ("信", "书信"),
    ("隶书", "书法"),
    ("篆书", "书法"),
    ("楷书", "书法"),
    ("行书", "书法"),
    ("草书", "书法"),
    ("书", "书法"),
)

# form_type 特例：这些文物名称虽含关键词但不等于该形制
FORM_TYPE_OVERRIDE = {
    "300000001": None,  # 女书三朝书，非书法
    "300000012": None,  # 千字文拓本，形制以拓本为主，规则不覆盖
}

# 人工裁决的高风险字段（artifact_id -> {field: value}），仅限原文可依据者
MANUAL_EXTRAS: dict[str, dict[str, object]] = {
    "300000001": {"culture": "女书"},
    "300000002": {"related_person": ["易顺鼎"]},
    "300000003": {"related_person": ["易顺鼎"]},
    "300000005": {
        "related_person": ["史晨", "孔子"],
        "site": "孔庙",
    },
    "300000007": {"related_person": ["杨钧", "尹处士"]},
    "300000008": {"related_person": ["谭嗣同"]},
    "300000009": {"related_person": ["卞鉴"]},
    "300000010": {"site": "等慈寺"},
    "300000011": {"related_person": ["杨君"], "site": "石门", "region": "楗为"},
    "300000012": {"related_person": ["褚遂良"]},
    "300000013": {"site": "多宝塔"},
    "300000015": {"site": "国山"},
    "300000017": {"related_person": ["曹真"]},
    "300000018": {"related_person": ["何绍基"]},
    "300000019": {"site": "嵩山"},
    "300000022": {"related_person": ["孔羡"]},
    "300000024": {"related_person": ["张玄"], "region": "南阳"},
    "300000025": {
        "related_person": ["陈公"],
        "site": "湖南电灯公司",
        "region": "湖南",
    },
    "300000026": {"related_person": ["曾熙"], "site": "船山书院"},
    "300000027": {"related_person": ["张猛龙"], "region": "鲁郡"},
    "300000028": {"site": "星凤楼"},
    "300000029": {"site": "峄山"},
    "300000030": {"site": "岳麓书院"},
    "300000031": {"related_person": ["道因法师"]},
    "300000032": {"related_person": ["郭嵩焘"]},
    "300000034": {"related_person": ["王代功"]},
    "300000036": {"related_person": ["夏寿田"]},
    "300000037": {"related_person": ["杨钧"]},
    "300000038": {"related_person": ["谭秉炎"]},
}


def dynasty_of(era: str) -> str | None:
    if not era:
        return None
    for token in DYNASTY_TOKENS:
        if token in era:
            return token
    return None


def form_type_of(artifact_id: str, name: str) -> str | None:
    if artifact_id in FORM_TYPE_OVERRIDE:
        return FORM_TYPE_OVERRIDE[artifact_id]
    for keyword, value in FORM_TYPE_KEYWORDS:
        if keyword in name:
            return value
    return None


def build_desc(era: str, category: str, name: str, size: str) -> str:
    parts = []
    if era:
        parts.append(era)
    if category:
        parts.append(category)
    head = "".join(parts) + "文物"
    tail = f"，名称为{name}" if name else ""
    if size:
        tail += f"，尺寸{size}"
    return head + tail + "。"


def build_reference(
    schema_order: list[str], artifact_id: str, raw: dict, use_manual: bool = True
) -> dict:
    name = str(raw.get("文物名称") or "").strip()
    era = str(raw.get("年代") or "").strip()
    category = str(raw.get("类型") or "").strip()
    size = str(raw.get("尺寸") or "").strip()

    result = {field: None for field in schema_order}
    result["name"] = name or None
    result["category"] = category or None
    result["dynasty"] = dynasty_of(era)
    result["period"] = era or None
    result["desc"] = build_desc(era, category, name, size)
    result["form_type"] = form_type_of(artifact_id, name)
    for field, value in MANUAL_EXTRAS.get(artifact_id, {}).items():
        result[field] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=ROOT / "data/hunan_museum_300000001_300000500_without_url",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=30)
    parser.add_argument(
        "--all-ids",
        action="store_true",
        help="使用全部 held-out id（否则只取前 --pilot 条）",
    )
    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="不使用 MANUAL_EXTRAS 人工裁决表（纯自动规则）",
    )
    args = parser.parse_args()
    cohort = args.cohort_dir
    output = args.output

    if output.exists():
        raise SystemExit(f"拒绝覆盖已有产物: {output}")

    schema_order = list(
        json.loads((ROOT / "simple2.json").read_text(encoding="utf-8")).keys()
    )
    prompt_hash = hashlib.sha256(RULE_PROMPT.read_bytes()).hexdigest()

    raw_by_id: dict[str, dict] = {}
    for line in (cohort / "snapshot.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            raw_by_id[str(row["artifact_id"])] = row["raw_data"]

    test_ids = [
        str(value)
        for value in json.loads((cohort / "test_ids.json").read_text(encoding="utf-8"))
    ]
    selected = test_ids if args.all_ids else test_ids[: args.pilot]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for artifact_id in selected:
            result = build_reference(
                schema_order,
                artifact_id,
                raw_by_id[str(int(artifact_id))],
                use_manual=not args.no_manual,
            )
            missing = [f for f in schema_order if f not in result]
            extra = [f for f in result if f not in schema_order]
            if missing or extra:
                raise SystemExit(
                    f"{artifact_id} 键不一致 missing={missing} extra={extra}"
                )
            handle.write(
                json.dumps(
                    {
                        "artifact_id": int(artifact_id),
                        "schema_version": "simple2-v1",
                        "reference_model": REFERENCE_MODEL,
                        "prompt_hash": prompt_hash,
                        "reference_result": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "records": len(selected),
                "output": str(output),
                "reference_model": REFERENCE_MODEL,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
