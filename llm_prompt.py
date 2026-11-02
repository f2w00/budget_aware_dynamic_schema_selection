#!/usr/bin/env python3
"""构造 Full、Fixed 和 Adaptive 共用的 LLM 抽取 Prompt。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FOCUS_FIELDS = {
    "dynasty",
    "category",
    "material",
    "form_type",
    "usage",
    "craft_school",
}

STABLE_FOCUS_DESCRIPTIONS = {
    "dynasty": "朝代或政权",
    "category": "文物类别",
    "material": "主要材质",
    "form_type": "器型或形制",
    "usage": "用途或功能",
    "craft_school": "流派、窑口或制作体系",
}


def build_prompt(
    template: str,
    schema: dict[str, str],
    selected_fields: list[str],
    raw_data: Any,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """按模板构造 Prompt，并支持普通字段名与重点字段说明分离的版本。"""
    target_structure = {
        field: schema[field]
        for field in selected_fields
    }
    focus_schema = {
        field: schema[field]
        for field in selected_fields
        if field in FOCUS_FIELDS
    }
    target_json = json.dumps(target_structure, ensure_ascii=False, indent=2)
    field_names_json = json.dumps(selected_fields, ensure_ascii=False, indent=2)
    focus_json = json.dumps(focus_schema, ensure_ascii=False, indent=2)
    raw_json = json.dumps(raw_data, ensure_ascii=False, separators=(",", ":"), default=str)
    if "%FIELD_NAMES_ONLY%" in template:
        # 稳定版只给普通字段名称，避免长字段定义干扰模型定位事实值。
        template = template.replace("%FIELD_NAMES_ONLY%", "")
        stable_focus = {
            field: STABLE_FOCUS_DESCRIPTIONS[field]
            for field in selected_fields
            if field in STABLE_FOCUS_DESCRIPTIONS
        }
        focus_schema = stable_focus
        prompt = template % (
            field_names_json,
            json.dumps(stable_focus, ensure_ascii=False, indent=2),
            raw_json,
        )
    elif template.count("%s") == 2:
        # 无重点字段版不再向模型暴露额外的优先级字段集合。
        focus_schema = {}
        prompt = template % (target_json, raw_json)
    else:
        prompt = template % (
            target_json,
            focus_json,
            raw_json,
        )
    return prompt, target_structure, focus_schema


def load_prompt_assets(
    template_path: Path,
    schema_path: Path,
) -> tuple[str, dict[str, str]]:
    return (
        template_path.read_text(encoding="utf-8"),
        json.loads(schema_path.read_text(encoding="utf-8")),
    )
