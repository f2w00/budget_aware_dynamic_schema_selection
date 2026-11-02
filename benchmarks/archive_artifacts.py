#!/usr/bin/env python3
"""为随稿复现材料生成版本化归档清单（D1）。

目标：回应审稿问题 32（提供稳定文件与版本标识）。
做法：
- 对主实验 main-comparison-pro 的逐次运行日志（raw/*.jsonl）、meta、summaries、
  validation，以及聚合产物、配置/数据输入、Gold/标注指南、各 seed 离线质量结果、
  分析脚本，逐文件计算 sha256。
- 生成 benchmarks/archive/SHA256SUMS（纯离线，不触发 LLM）。
- 生成 benchmarks/archive/manifest.json：git 提交、逐 run 的 workload_ids_sha256、
  输入文件哈希、环境信息。
- 生成 benchmarks/archive/README.md：结构说明 + 复现命令。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "benchmarks" / "runs" / "main-comparison-pro"
ANALYSIS_DIR = ROOT / "benchmarks" / "analysis" / "performance"
SWEEP_RUN_DIR = ROOT / "benchmarks" / "runs" / "a6-a3-sweep"
SWEEP_ANALYSIS_DIR = ROOT / "benchmarks" / "analysis" / "a6_a3"
SWEEP_EVIDENCE = ROOT / "tex" / "v6-lncs" / "补充实验2.md"
A4_ANALYSIS_DIR = ROOT / "benchmarks" / "analysis" / "a4_selection_ablation"
A5_ANALYSIS_DIR = ROOT / "benchmarks" / "analysis" / "a5_weight_sensitivity"
A5_EVIDENCE = ROOT / "tex" / "v6-lncs" / "补充实验3.md"
ARCHIVE_DIR = ROOT / "benchmarks" / "archive"
BENCH_DIR = ROOT / "benchmarks"

# 归档的输入/配置/数据文件（相对仓库根）。
INPUT_FILES = [
    "data/performance/performance_ids_2000.json",
    "data/performance/performance_ids_2000.manifest.json",
    "schema_costs.json",
    "field_weights.json",
    "1.prompt",
    "1.json",
]

# Gold 与标注指南（问题 1/2/3/29：独立事实标签的来源与判定依据）。
GOLD_DIR = ROOT / "data" / "gold"

# 各 seed 的离线质量结果（问题 32：逐字段、逐 seed 稳定文件）。
REPORTS_DIR = ROOT / "reports"

# 其余离线分析聚合产物（B4/B5/C1/C2/C3 脚本输出目录）。
EXTRA_ANALYSIS_DIRS = (
    "breakeven",
    "c1_near_duplicate",
    "c2_representativeness",
    "c3_scan",
    "initialization",
)

# 分析/绘图/归档脚本本身（随稿可运行、可复现）。
SCRIPTS_GLOB = "benchmarks/*.py"

# 汇总各 seed 离线结果的目录（仅收录结果 JSON/CSV，不含任何模型二进制）。
REPORTS_SUMMARY_DIRS = (
    "training_size_ablation",
    "model_strategy_ablation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"
    return out


def collect_run_artifacts() -> list[Path]:
    """收集主实验每次运行的全部待校验文件。"""
    files: list[Path] = []
    # 逐次运行：raw 日志 + meta
    for meta in sorted(RUN_DIR.glob("*.meta.json")):
        files.append(meta)
        base = meta.stem[: -len(".meta")]  # 去掉 .meta 后缀
        raw_dir = RUN_DIR / "raw"
        raw_candidates = sorted(raw_dir.glob(f"{base}*.jsonl"))
        files.extend(raw_candidates)
    # run 级汇总
    files.append(RUN_DIR / "summaries.json")
    files.append(RUN_DIR / "summaries.csv")
    files.append(RUN_DIR / "validation.json")
    return files


def collect_analysis_artifacts() -> list[Path]:
    return [
        ANALYSIS_DIR / "recompute_from_raw.json",
        ANALYSIS_DIR / "performance_statistics.json",
        ANALYSIS_DIR / "main_comparison.csv",
        ANALYSIS_DIR / "main_relative_to_full.csv",
        ANALYSIS_DIR / "endpoint_scaling.csv",
        ANALYSIS_DIR / "endpoint_relative_to_full.csv",
    ]


def collect_a6_a3_artifacts() -> list[Path]:
    """收集 A6/A3 冻结负载重跑日志、汇总结果与证据文档。"""
    files: list[Path] = []
    raw_dir = SWEEP_RUN_DIR / "raw"
    if raw_dir.is_dir():
        files.extend(sorted(raw_dir.glob("*.jsonl")))
    if SWEEP_ANALYSIS_DIR.is_dir():
        for path in SWEEP_ANALYSIS_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in (".json", ".csv", ".png"):
                files.append(path)
    if SWEEP_EVIDENCE.exists():
        files.append(SWEEP_EVIDENCE)
    if A4_ANALYSIS_DIR.is_dir():
        for path in A4_ANALYSIS_DIR.glob("a4_selection_ablation_budget_*"):
            if path.is_file() and path.suffix.lower() in (".json", ".csv", ".png"):
                files.append(path)
    return files


def collect_a5_artifacts() -> list[Path]:
    """收集 A5 权重敏感性结果与独立证据文档。"""
    files: list[Path] = []
    if A5_ANALYSIS_DIR.is_dir():
        for path in A5_ANALYSIS_DIR.glob("a5_weight_sensitivity.*"):
            if path.is_file() and path.suffix.lower() in (".json", ".csv", ".png"):
                files.append(path)
    if A5_EVIDENCE.exists():
        files.append(A5_EVIDENCE)
    return files


def collect_gold_artifacts() -> list[Path]:
    """收集 Gold 数据与标注指南（问题 1/2/3/29）。"""
    files: list[Path] = []
    if GOLD_DIR.is_dir():
        for pattern in ("*.json", "*.jsonl"):
            files.extend(sorted(GOLD_DIR.glob(pattern)))
    # A4 使用的独立 Gold 主文件位于任务子目录，需显式随归档提供。
    gold_main = GOLD_DIR / "agent_tasks" / "gold_main.jsonl"
    if gold_main.exists():
        files.append(gold_main)
    prompt = ROOT / "gold_reference.prompt"
    if prompt.exists():
        files.append(prompt)
    return files


def collect_report_artifacts() -> list[Path]:
    """收集每 seed 离线质量结果，排除体积大且可重建的模型二进制。

    reports/ 中只含 metrics.json / summary.* / thresholds.json / comparison.json
    与 figures/*.png，无 joblib，可直接递归收录。
    """
    files: list[Path] = []
    extensions = (".json", ".csv", ".md", ".png")
    for sub in REPORTS_SUMMARY_DIRS:
        base = REPORTS_DIR / sub
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    files.append(path)
    # reports/ 顶层各离线质量报告
    if REPORTS_DIR.is_dir():
        for path in sorted(REPORTS_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    return files


def collect_extra_analysis() -> list[Path]:
    """收集 B4/B5/C1/C2/C3 的离线分析产物（JSON/CSV/图）。"""
    files: list[Path] = []
    extensions = (".json", ".csv", ".png", ".pdf")
    for name in EXTRA_ANALYSIS_DIRS:
        base = BENCH_DIR / "analysis" / name
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    files.append(path)
    # performance 分析目录下的图（原 collect_analysis_artifacts 未含）
    fig_dir = ANALYSIS_DIR / "figures"
    if fig_dir.is_dir():
        for path in fig_dir.iterdir():
            if path.is_file():
                files.append(path)
    return files


def collect_scripts() -> list[Path]:
    """收集分析/绘图/归档脚本本体。"""
    return sorted(BENCH_DIR.glob("*.py"))


def build_sha256sums(files: list[Path], output: Path) -> dict[Path, str]:
    """逐文件计算 sha256 并写入 SHA256SUMS，返回 {相对路径: 哈希}。"""
    lines: list[str] = []
    hashes: dict[Path, str] = {}
    for path in files:
        rel = path.relative_to(ROOT)
        digest = sha256_file(path)
        hashes[rel] = digest
        lines.append(f"{digest}  {rel}")
    lines.sort()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashes


def collect_workload_hashes() -> list[dict[str, Any]]:
    """从每个 meta.json 提取 run 级版本标识。"""
    runs: list[dict[str, Any]] = []
    for meta in sorted(RUN_DIR.glob("*.meta.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        runs.append({
            "run_id": data.get("run_id"),
            "method": data.get("method"),
            "concurrency": data.get("concurrency"),
            "queue_size": data.get("queue_size"),
            "repeat": data.get("repeat"),
            "limit": data.get("limit"),
            "temperature": data.get("temperature"),
            "workload_type": data.get("workload_type"),
            "workload_ids_sha256": data.get("workload_ids_sha256"),
            "workload_records": data.get("workload_records"),
            "ids_file": data.get("ids_file"),
            "wall_clock_ms": data.get("wall_clock_ms"),
            "returncode": data.get("returncode"),
            "command": data.get("command"),
        })
    return runs


def build_manifest(hashes: dict[Path, str], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "git_commit": git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "archived_runs": RUN_DIR.name,
        "additional_experiment_sets": {
            "a6-a3-sweep": {
                "raw_dir": str(SWEEP_RUN_DIR.relative_to(ROOT)),
                "analysis_dir": str(SWEEP_ANALYSIS_DIR.relative_to(ROOT)),
                "methods": [
                    "budget-60", "budget-100", "budget-150", "budget-200",
                    "budget-300", "budget-400", "budget-600", "budget-800",
                    "budget-1200", "fixed-5", "fixed-10", "prompt-null",
                    "prompt-omit", "lambda-0.003",
                ],
                "records_per_method": 2000,
                "concurrency": 16,
                "queue_size": 16,
                "evidence": str(SWEEP_EVIDENCE.relative_to(ROOT)),
            },
            "a4-selection-ablation": {
                "analysis_dir": str(A4_ANALYSIS_DIR.relative_to(ROOT)),
                "budgets": [200, 400],
                "model_size": 100,
                "model_seeds": [42, 43, 44, 45, 46],
                "gold_reference": "data/gold/agent_tasks/gold_main.jsonl",
                "evidence": str(SWEEP_EVIDENCE.relative_to(ROOT)),
            },
            "a5-weight-sensitivity": {
                "analysis_dir": str(A5_ANALYSIS_DIR.relative_to(ROOT)),
                "budgets": [200, 400],
                "model_size": 100,
                "model_seeds": [42, 43, 44, 45, 46],
                "perturbation_seeds": list(range(200, 220)),
                "gold_reference": "data/gold/agent_tasks/gold_main.jsonl",
                "evidence": str(A5_EVIDENCE.relative_to(ROOT)),
            }
        },
        "run_count": len(runs),
        "runs": runs,
        "input_hashes": {
            rel: hashes.get(Path(rel))
            for rel in INPUT_FILES if (ROOT / rel).exists()
        },
        "input_files": INPUT_FILES,
    }


def write_readme(runs: list[dict[str, Any]]) -> str:
    methods = sorted({r["method"] for r in runs})
    lines = [
        "# 随稿复现材料归档（D1）",
        "",
        "本目录冻结主实验 `main-comparison-pro` 的完整复现证据，回应审稿问题 32",
        "（提供稳定文件与版本标识）。所有文件均离线生成，不触发 LLM/embedding。",
        "",
        "## 内容",
        "",
        "- `SHA256SUMS`：全部归档文件（raw 日志、meta、汇总、聚合产物、Gold/标注、" 
        "各 seed 离线质量结果、分析脚本）的校验和。",
        "- `manifest.json`：git 提交、逐次运行的 `workload_ids_sha256`、输入文件哈希、环境。",
        "- `README.md`：本说明。",
        "",
        "## 数据来源",
        "",
        "- 主实验运行目录：`benchmarks/runs/main-comparison-pro/`（git 忽略），",
        "  方法 " + "、".join(methods) + "，每次 2000 条记录、3 次重复、",
        "  concurrency=16、queue_size=16。",
        "- 聚合产物：`benchmarks/analysis/performance/` 及各离线分析目录",
        "  （breakeven、c1_near_duplicate、c2_representativeness、c3_scan、initialization）。",
        "- A6/A3 在线重跑：`benchmarks/runs/a6-a3-sweep/` 的 13 份逐条日志及",
        "  `benchmarks/analysis/a6_a3/` 的预算曲线、基线表和图。",
        "- A4 离线选择策略消融：`benchmarks/analysis/a4_selection_ablation/` 的固定测试集 Gold",
        "  汇总、表格和图；Gold 真值为 `data/gold/agent_tasks/gold_main.jsonl`。",
        "- A5 离线权重敏感性分析：`benchmarks/analysis/a5_weight_sensitivity/` 与",
        "  `tex/v6-lncs/补充实验3.md`；比较人工、等权、训练频率、反频率压力和扰动权重。",
        "- Gold 与标注：`data/gold/` + `gold_reference.prompt`。",
        "- 各 seed 离线质量结果：`reports/`（quality_token_comparison 与训练/模型消融的逐 seed metrics）。",
        "- 分析/绘图/归档脚本：`benchmarks/*.py`。",
        "",
        "## 复现命令（来自各 run 的 meta.json command 字段）",
        "",
        "```bash",
        "# 从逐条日志重算性能指标（B1/B3，问题 9/11/12/13/26/31）",
        ".venv/bin/python benchmarks/recompute_from_raw.py",
        "",
        "# 离线汇总统计表（原有）",
        ".venv/bin/python benchmarks/analyze_performance_results.py",
        "",
        "# 盈亏平衡点分析（B5，问题 17）",
        ".venv/bin/python benchmarks/analyze_breakeven.py",
        "",
        "# 近重复检测与分组隔离可行性（C1，问题 19/20）",
        ".venv/bin/python benchmarks/analyze_c1_near_duplicate.py",
        "",
        "# 数据代表性与分层分析（C2，问题 20/21）",
        ".venv/bin/python benchmarks/analyze_c2_representativeness.py",
        "",
        "# 并发/队列解耦扫描（C3，问题 15）",
        ".venv/bin/python benchmarks/analyze_c3_scan.py",
        "",
        "# 预算响应与匹配基线（A6/A3，问题 5/7/8/18/25/30/34）",
        ".venv/bin/python benchmarks/analyze_a6_a3.py",
        "",
        "# 固定测试集 Gold 的选择策略消融（A4，问题 6/7/25/34）",
        ".venv/bin/python benchmarks/analyze_a4_selection_ablation.py",
        "A4_BUDGET=400 .venv/bin/python benchmarks/analyze_a4_selection_ablation.py",
        "",
        "# 字段权重敏感性分析（A5，问题 6/25/34）",
        ".venv/bin/python benchmarks/analyze_a5_weight_sensitivity.py",
        "",
        "# 重新生成本归档清单",
        ".venv/bin/python benchmarks/archive_artifacts.py",
        "```",
        "",
        "## 验证归档完整性",
        "",
        "```bash",
        "cd 仓库根目录",
        "sha256sum -c benchmarks/archive/SHA256SUMS",
        "```",
        "",
    ]
    return "\n".join(lines)


def collect_input_files() -> list[Path]:
    return [ROOT / rel for rel in INPUT_FILES if (ROOT / rel).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成随稿复现材料归档清单（D1）")
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE_DIR)
    args = parser.parse_args()

    files = (
        collect_run_artifacts()
        + collect_analysis_artifacts()
        + collect_a6_a3_artifacts()
        + collect_a5_artifacts()
        + collect_input_files()
        + collect_gold_artifacts()
        + collect_report_artifacts()
        + collect_extra_analysis()
        + collect_scripts()
    )
    # 去重并校验存在
    seen: set[Path] = set()
    existing: list[Path] = []
    for f in files:
        if f not in seen and f.exists():
            seen.add(f)
            existing.append(f)

    hashes = build_sha256sums(existing, args.output_dir / "SHA256SUMS")
    runs = collect_workload_hashes()
    manifest = build_manifest(hashes, runs)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        write_readme(runs), encoding="utf-8"
    )

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "archived_files": len(existing),
        "runs": len(runs),
        "git_commit": manifest["git_commit"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
