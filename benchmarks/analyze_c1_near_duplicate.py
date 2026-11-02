#!/usr/bin/env python3
"""C1 近重复检测与分组隔离可行性报告（纯离线，不触发 LLM/embedding）。

回应审稿问题 19/20（artifact_id 不重叠无法排除同藏品、机构、模板、批次或
近重复文本带来的泄漏）。方法：
- 对每条性能负载记录拼接(类型+文物名称+详细描述)，做字符 n-gram(3-gram) Jaccard
  相似度。
- 用倒排索引限定候选对（仅比较共享至少一个 n-gram 的记录），避免 O(n^2)。
- 报告近重复分布、最大近重复簇、类型×年代批次代理聚类，并评估 group-held-out
  划分的可行性。

说明：raw_data 仅 5 字段（artifact_id、年代、文物名称、类型、详细描述），
来源/机构/模板元数据脱敏不可得，故只能基于文本与类型×年代做近重复代理检测。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RAW_GLOB = "benchmarks/runs/main-comparison-pro/raw/*.jsonl"
OUTPUT_DIR = Path("benchmarks/analysis/c1_near_duplicate")
NGRAM_N = 3
# 近重复相似度阈值（Jaccard）
NEAR_DUP_THRESHOLD = 0.6


def norm(text: str) -> str:
    """归一化：去空白与标点，用于 n-gram。"""
    return re.sub(r"[\s\W_]+", "", str(text))


def ngrams(text: str, n: int) -> set[str]:
    s = norm(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a | b))


def load_records() -> list[dict[str, Any]]:
    """从主实验 raw 日志加载去重后的性能记录（每个 artifact_id 一条）。"""
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(Path().glob(RAW_GLOB)):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.update(rec.get("raw_data") or {})
                aid = rec.get("artifact_id")
                if aid is not None:
                    records.setdefault(aid, rec)
    return list(records.values())


def find_near_duplicates(
    records: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    """倒排索引加速：仅比较共享至少一个 n-gram 的记录对。"""
    index: dict[str, list[int]] = defaultdict(list)
    texts: list[str] = []
    for i, rec in enumerate(records):
        text = " ".join(
            [str(rec.get("类型") or ""), str(rec.get("文物名称") or ""),
             str(rec.get("详细描述") or "")]
        )
        texts.append(text)
        grams = ngrams(text, NGRAM_N)
        for g in grams:
            index[g].append(i)
    # 统计每个记录的总 gram 数用于提前剪枝
    gram_count = [len(ngrams(t, NGRAM_N)) for t in texts]
    # 候选去重
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for g, ids in index.items():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((i, j))
    pairs: list[tuple[int, int, float]] = []
    for i, j in candidates:
        ai, aj = ngrams(texts[i], NGRAM_N), ngrams(texts[j], NGRAM_N)
        # 提前剪枝：交集不可能超过较小 gram 数
        if gram_count[i] < NEAR_DUP_THRESHOLD * gram_count[j] and \
           gram_count[j] < NEAR_DUP_THRESHOLD * gram_count[i]:
            continue
        sim = jaccard(ai, aj)
        if sim >= NEAR_DUP_THRESHOLD:
            pairs.append((i, j, sim))
    return pairs


def union_find_clusters(
    n: int, pairs: list[tuple[int, int, float]]
) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)
    return [g for g in groups.values() if len(g) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="C1 近重复检测与分组隔离报告")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    records = load_records()
    n = len(records)
    print(f"记录数: {n}, 唯一 artifact_id: {len({r['artifact_id'] for r in records})}")

    pairs = find_near_duplicates(records)
    print(f"近重复对 (Jaccard>={NEAR_DUP_THRESHOLD}): {len(pairs)}")
    if pairs:
        sims = [p[2] for p in pairs]
        print(f"  相似度 min={min(sims):.3f} max={max(sims):.3f} "
              f"mean={sum(sims)/len(sims):.3f}")

    clusters = union_find_clusters(n, pairs)
    print(f"近重复簇 (size>1): {len(clusters)}")
    cluster_sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"  最大簇大小: {cluster_sizes[:10] if cluster_sizes else '无'}")

    # 类型×年代 批次代理聚类（机构/模板元数据脱敏不可得的代理）
    type_epoch: Counter = Counter()
    for rec in records:
        type_epoch[(str(rec.get("类型") or ""), str(rec.get("年代") or ""))] += 1
    batch_dup = {k: v for k, v in type_epoch.items() if v > 1}
    print(f"类型×年代组合数: {len(type_epoch)}，其中重复(>1条)的组合数: {len(batch_dup)}")

    # 输出
    payload = {
        "analysis": {
            "description": "C1 近重复检测；字符3-gram Jaccard；纯离线。",
            "text_source": "raw_data(类型+文物名称+详细描述)",
            "ngram": NGRAM_N,
            "near_dup_threshold": NEAR_DUP_THRESHOLD,
            "limitation": (
                "来源/机构/模板元数据脱敏不可得，仅能做文本近重复 + 类型×年代批次代理；"
                "同藏品跨记录但文本改写充分的近重复可能漏检。"
            ),
        },
        "record_count": n,
        "unique_artifact_ids": len({r["artifact_id"] for r in records}),
        "near_dup_pairs": len(pairs),
        "near_dup_sim_range": [round(min(p[2] for p in pairs), 4),
                               round(max(p[2] for p in pairs), 4)] if pairs else None,
        "near_dup_clusters": len(clusters),
        "largest_cluster_sizes": cluster_sizes[:10],
        "type_epoch_batches": len(type_epoch),
        "type_epoch_dup_batches": len(batch_dup),
        "top_type_epoch_batches": [
            {"type": k[0], "epoch": k[1], "count": v}
            for k, v in type_epoch.most_common(15)
        ],
        "group_held_out_feasibility": (
            "存在 {dup} 个类型×年代批次含 >1 条记录，可按类型×年代做 group-held-out 划分；"
            "但真实来源/机构分组需原始元数据，当前脱敏数据仅能文本+类型×年代代理。"
        ).format(dup=len(batch_dup)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "c1_near_duplicate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"输出: {args.output_dir / 'c1_near_duplicate.json'}")


if __name__ == "__main__":
    main()