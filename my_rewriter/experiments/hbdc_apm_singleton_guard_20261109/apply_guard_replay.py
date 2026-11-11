#!/usr/bin/env python3
"""Apply the Calcite singleton-APM selection guard to frozen crossover rows.

The guard selects the original query whenever the complete selected rule
sequence is exactly ``[AGGREGATE_PROJECT_MERGE]``.  No query is re-executed:
the original latency samples were already collected in the same workload-
crossover session as the rewrite and are therefore the exact latency that the
guarded policy would have selected.  Multi-rule sequences remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(
    os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])
).resolve()
SOURCE = ROOT / "results/hbdc_v2_selective_replay_20261108"
OUTPUT = ROOT / "results/hbdc_apm_singleton_guard_20261109"
KNOWLEDGE = ROOT / "knowledge/hbdc/hbdc_calcite_sequence_guard_v1.json"
JOBS = (
    "gpt4o_uniform",
    "gpt4o_zipf",
    "claude46_uniform",
    "claude46_zipf",
    "deepseek_v4pro_uniform",
    "deepseek_v4pro_zipf",
)
TARGET_RULES = ["AGGREGATE_PROJECT_MERGE"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats_ms(values: list[float]) -> dict[str, float]:
    p50, p75, p90, p95 = np.percentile(values, [50, 75, 90, 95])
    return {
        "mean": float(np.mean(values)),
        "median": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, Any] = {
        "method": "deterministic candidate-selection replay",
        "knowledge_path": str(KNOWLEDGE),
        "knowledge_sha256": sha256(KNOWLEDGE),
        "policy": {
            "workload": "calcite",
            "exact_rules": TARGET_RULES,
            "decision": "original",
            "multi_rule_sequences_affected": False,
        },
        "measurement_note": (
            "No fresh DB execution was needed. For a rejected rewrite, effective "
            "latency is the original-query latency already measured in that row's "
            "same workload-crossover session."
        ),
        "jobs": {},
    }

    for job in JOBS:
        source = SOURCE / job / "merged_44"
        target = OUTPUT / job / "guarded_44"
        rewrites = read_jsonl(source / "rewrites.jsonl")
        equivalence = read_jsonl(source / "equivalence.jsonl")
        latency = read_jsonl(source / "latency_per_query.jsonl")
        rewrite_by_id = {row["query_id"]: row for row in rewrites}
        eq_by_id = {row["query_id"]: row for row in equivalence}
        latency_by_id = {row["query_id"]: row for row in latency}
        if not (len(rewrite_by_id) == len(eq_by_id) == len(latency_by_id) == 44):
            raise RuntimeError(f"{job}: expected aligned 44-query inputs")

        matched_ids = [
            row["query_id"] for row in rewrites if row.get("used_rules") == TARGET_RULES
        ]
        affected_ids = [
            query_id for query_id in matched_ids
            if latency_by_id[query_id].get("adopted")
        ]

        guarded_rewrites = []
        guarded_equivalence = []
        guarded_latency = []
        effects = []
        for original_rewrite in rewrites:
            query_id = original_rewrite["query_id"]
            rewrite = dict(original_rewrite)
            eq = dict(eq_by_id[query_id])
            lat = dict(latency_by_id[query_id])
            if query_id in affected_ids:
                before_ms = float(lat["effective_latency_ms"])
                after_ms = float(lat["original_latency_ms"])
                effects.append({
                    "query_id": query_id,
                    "query_family": rewrite.get("query_family"),
                    "before_effective_latency_ms": before_ms,
                    "after_effective_latency_ms": after_ms,
                    "delta_ms": after_ms - before_ms,
                })

                rewrite["pre_guard_rewritten_query"] = rewrite.get("rewritten_query")
                rewrite["pre_guard_used_rules"] = rewrite.get("used_rules")
                rewrite["rewritten_query"] = rewrite["original_query"]
                rewrite["used_rules"] = ["HBDC_APM_SINGLETON_GUARD"]
                rewrite["rewrite_generated"] = False
                rewrite["rewrite_same_as_original"] = True
                rewrite["hbdc_sequence_override"] = {
                    "id": "calcite_suppress_singleton_aggregate_project_merge_v1",
                    "decision": "original",
                    "exact_rules": TARGET_RULES,
                }

                eq["pre_guard_equivalence_status"] = eq.get("equivalence_status")
                eq["equivalence_status"] = "original_by_hbdc_sequence_override"
                eq["adopted"] = False
                eq["fallback"] = False
                eq["reason"] = "exact_singleton_sequence_guard"

                lat["pre_guard_effective_latency_ms"] = before_ms
                lat["pre_guard_adopted"] = lat.get("adopted")
                lat["pre_guard_rewritten_query"] = lat.get("rewritten_query")
                lat["rewritten_query"] = lat["original_query"]
                lat["effective_latency_ms"] = after_ms
                lat["improvement_pct"] = 0.0
                lat["adopted"] = False
                lat["fallback"] = False
                lat["rewrite_same_as_original"] = True
                lat["hbdc_sequence_override"] = (
                    "calcite_suppress_singleton_aggregate_project_merge_v1"
                )

            guarded_rewrites.append(rewrite)
            guarded_equivalence.append(eq)
            guarded_latency.append(lat)

        write_jsonl(target / "rewrites.jsonl", guarded_rewrites)
        write_jsonl(target / "equivalence.jsonl", guarded_equivalence)
        write_jsonl(target / "latency_per_query.jsonl", guarded_latency)

        original_values = [float(row["original_latency_ms"]) for row in guarded_latency]
        before_values = [float(row["effective_latency_ms"]) for row in latency]
        after_values = [float(row["effective_latency_ms"]) for row in guarded_latency]
        original_stats = stats_ms(original_values)
        before_stats = stats_ms(before_values)
        after_stats = stats_ms(after_values)
        reduction = {
            key: (original_stats[key] - after_stats[key]) / original_stats[key] * 100.0
            for key in original_stats
        }
        summary = {
            "job": job,
            "query_count": 44,
            "source": str(source),
            "knowledge_path": str(KNOWLEDGE),
            "exact_sequence_matched_count": len(matched_ids),
            "adopted_rewrite_suppressed_count": len(affected_ids),
            "matched_query_ids": matched_ids,
            "suppressed_query_ids": affected_ids,
            "original_latency_ms": original_stats,
            "before_guard_effective_latency_ms": before_stats,
            "effective_latency_ms": after_stats,
            "reduction_vs_original_pct": reduction,
            "per_query_effects": effects,
            "measurement_mode": "same-session original-arm deterministic replay",
        }
        (target / "latency_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        all_summary["jobs"][job] = summary

    (OUTPUT / "summary_all.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n"
    )
    report = [
        "# Calcite singleton AGGREGATE_PROJECT_MERGE guard\n",
        "정확히 `[AGGREGATE_PROJECT_MERGE]`인 Calcite 후보만 원본으로 되돌린 "
        "결정론적 selection replay 결과다. Multi-rule sequence는 변경하지 않았다.\n",
        "| Job | Suppressed | Mean (s) | Median (s) | P75 (s) | P90 (s) | P95 (s) |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for job, row in all_summary["jobs"].items():
        values = row["effective_latency_ms"]
        report.append(
            f"| {job} | {row['adopted_rewrite_suppressed_count']} | "
            f"{values['mean'] / 1000:.3f} | {values['median'] / 1000:.3f} | "
            f"{values['p75'] / 1000:.3f} | {values['p90'] / 1000:.3f} | "
            f"{values['p95'] / 1000:.3f} |\n"
        )
    report.extend([
        "\n## 측정 해석\n",
        "새 DB 실행은 하지 않았다. Guard가 rewrite를 거절하면 실행되는 것은 원본 "
        "SQL이므로, 각 query의 동일 workload-crossover 세션에서 이미 수집한 original "
        "samples를 effective latency로 선택했다. 따라서 이는 새 후보의 latency 추정이 "
        "아니라 저장된 실측값을 이용한 결정론적 policy replay다.\n",
    ])
    (OUTPUT / "REPORT_KO.md").write_text("".join(report))
    print(json.dumps({
        job: {
            "suppressed": row["adopted_rewrite_suppressed_count"],
            "effective_latency_ms": row["effective_latency_ms"],
        }
        for job, row in all_summary["jobs"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
