#!/usr/bin/env python3
"""Leakage-resistant candidate-level shadow replay for a mined HBDC delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from my_rewriter.candidate_selection.cost_policy import CostOnlyPolicy
from my_rewriter.candidate_selection.interfaces import CandidateFeatures
from my_rewriter.candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
from my_rewriter.candidate_selection.mvm.policy import MVMConfig
from my_rewriter.experiments.mvm_reliability.dataset import load_candidates_jsonl
from my_rewriter.experiments.mvm_reliability.schemas import CandidateRecord
from my_rewriter.experiments.hbdc_context_prior_20261102.shadow_replay import (
    HierarchicalContextPolicy,
)


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
BASE = ROOT / "knowledge/hbdc/mvm_v2/candidates.jsonl"
CURRENT_DELTA = ROOT / "knowledge/hbdc/calcite_synthetic25_v1/delta_candidates.jsonl"
NEW_DIR = ROOT / "knowledge/hbdc/calcite_log_mined_v2"


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def sql_hash(sql: str) -> str:
    return hashlib.sha256(normalize_sql(sql).encode()).hexdigest()


def raw_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def features(row: CandidateRecord) -> CandidateFeatures:
    return CandidateFeatures(
        candidate_id=row.id,
        workload=row.workload,
        query=row.original_sql,
        query_family=row.query_family,
        rules=row.rules,
        original_cost=row.original_cost,
        rewritten_cost=row.rewritten_cost,
    )


def safe_train(history: list[CandidateRecord], test: CandidateRecord,
               distribution: str | None = None) -> list[CandidateRecord]:
    test_hash = sql_hash(test.original_sql)
    return [
        row for row in history
        if row.usable
        and (distribution is None or row.distribution in {None, distribution})
        and not (row.workload == test.workload and row.query_family == test.query_family)
        and not (row.workload == test.workload and sql_hash(row.original_sql) == test_hash)
    ]


def decision(policy_name: str, history: list[CandidateRecord], test: CandidateRecord,
             config: MVMConfig, cost: CostOnlyPolicy,
             distribution: str | None = None) -> dict[str, Any]:
    if policy_name == "cost_only":
        chosen = cost.decide(features(test))
        detail = None
    else:
        training = safe_train(history, test, distribution)
        policy = (
            HierarchicalContextPolicy(training, config)
            if policy_name == "mined_context_hbdc"
            else ConditionalMVMPolicy.build(training, config, cost)
        )
        detail = policy.decide_detailed(features(test))
        chosen = detail.decision
    effective = (
        test.rewritten_latency_ms if chosen == "rewrite" else test.original_latency_ms
    )
    actual = "rewrite" if test.rewritten_latency_ms < test.original_latency_ms else "original"
    return {
        "decision": chosen,
        "actual_best": actual,
        "correct": chosen == actual,
        "effective_latency_ms": effective,
        "prior_source": getattr(detail, "prior_source", None),
        "prior_key": getattr(detail, "prior_key", None),
        "support": getattr(detail, "support", None),
        "reliability": getattr(detail, "reliability", None),
    }


def summarize(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    total = sum(row[policy]["effective_latency_ms"] for row in rows)
    oracle = sum(
        min(row["original_latency_ms"], row["rewritten_latency_ms"]) for row in rows
    )
    return {
        "candidates": len(rows),
        "correct": sum(row[policy]["correct"] for row in rows),
        "accuracy": sum(row[policy]["correct"] for row in rows) / len(rows) if rows else None,
        "effective_latency_sum_ms": total,
        "oracle_latency_sum_ms": oracle,
        "regret_ms": total - oracle,
        "decisions": dict(Counter(row[policy]["decision"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=NEW_DIR / "shadow_evaluation.json")
    args = parser.parse_args()

    base = load_candidates_jsonl(BASE)
    current = base + load_candidates_jsonl(CURRENT_DELTA)
    updated = base + load_candidates_jsonl(NEW_DIR / "delta_candidates.jsonl")
    audit_raw = raw_jsonl(NEW_DIR / "audit_all_candidates.jsonl")
    audit = load_candidates_jsonl(NEW_DIR / "audit_all_candidates.jsonl")
    tests = [
        row for raw, row in zip(audit_raw, audit)
        if raw.get("history_role") == "heldout_eval" and raw.get("quality_eligible")
    ]
    raw_by_id = {row["id"]: row for row in audit_raw}

    config = MVMConfig(
        alpha=1.0,
        beta=1.0,
        min_support=5,
        fp_guard_threshold=0.4,
        fn_recovery_threshold=0.6,
        cost_strict=True,
    )
    cost = CostOnlyPolicy(strict=True)
    policies = {
        "cost_only": [],
        "base_hbdc": base,
        "current_delta_hbdc": current,
        "mined_delta_hbdc": updated,
        "mined_distribution_hbdc": updated,
        "mined_context_hbdc": updated,
    }
    rows = []
    for test in tests:
        raw = raw_by_id[test.id]
        row = {
            "candidate_id": test.id,
            "query_family": test.query_family,
            "distribution": raw.get("distribution"),
            "rules": list(test.rules),
            "label": test.label,
            "original_latency_ms": test.original_latency_ms,
            "rewritten_latency_ms": test.rewritten_latency_ms,
        }
        for name, history in policies.items():
            distribution = (
                raw.get("distribution")
                if name in {"mined_distribution_hbdc", "mined_context_hbdc"}
                else None
            )
            row[name] = decision(name, history, test, config, cost, distribution)
        rows.append(row)

    summaries = {name: summarize(rows, name) for name in policies}
    cost_total = summaries["cost_only"]["effective_latency_sum_ms"]
    for name, summary in summaries.items():
        summary["reduction_vs_cost_only_pct"] = (
            (cost_total - summary["effective_latency_sum_ms"]) / cost_total * 100.0
            if cost_total else None
        )

    by_distribution = {}
    for distribution in sorted({row["distribution"] for row in rows}):
        subset = [row for row in rows if row["distribution"] == distribution]
        by_distribution[distribution] = {
            name: summarize(subset, name) for name in policies
        }

    flip_rows = []
    for row in rows:
        old = row["current_delta_hbdc"]
        new = row["mined_distribution_hbdc"]
        if old["decision"] != new["decision"]:
            flip_rows.append({
                "candidate_id": row["candidate_id"],
                "query_family": row["query_family"],
                "distribution": row["distribution"],
                "rules": row["rules"],
                "old_decision": old["decision"],
                "new_decision": new["decision"],
                "actual_best": new["actual_best"],
                "beneficial": new["correct"] and not old["correct"],
                "harmful": old["correct"] and not new["correct"],
                "old_prior": {
                    "source": old["prior_source"], "key": old["prior_key"],
                    "support": old["support"], "reliability": old["reliability"],
                },
                "new_prior": {
                    "source": new["prior_source"], "key": new["prior_key"],
                    "support": new["support"], "reliability": new["reliability"],
                },
            })

    report = {
        "method": "candidate-level strict shadow replay",
        "test_scope": "Calcite44 SQL only; quality-filtered, strict equivalence, measured latency",
        "leakage_controls": [
            "evaluated query family excluded from every HBDC history",
            "normalized original SQL hash excluded from every HBDC history",
            "modern Calcite44 campaign rows excluded from training delta",
            "historical crossover rows used only across query families",
            "micro-equivalence evidence excluded",
        ],
        "config": config.__dict__,
        "summaries": summaries,
        "by_distribution": by_distribution,
        "current_to_mined_flips": {
            "count": len(flip_rows),
            "beneficial": sum(row["beneficial"] for row in flip_rows),
            "harmful": sum(row["harmful"] for row in flip_rows),
            "details": flip_rows,
        },
        "candidate_details": rows,
        "limitation": (
            "This replays logged final candidates; it does not regenerate candidates or claim an end-to-end gain."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_details"}, indent=2))


if __name__ == "__main__":
    main()
