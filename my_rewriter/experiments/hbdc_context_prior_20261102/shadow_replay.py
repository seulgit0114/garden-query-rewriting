#!/usr/bin/env python3
"""Family-excluded shadow replay for a hierarchical context HBDC policy.

This experiment is deliberately isolated from the live GARDEN policy.  It
replays the final Calcite candidates found in completed runs and reports only
decision changes.  No observed latency from the evaluated query is exposed to
the policy.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from sqlglot import exp, parse_one

from my_rewriter.candidate_selection.cost_policy import CostOnlyPolicy
from my_rewriter.candidate_selection.interfaces import CandidateFeatures, MVMDecision
from my_rewriter.candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
from my_rewriter.candidate_selection.mvm.policy import MVMConfig
from my_rewriter.experiments.mvm_reliability.dataset import load_candidates_jsonl


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
HISTORY = ROOT / "knowledge/hbdc/mvm_v2/candidates.jsonl"
UNIFORM = ROOT / "results/llm-agent/261101/garden_full_calcite_uniform_claude46_44/arms/full_pipeline_refined_hbdc"
ZIPF = ROOT / "results/llm-agent/261029/garden_gate1_calcite_zipf_40m_pool_dcrp_repair_claude46_69/arms/full_gate1_on"


@lru_cache(maxsize=8192)
def operator_pattern(sql: str) -> str:
    try:
        tree = parse_one(sql, read="postgres")
    except Exception:
        return "parse_error"
    aggs = list(tree.find_all(exp.AggFunc))
    distinct_aggs = [node for node in aggs if node.find(exp.Distinct) is not None]
    nondistinct_aggs = [node for node in aggs if node.find(exp.Distinct) is None]
    joins = list(tree.find_all(exp.Join))
    outer = sum(
        str(node.args.get("side") or "").upper() in {"LEFT", "RIGHT", "FULL"}
        for node in joins
    )
    subqueries = list(tree.find_all(exp.Subquery))
    exists = list(tree.find_all(exp.Exists))
    not_in = "NOT IN" in sql.upper()
    any_some = bool(re.search(r"\b(?:ANY|SOME)\s*\(", sql, re.I))
    grouping = bool(list(tree.find_all(exp.Group)))
    bucket = lambda value: "0" if value == 0 else "1" if value == 1 else "2+"
    return "|".join(
        (
            f"agg{bucket(len(aggs))}",
            f"d{bucket(len(distinct_aggs))}",
            f"nd{bucket(len(nondistinct_aggs))}",
            f"join{bucket(len(joins))}",
            f"outer{bucket(outer)}",
            f"subq{bucket(len(subqueries))}",
            f"exists{bucket(len(exists))}",
            f"notin{int(not_in)}",
            f"anysome{int(any_some)}",
            f"group{int(grouping)}",
        )
    )


def cost_decision(row: Any, policy: CostOnlyPolicy) -> str:
    return policy.decide(
        CandidateFeatures(
            candidate_id=getattr(row, "id", ""),
            workload=row.workload,
            query=row.original_sql,
            query_family=row.query_family,
            rules=row.rules,
            original_cost=row.original_cost,
            rewritten_cost=row.rewritten_cost,
        )
    )


@dataclass(frozen=True)
class Entry:
    support: int
    reliability: float
    mean_improvement_pct: float
    median_improvement_pct: float


def make_entry(rows: Iterable[Any], alpha: float, beta: float) -> Entry:
    rows = list(rows)
    improvements = [
        (row.original_latency_ms - row.rewritten_latency_ms)
        / row.original_latency_ms
        * 100.0
        for row in rows
        if row.original_latency_ms > 0
    ]
    positives = sum(row.rewritten_latency_ms < row.original_latency_ms for row in rows)
    return Entry(
        support=len(rows),
        reliability=(positives + alpha) / (len(rows) + alpha + beta),
        mean_improvement_pct=statistics.mean(improvements),
        median_improvement_pct=statistics.median(improvements),
    )


class HierarchicalContextPolicy:
    """Context-first policy with a decision-preserving legacy fallback."""

    def __init__(self, rows: list[Any], config: MVMConfig):
        self.config = config
        self.cost = CostOnlyPolicy(strict=config.cost_strict)
        self.legacy = ConditionalMVMPolicy.build(rows, config, self.cost)
        grouped = [defaultdict(list) for _ in range(4)]
        for row in rows:
            decision = cost_decision(row, self.cost)
            pattern = operator_pattern(row.original_sql)
            sequence = tuple(row.rules)
            grouped[0][(decision, sequence, pattern)].append(row)
            grouped[1][(decision, sequence)].append(row)
            for rule in dict.fromkeys(row.rules):
                grouped[2][(decision, rule, pattern)].append(row)
                grouped[3][(decision, rule)].append(row)
        self.tables = [
            {key: make_entry(items, config.alpha, config.beta) for key, items in table.items()}
            for table in grouped
        ]

    def decide_detailed(self, candidate: CandidateFeatures) -> MVMDecision:
        decision = self.cost.decide(candidate)
        threshold = (
            self.config.fp_guard_threshold
            if decision == "rewrite"
            else self.config.fn_recovery_threshold
        )
        pattern = operator_pattern(candidate.query)
        sequence = tuple(candidate.rules)
        chain = (
            (self.tables[0].get((decision, sequence, pattern)), "sequence_operator", ">".join(sequence)),
            (self.tables[1].get((decision, sequence)), "rule_sequence", ">".join(sequence)),
        )
        for entry, source, key in chain:
            if entry is not None and entry.support >= self.config.min_support:
                chosen = "rewrite" if entry.reliability >= threshold else "original"
                return MVMDecision(
                    decision=chosen,
                    reliability=entry.reliability,
                    support=entry.support,
                    prior_source=source,
                    prior_key=f"{key}:{pattern}" if source == "sequence_operator" else key,
                    reason=f"hierarchical {source} support={entry.support}",
                    mean_latency_improvement_pct=entry.mean_improvement_pct,
                    median_latency_improvement_pct=entry.median_improvement_pct,
                )

        per_rule = []
        for rule in dict.fromkeys(candidate.rules):
            entry = self.tables[2].get((decision, rule, pattern))
            source = "combined_rule_operator"
            if entry is None or entry.support < self.config.min_support:
                entry = self.tables[3].get((decision, rule))
                source = "combined_rule"
            if entry is not None and entry.support >= self.config.min_support:
                per_rule.append((rule, entry, source))
        if per_rule:
            # False-positive prevention is conjunctive (weakest rule).  False-
            # negative recovery is disjunctive (strongest supported rule).
            rule, entry, source = (
                min(per_rule, key=lambda item: (item[1].reliability, item[1].median_improvement_pct))
                if decision == "rewrite"
                else max(per_rule, key=lambda item: (item[1].reliability, item[1].median_improvement_pct))
            )
            chosen = "rewrite" if entry.reliability >= threshold else "original"
            return MVMDecision(
                decision=chosen,
                reliability=entry.reliability,
                support=entry.support,
                prior_source=source,
                prior_key=f"{rule}:{pattern}" if source.endswith("operator") else rule,
                reason=f"hierarchical {source} support={entry.support}",
                mean_latency_improvement_pct=entry.mean_improvement_pct,
                median_latency_improvement_pct=entry.median_improvement_pct,
            )
        fallback = self.legacy.decide_detailed(candidate)
        return MVMDecision(
            decision=fallback.decision,
            reliability=fallback.reliability,
            support=fallback.support,
            prior_source=f"legacy_fallback:{fallback.prior_source}",
            prior_key=fallback.prior_key,
            reason=f"no supported context; {fallback.reason}",
            mean_latency_improvement_pct=fallback.mean_latency_improvement_pct,
            median_latency_improvement_pct=fallback.median_latency_improvement_pct,
        )


def parse_result_dict(line: str) -> dict[str, Any] | None:
    marker = "Rewrite Execution Results: "
    if marker not in line:
        return None
    try:
        return ast.literal_eval(line.split(marker, 1)[1])
    except Exception:
        return None


def extract_candidate(log: Path, original: str) -> tuple[float, dict[str, Any]] | None:
    text = log.read_text(errors="replace")
    match = re.search(r"Input Cost: ([0-9.]+)", text)
    if not match:
        return None
    input_cost = float(match.group(1))
    candidates = []
    for line in text.splitlines():
        result = parse_result_dict(line)
        if not result:
            continue
        rules = result.get("used_rules") or []
        sql = result.get("output_sql")
        cost = result.get("output_cost")
        if (
            sql not in (None, "None", original)
            and cost is not None
            and "DCRP_REJECT_FALLBACK" not in rules
        ):
            candidates.append(result)
    return (input_cost, candidates[-1]) if candidates else None


def load_arm(arm: Path, workload_name: str, history: list[Any], config: MVMConfig) -> list[dict[str, Any]]:
    rewrites = {
        row["query_id"]: row
        for row in map(json.loads, (arm / "01_rewrite/rewrites.jsonl").open())
        if row["query_id"].startswith("base_")
    }
    output = []
    for query_id, row in rewrites.items():
        family = row["query_family"]
        extracted = extract_candidate(
            arm / "01_rewrite/query_logs" / f"{query_id}.log", row["original_query"]
        )
        if extracted is None:
            continue
        original_cost, candidate = extracted
        rules = tuple(candidate.get("used_rules") or ())
        features = CandidateFeatures(
            candidate_id=query_id,
            workload="calcite",
            query=row["original_query"],
            query_family=family,
            rules=rules,
            original_cost=original_cost,
            rewritten_cost=float(candidate["output_cost"]),
        )
        family_excluded = [
            item for item in history
            if not (item.workload == "calcite" and item.query_family == family)
        ]
        legacy = ConditionalMVMPolicy.build(family_excluded, config).decide_detailed(features)
        contextual = HierarchicalContextPolicy(family_excluded, config).decide_detailed(features)
        current = "rewrite" if row["rewrite_generated"] else "original"
        output.append(
            {
                "workload": workload_name,
                "query_id": query_id,
                "query_family": family,
                "rules": list(rules),
                "cost_ratio": features.rewritten_cost / features.original_cost,
                "operator_pattern": operator_pattern(features.query),
                "current_pipeline_decision": current,
                "legacy_family_excluded": legacy.__dict__,
                "hierarchical_context": contextual.__dict__,
                "decision_changed_vs_legacy": contextual.decision != legacy.decision,
                "decision_changed_vs_current": contextual.decision != current,
                "candidate_sql": candidate["output_sql"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = [row for row in load_candidates_jsonl(HISTORY) if row.usable]
    config = MVMConfig(
        alpha=1.0,
        beta=1.0,
        min_support=5,
        fp_guard_threshold=0.4,
        fn_recovery_threshold=0.6,
        cost_strict=True,
    )
    rows = load_arm(UNIFORM, "calcite_uniform", history, config)
    rows += load_arm(ZIPF, "calcite_zipf", history, config)
    report = {
        "method": "family-excluded hierarchical context HBDC shadow replay",
        "distribution_bucket_status": "not available in historical mvm_v2 records; not used",
        "config": config.__dict__,
        "counts": {
            workload: {
                "candidates": sum(row["workload"] == workload for row in rows),
                "changed_vs_legacy": sum(
                    row["workload"] == workload and row["decision_changed_vs_legacy"]
                    for row in rows
                ),
                "changed_vs_current": sum(
                    row["workload"] == workload and row["decision_changed_vs_current"]
                    for row in rows
                ),
            }
            for workload in ("calcite_uniform", "calcite_zipf")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    for row in rows:
        if row["decision_changed_vs_legacy"] or row["decision_changed_vs_current"]:
            print(
                row["workload"], row["query_id"],
                "current=", row["current_pipeline_decision"],
                "legacy=", row["legacy_family_excluded"]["decision"],
                "context=", row["hierarchical_context"]["decision"],
                "source=", row["hierarchical_context"]["prior_source"],
            )


if __name__ == "__main__":
    main()
