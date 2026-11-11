"""Train-only, structure-conditioned DCRP for Correlate candidates.

This module is deliberately experiment-local.  It does not modify the shared
GARDEN implementation while another campaign is running.  Non-Correlate
candidates are delegated unchanged to ``ConditionalMVMPolicy``.

The Correlate policy fixes two limitations of the current prior lookup:

* a mixed rule sequence can select a high-support *other* rule instead of
  ``FILTER_SUB_QUERY_TO_CORRELATE``;
* cost-rejected Correlate candidates pool structurally different subqueries.

Only frozen training-history latency is used to build priors.  Runtime inputs
are the original SQL, rule names, workload, and estimated costs; no runtime
latency or query-family identity is used.
"""

from __future__ import annotations

import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from sqlglot import exp, parse_one

from my_rewriter.candidate_selection.cost_policy import CostOnlyPolicy
from my_rewriter.candidate_selection.interfaces import CandidateFeatures, MVMDecision
from my_rewriter.candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
from my_rewriter.candidate_selection.mvm.policy import MVMConfig


CORRELATE_RULE = "FILTER_SUB_QUERY_TO_CORRELATE"


def _is_positive(row: t.Any) -> bool:
    return row.rewritten_latency_ms < row.original_latency_ms


def _improvement_pct(row: t.Any) -> float:
    return (row.original_latency_ms - row.rewritten_latency_ms) / row.original_latency_ms * 100.0


def subquery_motif(sql: str) -> str:
    """Return a coarse semantic motif without literals or template identity."""
    try:
        tree = parse_one(sql, read="postgres")
    except Exception:
        return "parse_error"

    cte_count = sum(1 for _ in tree.find_all(exp.CTE))
    setop_count = sum(
        sum(1 for _ in tree.find_all(kind))
        for kind in (exp.Union, exp.Intersect, exp.Except)
    )
    exists_count = sum(1 for _ in tree.find_all(exp.Exists))
    scalar_aggregate_subqueries = [
        node
        for node in tree.find_all(exp.Subquery)
        if any(True for _ in node.find_all(exp.AggFunc))
    ]
    where_scalar_count = sum(
        node.find_ancestor(exp.Where) is not None
        for node in scalar_aggregate_subqueries
    )
    having_scalar_count = sum(
        node.find_ancestor(exp.Having) is not None
        for node in scalar_aggregate_subqueries
    )

    if scalar_aggregate_subqueries and where_scalar_count and not cte_count and not setop_count:
        return "scalar_aggregate_where_plain"
    if scalar_aggregate_subqueries and having_scalar_count and not cte_count:
        return "scalar_aggregate_having_plain"
    if scalar_aggregate_subqueries and cte_count:
        return "scalar_aggregate_with_cte"
    if scalar_aggregate_subqueries:
        return "scalar_aggregate_other"
    if exists_count >= 2:
        return "multiple_exists"
    if exists_count == 1:
        return "single_exists"
    return "other_subquery"


@dataclass(frozen=True)
class _Entry:
    support: int
    reliability: float
    mean_improvement_pct: float
    median_improvement_pct: float


def _entry(rows: t.Sequence[t.Any], alpha: float, beta: float) -> _Entry:
    positives = sum(_is_positive(row) for row in rows)
    improvements = [_improvement_pct(row) for row in rows]
    return _Entry(
        support=len(rows),
        reliability=(positives + alpha) / (len(rows) + alpha + beta),
        mean_improvement_pct=sum(improvements) / len(improvements),
        median_improvement_pct=statistics.median(improvements),
    )


def _features_for_row(row: t.Any) -> CandidateFeatures:
    return CandidateFeatures(
        candidate_id=getattr(row, "id", ""),
        workload=row.workload,
        query=row.original_sql,
        query_family="",
        rules=row.rules,
        original_cost=row.original_cost,
        rewritten_cost=row.rewritten_cost,
    )


@dataclass
class CorrelateAwareDCRPPolicy:
    """DCRP with an explicit Correlate-specific hierarchical prior."""

    base_policy: ConditionalMVMPolicy
    config: MVMConfig
    cost_policy: CostOnlyPolicy
    tables: t.Tuple[t.Dict[t.Any, _Entry], ...]
    correlate_train_count: int

    @classmethod
    def build(
        cls,
        train_rows: t.Sequence[t.Any],
        config: MVMConfig,
        cost_policy: t.Optional[CostOnlyPolicy] = None,
    ) -> "CorrelateAwareDCRPPolicy":
        cost_policy = cost_policy or CostOnlyPolicy(strict=config.cost_strict)
        base_policy = ConditionalMVMPolicy.build(train_rows, config, cost_policy)
        grouped: t.List[t.DefaultDict[t.Any, list]] = [defaultdict(list) for _ in range(5)]
        correlate_rows = [row for row in train_rows if CORRELATE_RULE in row.rules]
        for row in correlate_rows:
            decision = cost_policy.decide(_features_for_row(row))
            motif = subquery_motif(row.original_sql)
            keys = (
                (row.workload, decision, motif),
                (decision, motif),
                motif,
                (row.workload, decision),
                decision,
            )
            for table, key in zip(grouped, keys):
                table[key].append(row)
        tables = tuple(
            {
                key: _entry(rows, config.alpha, config.beta)
                for key, rows in table.items()
            }
            for table in grouped
        )
        return cls(base_policy, config, cost_policy, tables, len(correlate_rows))

    @property
    def accept_train_count(self) -> int:
        return self.base_policy.accept_train_count

    @property
    def reject_train_count(self) -> int:
        return self.base_policy.reject_train_count

    def decide(self, candidate: CandidateFeatures) -> str:
        return self.decide_detailed(candidate).decision

    def decide_detailed(self, candidate: CandidateFeatures) -> MVMDecision:
        if CORRELATE_RULE not in candidate.rules:
            return self.base_policy.decide_detailed(candidate)

        cost_decision = self.cost_policy.decide(candidate)
        motif = subquery_motif(candidate.query)
        workload_motif, decision_motif, motif_only, workload_decision, decision_only = self.tables

        # A cost-accepted Correlate rewrite should be judged by the explicit
        # Correlate history, not by an unrelated high-support companion rule.
        if cost_decision == "rewrite":
            chain = (
                (workload_decision.get((candidate.workload, cost_decision)), 2,
                 "correlate_workload_cost_decision", f"{candidate.workload}:{cost_decision}"),
                (decision_only.get(cost_decision), self.config.min_support,
                 "correlate_cost_decision", cost_decision),
            )
            threshold = self.config.fp_guard_threshold
        else:
            # For false-negative recovery, increasingly broad structural
            # priors precede the rule-wide fallback.  The decision-agnostic
            # motif is intentional: it transfers evidence for the same SQL
            # construct when the cost estimator flips sides of 1.0.
            chain = (
                (workload_motif.get((candidate.workload, cost_decision, motif)), 3,
                 "correlate_workload_cost_motif", f"{candidate.workload}:{cost_decision}:{motif}"),
                (decision_motif.get((cost_decision, motif)), 3,
                 "correlate_cost_motif", f"{cost_decision}:{motif}"),
                (motif_only.get(motif), 3,
                 "correlate_motif_transfer", motif),
                (workload_decision.get((candidate.workload, cost_decision)), self.config.min_support,
                 "correlate_workload_cost_decision", f"{candidate.workload}:{cost_decision}"),
                (decision_only.get(cost_decision), self.config.min_support,
                 "correlate_cost_decision", cost_decision),
            )
            threshold = self.config.fn_recovery_threshold

        for entry, minimum, source, key in chain:
            if entry is None or entry.support < minimum:
                continue
            decision = "rewrite" if entry.reliability >= threshold else "original"
            return MVMDecision(
                decision=decision,
                reliability=entry.reliability,
                support=entry.support,
                prior_source=source,
                prior_key=key,
                reason=(
                    f"correlate-aware DCRP: cost={cost_decision}, motif={motif}, "
                    f"reliability={entry.reliability:.3f}, threshold={threshold:.3f}, "
                    f"support={entry.support}"
                ),
                mean_latency_improvement_pct=entry.mean_improvement_pct,
                median_latency_improvement_pct=entry.median_improvement_pct,
            )

        fallback = self.base_policy.decide_detailed(candidate)
        return MVMDecision(
            decision=fallback.decision,
            reliability=fallback.reliability,
            support=fallback.support,
            prior_source=f"correlate_fallback:{fallback.prior_source}",
            prior_key=fallback.prior_key,
            reason=f"no supported correlate-aware bucket; {fallback.reason}",
            mean_latency_improvement_pct=fallback.mean_latency_improvement_pct,
            median_latency_improvement_pct=fallback.median_latency_improvement_pct,
        )
