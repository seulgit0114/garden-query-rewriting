"""Cost-decision-conditioned MVM reliability policy (the MVE from the
experiment design review).

Motivation: ``PriorTable.build()`` (prior.py) pools ALL train rows for a
rule/sequence into one unconditional ``P(actual_positive | rule)`` value,
and ``MVMPolicy.decide_detailed()`` (policy.py) reuses that same value for
both ``fp_guard`` (only ever evaluated when cost already said 'rewrite')
and ``fn_recovery`` (only ever evaluated when cost already said
'original'). Those are two different conditional distributions --
``P(actual_positive | rule, cost=accept)`` and
``P(actual_positive | rule, cost=reject)`` -- and pooling them can dilute
or invert the signal actually relevant to each guard.

Concretely reproduced case (leave-one-correlate-family-out,
dsb/query010 held out): FILTER_SUB_QUERY_TO_CORRELATE's only
"cost accepted" train instance elsewhere was a true positive (reliability
1.0 conditioned on accept), but the pooled/reject-heavy unconditional
reliability (~0.14-0.21, driven by the 30+ "cost rejected" rows) is what
``fp_guard`` actually saw, and it overrode a correct 'rewrite' decision.

``ConditionalMVMPolicy`` fixes this by building two independent
``PriorTable`` instances from a cost-decision partition of the same train
rows, and using the accept-side table only for ``fp_guard`` and the
reject-side table only for ``fn_recovery``. Everything else (fallback
chain, smoothing, min_support, safe-by-default when no evidence) is
unchanged -- reuses ``PriorTable`` from prior.py as-is, no duplicated
prior-computation logic.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from ..cost_policy import CostOnlyPolicy
from ..interfaces import CandidateFeatures, MVMDecision, SelectionPolicy
from .policy import MVMConfig
from .prior import PriorTable


def _as_features(row: t.Any) -> CandidateFeatures:
    return CandidateFeatures(
        candidate_id=getattr(row, 'id', ''),
        workload=row.workload,
        query=getattr(row, 'query', ''),
        query_family=getattr(row, 'query_family', ''),
        rules=row.rules,
        original_cost=row.original_cost,
        rewritten_cost=row.rewritten_cost,
    )


def partition_by_cost_decision(
    train_rows: t.Sequence[t.Any], cost_policy: CostOnlyPolicy
) -> t.Tuple[t.List[t.Any], t.List[t.Any]]:
    """Split train rows (CandidateRecord-like) into (accept_rows,
    reject_rows) using the *same* cost decision function the runtime
    policy uses, so the conditioning is exactly consistent with what
    fp_guard/fn_recovery will see at decision time."""
    accept: t.List[t.Any] = []
    reject: t.List[t.Any] = []
    for row in train_rows:
        bucket = accept if cost_policy.decide(_as_features(row)) == 'rewrite' else reject
        bucket.append(row)
    return accept, reject


@dataclass(frozen=True)
class ConditionalMVMPolicy(SelectionPolicy):
    accept_prior_table: PriorTable
    reject_prior_table: PriorTable
    config: MVMConfig
    cost_policy: CostOnlyPolicy
    accept_train_count: int
    reject_train_count: int

    @classmethod
    def build(
        cls,
        train_rows: t.Sequence[t.Any],
        config: MVMConfig,
        cost_policy: t.Optional[CostOnlyPolicy] = None,
    ) -> 'ConditionalMVMPolicy':
        cost_policy = cost_policy or CostOnlyPolicy(strict=config.cost_strict)
        accept_rows, reject_rows = partition_by_cost_decision(train_rows, cost_policy)
        accept_table = PriorTable.build(accept_rows, alpha=config.alpha, beta=config.beta)
        reject_table = PriorTable.build(reject_rows, alpha=config.alpha, beta=config.beta)
        return cls(
            accept_prior_table=accept_table,
            reject_prior_table=reject_table,
            config=config,
            cost_policy=cost_policy,
            accept_train_count=len(accept_rows),
            reject_train_count=len(reject_rows),
        )

    def decide(self, candidate: CandidateFeatures) -> str:
        return self.decide_detailed(candidate).decision

    def decide_detailed(self, candidate: CandidateFeatures) -> MVMDecision:
        cost_decision = self.cost_policy.decide(candidate)

        if cost_decision == 'rewrite':
            prior = self.accept_prior_table.lookup(candidate.rules, candidate.workload, self.config.min_support)
            guard_name, threshold = 'fp_guard', self.config.fp_guard_threshold
            if prior.source == 'none':
                return MVMDecision(
                    decision=cost_decision, reliability=None, support=0,
                    prior_source='none', prior_key=None,
                    reason=f"no accept-conditional prior evidence ({prior.fallback_reason}); "
                           f"keeping cost decision '{cost_decision}'",
                    mean_latency_improvement_pct=None,
                    median_latency_improvement_pct=None,
                )
            if prior.reliability < threshold:
                decision = 'original'
                reason = (
                    f"{guard_name}(conditional): cost recommended 'rewrite' but accept-conditional "
                    f"reliability {prior.reliability:.3f} < fp_guard_threshold {threshold} "
                    f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
                )
            else:
                decision = cost_decision
                reason = (
                    f"cost decision retained: accept-conditional reliability {prior.reliability:.3f} "
                    f">= fp_guard_threshold {threshold} "
                    f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
                )
        else:
            prior = self.reject_prior_table.lookup(candidate.rules, candidate.workload, self.config.min_support)
            threshold = self.config.fn_recovery_threshold
            if prior.source == 'none':
                return MVMDecision(
                    decision=cost_decision, reliability=None, support=0,
                    prior_source='none', prior_key=None,
                    reason=f"no reject-conditional prior evidence ({prior.fallback_reason}); "
                           f"keeping cost decision '{cost_decision}'",
                    mean_latency_improvement_pct=None,
                    median_latency_improvement_pct=None,
                )
            if prior.reliability >= threshold:
                decision = 'rewrite'
                reason = (
                    f"fn_recovery(conditional): cost recommended 'original' but reject-conditional "
                    f"reliability {prior.reliability:.3f} >= fn_recovery_threshold {threshold} "
                    f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
                )
            else:
                decision = cost_decision
                reason = (
                    f"cost decision retained: reject-conditional reliability {prior.reliability:.3f} "
                    f"< fn_recovery_threshold {threshold} "
                    f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
                )

        return MVMDecision(
            decision=decision,
            reliability=prior.reliability,
            support=prior.support,
            prior_source=prior.source,
            prior_key=prior.key,
            reason=reason,
            mean_latency_improvement_pct=prior.mean_latency_improvement_pct,
            median_latency_improvement_pct=prior.median_latency_improvement_pct,
        )
