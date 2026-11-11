"""MVM reliability policy: cost decision, corrected by historical prior.

Deliberately not a learned/weighted model. It does exactly two things to
the cost-only decision:

  * FP guard   -- cost says 'rewrite', but the matching prior's
                  reliability is below ``fp_guard_threshold`` -> override
                  to 'original' (this is the case the offline analysis
                  calls a False Positive: lower cost, but historically
                  the rewrite tends not to actually be faster).
  * FN recovery -- cost says 'original', but the matching prior's
                  reliability is at or above ``fn_recovery_threshold`` ->
                  override to 'rewrite' (this is the False Negative case:
                  cost looks non-improving, but this rule/sequence has
                  historically delivered a real latency win anyway).

Everywhere else the cost-only decision is kept unchanged. When the prior
lookup falls all the way through to 'none' (no train data whatsoever),
MVM also keeps the cost-only decision -- this is the safe-by-default
behavior required so that missing data degrades to current pipeline
behavior rather than an arbitrary guess.

Only two thresholds and one min_support cutoff are used; there is no
additional hidden weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import CandidateFeatures, MVMDecision, SelectionPolicy
from ..cost_policy import CostOnlyPolicy
from .prior import PriorTable


@dataclass(frozen=True)
class MVMConfig:
    alpha: float = 1.0
    beta: float = 1.0
    min_support: int = 5
    fp_guard_threshold: float = 0.4
    fn_recovery_threshold: float = 0.6
    cost_strict: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.fp_guard_threshold <= self.fn_recovery_threshold <= 1.0):
            raise ValueError(
                'MVMConfig requires 0 <= fp_guard_threshold <= fn_recovery_threshold <= 1, '
                f'got fp_guard_threshold={self.fp_guard_threshold}, '
                f'fn_recovery_threshold={self.fn_recovery_threshold}'
            )
        if self.min_support < 0:
            raise ValueError(f'MVMConfig.min_support must be >= 0, got {self.min_support}')

    @classmethod
    def from_dict(cls, data: dict) -> 'MVMConfig':
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


class MVMPolicy(SelectionPolicy):
    def __init__(self, prior_table: PriorTable, config: MVMConfig, cost_policy: CostOnlyPolicy = None) -> None:
        self.prior_table = prior_table
        self.config = config
        self.cost_policy = cost_policy or CostOnlyPolicy(strict=config.cost_strict)

    def decide(self, candidate: CandidateFeatures) -> str:
        return self.decide_detailed(candidate).decision

    def decide_detailed(self, candidate: CandidateFeatures) -> MVMDecision:
        cost_decision = self.cost_policy.decide(candidate)
        prior = self.prior_table.lookup(candidate.rules, candidate.workload, self.config.min_support)

        if prior.source == 'none':
            return MVMDecision(
                decision=cost_decision,
                reliability=None,
                support=0,
                prior_source='none',
                prior_key=None,
                reason=f"no prior evidence available ({prior.fallback_reason}); "
                       f"keeping cost decision '{cost_decision}'",
                mean_latency_improvement_pct=None,
                median_latency_improvement_pct=None,
            )

        if cost_decision == 'rewrite' and prior.reliability < self.config.fp_guard_threshold:
            decision = 'original'
            reason = (
                f"fp_guard: cost recommended 'rewrite' but reliability {prior.reliability:.3f} "
                f"< fp_guard_threshold {self.config.fp_guard_threshold} "
                f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
            )
        elif cost_decision == 'original' and prior.reliability >= self.config.fn_recovery_threshold:
            decision = 'rewrite'
            reason = (
                f"fn_recovery: cost recommended 'original' but reliability {prior.reliability:.3f} "
                f">= fn_recovery_threshold {self.config.fn_recovery_threshold} "
                f"(source={prior.source}, key={prior.key!r}, support={prior.support})"
            )
        else:
            decision = cost_decision
            reason = (
                f"cost decision retained: reliability {prior.reliability:.3f} does not cross "
                f"fp_guard({self.config.fp_guard_threshold})/fn_recovery({self.config.fn_recovery_threshold}) "
                f"thresholds (source={prior.source}, key={prior.key!r}, support={prior.support})"
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
