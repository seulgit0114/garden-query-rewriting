"""Shared types for candidate-selection policies.

These types are the single contract shared by the offline evaluator
(`experiments/mvm_reliability`) and any future online adapter inside the
live pipeline. Neither side should redefine its own copy of a candidate or
decision shape -- both import from here.
"""

from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateFeatures:
    """Everything a selection policy is allowed to see for one rewrite
    candidate.

    This type intentionally excludes actual latency and ground-truth
    labels. A policy that only receives ``CandidateFeatures`` objects
    cannot leak test-time answers into its decision, by construction.
    """

    candidate_id: str
    workload: str
    query: str
    query_family: str
    rules: t.Tuple[str, ...]
    original_cost: float
    rewritten_cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple):
            object.__setattr__(self, 'rules', tuple(self.rules))
        for name, value in (
            ('original_cost', self.original_cost),
            ('rewritten_cost', self.rewritten_cost),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'CandidateFeatures.{name} must be a finite number, got {value!r}')


@dataclass(frozen=True)
class PriorLookup:
    """Provenance-carrying result of a prior lookup for one rule/sequence key."""

    reliability: float
    support: int
    source: str  # 'rule_sequence' | 'rule' | 'workload_global' | 'global' | 'none'
    key: t.Optional[str]
    fallback_reason: str
    # Train-only mean (original_latency_ms - rewritten_latency_ms) / original_latency_ms
    # * 100 for the same entry `reliability`/`support` came from (see
    # PriorEntry.mean_latency_improvement_pct in candidate_selection/mvm/prior.py).
    # None for the 'none' source (no train data at all). Optional and
    # additive so existing callers that only read reliability/support are
    # unaffected -- added for latency-aware group selection (see
    # experiments/mvm_reliability/conditional_evaluate.py::_select_group_latency_aware).
    mean_latency_improvement_pct: t.Optional[float] = None
    # Median counterpart, robust to a handful of severe-failure rows
    # dragging the mean down (see PriorEntry.median_latency_improvement_pct
    # in candidate_selection/mvm/prior.py).
    median_latency_improvement_pct: t.Optional[float] = None


@dataclass(frozen=True)
class MVMDecision:
    """Output of MVMPolicy.decide() for a single candidate."""

    decision: str  # 'rewrite' | 'original'
    reliability: t.Optional[float]
    support: t.Optional[int]
    prior_source: str
    prior_key: t.Optional[str]
    reason: str
    # Carried through from the PriorLookup that produced this decision;
    # None when no prior evidence was available. See PriorLookup above.
    mean_latency_improvement_pct: t.Optional[float] = None
    median_latency_improvement_pct: t.Optional[float] = None


class SelectionPolicy(t.Protocol):
    """Protocol both CostOnlyPolicy and MVMPolicy satisfy.

    ``decide`` is intentionally a per-candidate binary call (rewrite vs.
    original) rather than a whole-group ranking call. Group-level
    (multi-candidate) selection is composed on top of per-candidate
    decisions by the caller (see experiments/mvm_reliability/evaluate.py
    and, in the future, candidate_selection/mvm/adapter.py) so that the
    exact same per-candidate decision function can be reused unchanged
    whether the pipeline is comparing one rewrite against the original or
    several rewrites at once.
    """

    def decide(self, candidate: CandidateFeatures) -> str:
        ...
