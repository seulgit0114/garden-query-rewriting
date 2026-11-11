"""Offline replay control flow: Cost-only vs Cost+MVM, scored against
already-logged actual latency. No SQL execution or LLM calls happen here
-- every value used comes from mvm_v1/candidates.jsonl.

Decision generation (steps 1-4) only ever sees ``CandidateFeatures``,
which has no latency or label fields. Ground truth (``actual_positive``,
the two latency values) is attached to the trace record only after both
policies have already produced their decision -- see ``decide_candidate``.
"""

from __future__ import annotations

import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

from candidate_selection.cost_policy import CostOnlyPolicy
from candidate_selection.interfaces import CandidateFeatures
from candidate_selection.mvm.policy import MVMConfig, MVMPolicy
from candidate_selection.mvm.prior import PriorTable

from .dataset import filter_split, group_usable_by_query
from .schemas import CandidateRecord, SplitInfo


def to_features(record: CandidateRecord) -> CandidateFeatures:
    return CandidateFeatures(
        candidate_id=record.id,
        workload=record.workload,
        query=record.query,
        query_family=record.query_family,
        rules=record.rules,
        original_cost=record.original_cost,
        rewritten_cost=record.rewritten_cost,
    )


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    workload: str
    query: str
    query_family: str
    rules: t.Tuple[str, ...]
    original_cost: float
    rewritten_cost: float
    decision: str
    cost_decision: str
    mvm_decision: str
    reliability: t.Optional[float]
    support: t.Optional[int]
    prior_source: str
    prior_key: t.Optional[str]
    reason: str
    changed_from_cost: bool
    actual_positive: bool
    original_latency_ms: float
    rewritten_latency_ms: float

    def to_dict(self) -> dict:
        return {
            'candidate_id': self.candidate_id,
            'workload': self.workload,
            'query': self.query,
            'query_family': self.query_family,
            'rules': list(self.rules),
            'original_cost': self.original_cost,
            'rewritten_cost': self.rewritten_cost,
            'decision': self.decision,
            'cost_decision': self.cost_decision,
            'mvm_decision': self.mvm_decision,
            'reliability': self.reliability,
            'support': self.support,
            'prior_source': self.prior_source,
            'prior_key': self.prior_key,
            'reason': self.reason,
            'changed_from_cost': self.changed_from_cost,
            'actual_positive': self.actual_positive,
            'original_latency_ms': self.original_latency_ms,
            'rewritten_latency_ms': self.rewritten_latency_ms,
        }


def decide_candidate(record: CandidateRecord, cost_policy: CostOnlyPolicy, mvm_policy: MVMPolicy) -> CandidateDecision:
    """Steps 1-4: build policy-visible features, get both decisions, THEN
    attach ground truth for later scoring."""
    features = to_features(record)
    cost_decision = cost_policy.decide(features)
    mvm_result = mvm_policy.decide_detailed(features)
    return CandidateDecision(
        candidate_id=record.id,
        workload=record.workload,
        query=record.query,
        query_family=record.query_family,
        rules=record.rules,
        original_cost=record.original_cost,
        rewritten_cost=record.rewritten_cost,
        decision=mvm_result.decision,
        cost_decision=cost_decision,
        mvm_decision=mvm_result.decision,
        reliability=mvm_result.reliability,
        support=mvm_result.support,
        prior_source=mvm_result.prior_source,
        prior_key=mvm_result.prior_key,
        reason=mvm_result.reason,
        changed_from_cost=(cost_decision != mvm_result.decision),
        actual_positive=record.rewritten_latency_ms < record.original_latency_ms,
        original_latency_ms=record.original_latency_ms,
        rewritten_latency_ms=record.rewritten_latency_ms,
    )


def select_group_cost_only(decisions: t.Sequence[CandidateDecision]) -> t.Optional[CandidateDecision]:
    """Among candidates the cost-only policy would accept, pick the one
    with the lowest rewritten cost (mirrors _select_best_candidate()'s
    min-cost rule in portfolio_rewrite.py). None means 'keep original'."""
    accepted = [d for d in decisions if d.cost_decision == 'rewrite']
    if not accepted:
        return None
    return min(accepted, key=lambda d: d.rewritten_cost)


def select_group_mvm(decisions: t.Sequence[CandidateDecision]) -> t.Optional[CandidateDecision]:
    """Among candidates MVM would accept, pick the one with the highest
    prior reliability, breaking ties by lower cost. None means 'keep
    original'."""
    accepted = [d for d in decisions if d.mvm_decision == 'rewrite']
    if not accepted:
        return None
    return max(accepted, key=lambda d: (d.reliability if d.reliability is not None else 0.0, -d.rewritten_cost))


def select_oracle(decisions: t.Sequence[CandidateDecision], group_original_latency_ms: float) -> t.Optional[CandidateDecision]:
    """The candidate with the lowest actual rewritten latency, if it beats
    the group's original latency; otherwise None (original wins)."""
    best = min(decisions, key=lambda d: d.rewritten_latency_ms)
    return best if best.rewritten_latency_ms < group_original_latency_ms else None


@dataclass(frozen=True)
class GroupResult:
    workload: str
    query: str
    query_family: str
    candidates: t.Tuple[CandidateDecision, ...]
    original_latency_ms: float
    cost_selected_id: t.Optional[str]
    mvm_selected_id: t.Optional[str]
    oracle_selected_id: t.Optional[str]
    cost_selected_latency_ms: float
    mvm_selected_latency_ms: float
    oracle_latency_ms: float


def build_group_result(
    workload: str,
    query: str,
    query_family: str,
    records: t.Sequence[CandidateRecord],
    cost_policy: CostOnlyPolicy,
    mvm_policy: MVMPolicy,
) -> GroupResult:
    decisions = [decide_candidate(r, cost_policy, mvm_policy) for r in records]
    # All rows share the same original_sql; original_latency_ms is
    # measured independently per logged row (different ablation-arm runs),
    # so the group-level estimate is the mean across those measurements.
    original_latency_ms = sum(d.original_latency_ms for d in decisions) / len(decisions)

    cost_pick = select_group_cost_only(decisions)
    mvm_pick = select_group_mvm(decisions)
    oracle_pick = select_oracle(decisions, original_latency_ms)

    return GroupResult(
        workload=workload,
        query=query,
        query_family=query_family,
        candidates=tuple(decisions),
        original_latency_ms=original_latency_ms,
        cost_selected_id=cost_pick.candidate_id if cost_pick else None,
        mvm_selected_id=mvm_pick.candidate_id if mvm_pick else None,
        oracle_selected_id=oracle_pick.candidate_id if oracle_pick else None,
        cost_selected_latency_ms=cost_pick.rewritten_latency_ms if cost_pick else original_latency_ms,
        mvm_selected_latency_ms=mvm_pick.rewritten_latency_ms if mvm_pick else original_latency_ms,
        oracle_latency_ms=oracle_pick.rewritten_latency_ms if oracle_pick else original_latency_ms,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    cost_strict: bool = True
    mvm: MVMConfig = field(default_factory=MVMConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        cost_strict = bool(data.get('cost_policy', {}).get('strict', True))
        mvm_config = MVMConfig.from_dict(data.get('mvm_policy', {}))
        return cls(cost_strict=cost_strict, mvm=mvm_config)


@dataclass(frozen=True)
class EvaluationResult:
    config: ExperimentConfig
    prior_table: PriorTable
    train_candidate_count: int
    groups: t.Tuple[GroupResult, ...]

    @property
    def all_candidate_decisions(self) -> t.List[CandidateDecision]:
        out: t.List[CandidateDecision] = []
        for group in self.groups:
            out.extend(group.candidates)
        return out


def select_pilot_families(
    records: t.Sequence[CandidateRecord], split: SplitInfo, families_per_workload: int = 1
) -> t.Dict[str, t.List[str]]:
    """Deterministically pick a small set of TEST query families for a
    pilot run, using only structural information (per-family usable test
    candidate *counts*) -- never test latency or label values. Families
    with more logged candidates are preferred (more chance of covering
    multiple TP/FP/FN/TN outcomes), ties broken by family name for
    determinism. A workload contributes no family if it has zero usable
    test candidates."""
    counts: t.Dict[t.Tuple[str, str], int] = defaultdict(int)
    for record in records:
        if record.usable and split.split_of(record.workload, record.query_family) == 'test':
            counts[(record.workload, record.query_family)] += 1

    selected: t.Dict[str, t.List[str]] = {}
    for workload, families in split.test_families.items():
        ranked = sorted(
            ((fam, counts.get((workload, fam), 0)) for fam in families),
            key=lambda pair: (-pair[1], pair[0]),
        )
        selected[workload] = [fam for fam, cnt in ranked[:families_per_workload] if cnt > 0]
    return selected


def run_evaluation(
    records: t.Sequence[CandidateRecord],
    split: SplitInfo,
    config: ExperimentConfig,
    test_family_filter: t.Optional[t.Dict[str, t.Sequence[str]]] = None,
) -> EvaluationResult:
    """Full control flow, test split only:

    1. train_rows = usable train-split candidates (train latency/label
       used here, never test).
    2. prior_table computed from train_rows only.
    3. cost_policy / mvm_policy constructed (mvm_policy internally reuses
       the *same* cost_policy instance for its own cost_decision, so
       'changed_from_cost' compares against the identical baseline).
    4. test candidates grouped by original query, optionally restricted to
       `test_family_filter` (workload -> allowed query_family list) for a
       small pilot run. The filter is applied on family *names* only
       (e.g. from select_pilot_families), never on test latency/label.
    5. each group replayed: per-candidate decisions, then group-level
       cost-only / mvm / oracle selection, then latency ground truth
       attached for scoring only.
    """
    train_rows = filter_split(records, split, 'train')
    prior_table = PriorTable.build(train_rows, alpha=config.mvm.alpha, beta=config.mvm.beta)

    cost_policy = CostOnlyPolicy(strict=config.cost_strict)
    mvm_policy = MVMPolicy(prior_table, config.mvm, cost_policy=cost_policy)

    test_records = filter_split(records, split, 'test')
    if test_family_filter is not None:
        test_records = [
            r for r in test_records if r.query_family in set(test_family_filter.get(r.workload, ()))
        ]
    test_groups = group_usable_by_query(test_records)

    group_results = []
    for (workload, query), group_records in sorted(test_groups.items()):
        query_family = group_records[0].query_family
        group_results.append(
            build_group_result(workload, query, query_family, group_records, cost_policy, mvm_policy)
        )

    return EvaluationResult(
        config=config,
        prior_table=prior_table,
        train_candidate_count=len(train_rows),
        groups=tuple(group_results),
    )
