"""MVE control flow: Cost-only vs global MVM vs cost-decision-conditional
MVM, scored against already-logged actual latency. Mirrors
evaluate.py's replay design (no SQL/LLM calls, ground truth attached only
after all three decisions are made) but adds the conditional policy and
records the two plan-free residual definitions (A, B) from the experiment
design review for descriptive analysis only -- they are never fed back
into any decision.

Residual definitions (see experiments/mvm_reliability/outputs/.../report.md
of the design-review conversation for the derivation):
  A. log-ratio residual:  log(L_o / L_r) - log(C_o / C_r)
  B. cost-decision-conditioned signed residual: (L_o - L_r) / L_o,
     always reported together with cost_decision so consumers can split
     it into the fp_guard population (cost='rewrite') and the
     fn_recovery population (cost='original') rather than pooling them.
Both use only original_cost/rewritten_cost/original_latency_ms/
rewritten_latency_ms, which are already in candidates.jsonl -- no plan
features required, and both are computed from the *train* row's own
already-known latency for descriptive/reporting purposes only, never used
as a decision input.
"""

from __future__ import annotations

import math
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass

from candidate_selection.cost_policy import CostOnlyPolicy
from candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
from candidate_selection.mvm.policy import MVMConfig, MVMPolicy
from candidate_selection.mvm.prior import PriorTable

from .dataset import filter_split, group_usable_by_query
from .evaluate import ExperimentConfig, select_oracle, to_features
from .schemas import CandidateRecord, SplitInfo


def residual_log_ratio(record: CandidateRecord) -> t.Optional[float]:
    """Definition A. None when any value is non-positive (log undefined)."""
    lo, lr, co, cr = record.original_latency_ms, record.rewritten_latency_ms, record.original_cost, record.rewritten_cost
    if not all(isinstance(v, (int, float)) and v > 0 for v in (lo, lr, co, cr)):
        return None
    return math.log(lo / lr) - math.log(co / cr)


def residual_signed(record: CandidateRecord) -> t.Optional[float]:
    """Definition B. None when original_latency_ms is non-positive."""
    lo, lr = record.original_latency_ms, record.rewritten_latency_ms
    if not (isinstance(lo, (int, float)) and lo > 0):
        return None
    return (lo - lr) / lo


@dataclass(frozen=True)
class MultiCandidateDecision:
    candidate_id: str
    workload: str
    query: str
    query_family: str
    rules: t.Tuple[str, ...]
    original_cost: float
    rewritten_cost: float
    cost_decision: str
    global_mvm_decision: str
    global_mvm_reliability: t.Optional[float]
    global_mvm_support: t.Optional[int]
    global_mvm_source: str
    cond_mvm_decision: str
    cond_mvm_reliability: t.Optional[float]
    cond_mvm_support: t.Optional[int]
    cond_mvm_source: str
    cond_mvm_mean_latency_improvement_pct: t.Optional[float]
    cond_mvm_median_latency_improvement_pct: t.Optional[float]
    actual_positive: bool
    original_latency_ms: float
    rewritten_latency_ms: float
    residual_log_ratio: t.Optional[float]
    residual_signed: t.Optional[float]

    def to_dict(self) -> dict:
        return {
            'candidate_id': self.candidate_id,
            'workload': self.workload,
            'query': self.query,
            'query_family': self.query_family,
            'rules': list(self.rules),
            'original_cost': self.original_cost,
            'rewritten_cost': self.rewritten_cost,
            'cost_decision': self.cost_decision,
            'global_mvm_decision': self.global_mvm_decision,
            'global_mvm_reliability': self.global_mvm_reliability,
            'global_mvm_support': self.global_mvm_support,
            'global_mvm_source': self.global_mvm_source,
            'cond_mvm_decision': self.cond_mvm_decision,
            'cond_mvm_reliability': self.cond_mvm_reliability,
            'cond_mvm_support': self.cond_mvm_support,
            'cond_mvm_source': self.cond_mvm_source,
            'cond_mvm_mean_latency_improvement_pct': self.cond_mvm_mean_latency_improvement_pct,
            'cond_mvm_median_latency_improvement_pct': self.cond_mvm_median_latency_improvement_pct,
            'actual_positive': self.actual_positive,
            'original_latency_ms': self.original_latency_ms,
            'rewritten_latency_ms': self.rewritten_latency_ms,
            'residual_log_ratio': self.residual_log_ratio,
            'residual_signed': self.residual_signed,
        }


def decide_candidate_multi(
    record: CandidateRecord,
    cost_policy: CostOnlyPolicy,
    global_mvm_policy: MVMPolicy,
    cond_mvm_policy: ConditionalMVMPolicy,
) -> MultiCandidateDecision:
    features = to_features(record)
    cost_decision = cost_policy.decide(features)
    global_result = global_mvm_policy.decide_detailed(features)
    cond_result = cond_mvm_policy.decide_detailed(features)
    return MultiCandidateDecision(
        candidate_id=record.id,
        workload=record.workload,
        query=record.query,
        query_family=record.query_family,
        rules=record.rules,
        original_cost=record.original_cost,
        rewritten_cost=record.rewritten_cost,
        cost_decision=cost_decision,
        global_mvm_decision=global_result.decision,
        global_mvm_reliability=global_result.reliability,
        global_mvm_support=global_result.support,
        global_mvm_source=global_result.prior_source,
        cond_mvm_decision=cond_result.decision,
        cond_mvm_reliability=cond_result.reliability,
        cond_mvm_support=cond_result.support,
        cond_mvm_source=cond_result.prior_source,
        cond_mvm_mean_latency_improvement_pct=cond_result.mean_latency_improvement_pct,
        cond_mvm_median_latency_improvement_pct=cond_result.median_latency_improvement_pct,
        actual_positive=record.rewritten_latency_ms < record.original_latency_ms,
        original_latency_ms=record.original_latency_ms,
        rewritten_latency_ms=record.rewritten_latency_ms,
        residual_log_ratio=residual_log_ratio(record),
        residual_signed=residual_signed(record),
    )


def _select_group(
    decisions: t.Sequence[MultiCandidateDecision], decision_attr: str, reliability_attr: t.Optional[str]
) -> t.Optional[MultiCandidateDecision]:
    accepted = [d for d in decisions if getattr(d, decision_attr) == 'rewrite']
    if not accepted:
        return None
    if reliability_attr is None:
        return min(accepted, key=lambda d: d.rewritten_cost)
    return max(accepted, key=lambda d: (getattr(d, reliability_attr) or 0.0, -d.rewritten_cost))


def _select_group_latency_aware(decisions: t.Sequence[MultiCandidateDecision]) -> t.Optional[MultiCandidateDecision]:
    """Among candidates cond_mvm accepts, rank by the train-only
    mean_latency_improvement_pct attached to the prior entry that was
    actually used for that candidate's decision (see PriorLookup /
    MVMDecision.mean_latency_improvement_pct) -- NOT by reliability. This
    is a train-only expectation of magnitude, never the candidate's own
    (unknown, test-time) actual latency, so it carries no leakage.
    Candidates with no such value (prior_source='none') fall back to
    reliability, then lowest cost, same as _select_group().

    Known failure mode (see tpch/query18_0 in the k-fold CV run): the
    MEAN can be dragged near/above 0 by a thin-support (e.g. n=10)
    sequence whose few observed rows happened to be mildly positive,
    while a much better-supported but higher-variance rule (severe
    occasional failures pulling its mean negative) is actually the
    stronger real choice. _select_group_latency_aware_median() below
    ranks by median instead, which is robust to that."""
    accepted = [d for d in decisions if d.cond_mvm_decision == 'rewrite']
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda d: (
            d.cond_mvm_mean_latency_improvement_pct is not None,
            d.cond_mvm_mean_latency_improvement_pct if d.cond_mvm_mean_latency_improvement_pct is not None else 0.0,
            d.cond_mvm_reliability or 0.0,
            -d.rewritten_cost,
        ),
    )


def _select_group_latency_aware_median(decisions: t.Sequence[MultiCandidateDecision]) -> t.Optional[MultiCandidateDecision]:
    """Same as _select_group_latency_aware() but ranks by
    median_latency_improvement_pct instead of the mean -- robust to a
    handful of severe-failure train rows dragging the mean down (or a
    handful of lucky rows dragging a thin-support mean up)."""
    accepted = [d for d in decisions if d.cond_mvm_decision == 'rewrite']
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda d: (
            d.cond_mvm_median_latency_improvement_pct is not None,
            d.cond_mvm_median_latency_improvement_pct if d.cond_mvm_median_latency_improvement_pct is not None else 0.0,
            d.cond_mvm_reliability or 0.0,
            -d.rewritten_cost,
        ),
    )


def _select_group_expected_value(decisions: t.Sequence[MultiCandidateDecision]) -> t.Optional[MultiCandidateDecision]:
    """Rank by reliability * median_latency_improvement_pct instead of
    magnitude alone. Motivation: pure-magnitude ranking
    (_select_group_latency_aware_median) can pick a thin-support entry
    whose median happens to be barely positive over a much better
    supported entry whose median is negative but whose reliability
    (P(actual_positive)) is high -- min_support-scan results showed this
    trade-off concretely: lowering min_support improved the k-fold
    latency metric but *hurt* LOFO accuracy (0.736 at min_support=5 vs
    0.705 at min_support<=1). Multiplying by reliability down-weights
    low-confidence magnitude estimates instead of ignoring confidence
    entirely. Still zero test-time leakage -- both factors come from the
    train-only prior lookup attached to this candidate's rule/sequence
    key, never from the candidate's own (unknown) actual latency."""
    accepted = [d for d in decisions if d.cond_mvm_decision == 'rewrite']
    if not accepted:
        return None

    def _score(d: MultiCandidateDecision) -> float:
        if d.cond_mvm_median_latency_improvement_pct is None or d.cond_mvm_reliability is None:
            return 0.0
        return d.cond_mvm_reliability * d.cond_mvm_median_latency_improvement_pct

    return max(
        accepted,
        key=lambda d: (
            d.cond_mvm_median_latency_improvement_pct is not None,
            _score(d),
            d.cond_mvm_reliability or 0.0,
            -d.rewritten_cost,
        ),
    )


@dataclass(frozen=True)
class ConditionalGroupResult:
    workload: str
    query: str
    query_family: str
    candidates: t.Tuple[MultiCandidateDecision, ...]
    original_latency_ms: float
    cost_selected_latency_ms: float
    global_mvm_selected_latency_ms: float
    cond_mvm_selected_latency_ms: float
    cond_mvm_latency_aware_selected_latency_ms: float
    cond_mvm_latency_aware_median_selected_latency_ms: float
    cond_mvm_expected_value_selected_latency_ms: float
    oracle_latency_ms: float


def build_conditional_group_result(
    workload: str,
    query: str,
    query_family: str,
    records: t.Sequence[CandidateRecord],
    cost_policy: CostOnlyPolicy,
    global_mvm_policy: MVMPolicy,
    cond_mvm_policy: ConditionalMVMPolicy,
) -> ConditionalGroupResult:
    decisions = [decide_candidate_multi(r, cost_policy, global_mvm_policy, cond_mvm_policy) for r in records]
    original_latency_ms = sum(d.original_latency_ms for d in decisions) / len(decisions)

    cost_pick = _select_group(decisions, 'cost_decision', None)
    global_pick = _select_group(decisions, 'global_mvm_decision', 'global_mvm_reliability')
    cond_pick = _select_group(decisions, 'cond_mvm_decision', 'cond_mvm_reliability')
    cond_latency_aware_pick = _select_group_latency_aware(decisions)
    cond_latency_aware_median_pick = _select_group_latency_aware_median(decisions)
    cond_expected_value_pick = _select_group_expected_value(decisions)

    # select_oracle() from evaluate.py expects objects with rewritten_latency_ms;
    # MultiCandidateDecision has it, so it is reused unchanged.
    oracle_pick = select_oracle(decisions, original_latency_ms)

    return ConditionalGroupResult(
        workload=workload,
        query=query,
        query_family=query_family,
        candidates=tuple(decisions),
        original_latency_ms=original_latency_ms,
        cost_selected_latency_ms=cost_pick.rewritten_latency_ms if cost_pick else original_latency_ms,
        global_mvm_selected_latency_ms=global_pick.rewritten_latency_ms if global_pick else original_latency_ms,
        cond_mvm_selected_latency_ms=cond_pick.rewritten_latency_ms if cond_pick else original_latency_ms,
        cond_mvm_latency_aware_selected_latency_ms=(
            cond_latency_aware_pick.rewritten_latency_ms if cond_latency_aware_pick else original_latency_ms
        ),
        cond_mvm_latency_aware_median_selected_latency_ms=(
            cond_latency_aware_median_pick.rewritten_latency_ms
            if cond_latency_aware_median_pick else original_latency_ms
        ),
        cond_mvm_expected_value_selected_latency_ms=(
            cond_expected_value_pick.rewritten_latency_ms if cond_expected_value_pick else original_latency_ms
        ),
        oracle_latency_ms=oracle_pick.rewritten_latency_ms if oracle_pick else original_latency_ms,
    )


@dataclass(frozen=True)
class MultiEvaluationResult:
    train_candidate_count: int
    accept_train_count: int
    reject_train_count: int
    groups: t.Tuple[ConditionalGroupResult, ...]

    @property
    def all_candidate_decisions(self) -> t.List[MultiCandidateDecision]:
        out: t.List[MultiCandidateDecision] = []
        for group in self.groups:
            out.extend(group.candidates)
        return out


def run_multi_evaluation(
    records: t.Sequence[CandidateRecord],
    split: SplitInfo,
    config: ExperimentConfig,
) -> MultiEvaluationResult:
    train_rows = filter_split(records, split, 'train')
    global_prior_table = PriorTable.build(train_rows, alpha=config.mvm.alpha, beta=config.mvm.beta)

    cost_policy = CostOnlyPolicy(strict=config.cost_strict)
    global_mvm_policy = MVMPolicy(global_prior_table, config.mvm, cost_policy=cost_policy)
    cond_mvm_policy = ConditionalMVMPolicy.build(train_rows, config.mvm, cost_policy=cost_policy)

    test_records = filter_split(records, split, 'test')
    test_groups = group_usable_by_query(test_records)

    group_results = []
    for (workload, query), group_records in sorted(test_groups.items()):
        query_family = group_records[0].query_family
        group_results.append(
            build_conditional_group_result(
                workload, query, query_family, group_records, cost_policy, global_mvm_policy, cond_mvm_policy
            )
        )

    return MultiEvaluationResult(
        train_candidate_count=len(train_rows),
        accept_train_count=cond_mvm_policy.accept_train_count,
        reject_train_count=cond_mvm_policy.reject_train_count,
        groups=tuple(group_results),
    )


CONFUSION_LABELS = ('TP', 'FP', 'FN', 'TN')


def _classify(decision: str, actual_positive: bool) -> str:
    if decision == 'rewrite':
        return 'TP' if actual_positive else 'FP'
    return 'FN' if actual_positive else 'TN'


def _confusion(labels: t.Iterable[str]) -> t.Dict[str, int]:
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in CONFUSION_LABELS}


def _accuracy(counts: t.Dict[str, int]) -> t.Optional[float]:
    total = sum(counts.values())
    return (counts['TP'] + counts['TN']) / total if total else None


def _precision_recall_f1(counts: t.Dict[str, int]) -> t.Dict[str, t.Optional[float]]:
    tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _policy_summary(decisions: t.Sequence[MultiCandidateDecision], decision_attr: str) -> dict:
    labels = [_classify(getattr(d, decision_attr), d.actual_positive) for d in decisions]
    counts = _confusion(labels)
    return {'confusion': counts, 'accuracy': _accuracy(counts), **_precision_recall_f1(counts)}


def _percentile(sorted_values: t.Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _latency_stats(values: t.Sequence[float]) -> dict:
    if not values:
        return {'count': 0, 'sum': 0.0, 'mean': None, 'median': None, 'p75': None, 'p95': None}
    ordered = sorted(values)
    return {
        'count': len(values),
        'sum': sum(values),
        'mean': sum(values) / len(values),
        'median': _percentile(ordered, 0.5),
        'p75': _percentile(ordered, 0.75),
        'p95': _percentile(ordered, 0.95),
    }


def _oracle_gap_recovered_rate(cost_lat, other_lat, oracle_lat) -> t.Optional[float]:
    gap_available = [c - o for c, o in zip(cost_lat, oracle_lat)]
    total_gap_available = sum(g for g in gap_available if g > 0)
    total_gap_recovered = sum(c - m for c, m in zip(cost_lat, other_lat))
    return (total_gap_recovered / total_gap_available) if total_gap_available > 0 else None


def summarize_multi(result: MultiEvaluationResult) -> dict:
    decisions = result.all_candidate_decisions
    groups = list(result.groups)

    cost_lat = [g.cost_selected_latency_ms for g in groups]
    global_lat = [g.global_mvm_selected_latency_ms for g in groups]
    cond_lat = [g.cond_mvm_selected_latency_ms for g in groups]
    cond_la_lat = [g.cond_mvm_latency_aware_selected_latency_ms for g in groups]
    cond_la_med_lat = [g.cond_mvm_latency_aware_median_selected_latency_ms for g in groups]
    cond_ev_lat = [g.cond_mvm_expected_value_selected_latency_ms for g in groups]
    original_lat = [g.original_latency_ms for g in groups]
    oracle_lat = [g.oracle_latency_ms for g in groups]

    return {
        'test_group_count': len(groups),
        'test_candidate_count': len(decisions),
        'train_candidate_count': result.train_candidate_count,
        'accept_train_count': result.accept_train_count,
        'reject_train_count': result.reject_train_count,
        'candidates': {
            'cost_only': _policy_summary(decisions, 'cost_decision'),
            'global_mvm': _policy_summary(decisions, 'global_mvm_decision'),
            'cond_mvm': _policy_summary(decisions, 'cond_mvm_decision'),
        },
        'groups': {
            'cost_selected_latency_ms': _latency_stats(cost_lat),
            'global_mvm_selected_latency_ms': _latency_stats(global_lat),
            'cond_mvm_selected_latency_ms': _latency_stats(cond_lat),
            'cond_mvm_latency_aware_selected_latency_ms': _latency_stats(cond_la_lat),
            'cond_mvm_latency_aware_median_selected_latency_ms': _latency_stats(cond_la_med_lat),
            'cond_mvm_expected_value_selected_latency_ms': _latency_stats(cond_ev_lat),
            'original_only_latency_ms': _latency_stats(original_lat),
            'oracle_latency_ms': _latency_stats(oracle_lat),
            'global_mvm_oracle_gap_recovered_rate': _oracle_gap_recovered_rate(cost_lat, global_lat, oracle_lat),
            'cond_mvm_oracle_gap_recovered_rate': _oracle_gap_recovered_rate(cost_lat, cond_lat, oracle_lat),
            'cond_mvm_latency_aware_oracle_gap_recovered_rate': _oracle_gap_recovered_rate(
                cost_lat, cond_la_lat, oracle_lat
            ),
            'cond_mvm_latency_aware_median_oracle_gap_recovered_rate': _oracle_gap_recovered_rate(
                cost_lat, cond_la_med_lat, oracle_lat
            ),
            'cond_mvm_expected_value_oracle_gap_recovered_rate': _oracle_gap_recovered_rate(
                cost_lat, cond_ev_lat, oracle_lat
            ),
        },
        # Raw per-group latency, kept for pooled tail-sensitivity analysis across folds
        # (median / trimmed-mean / top-contributor) -- see run_conditional_mve.py.
        'group_rows': [
            {
                'workload': g.workload, 'query': g.query, 'query_family': g.query_family,
                'original_latency_ms': g.original_latency_ms,
                'cost_selected_latency_ms': g.cost_selected_latency_ms,
                'global_mvm_selected_latency_ms': g.global_mvm_selected_latency_ms,
                'cond_mvm_selected_latency_ms': g.cond_mvm_selected_latency_ms,
                'cond_mvm_latency_aware_selected_latency_ms': g.cond_mvm_latency_aware_selected_latency_ms,
                'cond_mvm_latency_aware_median_selected_latency_ms': g.cond_mvm_latency_aware_median_selected_latency_ms,
                'cond_mvm_expected_value_selected_latency_ms': g.cond_mvm_expected_value_selected_latency_ms,
                'oracle_latency_ms': g.oracle_latency_ms,
            }
            for g in groups
        ],
    }
