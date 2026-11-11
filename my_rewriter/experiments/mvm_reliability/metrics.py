"""Aggregate metrics over an EvaluationResult.

Two independent views are computed:

  * candidate-level confusion metrics (TP/FP/FN/TN, accuracy, FP
    prevented, FN recovered, newly broken decisions) -- one row per
    logged rewrite candidate, using the same TP/FP/FN/TN definition as
    the source dataset (decision='rewrite' + actual_positive -> TP, etc.)
    but reclassified under each policy's own decision.
  * group-level latency metrics (selected/oracle/original latency
    sum/mean/median/p75/p95) -- one row per original query, since summing
    per-candidate latencies would double count queries with multiple
    logged rewrite candidates.
"""

from __future__ import annotations

import math
import statistics
import typing as t
from collections import Counter, defaultdict

from .evaluate import CandidateDecision, EvaluationResult, GroupResult

CONFUSION_LABELS = ('TP', 'FP', 'FN', 'TN')
SUPPORT_BUCKETS = ((0, 0), (1, 4), (5, 9), (10, 19), (20, math.inf))


def classify(decision: str, actual_positive: bool) -> str:
    if decision == 'rewrite':
        return 'TP' if actual_positive else 'FP'
    return 'FN' if actual_positive else 'TN'


def _confusion_counts(labels: t.Iterable[str]) -> t.Dict[str, int]:
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in CONFUSION_LABELS}


def _accuracy(counts: t.Dict[str, int]) -> t.Optional[float]:
    total = sum(counts.values())
    if total == 0:
        return None
    return (counts['TP'] + counts['TN']) / total


def summarize_candidates(decisions: t.Sequence[CandidateDecision]) -> dict:
    if not decisions:
        return {
            'candidate_count': 0,
            'cost_only': {'confusion': _confusion_counts([]), 'accuracy': None},
            'mvm': {'confusion': _confusion_counts([]), 'accuracy': None},
            'fp_prevented': {'count': 0, 'of_cost_fp': 0, 'rate': None},
            'fn_recovered': {'count': 0, 'of_cost_fn': 0, 'rate': None},
            'broken_correct_decisions': 0,
            'new_fp': 0,
            'new_fn': 0,
        }

    cost_labels = [classify(d.cost_decision, d.actual_positive) for d in decisions]
    mvm_labels = [classify(d.mvm_decision, d.actual_positive) for d in decisions]
    cost_counts = _confusion_counts(cost_labels)
    mvm_counts = _confusion_counts(mvm_labels)

    fp_total = cost_counts['FP']
    fp_prevented = sum(
        1 for d, cl in zip(decisions, cost_labels) if cl == 'FP' and d.mvm_decision == 'original'
    )
    fn_total = cost_counts['FN']
    fn_recovered = sum(
        1 for d, cl in zip(decisions, cost_labels) if cl == 'FN' and d.mvm_decision == 'rewrite'
    )
    broken_correct = sum(
        1 for d, cl in zip(decisions, cost_labels) if cl in ('TP', 'TN') and d.changed_from_cost
    )
    new_fp = sum(1 for cl, ml in zip(cost_labels, mvm_labels) if ml == 'FP' and cl != 'FP')
    new_fn = sum(1 for cl, ml in zip(cost_labels, mvm_labels) if ml == 'FN' and cl != 'FN')

    return {
        'candidate_count': len(decisions),
        'cost_only': {'confusion': cost_counts, 'accuracy': _accuracy(cost_counts)},
        'mvm': {'confusion': mvm_counts, 'accuracy': _accuracy(mvm_counts)},
        'fp_prevented': {
            'count': fp_prevented,
            'of_cost_fp': fp_total,
            'rate': (fp_prevented / fp_total) if fp_total else None,
        },
        'fn_recovered': {
            'count': fn_recovered,
            'of_cost_fn': fn_total,
            'rate': (fn_recovered / fn_total) if fn_total else None,
        },
        'broken_correct_decisions': broken_correct,
        'new_fp': new_fp,
        'new_fn': new_fn,
    }


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
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p75': _percentile(ordered, 0.75),
        'p95': _percentile(ordered, 0.95),
    }


def summarize_groups(groups: t.Sequence[GroupResult]) -> dict:
    cost_lat = [g.cost_selected_latency_ms for g in groups]
    mvm_lat = [g.mvm_selected_latency_ms for g in groups]
    original_lat = [g.original_latency_ms for g in groups]
    oracle_lat = [g.oracle_latency_ms for g in groups]

    cost_vs_mvm_delta = [c - m for c, m in zip(cost_lat, mvm_lat)]  # positive => MVM faster
    original_vs_mvm_improvement_pct = [
        ((o - m) / o * 100.0) if o > 0 else 0.0 for o, m in zip(original_lat, mvm_lat)
    ]

    gap_available = [c - o for c, o in zip(cost_lat, oracle_lat)]  # oracle's headroom over cost-only
    total_gap_available = sum(g for g in gap_available if g > 0)
    total_gap_recovered = sum(cost_vs_mvm_delta)
    oracle_recovered_rate = (
        (total_gap_recovered / total_gap_available) if total_gap_available > 0 else None
    )

    return {
        'group_count': len(groups),
        'cost_selected_latency_ms': _latency_stats(cost_lat),
        'mvm_selected_latency_ms': _latency_stats(mvm_lat),
        'original_only_latency_ms': _latency_stats(original_lat),
        'oracle_latency_ms': _latency_stats(oracle_lat),
        'cost_vs_mvm_delta_ms': _latency_stats(cost_vs_mvm_delta),
        'original_vs_mvm_improvement_pct': _latency_stats(original_vs_mvm_improvement_pct),
        'oracle_gap_available_ms': total_gap_available,
        'oracle_gap_recovered_ms': total_gap_recovered,
        'oracle_gap_recovered_rate': oracle_recovered_rate,
    }


def _support_bucket(support: t.Optional[int]) -> str:
    if support is None:
        return 'none'
    for lo, hi in SUPPORT_BUCKETS:
        if lo <= support <= hi:
            return f'{lo}-{hi}' if math.isfinite(hi) else f'{lo}+'
    return 'unknown'


def breakdown_by(
    decisions: t.Sequence[CandidateDecision], key_fn: t.Callable[[CandidateDecision], t.Sequence[str]]
) -> t.Dict[str, dict]:
    """Bucket candidate decisions by one or more keys per decision (e.g. a
    decision with 3 rules contributes to 3 rule buckets, matching the
    per-rule prior aggregation convention)."""
    buckets: t.Dict[str, list] = defaultdict(list)
    for decision in decisions:
        for key in key_fn(decision):
            buckets[key].append(decision)
    return {key: summarize_candidates(rows) for key, rows in buckets.items()}


def summarize_result(result: EvaluationResult) -> dict:
    decisions = result.all_candidate_decisions
    groups = list(result.groups)

    by_workload_candidates = breakdown_by(decisions, lambda d: (d.workload,))
    by_rule = breakdown_by(decisions, lambda d: tuple(dict.fromkeys(d.rules)) or ('(no_rule)',))
    by_sequence = breakdown_by(
        decisions, lambda d: (' > '.join(d.rules) if d.rules else '(no_rule)',)
    )
    by_support_bucket = breakdown_by(decisions, lambda d: (_support_bucket(d.support),))

    groups_by_workload: t.Dict[str, list] = defaultdict(list)
    for group in groups:
        groups_by_workload[group.workload].append(group)
    by_workload_groups = {wl: summarize_groups(rows) for wl, rows in groups_by_workload.items()}

    return {
        'overall': {
            'candidates': summarize_candidates(decisions),
            'groups': summarize_groups(groups),
        },
        'by_workload': {
            'candidates': by_workload_candidates,
            'groups': by_workload_groups,
        },
        'by_rule': by_rule,
        'by_rule_sequence': by_sequence,
        'by_prior_support_bucket': by_support_bucket,
    }
