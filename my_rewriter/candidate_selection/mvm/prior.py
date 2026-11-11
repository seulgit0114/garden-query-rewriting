"""MVM reliability prior: train-only statistics over rule / rule-sequence
history, with a deterministic fallback chain.

This module is the single source of truth for "how reliable has this rule
(or rule sequence) historically been at turning a lower estimated cost
into a lower actual latency". Both the offline evaluator
(``experiments/mvm_reliability``) and any future online adapter must
import ``PriorTable`` from here rather than reimplementing this
computation -- see candidate_selection/mvm/policy.py for the consumer.

Success definition (fixed, not configurable): a training row is
"positive" iff ``rewritten_latency_ms < original_latency_ms``. This is
computed directly from the two latency fields, never from the
precomputed ``label`` field, so the definition stays correct even if the
labeling policy upstream changes.
"""

from __future__ import annotations

import statistics
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..interfaces import PriorLookup

NO_RULE_KEY = '(no_rule)'


@dataclass(frozen=True)
class PriorEntry:
    count: int
    positive: int
    negative: int
    success_rate: float
    laplace_success_rate: float
    mean_latency_improvement_pct: float
    # Median of the same per-row improvement values as
    # mean_latency_improvement_pct. Added after a k-fold CV failure case
    # (tpch/query18_0) showed the mean statistic can be dominated by a
    # handful of severe-failure rows even when most rows for that
    # rule/sequence succeed -- median is robust to that and is what
    # latency-aware group selection should rank by (see
    # experiments/mvm_reliability/conditional_evaluate.py::_select_group_latency_aware_median).
    median_latency_improvement_pct: float
    labels: t.Dict[str, int]

    def to_dict(self) -> dict:
        return {
            'count': self.count,
            'positive': self.positive,
            'negative': self.negative,
            'success_rate': self.success_rate,
            'laplace_success_rate': self.laplace_success_rate,
            'mean_latency_improvement_pct': self.mean_latency_improvement_pct,
            'median_latency_improvement_pct': self.median_latency_improvement_pct,
            'labels': dict(self.labels),
        }


def is_positive(row: t.Any) -> bool:
    """Success definition: strictly lower actual latency after rewrite.

    ``row`` is any object exposing ``rewritten_latency_ms`` and
    ``original_latency_ms`` attributes (a ``CandidateRecord``); this is
    computed directly from the two latency fields, never from the
    precomputed ``label`` field.
    """
    return row.rewritten_latency_ms < row.original_latency_ms


def _latency_improvement_pct(row: t.Any) -> t.Optional[float]:
    original = row.original_latency_ms
    if not (isinstance(original, (int, float)) and original > 0):
        return None
    return (original - row.rewritten_latency_ms) / original * 100.0


def _make_entry(rows: t.Sequence[t.Any], alpha: float, beta: float) -> PriorEntry:
    count = len(rows)
    positive = sum(1 for row in rows if is_positive(row))
    negative = count - positive
    success_rate = (positive / count) if count else 0.0
    laplace_success_rate = (positive + alpha) / (count + alpha + beta)
    improvements = [x for x in (_latency_improvement_pct(row) for row in rows) if x is not None]
    mean_improvement = (sum(improvements) / len(improvements)) if improvements else 0.0
    median_improvement = statistics.median(improvements) if improvements else 0.0
    labels = Counter(row.label for row in rows)
    return PriorEntry(
        count=count,
        positive=positive,
        negative=negative,
        success_rate=success_rate,
        median_latency_improvement_pct=median_improvement,
        laplace_success_rate=laplace_success_rate,
        mean_latency_improvement_pct=mean_improvement,
        labels=dict(labels),
    )


def sequence_key(rules: t.Sequence[str]) -> str:
    return ' > '.join(rules) if rules else NO_RULE_KEY


def compute_rule_priors(train_rows: t.Sequence[t.Any], alpha: float = 1.0, beta: float = 1.0) -> t.Dict[str, PriorEntry]:
    """One entry per distinct rule name. A row that applies the same rule
    is only counted once for that rule (``dict.fromkeys`` dedup); a row
    that applies N distinct rules contributes to N rule entries."""
    by_rule: t.Dict[str, list] = defaultdict(list)
    for row in train_rows:
        for rule in dict.fromkeys(row.rules):
            by_rule[rule].append(row)
    return {rule: _make_entry(rows, alpha, beta) for rule, rows in by_rule.items()}


def compute_sequence_priors(train_rows: t.Sequence[t.Any], alpha: float = 1.0, beta: float = 1.0) -> t.Dict[str, PriorEntry]:
    """One entry per exact, order-preserving rule sequence (as stored)."""
    by_seq: t.Dict[str, list] = defaultdict(list)
    for row in train_rows:
        by_seq[sequence_key(row.rules)].append(row)
    return {seq: _make_entry(rows, alpha, beta) for seq, rows in by_seq.items()}


def compute_workload_priors(train_rows: t.Sequence[t.Any], alpha: float = 1.0, beta: float = 1.0) -> t.Dict[str, PriorEntry]:
    by_workload: t.Dict[str, list] = defaultdict(list)
    for row in train_rows:
        by_workload[row.workload].append(row)
    return {workload: _make_entry(rows, alpha, beta) for workload, rows in by_workload.items()}


def compute_global_prior(train_rows: t.Sequence[t.Any], alpha: float = 1.0, beta: float = 1.0) -> PriorEntry:
    return _make_entry(list(train_rows), alpha, beta)


class PriorTable:
    """Train-only prior statistics plus the fixed fallback lookup chain:

    1. exact rule-sequence prior (``' > '.join(rules)`` or ``'(no_rule)'``)
    2. individual-rule prior -- among the candidate's distinct rules, the
       one with the most train support is used as the representative
       (highest-evidence rule, not an average across heterogeneous rules)
    3. workload-level global prior (all train rows for that workload)
    4. dataset-wide global prior (all train rows)

    A step is skipped (falls through) when either no entry exists for the
    key, or the entry's ``count`` is below ``min_support``. Every lookup
    returns a :class:`PriorLookup` with a human-readable ``fallback_reason``
    documenting which steps were tried and why they were skipped.
    """

    def __init__(
        self,
        rule_priors: t.Dict[str, PriorEntry],
        sequence_priors: t.Dict[str, PriorEntry],
        workload_priors: t.Dict[str, PriorEntry],
        global_prior: PriorEntry,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.rules = rule_priors
        self.sequences = sequence_priors
        self.workloads = workload_priors
        self.global_entry = global_prior
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def build(cls, train_rows: t.Sequence[t.Any], alpha: float = 1.0, beta: float = 1.0) -> 'PriorTable':
        return cls(
            rule_priors=compute_rule_priors(train_rows, alpha, beta),
            sequence_priors=compute_sequence_priors(train_rows, alpha, beta),
            workload_priors=compute_workload_priors(train_rows, alpha, beta),
            global_prior=compute_global_prior(train_rows, alpha, beta),
            alpha=alpha,
            beta=beta,
        )

    def lookup(self, rules: t.Sequence[str], workload: str, min_support: int) -> PriorLookup:
        rules = tuple(rules)
        seq_key = sequence_key(rules)
        seq_entry = self.sequences.get(seq_key)
        if seq_entry is not None and seq_entry.count >= min_support:
            return PriorLookup(
                reliability=seq_entry.laplace_success_rate,
                support=seq_entry.count,
                source='rule_sequence',
                key=seq_key,
                fallback_reason=(
                    f"exact rule_sequence '{seq_key}' prior used "
                    f"(support={seq_entry.count} >= min_support={min_support})"
                ),
                mean_latency_improvement_pct=seq_entry.mean_latency_improvement_pct,
                median_latency_improvement_pct=seq_entry.median_latency_improvement_pct,
            )
        seq_reason = (
            f"rule_sequence '{seq_key}' support {seq_entry.count} < min_support {min_support}"
            if seq_entry is not None
            else f"rule_sequence '{seq_key}' not observed in train priors"
        )

        best_rule, best_entry = None, None
        for rule in dict.fromkeys(rules):
            entry = self.rules.get(rule)
            if entry is not None and (best_entry is None or entry.count > best_entry.count):
                best_rule, best_entry = rule, entry
        if best_entry is not None and best_entry.count >= min_support:
            return PriorLookup(
                reliability=best_entry.laplace_success_rate,
                support=best_entry.count,
                source='rule',
                key=best_rule,
                fallback_reason=(
                    f"{seq_reason}; individual rule '{best_rule}' prior used "
                    f"(support={best_entry.count} >= min_support={min_support}, "
                    f"highest-support rule among {list(dict.fromkeys(rules))})"
                ),
                mean_latency_improvement_pct=best_entry.mean_latency_improvement_pct,
                median_latency_improvement_pct=best_entry.median_latency_improvement_pct,
            )
        rule_reason = (
            f"; individual rule '{best_rule}' support {best_entry.count} < min_support {min_support}"
            if best_entry is not None
            else "; no individual rule prior with data for this candidate's rules"
        )

        wl_entry = self.workloads.get(workload)
        if wl_entry is not None and wl_entry.count >= min_support:
            return PriorLookup(
                reliability=wl_entry.laplace_success_rate,
                support=wl_entry.count,
                source='workload_global',
                key=workload,
                fallback_reason=(
                    f"{seq_reason}{rule_reason}; workload '{workload}' global prior used "
                    f"(support={wl_entry.count} >= min_support={min_support})"
                ),
                mean_latency_improvement_pct=wl_entry.mean_latency_improvement_pct,
                median_latency_improvement_pct=wl_entry.median_latency_improvement_pct,
            )
        wl_reason = (
            f"; workload '{workload}' support {wl_entry.count} < min_support {min_support}"
            if wl_entry is not None
            else f"; no train rows for workload '{workload}'"
        )

        if self.global_entry.count > 0:
            return PriorLookup(
                reliability=self.global_entry.laplace_success_rate,
                support=self.global_entry.count,
                source='global',
                key=None,
                fallback_reason=f"{seq_reason}{rule_reason}{wl_reason}; dataset-wide global prior used "
                                f"(support={self.global_entry.count})",
                mean_latency_improvement_pct=self.global_entry.mean_latency_improvement_pct,
                median_latency_improvement_pct=self.global_entry.median_latency_improvement_pct,
            )
        return PriorLookup(
            reliability=0.5,
            support=0,
            source='none',
            key=None,
            fallback_reason=f"{seq_reason}{rule_reason}{wl_reason}; no train data available at all",
            mean_latency_improvement_pct=None,
            median_latency_improvement_pct=None,
        )

    def to_dict(self, train_candidate_count: int) -> dict:
        return {
            'policy': {
                'source': 'train split only',
                'success': 'rewritten_latency_ms < original_latency_ms',
                'smoothing': f'Laplace/Beta({self.alpha},{self.beta})',
                'fallback_order': ['rule_sequence', 'rule', 'workload_global', 'global'],
            },
            'train_candidate_count': train_candidate_count,
            'rules': {k: v.to_dict() for k, v in self.rules.items()},
            'rule_sequences': {k: v.to_dict() for k, v in self.sequences.items()},
            'workloads': {k: v.to_dict() for k, v in self.workloads.items()},
            'global': self.global_entry.to_dict(),
        }
