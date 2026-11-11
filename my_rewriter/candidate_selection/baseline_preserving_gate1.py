"""Baseline-preserving Gate1 candidate generation and asymmetric DCRP selection.

The preserved baseline is the *actual output of the existing Gate1-OFF
pipeline*.  It is never reconstructed by a deterministic surrogate.  Gate1 may
generate one additional skew-aware candidate only when its activation predicate
fires.  The alternative replaces the baseline only when all safety checks pass
and a history-corrected latency-ratio upper bound clears a one-sided margin.

This module deliberately does not read measured latency at decision time.
Latency appears only in train-history observations used to build a DCRP prior.
"""

from __future__ import annotations

import math
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field


# The 13 pre-execution plan fields used by exp6.  Cost is handled separately by
# DCRP; these fields decide whether the two plans are structurally distinguishable.
DEFAULT_PLAN_FIELDS = (
    "agg_count",
    "distinct_node_types",
    "hash_join_count",
    "index_scan_count",
    "merge_join_count",
    "nested_loop_count",
    "node_count",
    "plan_depth",
    "root_plan_rows",
    "root_plan_width",
    "seq_scan_count",
    "sort_count",
    "sum_plan_rows",
)


@dataclass(frozen=True)
class RewriteCandidate:
    candidate_id: str
    sql: str
    rules: t.Tuple[str, ...]
    estimated_cost: float
    plan_features: t.Mapping[str, t.Any] = field(default_factory=dict)
    equivalent: bool = True
    valid: bool = True
    metadata: t.Mapping[str, t.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple):
            object.__setattr__(self, "rules", tuple(self.rules))
        if not isinstance(self.estimated_cost, (int, float)) or not math.isfinite(
            self.estimated_cost
        ):
            raise ValueError("estimated_cost must be finite")


@dataclass(frozen=True)
class DCRPObservation:
    """One train-only baseline/alternative outcome.

    ``query_family`` is carried so evaluation code can remove the held-out
    family before constructing a prior.
    """

    workload: str
    query_family: str
    alternative_rules: t.Tuple[str, ...]
    baseline_cost: float
    alternative_cost: float
    baseline_latency_ms: float
    alternative_latency_ms: float
    gate1_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.alternative_rules, tuple):
            object.__setattr__(self, "alternative_rules", tuple(self.alternative_rules))
        for name in (
            "baseline_cost", "alternative_cost", "baseline_latency_ms",
            "alternative_latency_ms",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

    @property
    def residual(self) -> float:
        """Cost error in log-ratio space; negative means cost was pessimistic."""
        latency_ratio = self.alternative_latency_ms / self.baseline_latency_ms
        cost_ratio = self.alternative_cost / self.baseline_cost
        return math.log(latency_ratio) - math.log(cost_ratio)


@dataclass(frozen=True)
class DCRPPrior:
    center: float
    robust_sigma: float
    support: int
    source: str
    key: t.Optional[str]


@dataclass(frozen=True)
class ConservativeDCRPConfig:
    min_support: int = 5
    shrinkage_k: float = 10.0
    confidence_z: float = 1.645
    uncertainty_floor: float = 0.02
    min_improvement_ratio: float = 0.05
    require_distinct_plan: bool = True
    allow_workload_fallback: bool = False
    plan_fields: t.Tuple[str, ...] = DEFAULT_PLAN_FIELDS

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if self.shrinkage_k < 0 or self.confidence_z < 0 or self.uncertainty_floor < 0:
            raise ValueError("shrinkage/confidence/uncertainty values must be non-negative")
        if not 0 <= self.min_improvement_ratio < 1:
            raise ValueError("min_improvement_ratio must be in [0, 1)")


class DCRPHistory:
    """Leakage-safe residual lookup with sequence -> rule -> optional workload backoff."""

    def __init__(
        self,
        observations: t.Iterable[DCRPObservation],
        held_out_family: t.Optional[str] = None,
    ) -> None:
        self.observations = [
            item for item in observations
            if item.gate1_active and item.query_family != held_out_family
        ]

    @staticmethod
    def _sequence_key(rules: t.Iterable[str]) -> str:
        return " > ".join(rules) if rules else "(no_rule)"

    @staticmethod
    def _prior(values: list[float], fallback: list[float], config: ConservativeDCRPConfig,
               source: str, key: t.Optional[str]) -> DCRPPrior:
        fallback_center = statistics.median(fallback) if fallback else 0.0
        local_center = statistics.median(values)
        n = len(values)
        center = (
            n / (n + config.shrinkage_k) * local_center
            + config.shrinkage_k / (n + config.shrinkage_k) * fallback_center
        )
        deviations = [abs(value - local_center) for value in values]
        mad = statistics.median(deviations) if deviations else 0.0
        sigma = max(config.uncertainty_floor, 1.4826 * mad)
        return DCRPPrior(center=center, robust_sigma=sigma, support=n, source=source, key=key)

    def lookup(
        self,
        workload: str,
        rules: t.Iterable[str],
        config: ConservativeDCRPConfig,
    ) -> t.Optional[DCRPPrior]:
        rules = tuple(rules)
        workload_rows = [row for row in self.observations if row.workload == workload]
        fallback = [row.residual for row in workload_rows]
        sequence = self._sequence_key(rules)
        exact = [
            row.residual for row in workload_rows
            if self._sequence_key(row.alternative_rules) == sequence
        ]
        if len(exact) >= config.min_support:
            return self._prior(exact, fallback, config, "rule_sequence", sequence)

        best_rule, best_values = None, []
        for rule in dict.fromkeys(rules):
            values = [row.residual for row in workload_rows if rule in row.alternative_rules]
            if len(values) > len(best_values):
                best_rule, best_values = rule, values
        if len(best_values) >= config.min_support:
            return self._prior(best_values, fallback, config, "rule", best_rule)

        if config.allow_workload_fallback and len(fallback) >= config.min_support:
            return self._prior(fallback, fallback, config, "workload", workload)
        return None


@dataclass(frozen=True)
class Gate1SelectionDecision:
    selected: RewriteCandidate
    decision: str  # baseline | alternative
    reason: str
    estimated_log_cost_ratio: t.Optional[float] = None
    corrected_log_latency_ratio: t.Optional[float] = None
    conservative_upper_bound: t.Optional[float] = None
    required_upper_bound: t.Optional[float] = None
    prior: t.Optional[DCRPPrior] = None


def normalized_sql(sql: str) -> str:
    return " ".join((sql or "").strip().rstrip(";").lower().split())


def plan_features_identical(
    baseline: t.Mapping[str, t.Any],
    alternative: t.Mapping[str, t.Any],
    fields: t.Iterable[str] = DEFAULT_PLAN_FIELDS,
) -> t.Optional[bool]:
    """Return None when the required pre-execution signal is unavailable."""
    fields = tuple(fields)
    if not baseline or not alternative:
        return None
    if any(field not in baseline or field not in alternative for field in fields):
        return None
    return all(baseline[field] == alternative[field] for field in fields)


class AsymmetricDCRPSelector:
    """One-sided baseline -> skew-alternative selector with abstention."""

    def __init__(self, history: DCRPHistory, config: ConservativeDCRPConfig = None) -> None:
        self.history = history
        self.config = config or ConservativeDCRPConfig()

    def select(
        self,
        baseline: RewriteCandidate,
        alternative: t.Optional[RewriteCandidate],
        workload: str,
        gate1_active: bool,
    ) -> Gate1SelectionDecision:
        def keep(reason: str, **kwargs: t.Any) -> Gate1SelectionDecision:
            return Gate1SelectionDecision(baseline, "baseline", reason, **kwargs)

        if not gate1_active:
            return keep("gate1_inactive")
        if alternative is None:
            return keep("alternative_missing")
        if not alternative.valid:
            return keep("alternative_invalid")
        if not alternative.equivalent:
            return keep("alternative_not_equivalent")
        if normalized_sql(alternative.sql) == normalized_sql(baseline.sql):
            return keep("same_sql")
        if baseline.estimated_cost <= 0 or alternative.estimated_cost <= 0:
            return keep("invalid_cost")

        identical = plan_features_identical(
            baseline.plan_features, alternative.plan_features, self.config.plan_fields
        )
        if identical is True:
            return keep("indistinguishable_plan_features")
        if self.config.require_distinct_plan and identical is None:
            return keep("plan_features_unavailable")

        prior = self.history.lookup(workload, alternative.rules, self.config)
        if prior is None:
            return keep("insufficient_dcrp_support")

        log_cost_ratio = math.log(alternative.estimated_cost / baseline.estimated_cost)
        corrected = log_cost_ratio + prior.center
        upper = corrected + self.config.confidence_z * prior.robust_sigma
        required = math.log(1.0 - self.config.min_improvement_ratio)
        details = {
            "estimated_log_cost_ratio": log_cost_ratio,
            "corrected_log_latency_ratio": corrected,
            "conservative_upper_bound": upper,
            "required_upper_bound": required,
            "prior": prior,
        }
        if upper >= required:
            return keep("uncertain_or_insufficient_improvement", **details)
        return Gate1SelectionDecision(
            alternative, "alternative", "dcrp_margin_cleared", **details
        )


@dataclass(frozen=True)
class BaselinePreservingOutcome:
    baseline: RewriteCandidate
    alternative: t.Optional[RewriteCandidate]
    selection: Gate1SelectionDecision
    gate1_active: bool
    extra_generation_calls: int


class BaselinePreservingGate1:
    """Orchestrate an unchanged baseline plus at most one extra Gate1 call.

    ``baseline_factory`` must invoke the existing Gate1-OFF pipeline unchanged.
    ``alternative_factory`` is the single distribution-aware planner call.  It
    receives the frozen baseline for provenance, but cannot mutate it because
    :class:`RewriteCandidate` is immutable.
    """

    def __init__(self, selector: AsymmetricDCRPSelector) -> None:
        self.selector = selector

    def run(
        self,
        baseline_factory: t.Callable[[], RewriteCandidate],
        alternative_factory: t.Callable[[RewriteCandidate], RewriteCandidate],
        workload: str,
        gate1_active: bool,
    ) -> BaselinePreservingOutcome:
        baseline = baseline_factory()
        alternative = None
        calls = 0
        if gate1_active:
            calls = 1
            try:
                alternative = alternative_factory(baseline)
            except Exception:
                alternative = None
        selection = self.selector.select(baseline, alternative, workload, gate1_active)
        return BaselinePreservingOutcome(
            baseline=baseline,
            alternative=alternative,
            selection=selection,
            gate1_active=gate1_active,
            extra_generation_calls=calls,
        )
