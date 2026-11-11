"""Nested, family-grouped hyperparameter search for MVMConfig
(alpha/beta/min_support/fp_guard_threshold/fn_recovery_threshold).

Every one of these five values has been a fixed default (from
configs/mvm_v1.json) since the MVM Reliability Prior work started; this
module is the first attempt to actually tune them against data.

Leakage discipline (same as the rest of experiments/mvm_reliability/):
for each OUTER k-fold test fold, the candidate hyperparameter grid is
scored using ONLY an INNER k-fold built from that fold's own train
families (never touching the outer test fold), and only the
best-scoring config (by mean inner-fold oracle_gap_recovered_rate) is
then used to build the final policy from the *full* outer-train set and
evaluated once against the outer test fold. The outer test fold never
participates in config selection.
"""

from __future__ import annotations

import statistics
import typing as t
from dataclasses import asdict

from candidate_selection.mvm.policy import MVMConfig

from .conditional_evaluate import run_multi_evaluation, summarize_multi
from .cv_split import family_kfold_splits
from .dataset import filter_split
from .evaluate import ExperimentConfig
from .schemas import CandidateRecord, SplitInfo

# Kept intentionally modest (72 combinations minus invalid orderings) so a
# 5-outer x 5-inner nested search finishes in well under a minute -- the
# grid can be widened later once it is clear tuning is worth the extra
# search cost.
DEFAULT_GRID: t.Dict[str, t.List[float]] = {
    'alpha_beta': [1.0, 0.0],  # alpha and beta tied together (see mvm_v1_no_smoothing.json finding: barely matters)
    'min_support': [5, 3, 8, 15],
    'fp_guard_threshold': [0.4, 0.3, 0.5],
    'fn_recovery_threshold': [0.6, 0.5, 0.7],
}

DEFAULT_CONFIG_KEY = (1.0, 5, 0.4, 0.6)  # checked first so ties favor the status quo


def iter_grid(grid: t.Dict[str, t.List[float]]) -> t.Iterator[MVMConfig]:
    """Yields valid MVMConfig combinations (fp_guard <= fn_recovery), the
    current default first so a tie in scoring keeps today's behavior."""
    seen: t.Set[tuple] = set()

    def _combo_to_config(ab: float, ms: int, fpg: float, fnr: float) -> t.Optional[MVMConfig]:
        key = (ab, ms, fpg, fnr)
        if key in seen or fpg > fnr:
            return None
        seen.add(key)
        return MVMConfig(alpha=ab, beta=ab, min_support=ms, fp_guard_threshold=fpg, fn_recovery_threshold=fnr)

    default = _combo_to_config(*DEFAULT_CONFIG_KEY)
    if default is not None:
        yield default

    for ab in grid['alpha_beta']:
        for ms in grid['min_support']:
            for fpg in grid['fp_guard_threshold']:
                for fnr in grid['fn_recovery_threshold']:
                    cfg = _combo_to_config(ab, ms, fpg, fnr)
                    if cfg is not None:
                        yield cfg


def score_config(
    records: t.Sequence[CandidateRecord], k_inner: int, cost_strict: bool, mvm_config: MVMConfig
) -> t.Optional[float]:
    """Mean oracle_gap_recovered_rate (latency-aware, median selection)
    across k_inner family-grouped folds built ONLY from `records`
    (already restricted to one outer fold's train families)."""
    try:
        inner_splits = family_kfold_splits(records, k_inner)
    except ValueError:
        return None
    exp_config = ExperimentConfig(cost_strict=cost_strict, mvm=mvm_config)
    rates: t.List[float] = []
    for split in inner_splits:
        result = run_multi_evaluation(records, split, exp_config)
        summary = summarize_multi(result)
        rate = summary['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        if rate is not None:
            rates.append(rate)
    return statistics.mean(rates) if rates else None


def nested_search_one_outer_fold(
    records: t.Sequence[CandidateRecord],
    outer_split: SplitInfo,
    grid: t.Dict[str, t.List[float]],
    k_inner: int,
    cost_strict: bool,
) -> dict:
    # Usable records whose family this outer fold put in TRAIN -- the inner
    # search is built entirely from this subset, never from outer test.
    outer_train_records = filter_split(records, outer_split, 'train')

    best_config: t.Optional[MVMConfig] = None
    best_score = float('-inf')
    all_scores: t.List[dict] = []
    for cfg in iter_grid(grid):
        score = score_config(outer_train_records, k_inner, cost_strict, cfg)
        all_scores.append({'config': asdict(cfg), 'inner_score': score})
        if score is not None and score > best_score:
            best_score = score
            best_config = cfg

    if best_config is None:
        # No config produced a usable inner score anywhere -- fall back to default.
        best_config = MVMConfig()

    final_exp_config = ExperimentConfig(cost_strict=cost_strict, mvm=best_config)
    outer_result = run_multi_evaluation(records, outer_split, final_exp_config)
    outer_summary = summarize_multi(outer_result)

    default_exp_config = ExperimentConfig(cost_strict=cost_strict, mvm=MVMConfig())
    default_outer_result = run_multi_evaluation(records, outer_split, default_exp_config)
    default_outer_summary = summarize_multi(default_outer_result)

    return {
        'best_config': asdict(best_config),
        'best_inner_score': best_score if best_score != float('-inf') else None,
        'grid_size': len(all_scores),
        'outer_with_tuned_config': outer_summary,
        'outer_with_default_config': default_outer_summary,
    }
