#!/usr/bin/env python3
"""CLI entrypoint for the nested MVMConfig hyperparameter search (Priority-1
item: tune fp_guard_threshold/fn_recovery_threshold/min_support/alpha/beta,
which have been fixed defaults since the MVM Reliability Prior work began).

For each of the k outer family-grouped folds, a grid of MVMConfig values is
scored using an inner k-fold built ONLY from that fold's own train
families (see threshold_search.py for the leakage discipline), the
best-scoring config is applied to the outer test fold, and both the tuned
and the still-fixed-default config are evaluated on the SAME outer test
fold for a direct, paired comparison.

Also computes a percentile bootstrap CI (resampling test *groups*, not
individual candidates, so family-grouping is preserved) for the pooled
oracle_gap_recovered_rate of both the tuned and the default config, to
tell a real improvement apart from fold-to-fold noise.

Reads query_sets/mvm_v1/candidates.jsonl read-only. Writes a new run
directory under experiments/mvm_reliability/outputs/.

Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m experiments.mvm_reliability.run_threshold_search \\
        --run-name threshold_search --k 5 --k-inner 5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import typing as t
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.cv_split import family_kfold_splits, usable_families_by_workload, correlate_families  # noqa: E402
from experiments.mvm_reliability.dataset import load_candidates_jsonl, sha256_file  # noqa: E402
from experiments.mvm_reliability.report import atomic_write_json, atomic_write_text, git_info, make_run_dir  # noqa: E402
from experiments.mvm_reliability.threshold_search import DEFAULT_GRID, nested_search_one_outer_fold  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidates', type=Path, default=Path('query_sets/mvm_v1/candidates.jsonl'))
    parser.add_argument('--k', type=int, default=5, help='Number of OUTER folds')
    parser.add_argument('--k-inner', type=int, default=5, help='Number of INNER folds used for config selection')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260906, help='Bootstrap RNG seed (recorded in the run)')
    parser.add_argument('--run-name', default='threshold_search')
    parser.add_argument('--outputs-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    return parser.parse_args(argv)


def _mean_std(values: t.Sequence[t.Optional[float]]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {'n': 0, 'mean': None, 'stdev': None}
    return {'n': len(clean), 'mean': statistics.mean(clean), 'stdev': statistics.stdev(clean) if len(clean) >= 2 else None}


def _pooled_gap_recovered_rate(group_rows: t.List[dict], other_key: str) -> t.Optional[float]:
    gap_avail = sum(max(0.0, r['cost_selected_latency_ms'] - r['oracle_latency_ms']) for r in group_rows)
    gap_rec = sum(r['cost_selected_latency_ms'] - r[other_key] for r in group_rows)
    return (gap_rec / gap_avail) if gap_avail > 0 else None


def bootstrap_ci(
    group_rows: t.List[dict], other_key: str, n_samples: int, rng: random.Random, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap CI for the pooled oracle_gap_recovered_rate,
    resampling whole test GROUPS with replacement (never individual
    candidates) so the family-grouped nature of the data is respected."""
    n = len(group_rows)
    point = _pooled_gap_recovered_rate(group_rows, other_key)
    if n == 0:
        return {'point_estimate': None, 'ci_low': None, 'ci_high': None, 'n_bootstrap': 0, 'n_groups': 0}
    samples = []
    for _ in range(n_samples):
        resample = [group_rows[rng.randrange(n)] for _ in range(n)]
        rate = _pooled_gap_recovered_rate(resample, other_key)
        if rate is not None:
            samples.append(rate)
    samples.sort()
    if not samples:
        return {'point_estimate': point, 'ci_low': None, 'ci_high': None, 'n_bootstrap': 0, 'n_groups': n}
    lo_idx = int(len(samples) * (alpha / 2))
    hi_idx = min(len(samples) - 1, int(len(samples) * (1 - alpha / 2)))
    return {
        'point_estimate': point,
        'ci_low': samples[lo_idx],
        'ci_high': samples[hi_idx],
        'n_bootstrap': len(samples),
        'n_groups': n,
        'confidence_level': 1 - alpha,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    records = load_candidates_jsonl(args.candidates)
    print(f'[threshold_search] loaded {len(records)} candidates from {args.candidates}')

    outer_splits = family_kfold_splits(records, args.k)
    fold_results = []
    for idx, split in enumerate(outer_splits):
        print(f'[threshold_search] outer fold {idx + 1}/{args.k}: searching grid on inner folds ...')
        result = nested_search_one_outer_fold(records, split, DEFAULT_GRID, args.k_inner, cost_strict=True)
        result['fold_index'] = idx
        fold_results.append(result)
        tuned_rate = result['outer_with_tuned_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        default_rate = result['outer_with_default_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        print(
            f"  best_config={result['best_config']} inner_score={result['best_inner_score']} "
            f"outer tuned_rate={tuned_rate} default_rate={default_rate}"
        )

    tuned_group_rows = [row for f in fold_results for row in f['outer_with_tuned_config']['group_rows']]
    default_group_rows = [row for f in fold_results for row in f['outer_with_default_config']['group_rows']]

    tuned_accuracy = [f['outer_with_tuned_config']['candidates']['cond_mvm']['accuracy'] for f in fold_results]
    default_accuracy = [f['outer_with_default_config']['candidates']['cond_mvm']['accuracy'] for f in fold_results]
    tuned_rate_per_fold = [
        f['outer_with_tuned_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        for f in fold_results
    ]
    default_rate_per_fold = [
        f['outer_with_default_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        for f in fold_results
    ]

    rng = random.Random(args.seed)
    tuned_ci = bootstrap_ci(tuned_group_rows, 'cond_mvm_latency_aware_median_selected_latency_ms', args.bootstrap_samples, rng)
    rng2 = random.Random(args.seed)  # separate stream so both CIs use the same resampling indices per-call-order is not required, but keep reproducible
    default_ci = bootstrap_ci(default_group_rows, 'cond_mvm_latency_aware_median_selected_latency_ms', args.bootstrap_samples, rng2)

    aggregate = {
        'tuned_accuracy': _mean_std(tuned_accuracy),
        'default_accuracy': _mean_std(default_accuracy),
        'tuned_oracle_gap_recovered_rate': _mean_std(tuned_rate_per_fold),
        'default_oracle_gap_recovered_rate': _mean_std(default_rate_per_fold),
        'tuned_pooled_bootstrap_ci': tuned_ci,
        'default_pooled_bootstrap_ci': default_ci,
    }
    print(f"[threshold_search] tuned accuracy mean={aggregate['tuned_accuracy']['mean']}, "
          f"default accuracy mean={aggregate['default_accuracy']['mean']}")
    print(f"[threshold_search] tuned oracle_gap_recovered_rate mean={aggregate['tuned_oracle_gap_recovered_rate']['mean']}, "
          f"default mean={aggregate['default_oracle_gap_recovered_rate']['mean']}")
    print(f"[threshold_search] tuned pooled bootstrap 95% CI: [{tuned_ci['ci_low']}, {tuned_ci['ci_high']}] "
          f"(point={tuned_ci['point_estimate']})")
    print(f"[threshold_search] default pooled bootstrap 95% CI: [{default_ci['ci_low']}, {default_ci['ci_high']}] "
          f"(point={default_ci['point_estimate']})")

    usable_families = usable_families_by_workload(records)
    snapshot = {
        'candidates_path': str(args.candidates),
        'candidates_sha256': sha256_file(args.candidates),
        'candidate_count': len(records),
        'usable_family_count': sum(len(v) for v in usable_families.values()),
        'correlate_family_count': len(correlate_families(records)),
        'git': git_info(MY_REWRITER_DIR),
        'grid': DEFAULT_GRID,
        'k_outer': args.k,
        'k_inner': args.k_inner,
        'bootstrap_samples': args.bootstrap_samples,
        'bootstrap_seed': args.seed,
    }

    # Trim the very large per-config inner-score log out of the per-fold
    # record before writing (grid_size is enough context) to keep the run
    # directory small; nothing about leakage discipline depends on keeping it.
    for f in fold_results:
        f.pop('all_scores', None)

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_json(run_dir / 'dataset_snapshot.json', snapshot)
    atomic_write_json(run_dir / 'fold_results.json', fold_results)
    atomic_write_json(run_dir / 'aggregate.json', aggregate)

    best_configs = [f['best_config'] for f in fold_results]
    report_lines = [
        '# MVM Reliability Prior -- Nested Hyperparameter Search',
        '',
        f"candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"outer folds: {args.k}, inner folds: {args.k_inner}, bootstrap samples: {args.bootstrap_samples} (seed={args.seed})",
        '',
        '## Best config chosen per outer fold (via inner CV on that fold\'s train only)',
        '',
        '| fold | alpha=beta | min_support | fp_guard | fn_recovery | inner score |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for f in fold_results:
        c = f['best_config']
        report_lines.append(
            f"| {f['fold_index']} | {c['alpha']} | {c['min_support']} | {c['fp_guard_threshold']} | "
            f"{c['fn_recovery_threshold']} | {f['best_inner_score']} |"
        )
    report_lines += [
        '',
        '## Outer-fold accuracy and latency, tuned vs default config (paired, same outer test fold)',
        '',
        '| fold | tuned acc | default acc | tuned gap-rec | default gap-rec |',
        '|---:|---:|---:|---:|---:|',
    ]
    for f, ta, da in zip(fold_results, tuned_accuracy, default_accuracy):
        tr = f['outer_with_tuned_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        dr = f['outer_with_default_config']['groups']['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        report_lines.append(f"| {f['fold_index']} | {ta} | {da} | {tr} | {dr} |")
    report_lines += [
        '',
        '## Aggregate (mean +/- stdev across outer folds)',
        '',
        f"- Accuracy: tuned {aggregate['tuned_accuracy']['mean']} +/- {aggregate['tuned_accuracy']['stdev']}"
        f" vs default {aggregate['default_accuracy']['mean']} +/- {aggregate['default_accuracy']['stdev']}",
        f"- Oracle gap recovered rate: tuned {aggregate['tuned_oracle_gap_recovered_rate']['mean']} +/- "
        f"{aggregate['tuned_oracle_gap_recovered_rate']['stdev']} vs default "
        f"{aggregate['default_oracle_gap_recovered_rate']['mean']} +/- {aggregate['default_oracle_gap_recovered_rate']['stdev']}",
        '',
        '## Pooled bootstrap 95% CI on oracle_gap_recovered_rate (resampling groups, not candidates)',
        '',
        f"- Tuned: point={tuned_ci['point_estimate']}, CI=[{tuned_ci['ci_low']}, {tuned_ci['ci_high']}] (n_groups={tuned_ci['n_groups']})",
        f"- Default: point={default_ci['point_estimate']}, CI=[{default_ci['ci_low']}, {default_ci['ci_high']}] (n_groups={default_ci['n_groups']})",
        '',
        '## Limitations',
        '- Grid is intentionally modest (see threshold_search.DEFAULT_GRID); a wider grid was not run for turnaround time.',
        '- Inner CV score used to pick a config is oracle_gap_recovered_rate under latency-aware median selection only '
        '-- accuracy is reported for the tuned config but was not the selection criterion.',
        '- Read-only replay over already-logged data; no SQL/LLM/DB calls.',
        '',
    ]
    atomic_write_text(run_dir / 'report.md', '\n'.join(report_lines))
    print(f'[threshold_search] wrote run to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
