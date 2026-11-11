#!/usr/bin/env python3
"""CLI entrypoint for cross-validated MVM reliability offline evaluation.

Replaces the single fixed 80/20 split.json partition with two
family-grouped designs, both built in memory by cv_split.py (split.json
itself is never read or written here):

  1. Group-stratified k-fold CV over all usable query families
     (per-workload folds; a family's candidates are always entirely in
     one fold). Reports mean/stdev across folds.
  2. Leave-one-correlate-family-out: one fold per query family that has
     at least one FILTER_SUB_QUERY_TO_CORRELATE candidate, that family
     held out as the sole test family and every other usable family (any
     workload) used for the prior -- the most direct test of "does the
     correlate-rule prior generalize to a family it never saw".

Reads query_sets/mvm_v1/candidates.jsonl read-only. Writes a new,
independent run directory under experiments/mvm_reliability/outputs/.
Never touches query_sets/mvm_v1/results/mvm_evaluation.json.

Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m experiments.mvm_reliability.run_cv_evaluation \\
        --run-name cv --k 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import typing as t
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.cv_split import (  # noqa: E402
    correlate_families,
    family_kfold_splits,
    leave_one_family_out_splits,
    usable_families_by_workload,
)
from experiments.mvm_reliability.dataset import load_candidates_jsonl, sha256_file  # noqa: E402
from experiments.mvm_reliability.evaluate import ExperimentConfig, run_evaluation  # noqa: E402
from experiments.mvm_reliability.metrics import summarize_result  # noqa: E402
from experiments.mvm_reliability.report import atomic_write_json, atomic_write_text, git_info, make_run_dir  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'mvm_v1.json',
        help='Path to an experiment config JSON (default: configs/mvm_v1.json)',
    )
    parser.add_argument(
        '--candidates',
        type=Path,
        default=None,
        help='Override candidates.jsonl path (default: value in --config)',
    )
    parser.add_argument('--k', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--run-name', default='cv', help='Suffix for the run directory name')
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'outputs',
        help='Directory under which a new <timestamp>_<run-name> run directory is created',
    )
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (MY_REWRITER_DIR / path)


def _mean_std(values: t.Sequence[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {'n': 0, 'mean': None, 'stdev': None, 'min': None, 'max': None, 'values': []}
    return {
        'n': len(clean),
        'mean': statistics.mean(clean),
        'stdev': statistics.stdev(clean) if len(clean) >= 2 else None,
        'min': min(clean),
        'max': max(clean),
        'values': clean,
    }


def _fold_metrics(result) -> dict:
    summary = summarize_result(result)
    cand = summary['overall']['candidates']
    groups = summary['overall']['groups']
    return {
        'test_group_count': len(result.groups),
        'test_candidate_count': len(result.all_candidate_decisions),
        'train_candidate_count': result.train_candidate_count,
        'cost_only_accuracy': cand['cost_only']['accuracy'],
        'mvm_accuracy': cand['mvm']['accuracy'],
        'cost_only_confusion': cand['cost_only']['confusion'],
        'mvm_confusion': cand['mvm']['confusion'],
        'fp_prevented_rate': cand['fp_prevented']['rate'],
        'fn_recovered_rate': cand['fn_recovered']['rate'],
        'cost_selected_latency_mean_ms': groups['cost_selected_latency_ms']['mean'],
        'mvm_selected_latency_mean_ms': groups['mvm_selected_latency_ms']['mean'],
        'original_only_latency_mean_ms': groups['original_only_latency_ms']['mean'],
        'oracle_latency_mean_ms': groups['oracle_latency_ms']['mean'],
        'oracle_gap_available_ms': groups['oracle_gap_available_ms'],
        'oracle_gap_recovered_ms': groups['oracle_gap_recovered_ms'],
        'oracle_gap_recovered_rate': groups['oracle_gap_recovered_rate'],
    }


def run_kfold(records, config: ExperimentConfig, k: int) -> dict:
    splits = family_kfold_splits(records, k)
    folds = []
    for idx, split in enumerate(splits):
        result = run_evaluation(records, split, config)
        fold_metrics = _fold_metrics(result)
        fold_metrics['fold_index'] = idx
        fold_metrics['test_families'] = {wl: sorted(fams) for wl, fams in split.test_families.items()}
        folds.append(fold_metrics)

    aggregate = {
        'k': k,
        'cost_only_accuracy': _mean_std([f['cost_only_accuracy'] for f in folds]),
        'mvm_accuracy': _mean_std([f['mvm_accuracy'] for f in folds]),
        'oracle_gap_recovered_rate': _mean_std([f['oracle_gap_recovered_rate'] for f in folds]),
        'cost_selected_latency_mean_ms': _mean_std([f['cost_selected_latency_mean_ms'] for f in folds]),
        'mvm_selected_latency_mean_ms': _mean_std([f['mvm_selected_latency_mean_ms'] for f in folds]),
        'oracle_latency_mean_ms': _mean_std([f['oracle_latency_mean_ms'] for f in folds]),
    }
    return {'folds': folds, 'aggregate': aggregate}


def run_lofo(records, config: ExperimentConfig) -> dict:
    targets = correlate_families(records)
    lofo_splits = leave_one_family_out_splits(records, targets)
    folds = []
    for (workload, family), split in lofo_splits:
        result = run_evaluation(records, split, config)
        fold_metrics = _fold_metrics(result)
        fold_metrics['held_out_workload'] = workload
        fold_metrics['held_out_family'] = family
        folds.append(fold_metrics)

    aggregate = {
        'family_count': len(folds),
        'cost_only_accuracy': _mean_std([f['cost_only_accuracy'] for f in folds]),
        'mvm_accuracy': _mean_std([f['mvm_accuracy'] for f in folds]),
        'oracle_gap_recovered_rate': _mean_std([f['oracle_gap_recovered_rate'] for f in folds]),
        'cost_selected_latency_mean_ms': _mean_std([f['cost_selected_latency_mean_ms'] for f in folds]),
        'mvm_selected_latency_mean_ms': _mean_std([f['mvm_selected_latency_mean_ms'] for f in folds]),
        'oracle_latency_mean_ms': _mean_std([f['oracle_latency_mean_ms'] for f in folds]),
    }
    return {'folds': folds, 'aggregate': aggregate}


def render_report(snapshot: dict, kfold: dict, lofo: dict) -> str:
    lines = [
        '# MVM Reliability Prior -- Cross-Validated Offline Evaluation',
        '',
        f"candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"candidates: {snapshot['candidate_count']} (usable families: {snapshot['usable_family_count']})",
        f"git commit: {snapshot['git'].get('commit')} (dirty={snapshot['git'].get('dirty')}, branch={snapshot['git'].get('branch')})",
        '',
        f"## Group-stratified {kfold['aggregate']['k']}-fold CV (all usable query families)",
        '',
        '| metric | mean | stdev | n folds |',
        '|---|---:|---:|---:|',
    ]
    for key, label in (
        ('cost_only_accuracy', 'Cost-only accuracy'),
        ('mvm_accuracy', 'MVM accuracy'),
        ('oracle_gap_recovered_rate', 'Oracle gap recovered rate'),
        ('cost_selected_latency_mean_ms', 'Cost-only selected latency (mean, ms)'),
        ('mvm_selected_latency_mean_ms', 'MVM selected latency (mean, ms)'),
        ('oracle_latency_mean_ms', 'Oracle latency (mean, ms)'),
    ):
        stat = kfold['aggregate'][key]
        mean = f"{stat['mean']:.4f}" if stat['mean'] is not None else 'n/a'
        std = f"{stat['stdev']:.4f}" if stat['stdev'] is not None else 'n/a'
        lines.append(f"| {label} | {mean} | {std} | {stat['n']} |")

    lines += [
        '',
        '### Per-fold detail',
        '',
        '| fold | test groups | test candidates | cost-only acc | mvm acc | oracle_gap_recovered_rate |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for f in kfold['folds']:
        lines.append(
            f"| {f['fold_index']} | {f['test_group_count']} | {f['test_candidate_count']} | "
            f"{f['cost_only_accuracy']} | {f['mvm_accuracy']} | {f['oracle_gap_recovered_rate']} |"
        )

    lines += [
        '',
        f"## Leave-one-correlate-family-out ({lofo['aggregate']['family_count']} families)",
        '',
        '| metric | mean | stdev | n folds |',
        '|---|---:|---:|---:|',
    ]
    for key, label in (
        ('cost_only_accuracy', 'Cost-only accuracy'),
        ('mvm_accuracy', 'MVM accuracy'),
        ('oracle_gap_recovered_rate', 'Oracle gap recovered rate'),
        ('cost_selected_latency_mean_ms', 'Cost-only selected latency (mean, ms)'),
        ('mvm_selected_latency_mean_ms', 'MVM selected latency (mean, ms)'),
        ('oracle_latency_mean_ms', 'Oracle latency (mean, ms)'),
    ):
        stat = lofo['aggregate'][key]
        mean = f"{stat['mean']:.4f}" if stat['mean'] is not None else 'n/a'
        std = f"{stat['stdev']:.4f}" if stat['stdev'] is not None else 'n/a'
        lines.append(f"| {label} | {mean} | {std} | {stat['n']} |")

    lines += [
        '',
        '### Per-family detail',
        '',
        '| workload | family | test candidates | cost-only acc | mvm acc | oracle_gap_recovered_rate |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for f in lofo['folds']:
        lines.append(
            f"| {f['held_out_workload']} | {f['held_out_family']} | {f['test_candidate_count']} | "
            f"{f['cost_only_accuracy']} | {f['mvm_accuracy']} | {f['oracle_gap_recovered_rate']} |"
        )

    lines += [
        '',
        '## Limitations',
        '- Both designs are still a read-only replay over already-logged data; no SQL/LLM calls happen here.',
        '- k-fold cross-validation pools train priors from a fresh in-memory SplitInfo per fold; '
        'query_sets/mvm_v1/split.json is not read or modified.',
        '- Leave-one-correlate-family-out pools train candidates across *all* workloads other than the '
        'held-out family (prior computation is already workload-agnostic for rule/sequence priors -- see '
        'candidate_selection/mvm/prior.py), to maximize train support per fold.',
        '- Some held-out families have very few usable candidates (as few as 1), so their individual '
        'accuracy is a noisy single/few-sample estimate; read the aggregate mean/stdev, not any single row.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)

    config_dict = json.loads(args.config.read_text())
    candidates_path = args.candidates or _resolve(config_dict['dataset']['candidates_path'])

    print(f'[mvm_reliability_cv] loading {candidates_path}')
    records = load_candidates_jsonl(candidates_path)
    experiment_config = ExperimentConfig.from_dict(config_dict)

    print(f'[mvm_reliability_cv] running {args.k}-fold group-stratified CV ...')
    kfold = run_kfold(records, experiment_config, args.k)
    print(
        f"[mvm_reliability_cv] kfold cost-only acc mean={kfold['aggregate']['cost_only_accuracy']['mean']} "
        f"mvm acc mean={kfold['aggregate']['mvm_accuracy']['mean']} "
        f"oracle_gap_recovered_rate mean={kfold['aggregate']['oracle_gap_recovered_rate']['mean']}"
    )

    print('[mvm_reliability_cv] running leave-one-correlate-family-out ...')
    lofo = run_lofo(records, experiment_config)
    print(
        f"[mvm_reliability_cv] lofo cost-only acc mean={lofo['aggregate']['cost_only_accuracy']['mean']} "
        f"mvm acc mean={lofo['aggregate']['mvm_accuracy']['mean']} "
        f"oracle_gap_recovered_rate mean={lofo['aggregate']['oracle_gap_recovered_rate']['mean']}"
    )

    usable_families = usable_families_by_workload(records)
    snapshot = {
        'candidates_path': str(candidates_path),
        'candidates_sha256': sha256_file(candidates_path),
        'candidate_count': len(records),
        'usable_family_count': sum(len(v) for v in usable_families.values()),
        'usable_family_count_by_workload': {wl: len(v) for wl, v in usable_families.items()},
        'correlate_family_count': len(correlate_families(records)),
        'git': git_info(MY_REWRITER_DIR),
    }

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_json(run_dir / 'config.json', config_dict)
    atomic_write_json(run_dir / 'dataset_snapshot.json', snapshot)
    atomic_write_json(run_dir / 'kfold.json', kfold)
    atomic_write_json(run_dir / 'lofo_correlate.json', lofo)
    atomic_write_text(run_dir / 'report.md', render_report(snapshot, kfold, lofo))
    print(f'[mvm_reliability_cv] wrote run to {run_dir}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
