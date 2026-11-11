#!/usr/bin/env python3
"""CLI entrypoint for the SECONDARY within-family instance-level split
(Priority-1 item: "known-template generalization" -- does the prior
generalize to a new parameter/column variant of a query template it has
already partially seen -- as opposed to the PRIMARY family-grouped CV,
which tests generalization to an entirely unseen template).

Explicitly secondary: this evaluation is expected to look more optimistic
than the primary family-CV, because sibling instances of the same family
often share join structure/predicate shape (see the design-review
document); it must never be reported as if it were the primary
unseen-template result.

Reads query_sets/mvm_v1/candidates.jsonl read-only. Writes a new run
directory under experiments/mvm_reliability/outputs/.

Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m experiments.mvm_reliability.run_within_family_split \\
        --run-name within_family --k 5
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

from experiments.mvm_reliability.conditional_evaluate import summarize_multi  # noqa: E402
from experiments.mvm_reliability.dataset import load_candidates_jsonl, sha256_file  # noqa: E402
from experiments.mvm_reliability.evaluate import ExperimentConfig  # noqa: E402
from experiments.mvm_reliability.instance_split import (  # noqa: E402
    instance_kfold_splits,
    multi_instance_families,
    run_instance_split_evaluation,
    usable_instances_by_family,
)
from experiments.mvm_reliability.report import atomic_write_json, atomic_write_text, git_info, make_run_dir  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config', type=Path, default=Path(__file__).resolve().parent / 'configs' / 'mvm_v1.json'
    )
    parser.add_argument('--candidates', type=Path, default=None)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--run-name', default='within_family')
    parser.add_argument('--outputs-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (MY_REWRITER_DIR / path)


def _mean_std(values: t.Sequence[t.Optional[float]]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {'n': 0, 'mean': None, 'stdev': None}
    return {'n': len(clean), 'mean': statistics.mean(clean), 'stdev': statistics.stdev(clean) if len(clean) >= 2 else None}


def main(argv=None) -> int:
    args = parse_args(argv)
    config_dict = json.loads(args.config.read_text())
    candidates_path = args.candidates or _resolve(config_dict['dataset']['candidates_path'])

    records = load_candidates_jsonl(candidates_path)
    experiment_config = ExperimentConfig.from_dict(config_dict)
    print(f'[within_family] loaded {len(records)} candidates from {candidates_path}')

    eligible = multi_instance_families(records)
    by_fam = usable_instances_by_family(records)
    eligible_instance_count = {wl: sum(len(by_fam[wl][f]) for f in fams) for wl, fams in eligible.items()}
    eligible_family_count = {wl: len(f) for wl, f in eligible.items()}
    print(f'[within_family] multi-instance families by workload: {eligible_family_count}')
    print(f'[within_family] eligible instance count by workload: {eligible_instance_count}')

    splits = instance_kfold_splits(records, args.k)
    folds = []
    for idx, test_instances in enumerate(splits):
        result = run_instance_split_evaluation(records, test_instances, experiment_config)
        summary = summarize_multi(result)
        cand = summary['candidates']
        groups = summary['groups']
        folds.append({
            'fold_index': idx,
            'test_group_count': summary['test_group_count'],
            'test_candidate_count': summary['test_candidate_count'],
            'train_candidate_count': summary['train_candidate_count'],
            'cost_only_accuracy': cand['cost_only']['accuracy'],
            'global_mvm_accuracy': cand['global_mvm']['accuracy'],
            'cond_mvm_accuracy': cand['cond_mvm']['accuracy'],
            'cond_mvm_latency_aware_median_oracle_gap_recovered_rate': (
                groups['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
            ),
            'cond_mvm_latency_aware_median_selected_latency_mean_ms': (
                groups['cond_mvm_latency_aware_median_selected_latency_ms']['mean']
            ),
            'cost_selected_latency_mean_ms': groups['cost_selected_latency_ms']['mean'],
        })
        print(f"  fold {idx}: test_cand={summary['test_candidate_count']} "
              f"cost_acc={cand['cost_only']['accuracy']} cond_acc={cand['cond_mvm']['accuracy']} "
              f"gap_rec={groups['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']}")

    aggregate = {
        'cost_only_accuracy': _mean_std([f['cost_only_accuracy'] for f in folds]),
        'global_mvm_accuracy': _mean_std([f['global_mvm_accuracy'] for f in folds]),
        'cond_mvm_accuracy': _mean_std([f['cond_mvm_accuracy'] for f in folds]),
        'cond_mvm_latency_aware_median_oracle_gap_recovered_rate': _mean_std(
            [f['cond_mvm_latency_aware_median_oracle_gap_recovered_rate'] for f in folds]
        ),
        'cond_mvm_latency_aware_median_selected_latency_mean_ms': _mean_std(
            [f['cond_mvm_latency_aware_median_selected_latency_mean_ms'] for f in folds]
        ),
        'cost_selected_latency_mean_ms': _mean_std([f['cost_selected_latency_mean_ms'] for f in folds]),
    }
    print(f"[within_family] cost-only acc mean={aggregate['cost_only_accuracy']['mean']} "
          f"cond_mvm acc mean={aggregate['cond_mvm_accuracy']['mean']} "
          f"gap_rec mean={aggregate['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']['mean']}")

    snapshot = {
        'candidates_path': str(candidates_path),
        'candidates_sha256': sha256_file(candidates_path),
        'candidate_count': len(records),
        'eligible_multi_instance_families_by_workload': eligible_family_count,
        'eligible_instance_count_by_workload': eligible_instance_count,
        'git': git_info(MY_REWRITER_DIR),
        'k': args.k,
    }

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_json(run_dir / 'config.json', config_dict)
    atomic_write_json(run_dir / 'dataset_snapshot.json', snapshot)
    atomic_write_json(run_dir / 'folds.json', folds)
    atomic_write_json(run_dir / 'aggregate.json', aggregate)

    report_lines = [
        '# MVM Reliability Prior -- Within-Family Instance Split (SECONDARY, known-template generalization)',
        '',
        f"candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"eligible multi-instance families by workload: {snapshot['eligible_multi_instance_families_by_workload']}",
        f"eligible instance count by workload: {snapshot['eligible_instance_count_by_workload']} "
        f"(calcite has 0 -- its family names have no numeric-suffix variants, so it cannot participate)",
        '',
        '## Aggregate across folds',
        '',
        '| metric | mean | stdev | n |',
        '|---|---:|---:|---:|',
    ]
    for key, label in (
        ('cost_only_accuracy', 'Cost-only accuracy'),
        ('global_mvm_accuracy', 'Global MVM accuracy'),
        ('cond_mvm_accuracy', 'Conditional MVM accuracy'),
        ('cond_mvm_latency_aware_median_oracle_gap_recovered_rate', 'Conditional MVM (latency-aware median) oracle gap recovered rate'),
        ('cost_selected_latency_mean_ms', 'Cost-only selected latency (mean, ms)'),
        ('cond_mvm_latency_aware_median_selected_latency_mean_ms', 'Conditional MVM (latency-aware median) selected latency (mean, ms)'),
    ):
        s = aggregate[key]
        mean = f"{s['mean']:.4f}" if s['mean'] is not None else 'n/a'
        std = f"{s['stdev']:.4f}" if s['stdev'] is not None else 'n/a'
        report_lines.append(f"| {label} | {mean} | {std} | {s['n']} |")

    report_lines += [
        '',
        '### Per-fold detail',
        '',
        '| fold | test cand. | cost-only acc | cond MVM acc | gap-rec |',
        '|---:|---:|---:|---:|---:|',
    ]
    for f in folds:
        report_lines.append(
            f"| {f['fold_index']} | {f['test_candidate_count']} | {f['cost_only_accuracy']} | "
            f"{f['cond_mvm_accuracy']} | {f['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']} |"
        )

    report_lines += [
        '',
        '## Interpretation notes',
        '- SECONDARY evaluation only -- known-template (same family, new parameter/column instance) generalization.',
        '- Expected to look more optimistic than the primary family-grouped CV, because sibling instances of the '
        'same family often share join structure/predicate shape with train rows from the *same* family.',
        '- calcite cannot participate (no numeric-suffix instance variants in this dataset), so this result should '
        'not be read as covering calcite at all.',
        '- Read-only replay over already-logged data; no SQL/LLM/DB calls.',
        '',
    ]
    atomic_write_text(run_dir / 'report.md', '\n'.join(report_lines))
    print(f'[within_family] wrote run to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
