#!/usr/bin/env python3
"""CLI entrypoint for the conditional-MVM Minimum Viable Experiment (MVE).

Compares three policies -- Cost-only, global (unconditional) MVM, and
cost-decision-conditional MVM (candidate_selection/mvm/conditional_prior.py)
-- over the SAME family-grouped splits already used for the CV/LOFO
experiment (experiments/mvm_reliability/cv_split.py): group-stratified
k-fold CV (primary) and leave-one-correlate-family-out (correlate-specific).
Also computes the two plan-free residual definitions (A, B) for every
usable candidate, for descriptive analysis only (see conditional_evaluate.py
module docstring) -- this is NOT fed into any decision.

Reads query_sets/mvm_v1/candidates.jsonl read-only. Writes a new,
independent run directory under experiments/mvm_reliability/outputs/.
Never touches query_sets/mvm_v1/split.json or results/mvm_evaluation.json.

Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m experiments.mvm_reliability.run_conditional_mve \\
        --run-name conditional_mve --k 5
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

from experiments.mvm_reliability.conditional_evaluate import (  # noqa: E402
    residual_log_ratio,
    residual_signed,
    run_multi_evaluation,
    summarize_multi,
)
from experiments.mvm_reliability.cv_split import (  # noqa: E402
    correlate_families,
    family_kfold_splits,
    leave_one_family_out_splits,
    usable_families_by_workload,
)
from experiments.mvm_reliability.dataset import load_candidates_jsonl, sha256_file  # noqa: E402
from experiments.mvm_reliability.evaluate import ExperimentConfig  # noqa: E402
from experiments.mvm_reliability.report import atomic_write_json, atomic_write_jsonl, atomic_write_text, git_info, make_run_dir  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'mvm_v1.json',
        help='Path to an experiment config JSON (default: configs/mvm_v1.json)',
    )
    parser.add_argument('--candidates', type=Path, default=None, help='Override candidates.jsonl path')
    parser.add_argument('--k', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--run-name', default='conditional_mve', help='Suffix for the run directory name')
    parser.add_argument(
        '--outputs-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs',
        help='Directory under which a new <timestamp>_<run-name> run directory is created',
    )
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (MY_REWRITER_DIR / path)


def _mean_std(values: t.Sequence[t.Optional[float]]) -> dict:
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


def _fold_row(summary: dict) -> dict:
    cand = summary['candidates']
    groups = summary['groups']
    return {
        'test_group_count': summary['test_group_count'],
        'test_candidate_count': summary['test_candidate_count'],
        'train_candidate_count': summary['train_candidate_count'],
        'accept_train_count': summary['accept_train_count'],
        'reject_train_count': summary['reject_train_count'],
        'cost_only_accuracy': cand['cost_only']['accuracy'],
        'cost_only_f1': cand['cost_only']['f1'],
        'global_mvm_accuracy': cand['global_mvm']['accuracy'],
        'global_mvm_f1': cand['global_mvm']['f1'],
        'cond_mvm_accuracy': cand['cond_mvm']['accuracy'],
        'cond_mvm_f1': cand['cond_mvm']['f1'],
        'cost_only_confusion': cand['cost_only']['confusion'],
        'global_mvm_confusion': cand['global_mvm']['confusion'],
        'cond_mvm_confusion': cand['cond_mvm']['confusion'],
        'cost_selected_latency_mean_ms': groups['cost_selected_latency_ms']['mean'],
        'global_mvm_selected_latency_mean_ms': groups['global_mvm_selected_latency_ms']['mean'],
        'cond_mvm_selected_latency_mean_ms': groups['cond_mvm_selected_latency_ms']['mean'],
        'cond_mvm_latency_aware_selected_latency_mean_ms': groups['cond_mvm_latency_aware_selected_latency_ms']['mean'],
        'cond_mvm_latency_aware_median_selected_latency_mean_ms': (
            groups['cond_mvm_latency_aware_median_selected_latency_ms']['mean']
        ),
        'oracle_latency_mean_ms': groups['oracle_latency_ms']['mean'],
        'global_mvm_oracle_gap_recovered_rate': groups['global_mvm_oracle_gap_recovered_rate'],
        'cond_mvm_oracle_gap_recovered_rate': groups['cond_mvm_oracle_gap_recovered_rate'],
        'cond_mvm_latency_aware_oracle_gap_recovered_rate': groups['cond_mvm_latency_aware_oracle_gap_recovered_rate'],
        'cond_mvm_latency_aware_median_oracle_gap_recovered_rate': (
            groups['cond_mvm_latency_aware_median_oracle_gap_recovered_rate']
        ),
        'group_rows': summary['group_rows'],
    }


def _aggregate(folds: t.List[dict]) -> dict:
    return {
        'n_folds': len(folds),
        'cost_only_accuracy': _mean_std([f['cost_only_accuracy'] for f in folds]),
        'global_mvm_accuracy': _mean_std([f['global_mvm_accuracy'] for f in folds]),
        'cond_mvm_accuracy': _mean_std([f['cond_mvm_accuracy'] for f in folds]),
        'cost_only_f1': _mean_std([f['cost_only_f1'] for f in folds]),
        'global_mvm_f1': _mean_std([f['global_mvm_f1'] for f in folds]),
        'cond_mvm_f1': _mean_std([f['cond_mvm_f1'] for f in folds]),
        'global_mvm_oracle_gap_recovered_rate': _mean_std([f['global_mvm_oracle_gap_recovered_rate'] for f in folds]),
        'cond_mvm_oracle_gap_recovered_rate': _mean_std([f['cond_mvm_oracle_gap_recovered_rate'] for f in folds]),
        'cond_mvm_latency_aware_oracle_gap_recovered_rate': _mean_std(
            [f['cond_mvm_latency_aware_oracle_gap_recovered_rate'] for f in folds]
        ),
        'cond_mvm_latency_aware_median_oracle_gap_recovered_rate': _mean_std(
            [f['cond_mvm_latency_aware_median_oracle_gap_recovered_rate'] for f in folds]
        ),
        'cost_selected_latency_mean_ms': _mean_std([f['cost_selected_latency_mean_ms'] for f in folds]),
        'global_mvm_selected_latency_mean_ms': _mean_std([f['global_mvm_selected_latency_mean_ms'] for f in folds]),
        'cond_mvm_selected_latency_mean_ms': _mean_std([f['cond_mvm_selected_latency_mean_ms'] for f in folds]),
        'cond_mvm_latency_aware_selected_latency_mean_ms': _mean_std(
            [f['cond_mvm_latency_aware_selected_latency_mean_ms'] for f in folds]
        ),
        'cond_mvm_latency_aware_median_selected_latency_mean_ms': _mean_std(
            [f['cond_mvm_latency_aware_median_selected_latency_mean_ms'] for f in folds]
        ),
        'oracle_latency_mean_ms': _mean_std([f['oracle_latency_mean_ms'] for f in folds]),
    }


def run_kfold(records, config: ExperimentConfig, k: int) -> dict:
    splits = family_kfold_splits(records, k)
    folds = []
    for idx, split in enumerate(splits):
        result = run_multi_evaluation(records, split, config)
        row = _fold_row(summarize_multi(result))
        row['fold_index'] = idx
        row['test_families'] = {wl: sorted(fams) for wl, fams in split.test_families.items()}
        folds.append(row)
    return {'k': k, 'folds': folds, 'aggregate': _aggregate(folds)}


def run_lofo(records, config: ExperimentConfig) -> dict:
    targets = correlate_families(records)
    lofo_splits = leave_one_family_out_splits(records, targets)
    folds = []
    for (workload, family), split in lofo_splits:
        result = run_multi_evaluation(records, split, config)
        row = _fold_row(summarize_multi(result))
        row['held_out_workload'] = workload
        row['held_out_family'] = family
        folds.append(row)
    return {'family_count': len(folds), 'folds': folds, 'aggregate': _aggregate(folds)}


POOLED_POLICY_KEYS = (
    ('cost_selected_latency_ms', 'Cost-only'),
    ('global_mvm_selected_latency_ms', 'Global MVM'),
    ('cond_mvm_selected_latency_ms', 'Conditional MVM'),
    ('cond_mvm_latency_aware_selected_latency_ms', 'Conditional MVM (latency-aware, mean)'),
    ('cond_mvm_latency_aware_median_selected_latency_ms', 'Conditional MVM (latency-aware, median)'),
    ('oracle_latency_ms', 'Oracle'),
)


def _trimmed_mean(sorted_desc: t.List[float], drop_n: int) -> t.Optional[float]:
    remaining = sorted_desc[drop_n:]
    return (sum(remaining) / len(remaining)) if remaining else None


def pooled_tail_sensitivity(group_rows: t.List[dict]) -> dict:
    """Pooled across ALL fold test groups (for k-fold, every usable family
    appears in exactly one fold's test set, so the union covers all 112
    families once each -- for LOFO, the union covers the 10 correlate
    families once each). For each policy: plain mean, median, trimmed
    mean after dropping the top-1 / top-3 / top-5% highest-latency
    groups, and how much the single largest-latency group contributes to
    the total (sensitivity check requested after the -755x outlier found
    in the residual analysis)."""
    n = len(group_rows)
    top5pct_n = max(1, round(n * 0.05)) if n else 0
    out: t.Dict[str, dict] = {}
    for key, label in POOLED_POLICY_KEYS:
        values = [row[key] for row in group_rows]
        total = sum(values)
        sorted_desc = sorted(values, reverse=True)
        top1 = sorted_desc[0] if sorted_desc else None
        out[key] = {
            'label': label,
            'n': n,
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'trimmed_mean_drop_top1': _trimmed_mean(sorted_desc, 1),
            'trimmed_mean_drop_top3': _trimmed_mean(sorted_desc, 3),
            'trimmed_mean_drop_top5pct': _trimmed_mean(sorted_desc, top5pct_n),
            'top1_value_ms': top1,
            'top1_share_of_total': (top1 / total) if total > 0 and top1 is not None else None,
        }
    return {'n_groups': n, 'top5pct_n_dropped': top5pct_n, 'by_policy': out}


def residual_analysis(records) -> dict:
    """Descriptive only -- computed once from ALL usable candidates,
    independent of fold/split. Never used as a decision input anywhere in
    this codebase; see conditional_evaluate.py module docstring."""
    usable = [r for r in records if r.usable]
    rows = []
    for r in usable:
        a = residual_log_ratio(r)
        b = residual_signed(r)
        cost_decision = 'rewrite' if r.rewritten_cost < r.original_cost else 'original'
        actual_positive = r.rewritten_latency_ms < r.original_latency_ms
        rows.append({
            'candidate_id': r.id, 'workload': r.workload, 'rules': list(r.rules),
            'cost_decision': cost_decision, 'actual_positive': actual_positive,
            'residual_log_ratio': a, 'residual_signed': b,
        })

    import math as _math

    def _log_ratio_pair(r):
        if r.original_cost > 0 and r.rewritten_cost > 0 and r.original_latency_ms > 0 and r.rewritten_latency_ms > 0:
            return (_math.log(r.original_cost / r.rewritten_cost), _math.log(r.original_latency_ms / r.rewritten_latency_ms))
        return None

    pairs = [p for p in (_log_ratio_pair(r) for r in usable) if p is not None]
    correlation = None
    if len(pairs) >= 2:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        try:
            correlation = statistics.correlation(xs, ys)
        except statistics.StatisticsError:
            correlation = None

    by_bucket: t.Dict[str, t.List[float]] = {'accept': [], 'reject': []}
    for r in usable:
        b = residual_signed(r)
        if b is None:
            continue
        bucket = 'accept' if r.rewritten_cost < r.original_cost else 'reject'
        by_bucket[bucket].append(b)

    correlate_rows = [r for r in usable if 'FILTER_SUB_QUERY_TO_CORRELATE' in r.rules]
    correlate_by_bucket: t.Dict[str, t.List[float]] = {'accept': [], 'reject': []}
    for r in correlate_rows:
        b = residual_signed(r)
        if b is None:
            continue
        bucket = 'accept' if r.rewritten_cost < r.original_cost else 'reject'
        correlate_by_bucket[bucket].append(b)

    return {
        'usable_candidate_count': len(usable),
        'cost_log_ratio_vs_actual_log_ratio_pearson_r': correlation,
        'cost_log_ratio_vs_actual_log_ratio_n': len(pairs),
        'residual_signed_by_cost_decision': {
            'accept': _mean_std(by_bucket['accept']),
            'reject': _mean_std(by_bucket['reject']),
        },
        'correlate_rule_residual_signed_by_cost_decision': {
            'accept': _mean_std(correlate_by_bucket['accept']),
            'reject': _mean_std(correlate_by_bucket['reject']),
        },
        'rows': rows,
    }


def _tail_table(tail: dict) -> t.List[str]:
    lines = [
        f"Pooled across {tail['n_groups']} test groups (top-5% = {tail['top5pct_n_dropped']} groups dropped).",
        '',
        '| policy | mean | median | trimmed mean (drop top-1) | trimmed mean (drop top-3) | '
        'trimmed mean (drop top-5%) | top-1 value (ms) | top-1 share of total |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, label in POOLED_POLICY_KEYS:
        s = tail['by_policy'][key]
        def _fmt(v):
            return f"{v:.1f}" if v is not None else 'n/a'
        def _fmtp(v):
            return f"{v:.1%}" if v is not None else 'n/a'
        lines.append(
            f"| {label} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['trimmed_mean_drop_top1'])} | "
            f"{_fmt(s['trimmed_mean_drop_top3'])} | {_fmt(s['trimmed_mean_drop_top5pct'])} | "
            f"{_fmt(s['top1_value_ms'])} | {_fmtp(s['top1_share_of_total'])} |"
        )
    return lines


def render_report(snapshot: dict, kfold: dict, lofo: dict, residual: dict, kfold_tail: dict, lofo_tail: dict) -> str:
    def _table(agg: dict) -> t.List[str]:
        lines = ['| metric | mean | stdev | n |', '|---|---:|---:|---:|']
        for key, label in (
            ('cost_only_accuracy', 'Cost-only accuracy'),
            ('global_mvm_accuracy', 'Global MVM accuracy'),
            ('cond_mvm_accuracy', 'Conditional MVM accuracy'),
            ('cost_only_f1', 'Cost-only F1'),
            ('global_mvm_f1', 'Global MVM F1'),
            ('cond_mvm_f1', 'Conditional MVM F1'),
            ('global_mvm_oracle_gap_recovered_rate', 'Global MVM oracle gap recovered rate'),
            ('cond_mvm_oracle_gap_recovered_rate', 'Conditional MVM oracle gap recovered rate'),
            ('cond_mvm_latency_aware_oracle_gap_recovered_rate', 'Conditional MVM (latency-aware, mean) oracle gap recovered rate'),
            ('cond_mvm_latency_aware_median_oracle_gap_recovered_rate', 'Conditional MVM (latency-aware, median) oracle gap recovered rate'),
            ('cost_selected_latency_mean_ms', 'Cost-only selected latency (mean, ms)'),
            ('global_mvm_selected_latency_mean_ms', 'Global MVM selected latency (mean, ms)'),
            ('cond_mvm_selected_latency_mean_ms', 'Conditional MVM selected latency (mean, ms)'),
            ('cond_mvm_latency_aware_selected_latency_mean_ms', 'Conditional MVM (latency-aware, mean) selected latency (mean, ms)'),
            ('cond_mvm_latency_aware_median_selected_latency_mean_ms', 'Conditional MVM (latency-aware, median) selected latency (mean, ms)'),
            ('oracle_latency_mean_ms', 'Oracle latency (mean, ms)'),
        ):
            stat = agg[key]
            mean = f"{stat['mean']:.4f}" if stat['mean'] is not None else 'n/a'
            std = f"{stat['stdev']:.4f}" if stat['stdev'] is not None else 'n/a'
            lines.append(f"| {label} | {mean} | {std} | {stat['n']} |")
        return lines

    lines = [
        '# Conditional MVM -- Minimum Viable Experiment (MVE)',
        '',
        f"candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"candidates: {snapshot['candidate_count']} (usable families: {snapshot['usable_family_count']})",
        f"git commit: {snapshot['git'].get('commit')} (dirty={snapshot['git'].get('dirty')}, branch={snapshot['git'].get('branch')})",
        '',
        f"## Group-stratified {kfold['aggregate']['n_folds']}-fold CV (all usable query families)",
        '',
    ] + _table(kfold['aggregate']) + [
        '',
        '### Per-fold detail',
        '',
        '| fold | test cand. | cost-only acc | global MVM acc | cond MVM acc | global gap-rec. | cond gap-rec. |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for f in kfold['folds']:
        lines.append(
            f"| {f['fold_index']} | {f['test_candidate_count']} | {f['cost_only_accuracy']:.3f} | "
            f"{f['global_mvm_accuracy']:.3f} | {f['cond_mvm_accuracy']:.3f} | "
            f"{f['global_mvm_oracle_gap_recovered_rate']} | {f['cond_mvm_oracle_gap_recovered_rate']} |"
        )

    lines += [
        '',
        '### Tail sensitivity (pooled across all k-fold test groups)',
        '',
    ] + _tail_table(kfold_tail) + [
        '',
        f"## Leave-one-correlate-family-out ({lofo['aggregate']['n_folds']} families)",
        '',
    ] + _table(lofo['aggregate']) + [
        '',
        '### Per-family detail',
        '',
        '| workload | family | test cand. | cost-only acc | global MVM acc | cond MVM acc | global gap-rec. | cond gap-rec. |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for f in lofo['folds']:
        lines.append(
            f"| {f['held_out_workload']} | {f['held_out_family']} | {f['test_candidate_count']} | "
            f"{f['cost_only_accuracy']:.3f} | {f['global_mvm_accuracy']:.3f} | {f['cond_mvm_accuracy']:.3f} | "
            f"{f['global_mvm_oracle_gap_recovered_rate']} | {f['cond_mvm_oracle_gap_recovered_rate']} |"
        )

    lines += [
        '',
        '### Tail sensitivity (pooled across all LOFO test groups)',
        '',
    ] + _tail_table(lofo_tail)

    r = residual
    lines += [
        '',
        '## Residual analysis (descriptive only, not used in any decision)',
        '',
        f"- usable candidates: {r['usable_candidate_count']}",
        f"- Pearson r(cost log-ratio, actual log-ratio): {r['cost_log_ratio_vs_actual_log_ratio_pearson_r']} "
        f"(n={r['cost_log_ratio_vs_actual_log_ratio_n']})",
        f"- residual_signed (B) mean, cost accepted rows: {r['residual_signed_by_cost_decision']['accept']['mean']} "
        f"(n={r['residual_signed_by_cost_decision']['accept']['n']})",
        f"- residual_signed (B) mean, cost rejected rows: {r['residual_signed_by_cost_decision']['reject']['mean']} "
        f"(n={r['residual_signed_by_cost_decision']['reject']['n']})",
        f"- FILTER_SUB_QUERY_TO_CORRELATE residual_signed mean, cost accepted: "
        f"{r['correlate_rule_residual_signed_by_cost_decision']['accept']['mean']} "
        f"(n={r['correlate_rule_residual_signed_by_cost_decision']['accept']['n']})",
        f"- FILTER_SUB_QUERY_TO_CORRELATE residual_signed mean, cost rejected: "
        f"{r['correlate_rule_residual_signed_by_cost_decision']['reject']['mean']} "
        f"(n={r['correlate_rule_residual_signed_by_cost_decision']['reject']['n']})",
        '',
        '## Limitations',
        '- Read-only replay over already-logged data; no SQL/LLM/DB calls.',
        '- Residual definitions A and B use only original_cost/rewritten_cost/original_latency_ms/'
        'rewritten_latency_ms already in candidates.jsonl -- no plan features, computed for reporting only.',
        '- Conditional MVM partitions train rows by cost decision using the *same* CostOnlyPolicy the '
        'runtime policy uses (candidate_selection/mvm/conditional_prior.py), so accept/reject-side priors '
        'are exactly consistent with what fp_guard/fn_recovery see.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_dict = json.loads(args.config.read_text())
    candidates_path = args.candidates or _resolve(config_dict['dataset']['candidates_path'])

    print(f'[conditional_mve] loading {candidates_path}')
    records = load_candidates_jsonl(candidates_path)
    experiment_config = ExperimentConfig.from_dict(config_dict)

    print(f'[conditional_mve] running {args.k}-fold group-stratified CV ...')
    kfold = run_kfold(records, experiment_config, args.k)
    print(
        f"[conditional_mve] kfold cost-only={kfold['aggregate']['cost_only_accuracy']['mean']} "
        f"global_mvm={kfold['aggregate']['global_mvm_accuracy']['mean']} "
        f"cond_mvm={kfold['aggregate']['cond_mvm_accuracy']['mean']}"
    )

    print('[conditional_mve] running leave-one-correlate-family-out ...')
    lofo = run_lofo(records, experiment_config)
    print(
        f"[conditional_mve] lofo cost-only={lofo['aggregate']['cost_only_accuracy']['mean']} "
        f"global_mvm={lofo['aggregate']['global_mvm_accuracy']['mean']} "
        f"cond_mvm={lofo['aggregate']['cond_mvm_accuracy']['mean']}"
    )

    print('[conditional_mve] computing residual analysis (descriptive only) ...')
    residual = residual_analysis(records)

    kfold_pooled_rows = [row for fold in kfold['folds'] for row in fold['group_rows']]
    lofo_pooled_rows = [row for fold in lofo['folds'] for row in fold['group_rows']]
    kfold_tail = pooled_tail_sensitivity(kfold_pooled_rows)
    lofo_tail = pooled_tail_sensitivity(lofo_pooled_rows)
    print(
        f"[conditional_mve] kfold pooled ({kfold_tail['n_groups']} groups): "
        f"cost mean={kfold_tail['by_policy']['cost_selected_latency_ms']['mean']:.1f}ms "
        f"median={kfold_tail['by_policy']['cost_selected_latency_ms']['median']:.1f}ms | "
        f"cond mean={kfold_tail['by_policy']['cond_mvm_selected_latency_ms']['mean']:.1f}ms "
        f"median={kfold_tail['by_policy']['cond_mvm_selected_latency_ms']['median']:.1f}ms"
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
    atomic_write_json(run_dir / 'kfold_tail_sensitivity.json', kfold_tail)
    atomic_write_json(run_dir / 'lofo_tail_sensitivity.json', lofo_tail)
    residual_rows = residual.pop('rows')
    atomic_write_json(run_dir / 'residual_summary.json', residual)
    atomic_write_jsonl(run_dir / 'residual_rows.jsonl', residual_rows)
    atomic_write_text(run_dir / 'report.md', render_report(snapshot, kfold, lofo, residual, kfold_tail, lofo_tail))
    print(f'[conditional_mve] wrote run to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
