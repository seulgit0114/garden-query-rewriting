"""Writes one evaluation run's outputs to
experiments/mvm_reliability/outputs/<timestamp>_<run_name>/.

Every file is written via atomic_write_* (temp file + os.replace),
matching the pattern build_mvm_repository.py already uses for
query_sets/mvm_v1. make_run_dir() refuses to reuse or overwrite an
existing run directory.

write_official_adoption() is the *only* function in this package allowed
to touch query_sets/mvm_v1/results/mvm_evaluation.json, and it is only
ever called when the CLI is given --adopt-official explicitly (see
run_mvm_evaluation.py) -- never as a side effect of an ordinary run.
"""

from __future__ import annotations

import json
import os
import subprocess
import typing as t
from datetime import datetime, timezone
from pathlib import Path

from .dataset import DatasetSnapshot
from .evaluate import EvaluationResult
from .metrics import summarize_result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: t.Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + '\n')


def atomic_write_jsonl(path: Path, rows: t.Iterable[dict]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    atomic_write_text(path, '\n'.join(lines) + ('\n' if lines else ''))


def make_run_dir(outputs_dir: Path, run_name: str, now: t.Optional[datetime] = None) -> Path:
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%dT%H%M%S%f') + 'Z'
    run_dir = outputs_dir / f'{timestamp}_{run_name}'
    if run_dir.exists():
        raise FileExistsError(f'Run directory already exists, refusing to overwrite: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=False)
    return run_dir


def git_info(repo_dir: Path) -> dict:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_dir, text=True).strip())
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_dir, text=True
        ).strip()
        return {'commit': commit, 'dirty': dirty, 'branch': branch}
    except Exception as exc:  # pragma: no cover - defensive only, git is always present here
        return {'commit': None, 'dirty': None, 'branch': None, 'error': str(exc)}


def render_report_md(snapshot: dict, summary: dict) -> str:
    overall = summary['overall']
    cand = overall['candidates']
    groups = overall['groups']
    lines = [
        '# MVM Reliability Prior -- Offline Evaluation Report',
        '',
        f"Run directory: `{summary['run_dir']}`",
        '',
        '## Dataset snapshot',
        f"- candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"- split.json sha256: `{snapshot['split_sha256']}`",
        f"- candidates: {snapshot['candidate_count']} (usable: {snapshot['usable_count']})",
        f"- train candidates: {snapshot['train_candidate_count']}, "
        f"test candidates: {snapshot['test_candidate_count']}",
        f"- includes correlate-family rules: {snapshot['includes_correlate_rules']}",
        f"- git commit: {snapshot['git'].get('commit')} (dirty={snapshot['git'].get('dirty')}, "
        f"branch={snapshot['git'].get('branch')})",
        '',
        '## Candidate-level decision accuracy (test split, per logged rewrite candidate)',
        f"- candidates evaluated: {cand['candidate_count']}",
        f"- Cost-only accuracy: {cand['cost_only']['accuracy']}",
        f"- MVM accuracy: {cand['mvm']['accuracy']}",
        f"- Cost-only confusion: {cand['cost_only']['confusion']}",
        f"- MVM confusion: {cand['mvm']['confusion']}",
        f"- FP prevented by MVM: {cand['fp_prevented']}",
        f"- FN recovered by MVM: {cand['fn_recovered']}",
        f"- Correct cost-only decisions broken by MVM: {cand['broken_correct_decisions']}",
        f"- New FP introduced by MVM: {cand['new_fp']}",
        f"- New FN introduced by MVM: {cand['new_fn']}",
        '',
        '## Group-level latency (test split, one row per original query)',
        f"- groups evaluated: {groups['group_count']}",
        f"- Cost-only selected latency (ms): {groups['cost_selected_latency_ms']}",
        f"- MVM selected latency (ms): {groups['mvm_selected_latency_ms']}",
        f"- Original-only latency (ms): {groups['original_only_latency_ms']}",
        f"- Oracle latency (ms): {groups['oracle_latency_ms']}",
        f"- Oracle gap available over cost-only (ms): {groups['oracle_gap_available_ms']}",
        f"- Oracle gap recovered by MVM (ms): {groups['oracle_gap_recovered_ms']}",
        f"- Oracle gap recovered rate: {groups['oracle_gap_recovered_rate']}",
        '',
        '## Limitations',
        '- Candidates grouped under the same original query may come from different',
        '  ablation arms (source_ablations), so simultaneous availability as one',
        '  candidate pool is an approximation, not a replay of one single run.',
        '- Group-level "original latency" is the mean of that query\'s',
        '  per-row original_latency_ms measurements (paired-measurement noise).',
        '- This is a read-only replay over already-logged data; it does not run',
        '  or change the actual Query Rewriting pipeline.',
        '',
    ]
    return '\n'.join(lines)


def write_run(
    run_dir: Path,
    config_dict: dict,
    snapshot: DatasetSnapshot,
    result: EvaluationResult,
    repo_dir: Path,
) -> dict:
    atomic_write_json(run_dir / 'config.json', config_dict)

    snapshot_dict = dict(snapshot.to_dict())
    snapshot_dict['git'] = git_info(repo_dir)
    snapshot_dict['run_started_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(run_dir / 'dataset_snapshot.json', snapshot_dict)

    atomic_write_json(
        run_dir / 'rule_priors.json', result.prior_table.to_dict(result.train_candidate_count)
    )

    decisions = result.all_candidate_decisions
    atomic_write_jsonl(run_dir / 'candidate_decisions.jsonl', (d.to_dict() for d in decisions))

    metrics = summarize_result(result)
    atomic_write_json(run_dir / 'workload_metrics.json', metrics['by_workload'])

    summary = {
        'run_dir': str(run_dir),
        'test_candidate_count': len(decisions),
        'test_group_count': len(result.groups),
        'train_candidate_count': result.train_candidate_count,
        'overall': metrics['overall'],
        'by_rule': metrics['by_rule'],
        'by_rule_sequence': metrics['by_rule_sequence'],
        'by_prior_support_bucket': metrics['by_prior_support_bucket'],
    }
    atomic_write_json(run_dir / 'summary.json', summary)

    atomic_write_text(run_dir / 'report.md', render_report_md(snapshot_dict, summary))

    return summary


def write_official_adoption(mvm_v1_results_path: Path, run_dir: Path, summary: dict) -> None:
    """Only called when the CLI is given --adopt-official explicitly.
    Overwrites query_sets/mvm_v1/results/mvm_evaluation.json with a
    pointer to the officially-adopted run -- never called from the
    ordinary run flow."""
    cand = summary['overall']['candidates']
    groups = summary['overall']['groups']
    payload = {
        'status': 'evaluated',
        'official_run_id': Path(run_dir).name,
        'results_path': str(run_dir),
        'summary': {
            'test_candidate_count': summary['test_candidate_count'],
            'cost_only_accuracy': cand['cost_only']['accuracy'],
            'mvm_accuracy': cand['mvm']['accuracy'],
            'oracle_gap_recovered_rate': groups['oracle_gap_recovered_rate'],
        },
        'adopted_at': datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(mvm_v1_results_path, payload)
