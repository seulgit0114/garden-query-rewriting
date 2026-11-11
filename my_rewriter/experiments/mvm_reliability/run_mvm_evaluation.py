#!/usr/bin/env python3
"""CLI entrypoint for the MVM reliability offline evaluation.

Reads query_sets/mvm_v1/{candidates.jsonl,split.json} read-only, runs the
Cost-only vs Cost+MVM replay over the test split, and writes a new,
independent run directory under experiments/mvm_reliability/outputs/.

Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m experiments.mvm_reliability.run_mvm_evaluation \\
        --run-name pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.dataset import (  # noqa: E402
    DatasetIntegrityError,
    build_snapshot,
    load_candidates_jsonl,
    load_split,
    validate_dataset,
)
from experiments.mvm_reliability.evaluate import (  # noqa: E402
    ExperimentConfig,
    run_evaluation,
    select_pilot_families,
)
from experiments.mvm_reliability.report import make_run_dir, write_official_adoption, write_run  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'mvm_v1.json',
        help='Path to an experiment config JSON (default: configs/mvm_v1.json)',
    )
    parser.add_argument('--run-name', default='run', help="Suffix for the run directory name")
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'outputs',
        help='Directory under which a new <timestamp>_<run-name> run directory is created',
    )
    parser.add_argument(
        '--adopt-official',
        action='store_true',
        help='After writing the run, also overwrite '
             'query_sets/mvm_v1/results/mvm_evaluation.json with a pointer to this run. '
             'Off by default -- must be passed explicitly.',
    )
    parser.add_argument(
        '--pilot-families-per-workload',
        type=int,
        default=None,
        help='If set, restrict the test split to this many query families per workload, '
             'chosen deterministically from train/structural information only '
             '(never test latency or label) -- see evaluate.select_pilot_families(). '
             'Omit for a full test-split run.',
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=None,
        help='Use this exact, already-created run directory instead of creating a new '
             '<timestamp>_<run-name> one under --outputs-dir. Must not already contain '
             'summary.json (refuses to overwrite a completed run). Intended for nohup '
             'launches where the caller needs to know the log path before the process starts.',
    )
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (MY_REWRITER_DIR / path)


def main(argv=None) -> int:
    args = parse_args(argv)

    config_dict = json.loads(args.config.read_text())
    candidates_path = _resolve(config_dict['dataset']['candidates_path'])
    split_path = _resolve(config_dict['dataset']['split_path'])

    print(f'[mvm_reliability] loading {candidates_path}')
    records = load_candidates_jsonl(candidates_path)
    print(f'[mvm_reliability] loading {split_path}')
    split = load_split(split_path)

    print(f'[mvm_reliability] validating {len(records)} candidate records ...')
    try:
        validate_dataset(records, split)
    except DatasetIntegrityError as exc:
        print(f'[mvm_reliability] DATASET INTEGRITY CHECK FAILED:\n{exc}', file=sys.stderr)
        return 1
    print('[mvm_reliability] dataset integrity checks passed.')

    snapshot = build_snapshot(candidates_path, split_path, records, split)
    experiment_config = ExperimentConfig.from_dict(config_dict)

    pilot_families = None
    if args.pilot_families_per_workload is not None:
        pilot_families = select_pilot_families(records, split, args.pilot_families_per_workload)
        print(f'[mvm_reliability] PILOT MODE: restricting test split to families: {pilot_families}')
        config_dict = dict(config_dict)
        config_dict['pilot'] = {
            'families_per_workload': args.pilot_families_per_workload,
            'selected_families': pilot_families,
        }

    result = run_evaluation(records, split, experiment_config, test_family_filter=pilot_families)
    print(
        f'[mvm_reliability] train candidates: {result.train_candidate_count}, '
        f'test groups: {len(result.groups)}, '
        f'test candidates: {len(result.all_candidate_decisions)}'
    )

    if args.run_dir is not None:
        run_dir = args.run_dir
        if (run_dir / 'summary.json').exists():
            raise FileExistsError(f'Run directory already has a completed run, refusing to overwrite: {run_dir}')
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(args.outputs_dir, args.run_name)
    summary = write_run(run_dir, config_dict, snapshot, result, repo_dir=MY_REWRITER_DIR)
    print(f'[mvm_reliability] wrote run to {run_dir}')

    cand = summary['overall']['candidates']
    groups = summary['overall']['groups']
    print(f"[mvm_reliability] cost-only accuracy={cand['cost_only']['accuracy']} "
          f"mvm accuracy={cand['mvm']['accuracy']}")
    print(f"[mvm_reliability] oracle_gap_recovered_rate={groups['oracle_gap_recovered_rate']}")

    if args.adopt_official:
        results_path = MY_REWRITER_DIR / 'query_sets' / 'mvm_v1' / 'results' / 'mvm_evaluation.json'
        write_official_adoption(results_path, run_dir, summary)
        print(f'[mvm_reliability] adopted as official result: {results_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
