"""Audit items 1+2: AST structural fingerprints for every usable candidate,
on BOTH the original and the rewritten SQL.

The existing fingerprint corpus (query_sets/mvm_v2/family_audit/
query_fingerprints.json) was produced by the data-expansion audit and only
covers tpch/imdb *source* families on the ORIGINAL query side -- 63 families,
47.8% of usable candidates, and nothing at all for rewritten SQL. Plan
Residual Retrieval needs both sides for every candidate.

This script reuses `fingerprint_sql()` from
codex/mvm_data_expansion/initialize_and_audit.py verbatim (loaded by path,
since that directory is not an importable package) rather than
reimplementing it, so the features here are identical in definition to the
ones already audited.

Read-only with respect to every existing file: reads candidates.jsonl and
writes a new run directory under experiments/mvm_reliability/outputs/.

Leakage note (audit item 5): `template_fingerprint` and
`normalized_literal_free_sql` identify a query template almost exactly and
must never be used as retrieval features -- they are emitted here for
diagnostics and for the leakage check itself, and are listed in
RETRIEVAL_FORBIDDEN_FIELDS so downstream code can assert on them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import typing as t
from collections import Counter, defaultdict
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.report import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    git_info,
    make_run_dir,
)

GENERATOR_PATH = MY_REWRITER_DIR / 'codex' / 'mvm_data_expansion' / 'initialize_and_audit.py'

# Identify a template almost exactly -> never usable as a retrieval feature.
RETRIEVAL_FORBIDDEN_FIELDS = (
    'template_fingerprint',
    'normalized_literal_free_sql',
)

# Free-form / high-cardinality fields kept for diagnostics but not numeric features.
NON_NUMERIC_FIELDS = (
    'parse_ok', 'parse_error', 'structural_fingerprint', 'correlation_detection',
    'tables', 'join_edges', 'join_types', 'predicate_types',
    'aggregation_functions', 'correlation_edges',
)


def load_fingerprint_fn():
    """Load fingerprint_sql() from the expansion audit script by path."""
    spec = importlib.util.spec_from_file_location('_mvm_expansion_audit', GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load fingerprint generator at {GENERATOR_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fingerprint_sql


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--candidates', type=Path,
        default=MY_REWRITER_DIR / 'query_sets' / 'mvm_v2' / 'candidates.jsonl',
        help='Dataset to fingerprint (default: mvm_v2, 494 usable)',
    )
    parser.add_argument('--run-name', default='candidate_fingerprints')
    parser.add_argument(
        '--outputs-dir', type=Path,
        default=Path(__file__).resolve().parent / 'outputs',
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fingerprint_sql = load_fingerprint_fn()

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    usable = [r for r in rows if r.get('usable')]

    out_rows: t.List[dict] = []
    fail: t.List[dict] = []
    for record in usable:
        entry: dict = {
            'id': record['id'],
            'workload': record['workload'],
            'query': record['query'],
            'query_family': record['query_family'],
            'rules': record.get('rules') or [],
        }
        for side, sql_field in (('original', 'original_sql'), ('rewritten', 'rewritten_sql')):
            fp = fingerprint_sql(record[sql_field])
            entry[side] = fp
            if not fp.get('parse_ok'):
                fail.append({
                    'id': record['id'], 'workload': record['workload'],
                    'query': record['query'], 'side': side,
                    'parse_error': fp.get('parse_error', '')[:200],
                })
        out_rows.append(entry)

    # ---- verification -------------------------------------------------
    n = len(out_rows)
    ok_orig = sum(1 for r in out_rows if r['original'].get('parse_ok'))
    ok_rw = sum(1 for r in out_rows if r['rewritten'].get('parse_ok'))
    both = sum(1 for r in out_rows if r['original'].get('parse_ok') and r['rewritten'].get('parse_ok'))

    by_workload = defaultdict(lambda: {'n': 0, 'orig_ok': 0, 'rw_ok': 0})
    for r in out_rows:
        b = by_workload[r['workload']]
        b['n'] += 1
        b['orig_ok'] += bool(r['original'].get('parse_ok'))
        b['rw_ok'] += bool(r['rewritten'].get('parse_ok'))

    # Did fingerprinting actually distinguish anything? A feature set that
    # gives every candidate the same vector would be useless for retrieval.
    def numeric_vector(fp: dict) -> tuple:
        return tuple(
            (k, fp.get(k)) for k in sorted(fp)
            if k not in RETRIEVAL_FORBIDDEN_FIELDS
            and k not in NON_NUMERIC_FIELDS
        )

    distinct_orig = len({numeric_vector(r['original']) for r in out_rows if r['original'].get('parse_ok')})
    distinct_pairs = len({
        (numeric_vector(r['original']), numeric_vector(r['rewritten']))
        for r in out_rows if r['original'].get('parse_ok') and r['rewritten'].get('parse_ok')
    })
    # Does the rewrite actually change the AST? If not, the rewrite-side
    # fingerprint carries no information beyond the original.
    changed = sum(
        1 for r in out_rows
        if r['original'].get('parse_ok') and r['rewritten'].get('parse_ok')
        and numeric_vector(r['original']) != numeric_vector(r['rewritten'])
    )

    correlate_rows = [r for r in out_rows if 'FILTER_SUB_QUERY_TO_CORRELATE' in r['rules']]
    correlate_changed = sum(
        1 for r in correlate_rows
        if r['original'].get('parse_ok') and r['rewritten'].get('parse_ok')
        and numeric_vector(r['original']) != numeric_vector(r['rewritten'])
    )

    summary = {
        'candidates_path': str(args.candidates),
        'total_rows': len(rows),
        'usable_rows': n,
        'original_parse_ok': ok_orig,
        'rewritten_parse_ok': ok_rw,
        'both_parse_ok': both,
        'parse_failures': len(fail),
        'by_workload': {k: dict(v) for k, v in sorted(by_workload.items())},
        'distinct_original_feature_vectors': distinct_orig,
        'distinct_original_rewritten_pairs': distinct_pairs,
        'rewrite_changes_ast': changed,
        'correlate_candidates': len(correlate_rows),
        'correlate_rewrite_changes_ast': correlate_changed,
        'retrieval_forbidden_fields': list(RETRIEVAL_FORBIDDEN_FIELDS),
        'git': git_info(MY_REWRITER_DIR),
    }

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_jsonl(run_dir / 'candidate_fingerprints.jsonl', out_rows)
    atomic_write_json(run_dir / 'summary.json', summary)
    if fail:
        atomic_write_jsonl(run_dir / 'parse_failures.jsonl', fail)

    lines = [
        '# Candidate fingerprints (audit items 1 + 2)',
        '',
        f"source: `{args.candidates}`",
        f"generator: `{GENERATOR_PATH}` :: fingerprint_sql() (reused verbatim)",
        f"git: {summary['git']['commit']} (dirty={summary['git']['dirty']})",
        '',
        '| metric | value |',
        '|---|---:|',
        f"| usable candidates | {n} |",
        f"| original side parsed | {ok_orig} ({ok_orig / n * 100:.1f}%) |",
        f"| rewritten side parsed | {ok_rw} ({ok_rw / n * 100:.1f}%) |",
        f"| both sides parsed | {both} ({both / n * 100:.1f}%) |",
        f"| parse failures | {len(fail)} |",
        f"| distinct original feature vectors | {distinct_orig} |",
        f"| distinct (original, rewritten) pairs | {distinct_pairs} |",
        f"| rewrite changes the AST | {changed} / {both} |",
        f"| correlate candidates | {len(correlate_rows)} |",
        f"| ... whose rewrite changes the AST | {correlate_changed} / {len(correlate_rows)} |",
        '',
        '## Per workload',
        '',
        '| workload | usable | original parsed | rewritten parsed |',
        '|---|---:|---:|---:|',
    ]
    for wl, b in summary['by_workload'].items():
        lines.append(f"| {wl} | {b['n']} | {b['orig_ok']} | {b['rw_ok']} |")
    lines += [
        '',
        '## Leakage guard (audit item 5)',
        '',
        'These fields identify a template almost exactly and MUST NOT be used as '
        'retrieval features: ' + ', '.join(f'`{f}`' for f in RETRIEVAL_FORBIDDEN_FIELDS) + '.',
        '',
    ]
    report = '\n'.join(lines)
    atomic_write_text(run_dir / 'report.md', report)
    print(report)
    print(f'[fingerprints] run written to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
