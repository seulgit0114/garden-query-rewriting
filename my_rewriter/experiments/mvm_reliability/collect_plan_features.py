"""Audit item 3: collect PostgreSQL execution plans for every usable
candidate, on both the original and the rewritten SQL.

`EXPLAIN (FORMAT JSON)` only plans -- it never executes the statement --
so this is read-only against the workload databases and costs milliseconds
per query. A statement timeout is still set as a guard.

Why this exists: audit items 1/2 showed the AST fingerprint is constant
across the numeric instance variants of a family (60 of 80 multi-instance
families collapse to a single feature vector), so AST features cannot tell
`query054_0` from `query054_1` -- which differ by 58x in measured latency.
The planner sees the literal predicates and estimates their selectivity, so
the plan is the only remaining source of instance-level signal. This script
collects it and immediately tests exactly that: do plan features separate
instances that the AST cannot?

Workload -> database mapping matches run_correlate_cost_latency_collection.sh.
Writes a new run directory under experiments/mvm_reliability/outputs/;
no existing file is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import typing as t
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from my_rewriter.config import init_db_config  # noqa: E402

from experiments.mvm_reliability.report import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    git_info,
    make_run_dir,
)

WORKLOAD_DB = {'calcite': 'calcite10', 'dsb': 'dsb10', 'imdb': 'imdb', 'tpch': 'tpch10'}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--candidates', type=Path,
        default=MY_REWRITER_DIR / 'query_sets' / 'mvm_v2' / 'candidates.jsonl',
    )
    parser.add_argument('--timeout', type=int, default=30, help='statement_timeout seconds (planning only)')
    parser.add_argument('--run-name', default='plan_features')
    parser.add_argument('--outputs-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    return parser.parse_args(argv)


def walk_plan(node: dict) -> t.Iterator[t.Tuple[int, dict]]:
    stack = [(0, node)]
    while stack:
        depth, cur = stack.pop()
        yield depth, cur
        for child in cur.get('Plans') or []:
            stack.append((depth + 1, child))


def plan_features(plan_json: list) -> dict:
    """Numeric + categorical features from one EXPLAIN (FORMAT JSON) result."""
    root = plan_json[0]['Plan']
    node_types: Counter = Counter()
    scan_relations: t.Dict[str, str] = {}
    join_types: Counter = Counter()
    max_depth = 0
    total_rows = 0.0
    n_nodes = 0
    for depth, node in walk_plan(root):
        n_nodes += 1
        max_depth = max(max_depth, depth)
        ntype = node.get('Node Type', '')
        node_types[ntype] += 1
        total_rows += float(node.get('Plan Rows') or 0.0)
        if 'Relation Name' in node:
            scan_relations[node['Relation Name']] = ntype
        if 'Join Type' in node:
            join_types[f"{ntype}:{node['Join Type']}"] += 1
    return {
        'root_total_cost': float(root.get('Total Cost') or 0.0),
        'root_startup_cost': float(root.get('Startup Cost') or 0.0),
        'root_plan_rows': float(root.get('Plan Rows') or 0.0),
        'root_plan_width': int(root.get('Plan Width') or 0),
        'node_count': n_nodes,
        'plan_depth': max_depth,
        'sum_plan_rows': total_rows,
        'node_types': dict(sorted(node_types.items())),
        'join_types': dict(sorted(join_types.items())),
        'scan_relations': dict(sorted(scan_relations.items())),
        'distinct_node_types': len(node_types),
        'seq_scan_count': node_types.get('Seq Scan', 0),
        'index_scan_count': node_types.get('Index Scan', 0) + node_types.get('Index Only Scan', 0),
        'hash_join_count': node_types.get('Hash Join', 0),
        'nested_loop_count': node_types.get('Nested Loop', 0),
        'merge_join_count': node_types.get('Merge Join', 0),
        'sort_count': node_types.get('Sort', 0),
        'agg_count': node_types.get('Aggregate', 0) + node_types.get('HashAggregate', 0),
    }


NUMERIC_FEATURE_KEYS = (
    'root_total_cost', 'root_startup_cost', 'root_plan_rows', 'root_plan_width',
    'node_count', 'plan_depth', 'sum_plan_rows', 'distinct_node_types',
    'seq_scan_count', 'index_scan_count', 'hash_join_count',
    'nested_loop_count', 'merge_join_count', 'sort_count', 'agg_count',
)


def feature_vector(feat: dict) -> tuple:
    return tuple(feat.get(k) for k in NUMERIC_FEATURE_KEYS)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    usable = [r for r in rows if r.get('usable')]

    by_workload = defaultdict(list)
    for r in usable:
        by_workload[r['workload']].append(r)

    out: t.List[dict] = []
    failures: t.List[dict] = []
    started = time.time()

    for workload, records in sorted(by_workload.items()):
        dbname = WORKLOAD_DB[workload]
        cfg = init_db_config(dbname)
        conn = psycopg2.connect(
            database=dbname, user=cfg['user'], password=cfg['password'],
            host=cfg['host'], port=cfg['port'],
            options=f'-c statement_timeout={args.timeout}s',
        )
        t0 = time.time()
        ok = 0
        for record in records:
            entry = {
                'id': record['id'], 'workload': workload, 'database': dbname,
                'query': record['query'], 'query_family': record['query_family'],
                'rules': record.get('rules') or [],
            }
            for side, field in (('original', 'original_sql'), ('rewritten', 'rewritten_sql')):
                sql = record[field].rstrip().rstrip(';')
                try:
                    cur = conn.cursor()
                    cur.execute('EXPLAIN (FORMAT JSON) ' + sql)
                    plan = cur.fetchone()[0]
                    cur.close()
                    entry[side] = {'ok': True, **plan_features(plan), 'plan': plan}
                    ok += 1
                except Exception as exc:
                    conn.rollback()
                    entry[side] = {'ok': False, 'error': str(exc)[:300]}
                    failures.append({
                        'id': record['id'], 'workload': workload,
                        'query': record['query'], 'side': side, 'error': str(exc)[:300],
                    })
            out.append(entry)
        conn.close()
        print(f'[plan] {workload:8s} db={dbname:10s} candidates={len(records):3d} '
              f'explains_ok={ok}/{len(records) * 2} elapsed={time.time() - t0:.1f}s', flush=True)

    # ---- verification --------------------------------------------------
    n = len(out)
    both_ok = [r for r in out if r['original'].get('ok') and r['rewritten'].get('ok')]

    # THE decisive test: do plan features separate instances within a family,
    # where the AST fingerprint could not?
    fam_instances = defaultdict(lambda: defaultdict(set))
    for r in both_ok:
        fam_instances[(r['workload'], r['query_family'])][r['query']].add(
            feature_vector(r['original'])
        )
    multi = {k: v for k, v in fam_instances.items() if len(v) > 1}
    separable = 0
    for key, insts in multi.items():
        vectors = {next(iter(s)) for s in insts.values() if len(s) == 1}
        if len(vectors) == len(insts):
            separable += 1

    # Family-identifiability of the plan vector (leakage screen, item 5).
    vec2fam = defaultdict(set)
    for r in both_ok:
        vec2fam[feature_vector(r['original'])].add((r['workload'], r['query_family']))
    family_unique = sum(1 for r in both_ok if len(vec2fam[feature_vector(r['original'])]) == 1)

    summary = {
        'candidates_path': str(args.candidates),
        'workload_database_map': WORKLOAD_DB,
        'statement_timeout_s': args.timeout,
        'usable_candidates': n,
        'both_sides_ok': len(both_ok),
        'explain_failures': len(failures),
        'elapsed_s': round(time.time() - started, 1),
        'multi_instance_families': len(multi),
        'multi_instance_families_separable_by_plan': separable,
        'distinct_original_plan_vectors': len(vec2fam),
        'candidates_whose_plan_vector_is_family_unique': family_unique,
        'numeric_feature_keys': list(NUMERIC_FEATURE_KEYS),
        'git': git_info(MY_REWRITER_DIR),
    }

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_jsonl(run_dir / 'plan_features.jsonl', out)
    atomic_write_json(run_dir / 'summary.json', summary)
    if failures:
        atomic_write_jsonl(run_dir / 'explain_failures.jsonl', failures)

    pct = (lambda a, b: f'{a / b * 100:.1f}%' if b else '--')
    lines = [
        '# Plan feature collection (audit item 3)',
        '',
        f"source: `{args.candidates}`",
        f"EXPLAIN (FORMAT JSON) only -- no statement was executed. timeout={args.timeout}s",
        f"git: {summary['git']['commit']} (dirty={summary['git']['dirty']})",
        '',
        '| metric | value |',
        '|---|---:|',
        f"| usable candidates | {n} |",
        f"| both sides planned | {len(both_ok)} ({pct(len(both_ok), n)}) |",
        f"| EXPLAIN failures | {len(failures)} |",
        f"| elapsed | {summary['elapsed_s']}s |",
        '',
        '## Decisive test: can the plan separate what the AST could not?',
        '',
        '| metric | value |',
        '|---|---:|',
        f"| multi-instance families | {len(multi)} |",
        f"| ... fully separable by plan vector | {separable} ({pct(separable, len(multi))}) |",
        '',
        '(AST baseline from audit items 1/2: 60 of 80 multi-instance families '
        'collapsed to a single feature vector, i.e. only 20 were separable.)',
        '',
        '## Leakage screen (audit item 5)',
        '',
        f"| candidates whose original plan vector uniquely identifies its family | "
        f"{family_unique} ({pct(family_unique, len(both_ok))}) |",
        '',
    ]
    report = '\n'.join(lines)
    atomic_write_text(run_dir / 'report.md', report)
    print(report)
    print(f'[plan] run written to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
