#!/usr/bin/env python3
"""srdg collector CLI — bounded BFS transition collection.

    python3 bin/collect.py --dry-run --limit 5
    python3 bin/collect.py --adapter fake --self-test
    python3 bin/collect.py --workload dsb --limit 3 --max-depth 2

Writes only under rdg_refinement/claude/data/<run_id>/:
    transitions.jsonl   append-only, one row per transition (02 §D)
    checkpoint.json     query-level resume, atomically replaced
    manifest.json       git HEAD / config / seed / inputs
    stats.json          per-query search statistics
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from srdg import canonical, config, manifest as mf   # noqa: E402
from srdg.calcite_adapter import CalciteAdapter, FakeAdapter  # noqa: E402
from srdg.search import SearchLimits, search_query   # noqa: E402

WORKLOAD_ROOTS = {
    'dsb': config.PROJECT_ROOT / 'workloads/dsb',
    'tpch': config.PROJECT_ROOT / 'workloads/tpch',
    'calcite': config.PROJECT_ROOT / 'workloads/calcite',
    'imdb': config.PROJECT_ROOT / 'workloads/imdb',
}


def load_queries(workload: str, limit=None):
    """(query_id, family, sql, create_tables). Mirrors build_rdg_empirical.py's
    loader but never writes and never touches the existing graph."""
    root = WORKLOAD_ROOTS[workload]
    schema_file = root / 'create_tables.sql'
    if not root.is_dir() or not schema_file.exists():
        return [], f'missing root or schema for {workload}: {root}'
    create_tables = [x for x in schema_file.read_text().split(';') if x.strip()]
    out = []
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(tdir.glob('*.sql')):
            raw = re.sub(r'--[^\n]*\n', '', f.read_text())
            for i, stmt in enumerate(s.strip() for s in raw.split(';')):
                if not stmt:
                    continue
                out.append({
                    'query_id': f'{tdir.name}/{f.stem}#{i}',
                    'query_family': tdir.name,
                    'sql': stmt + ';',
                    'create_tables': create_tables,
                })
                if limit and len(out) >= limit:
                    return out, None
    if not out:
        # calcite ships `calcite.jsonl` rather than <template>/<name>.sql, so this
        # loader yields nothing for it. Reported explicitly instead of silently
        # returning an empty list — a workload that quietly contributes no data
        # would corrupt any coverage comparison that includes it.
        return [], (f'{workload}: no <template>/*.sql found under {root} '
                    f'(this loader does not read .jsonl workloads)')
    return out, None


def self_test_graph():
    """Synthetic graph exercising no-op, cycle, convergence, failure."""
    return {
        'S0': [('A', 'S1'), ('D', 'S4')],
        'S1': [('B', 'S2'), ('C', 'S3')],
        'S2': [('E', 'S1')],       # cycle back to an ancestor
        'S3': [('B', 'S2')],       # convergence onto S2
        'S4': [],                  # no children (terminal)
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter', choices=('calcite', 'fake'), default='calcite')
    ap.add_argument('--workload', nargs='*', default=['dsb'], choices=list(WORKLOAD_ROOTS))
    ap.add_argument('--limit', type=int, default=None, help='max queries per workload')
    ap.add_argument('--max-depth', type=int, default=config.DEFAULT_MAX_DEPTH)
    ap.add_argument('--beam', type=int, default=config.DEFAULT_BEAM)
    ap.add_argument('--max-states', type=int, default=config.DEFAULT_MAX_STATES_PER_QUERY)
    ap.add_argument('--max-calls', type=int, default=config.DEFAULT_MAX_CALLS_PER_QUERY)
    ap.add_argument('--explore-quota', type=float, default=config.DEFAULT_EXPLORE_QUOTA)
    ap.add_argument('--literal-insensitive-dedup', action='store_true')
    ap.add_argument('--seed', type=int, default=config.DEFAULT_SEED)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='plan the work and write the manifest; call no Calcite')
    ap.add_argument('--self-test', action='store_true',
                    help='run the synthetic graph through the real search controller')
    a = ap.parse_args(argv)

    run_id = a.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    outdir = Path(a.outdir) if a.outdir else config.DATA / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    limits = SearchLimits(max_depth=a.max_depth, beam=a.beam, max_states=a.max_states,
                          max_calls=a.max_calls, explore_quota=a.explore_quota,
                          literal_insensitive_dedup=a.literal_insensitive_dedup)

    # ── assemble work ────────────────────────────────────────────────────────
    if a.self_test:
        adapter = FakeAdapter(graph=self_test_graph())
        work = [{'workload': 'synthetic', 'query_id': 'q0', 'query_family': 'fam0',
                 'sql': 'S0', 'create_tables': []}]
        load_errors = {}
    else:
        adapter = FakeAdapter(graph={}) if a.adapter == 'fake' else CalciteAdapter()
        work, load_errors = [], {}
        for w in a.workload:
            qs, err = load_queries(w, a.limit)
            if err:
                load_errors[w] = err
                continue
            for q in qs:
                q['workload'] = w
            work.extend(qs)

    man = mf.build_manifest('collect', vars(a), {
        'n_queries': len(work),
        'workloads': sorted({q['workload'] for q in work}),
        'load_errors': load_errors,
        'adapter': getattr(adapter, 'name', a.adapter),
    }, a.seed)
    man['limits'].update({'max_depth': a.max_depth, 'beam': a.beam,
                          'max_states_per_query': a.max_states,
                          'max_calls_per_query': a.max_calls})
    mf.atomic_write_json(outdir / 'manifest.json', man)

    print(f'[srdg:collect] run_id={run_id}  outdir={outdir}', flush=True)
    print(f'[srdg:collect] queries={len(work)}  adapter={getattr(adapter,"name",a.adapter)}  '
          f'max_depth={a.max_depth} beam={a.beam or "unlimited"}', flush=True)
    if load_errors:
        print(f'[srdg:collect] load errors: {load_errors}', file=sys.stderr, flush=True)

    if a.dry_run:
        print('[srdg:collect] --dry-run: manifest written, no Calcite invoked.')
        print(f'[srdg:collect] would expand up to {a.max_calls} matcher calls/query '
              f'=> <= {a.max_calls * len(work)} total')
        return 0

    ckpt = mf.Checkpoint(outdir / 'checkpoint.json')
    stats_path = outdir / 'stats.json'
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    novelty: dict = {}
    started = time.time()
    n_done = n_fail = n_skip = 0

    with mf.TransitionLog(outdir / 'transitions.jsonl') as log:
        for q in work:
            key = (q['workload'], q['query_id'])
            if ckpt.is_done(key):
                n_skip += 1
                continue
            meta = {
                'workload': q['workload'], 'query_id': q['query_id'],
                'query_family': q['query_family'],
                'original_sql_sha256': canonical.sql_fingerprint(q['sql']),
                'collection_run_id': run_id,
            }
            try:
                stats, err = search_query(
                    adapter, meta=meta, original_sql=q['sql'],
                    create_tables=q['create_tables'], limits=limits,
                    seed=a.seed, novelty=novelty, on_row=log.write)
            except Exception as exc:                              # noqa: BLE001
                # A query must never take the run down.
                ckpt.mark_failed(key, f'{type(exc).__name__}: {exc}')
                n_fail += 1
                print(f'[warn] {key}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                continue
            all_stats[f'{key[0]}/{key[1]}'] = stats.as_dict()
            mf.atomic_write_json(stats_path, all_stats)
            if err:
                ckpt.mark_failed(key, err)
                n_fail += 1
                print(f'[warn] {key}: {err}', file=sys.stderr, flush=True)
            else:
                ckpt.mark_done(key)
                n_done += 1
                print(f'[{n_done}/{len(work)}] {key[0]}/{key[1]} '
                      f'states={stats.states} calls={stats.calls} rows={stats.rows} '
                      f'dedup={stats.dedup_hits} cyc={stats.cycles} '
                      f'({stats.termination})', flush=True)

    summary = {'run_id': run_id, 'done': n_done, 'failed': n_fail, 'skipped_resume': n_skip,
               'rows': log.n_written, 'elapsed_s': round(time.time() - started, 1)}
    mf.atomic_write_json(outdir / 'summary.json', summary)
    print(f'[srdg:collect] {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
