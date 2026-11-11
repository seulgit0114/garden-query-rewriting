#!/usr/bin/env python3
"""srdg builder CLI — transition log -> two-layer graph.

    python3 bin/build.py --run data/<run_id> --folds 5 --train-folds 0 1 2 3

Fold selection is mandatory-by-default: passing no --train-folds builds on
everything and stamps `train_folds: "ALL"` in the output so a graph built
without holdout can never be mistaken for one that had it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from srdg import builder, manifest as mf   # noqa: E402


def load_rows(run: Path):
    p = run / 'transitions.jsonl'
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='collection output directory')
    ap.add_argument('--out', default=None)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--train-folds', nargs='*', type=int, default=None)
    ap.add_argument('--salt', default='srdg-v1')
    ap.add_argument('--layer-a-min-support', type=int, default=2)
    ap.add_argument('--layer-b-min-support', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)

    run = Path(a.run)
    rows = load_rows(run)
    print(f'[srdg:build] rows={len(rows)} from {run}')
    if not rows:
        print('[srdg:build] nothing to build', file=sys.stderr)
        return 1

    folds = builder.assign_folds(rows, a.folds, a.salt)
    if a.train_folds is None:
        train, label = rows, 'ALL'
    else:
        keep = set(a.train_folds)
        train = [r for r in rows if folds[(r['workload'], r['query_family'])] in keep]
        label = sorted(keep)
    print(f'[srdg:build] train rows={len(train)} (folds={label}) '
          f'families={len({(r["workload"], r["query_family"]) for r in train})}')

    if a.dry_run:
        print('[srdg:build] --dry-run: no output written')
        return 0

    g = builder.build(train, layer_a_min_support=a.layer_a_min_support,
                      layer_b_min_support=a.layer_b_min_support)
    g['train_folds'] = label
    g['n_folds'] = a.folds
    g['salt'] = a.salt
    out = Path(a.out) if a.out else run / 'graph.json'
    mf.atomic_write_json(out, g)
    mf.atomic_write_json(run / 'folds.json',
                         {f'{k[0]}/{k[1]}': v for k, v in sorted(folds.items())})
    print(f'[srdg:build] layer_a edges={len(g["layer_a"]["edges"])} '
          f'layer_b uncond={g["layer_b"]["n_uncond"]} '
          f'disabled edges={len(g["disabled"]["edges"])}')
    print(f'[srdg:build] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
