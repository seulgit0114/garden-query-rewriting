#!/usr/bin/env python3
"""Aggregate the transition log into the two-layer graph (02 §H).

Layer A — rule-only coverage index.  support(A,B) counted over query FAMILIES,
          not queries, because the current builder's per-query count inflates
          DSB template variants (01 §7 P5: 29 of 43 edges sit exactly at the
          threshold). The ">0.5 whole-structure" filter of
          build_rdg_empirical.py:162 is deliberately NOT reproduced — 01 §7 C4.

Layer B — P(B newly-enabled | A, ctx) with Laplace smoothing and a backoff
          chain, mirroring the structure already validated for the reliability
          prior: specific -> less specific -> unconditional -> none, each step
          gated by min_support.

Fold discipline: `build()` takes the rows it is given. The caller is
responsible for passing TRAIN rows only. `split.assign_folds()` is provided so
that responsibility is easy to meet.
"""
from __future__ import annotations

import hashlib
import json
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

from . import config


# ── fold assignment (02 §I) ──────────────────────────────────────────────────
def assign_fold(workload: str, family: str, n_folds: int, salt: str) -> int:
    """Family-level, deterministic. Every state/sequence of one query lands in
    the same fold because the key ignores everything below the family."""
    h = hashlib.sha256(f'{salt}|{workload}|{family}'.encode()).hexdigest()
    return int(h[:8], 16) % n_folds


def assign_folds(rows: t.Iterable[dict], n_folds: int = 5,
                 salt: str = 'srdg-v1') -> t.Dict[t.Tuple[str, str], int]:
    keys = {(r['workload'], r['query_family']) for r in rows}
    return {k: assign_fold(k[0], k[1], n_folds, salt) for k in keys}


# ── Layer A ──────────────────────────────────────────────────────────────────
@dataclass
class LayerA:
    edges: t.Dict[t.Tuple[str, str], int] = field(default_factory=dict)   # (A,B) -> family support
    min_support: int = 2

    def successors(self, rule: str) -> t.List[str]:
        return sorted(b for (a, b), s in self.edges.items()
                      if a == rule and s >= self.min_support)

    def to_dict(self) -> dict:
        return {'min_support': self.min_support,
                'edges': [{'from': a, 'to': b, 'family_support': s}
                          for (a, b), s in sorted(self.edges.items()) if s >= self.min_support]}


def build_layer_a(rows: t.Iterable[dict], min_support: int = 2) -> LayerA:
    fams: t.Dict[t.Tuple[str, str], set] = defaultdict(set)
    for r in rows:
        ne = r.get('newly_enabled')
        if not ne:
            continue
        fam = (r['workload'], r['query_family'])
        for b in ne:
            fams[(r['attempted_rule'], b)].add(fam)
    return LayerA(edges={k: len(v) for k, v in fams.items()}, min_support=min_support)


# ── Layer B ──────────────────────────────────────────────────────────────────
CTX_KEYS = config.CTX_FEATURE_NAMES


def _ctx_tuple(d: dict) -> tuple:
    return tuple(d.get(k) for k in CTX_KEYS)


@dataclass
class LayerB:
    """P(B newly-enabled | A, ctx), Laplace-smoothed, with a backoff chain."""
    full: t.Dict[t.Tuple[str, tuple, str], t.Tuple[int, int]] = field(default_factory=dict)
    partial: t.Dict[t.Tuple[str, tuple, str], t.Tuple[int, int]] = field(default_factory=dict)
    uncond: t.Dict[t.Tuple[str, str], t.Tuple[int, int]] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 1.0
    min_support: int = 3
    partial_idx: t.Tuple[int, ...] = (0, 4)     # join_bin, has_correlate

    def _sm(self, pos: int, n: int) -> float:
        return (pos + self.alpha) / (n + self.alpha + self.beta)

    def lookup(self, rule: str, ctx: dict, target: str) -> dict:
        c = _ctx_tuple(ctx)
        cp = tuple(c[i] for i in self.partial_idx)
        for src, key, table in (('ctx_full', (rule, c, target), self.full),
                                ('ctx_partial', (rule, cp, target), self.partial),
                                ('unconditional', (rule, target), self.uncond)):
            e = table.get(key)
            if e and e[1] >= self.min_support:
                return {'p': self._sm(e[0], e[1]), 'support': e[1], 'source': src}
        return {'p': None, 'support': 0, 'source': 'none'}

    def to_dict(self) -> dict:
        return {
            'alpha': self.alpha, 'beta': self.beta, 'min_support': self.min_support,
            'ctx_keys': list(CTX_KEYS), 'partial_idx': list(self.partial_idx),
            'n_full': len(self.full), 'n_partial': len(self.partial), 'n_uncond': len(self.uncond),
            'unconditional': [{'from': a, 'to': b, 'pos': p, 'n': n,
                               'p': (p + self.alpha) / (n + self.alpha + self.beta)}
                              for (a, b), (p, n) in sorted(self.uncond.items())],
        }


def build_layer_b(rows: t.Iterable[dict], min_support: int = 3,
                  alpha: float = 1.0, beta: float = 1.0,
                  partial_idx: t.Tuple[int, ...] = (0, 4)) -> LayerB:
    """Denominator = every state where A was applied in that ctx; numerator =
    those where B appeared in `newly_enabled`. Rows whose child was pruned
    (`newly_enabled is None`) contribute to neither — a pruned child is missing
    data, not a negative observation."""
    full_n: t.Dict[t.Tuple[str, tuple], int] = defaultdict(int)
    full_p: t.Dict[t.Tuple[str, tuple, str], int] = defaultdict(int)
    part_n: t.Dict[t.Tuple[str, tuple], int] = defaultdict(int)
    part_p: t.Dict[t.Tuple[str, tuple, str], int] = defaultdict(int)
    unc_n: t.Dict[str, int] = defaultdict(int)
    unc_p: t.Dict[t.Tuple[str, str], int] = defaultdict(int)

    for r in rows:
        if r.get('newly_enabled') is None:
            continue
        a = r['attempted_rule']
        c = _ctx_tuple(r.get('ctx_before') or {})
        cp = tuple(c[i] for i in partial_idx)
        full_n[(a, c)] += 1
        part_n[(a, cp)] += 1
        unc_n[a] += 1
        for b in r['newly_enabled']:
            full_p[(a, c, b)] += 1
            part_p[(a, cp, b)] += 1
            unc_p[(a, b)] += 1

    return LayerB(
        full={k: (v, full_n[(k[0], k[1])]) for k, v in full_p.items()},
        partial={k: (v, part_n[(k[0], k[1])]) for k, v in part_p.items()},
        uncond={k: (v, unc_n[k[0]]) for k, v in unc_p.items()},
        alpha=alpha, beta=beta, min_support=min_support, partial_idx=partial_idx)


# ── disabled-side statistics (02 §E.2-2) ─────────────────────────────────────
def build_disabled_stats(rows: t.Iterable[dict], min_support: int = 2) -> dict:
    """A closes B. The existing RDG collects nothing equivalent."""
    fams: t.Dict[t.Tuple[str, str], set] = defaultdict(set)
    for r in rows:
        for b in (r.get('disabled') or []):
            fams[(r['attempted_rule'], b)].add((r['workload'], r['query_family']))
    return {'min_support': min_support,
            'edges': [{'from': a, 'to': b, 'family_support': len(v)}
                      for (a, b), v in sorted(fams.items()) if len(v) >= min_support]}


def build(rows: t.Sequence[dict], *, layer_a_min_support: int = 2,
          layer_b_min_support: int = 3) -> dict:
    a = build_layer_a(rows, layer_a_min_support)
    b = build_layer_b(rows, layer_b_min_support)
    return {
        'n_rows': len(rows),
        'n_families': len({(r['workload'], r['query_family']) for r in rows}),
        'layer_a': a.to_dict(),
        'layer_b': b.to_dict(),
        'disabled': build_disabled_stats(rows),
    }
