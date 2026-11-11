#!/usr/bin/env python3
"""State fingerprints, leakage-safe context features, cycle/dedup helpers.

TWO DIFFERENT THINGS LIVE HERE, AND CONFLATING THEM IS THE LEAK
---------------------------------------------------------------
`fingerprint()`  identifies a state exactly. It keeps table and column names,
                 so it DOES identify the query family. It may be used for
                 dedup and cycle detection, and it may be stored — but it must
                 never be a model input.
`ctx()`          is the only thing the builder is allowed to condition on
                 (02 §H.2, §I). Coarse structural bins, no identifiers.

The audit (02 §I) requires an explicit check that ctx cannot recover the
family; `evaluator.audit_ctx_leakage()` performs it.
"""
from __future__ import annotations

import hashlib
import re
import typing as t

from . import config

# ── plan normalisation ───────────────────────────────────────────────────────
# Calcite's explain output carries cost annotations whose values move with table
# statistics. Two loads of the same query must produce the same fingerprint, so
# those are stripped. Whether they appear at all is version dependent — the MVE
# checks fingerprint stability before anything else (02 §K, KF2).
_COST_ANNOT = re.compile(r'\s*:\s*rowcount\s*=.*$|\s*,\s*cumulative cost\s*=.*$', re.I)
_WS = re.compile(r'[ \t]+')
_NUM_LITERAL = re.compile(r'(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])')
_STR_LITERAL = re.compile(r"'(?:[^']|'')*'")


def normalize_plan(plan: str) -> str:
    lines = []
    for line in (plan or '').splitlines():
        line = _COST_ANNOT.sub('', line)
        line = _WS.sub(' ', line.rstrip())
        if line.strip():
            lines.append(line)
    return '\n'.join(lines)


def normalize_sql(sql: str) -> str:
    return _WS.sub(' ', (sql or '').strip().rstrip(';')).strip()


def fingerprint(plan: str) -> str:
    """Exact state identity. Family-identifying by construction — never a feature."""
    return hashlib.sha256(normalize_plan(plan).encode()).hexdigest()


def sql_fingerprint(sql: str) -> str:
    return hashlib.sha256(normalize_sql(sql).encode()).hexdigest()


def literal_masked_plan(plan: str) -> str:
    """Plan with numeric/string literals masked.

    Used for the *literal-insensitive* dedup variant: two DSB instances of one
    template differ only in constants, and treating them as distinct states
    inflates the state count without adding structure. Offered as an option
    rather than a default because masking can merge genuinely different plans.
    """
    p = normalize_plan(plan)
    p = _STR_LITERAL.sub("'?'", p)
    return _NUM_LITERAL.sub('?', p)


# ── structural features ──────────────────────────────────────────────────────
_NODE_PATTERNS = {
    'join_count': re.compile(r'\bLogical(Join|Correlate)\b'),
    'agg_count': re.compile(r'\bLogicalAggregate\b'),
    'filter_count': re.compile(r'\bLogicalFilter\b'),
    'project_count': re.compile(r'\bLogicalProject\b'),
    'sort_count': re.compile(r'\bLogicalSort\b'),
    'union_count': re.compile(r'\bLogical(Union|Intersect|Minus)\b'),
    'scan_count': re.compile(r'\bLogicalTableScan\b'),
    'correlate_count': re.compile(r'\bLogicalCorrelate\b'),
}


def structural_features(plan: str) -> t.Dict[str, int]:
    p = normalize_plan(plan)
    feats = {k: len(rx.findall(p)) for k, rx in _NODE_PATTERNS.items()}
    lines = [l for l in p.splitlines() if l.strip()]
    feats['node_count'] = len(lines)
    feats['plan_depth'] = max((len(l) - len(l.lstrip()) for l in lines), default=0) // 2
    return feats


def _bin(n: int, edges: t.Sequence[int]) -> int:
    """Monotone bucket index. Edges are chosen on TRAIN only (02 §I)."""
    for i, e in enumerate(edges):
        if n <= e:
            return i
    return len(edges)


DEFAULT_BINS = {'join_bin': (0, 1, 3), 'agg_bin': (0, 1), 'filter_bin': (0, 1, 3),
                'depth_bin': (3, 6, 10)}


def ctx(plan: str, bins: t.Optional[dict] = None) -> t.Tuple:
    """Leakage-safe context. Coarse counts only; no identifiers, no literals."""
    b = bins or DEFAULT_BINS
    f = structural_features(plan)
    return (
        _bin(f['join_count'], b['join_bin']),
        _bin(f['agg_count'], b['agg_bin']),
        _bin(f['filter_count'], b['filter_bin']),
        _bin(f['plan_depth'], b['depth_bin']),
        int(f['correlate_count'] > 0),
        int('LogicalCorrelate' in plan or 'variablesSet' in plan),
    )


def ctx_dict(plan: str, bins: t.Optional[dict] = None) -> t.Dict[str, int]:
    return dict(zip(config.CTX_FEATURE_NAMES, ctx(plan, bins)))


# ── transition set algebra (02 §E.2) ─────────────────────────────────────────
def newly_enabled(before: t.Iterable[str], after: t.Iterable[str],
                  applied: str) -> t.List[str]:
    """Enabled(S_after) - Enabled(S_before) - {applied}.

    The applied rule is subtracted because non-idempotent rules re-match on
    their own output; without this every such rule would appear to enable
    itself. Same correction the current builder makes
    (build_rdg_empirical.py:161).

    This is a set difference, NOT a causal claim: B may have been reachable
    without A. Never describe the result as "A-enabled".
    """
    return sorted(set(after) - set(before) - {applied})


def disabled(before: t.Iterable[str], after: t.Iterable[str],
             applied: str) -> t.List[str]:
    """Enabled(S_before) - Enabled(S_after) - {applied}.

    The current RDG collects nothing like this. A rule that CLOSES an option is
    as relevant to ordering as one that opens it (02 §E.2-2).
    """
    return sorted(set(before) - set(after) - {applied})
