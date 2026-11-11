#!/usr/bin/env python3
"""Arm comparison with coverage (G1) kept strictly apart from quality (G2).

The separation is not cosmetic. exp6 found that candidate sets are often
indistinguishable before execution (16 of 37 multi-candidate groups had
byte-identical plan features), so a method can enlarge coverage and still
change nothing downstream. Reporting one number that mixes the two hides
exactly that outcome, which 02 §L' lists as a likely "보류" verdict.

QUALITY IS DELIBERATELY NOT COMPUTED FROM LATENCY HERE.
02 §H.3 forbids attaching oracle labels to unmeasured states. This module
reports quality only over rows that carry a `measured_latency_ms`, and reports
how many rows that was. With none, the quality section is explicitly empty
rather than silently substituting Calcite cost.
"""
from __future__ import annotations

import random
import typing as t
from collections import defaultdict

from . import canonical


# ── G1: coverage ─────────────────────────────────────────────────────────────
def coverage(rows: t.Iterable[dict]) -> dict:
    """Which rules were ever observed as enabled, and at what depth first."""
    first_depth: t.Dict[str, int] = {}
    per_family: t.Dict[t.Tuple[str, str], set] = defaultdict(set)
    for r in rows:
        fam = (r['workload'], r['query_family'])
        for name in (r.get('enabled_before') or []):
            first_depth.setdefault(name, max(0, r['depth'] - 1))
            per_family[fam].add(name)
        for name in (r.get('enabled_after') or []):
            d = first_depth.get(name)
            if d is None or r['depth'] < d:
                first_depth[name] = r['depth']
            per_family[fam].add(name)
    by_depth: t.Dict[int, int] = defaultdict(int)
    for d in first_depth.values():
        by_depth[d] += 1
    return {
        'n_rules_seen': len(first_depth),
        'first_seen_depth_hist': dict(sorted(by_depth.items())),
        'rules_first_seen_at_depth_ge_1': sorted(k for k, v in first_depth.items() if v >= 1),
        'n_families': len(per_family),
        'mean_rules_per_family': (sum(len(v) for v in per_family.values()) / len(per_family)
                                  if per_family else 0.0),
    }


def depth_gain(rows: t.Iterable[dict]) -> dict:
    """THE MVE's primary measurement (02 §K).

    Fraction of observed rules that are first seen only at depth >= 1. If this
    is tiny, the 1-hop limitation (02 §A P1) is not real and the whole
    state-aware design loses its motivation — gate KF1.
    """
    c = coverage(rows)
    n = c['n_rules_seen']
    deep = len(c['rules_first_seen_at_depth_ge_1'])
    return {
        'n_rules_seen': n,
        'n_first_seen_deep': deep,
        'fraction_first_seen_deep': (deep / n) if n else 0.0,
        'rules': c['rules_first_seen_at_depth_ge_1'],
        'kf1_threshold': 0.05,
        'kf1_pass': ((deep / n) if n else 0.0) >= 0.05,
    }


# ── G2: quality (measured latency only) ──────────────────────────────────────
def quality(rows: t.Iterable[dict]) -> dict:
    measured = [r for r in rows if r.get('measured_latency_ms') is not None]
    if not measured:
        return {'n_measured': 0,
                'note': 'no measured latency present; quality intentionally not '
                        'inferred from Calcite cost (02 §H.3)'}
    by_fam = defaultdict(list)
    for r in measured:
        by_fam[(r['workload'], r['query_family'])].append(r['measured_latency_ms'])
    return {'n_measured': len(measured), 'n_families': len(by_fam)}


# ── leakage audit (02 §I) ────────────────────────────────────────────────────
def audit_ctx_leakage(rows: t.Sequence[dict], seed: int = 0) -> dict:
    """Can the family be recovered from ctx alone?

    Compares a majority-class predictor keyed on ctx against the trivial
    majority baseline. If ctx-conditioned accuracy is materially above the
    baseline, ctx is carrying identity and must be coarsened (gate LI3).

    The fingerprint is audited too — it is EXPECTED to leak, and that
    expectation is why fingerprints are never model inputs.
    """
    if not rows:
        return {'n': 0}
    fams = [f"{r['workload']}/{r['query_family']}" for r in rows]
    majority = max(defaultdict(int, {f: fams.count(f) for f in set(fams)}).values()) / len(fams)

    def acc(keyfn):
        groups = defaultdict(list)
        for r, f in zip(rows, fams):
            groups[keyfn(r)].append(f)
        correct = sum(max({x: g.count(x) for x in set(g)}.values()) for g in groups.values())
        return correct / len(fams)

    ctx_acc = acc(lambda r: tuple(sorted((r.get('ctx_before') or {}).items())))
    fp_acc = acc(lambda r: r['state_id'])
    return {
        'n': len(rows),
        'n_families': len(set(fams)),
        'majority_baseline_acc': majority,
        'ctx_family_acc': ctx_acc,
        'fingerprint_family_acc': fp_acc,
        'ctx_excess_over_baseline': ctx_acc - majority,
        'li3_pass': ctx_acc - majority < 0.30,
        'note': 'fingerprint_family_acc is expected to be ~1.0; that is why '
                'fingerprints are never used as features.',
    }


# ── arm harness ──────────────────────────────────────────────────────────────
ARMS = ('rbot_sequence', 'existing_rdg_closure', 'random_expansion',
        'original_state_only', 'bounded_bfs_no_rdg', 'cost_guided_beam',
        'state_aware_rdg', 'oracle_coverage')
REFERENCE_ONLY = ('oracle_coverage',)


def compare_coverage(per_arm: t.Dict[str, t.Sequence[dict]]) -> dict:
    """Coverage per arm plus the single contrast that defines G5.

    G5 is `state_aware_rdg` minus `bounded_bfs_no_rdg`: if the graph adds
    nothing over plain bounded BFS, the honest write-up is "bounded BFS",
    not "state-aware RDG" (02 §L' G5).
    """
    out = {a: coverage(rs) for a, rs in per_arm.items()}
    g5 = None
    if 'state_aware_rdg' in out and 'bounded_bfs_no_rdg' in out:
        g5 = (out['state_aware_rdg']['n_rules_seen']
              - out['bounded_bfs_no_rdg']['n_rules_seen'])
    return {'per_arm': out, 'g5_rules_gained_over_plain_bfs': g5,
            'reference_only': list(REFERENCE_ONLY)}


def bootstrap_family_diff(rows_a: t.Sequence[dict], rows_b: t.Sequence[dict],
                          stat=lambda rs: coverage(rs)['n_rules_seen'],
                          n: int = 1000, seed: int = 7) -> dict:
    """Paired bootstrap over families (exp6's procedure, reused)."""
    fam_a, fam_b = defaultdict(list), defaultdict(list)
    for r in rows_a:
        fam_a[(r['workload'], r['query_family'])].append(r)
    for r in rows_b:
        fam_b[(r['workload'], r['query_family'])].append(r)
    keys = sorted(set(fam_a) | set(fam_b))
    if not keys:
        return {'n': 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        diffs.append(stat([x for k in pick for x in fam_a.get(k, [])])
                     - stat([x for k in pick for x in fam_b.get(k, [])]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]
    return {'mean': sum(diffs) / len(diffs), 'ci_lo': lo, 'ci_hi': hi,
            'excludes_zero': lo > 0 or hi < 0}
