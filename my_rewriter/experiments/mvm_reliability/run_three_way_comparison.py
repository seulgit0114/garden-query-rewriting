"""Three-way end-to-end candidate-selection comparison on logged latency.

Compares the three selection policies that actually exist as pipeline
configurations, scored by the *measured* latency of whichever candidate
each policy ends up executing:

  A. cost_only            -- pick min rewritten_cost, then the cost
                             rejection gate (rewritten_cost < original_cost).
                             Mirrors portfolio_rewrite._select_best_candidate()
                             with both correlate switches off, followed by
                             _apply_cost_rejection_gate().

  B. correlate_realverify -- same, except a group containing any
                             FILTER_SUB_QUERY_TO_CORRELATE candidate picks the
                             cheapest *correlate* candidate, and that pick's
                             accept/reject is decided by its MEASURED latency
                             rather than by cost. Mirrors
                             correlate_real_verify_enabled=True.

  B2. correlate_override  -- same selection as B, but the correlate pick is
                             accepted unconditionally. Mirrors
                             correlate_override_enabled=True. Reported
                             alongside B because both switches ship together.

  C. mvm_conditional      -- per-candidate accept/reject from the
                             cost-decision-conditional prior, then rank the
                             accepted set by train-only
                             median_latency_improvement_pct.

A/B/B2 need no training. C does, so every policy is scored on the *test*
side of the same group-stratified k-fold splits and pooled, which keeps the
group set identical across all four columns.

Nothing is executed against a database: every candidate in the evaluated
pool already carries original_latency_ms / rewritten_latency_ms, and the
script asserts this rather than silently skipping.

Two candidate pools are reported:
  usable          -- the 453 candidates every earlier experiment used.
  usable+ruleless -- adds candidates that are equivalent_full_scale and have
                     measured latency but were excluded from the MVM dataset
                     only for `no_applied_rule_metadata` (LLM free-form
                     rewrites with no Calcite rule names). The live pipeline
                     would happily select these, so excluding them flatters
                     every policy equally but understates the real candidate
                     pool. C falls back to its workload/global prior for them.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[2]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from candidate_selection.cost_policy import CostOnlyPolicy  # noqa: E402
from candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy  # noqa: E402

from experiments.mvm_reliability.cv_split import CORRELATE_RULE_NAME, family_kfold_splits  # noqa: E402
from experiments.mvm_reliability.dataset import load_candidates_jsonl, sha256_file  # noqa: E402
from experiments.mvm_reliability.evaluate import ExperimentConfig, to_features  # noqa: E402
from experiments.mvm_reliability.report import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    git_info,
    make_run_dir,
)
from experiments.mvm_reliability.schemas import CandidateRecord, SplitInfo  # noqa: E402

POLICIES = (
    'cost_only',
    'correlate_realverify',
    'correlate_override',
    'mvm_conditional',
    'hybrid_measure_correlate',
    'oracle',
)
RULELESS_EXCLUDE_REASON = 'no_applied_rule_metadata'


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'mvm_v1_no_min_support.json',
        help='Experiment config JSON (default: configs/mvm_v1_no_min_support.json, the best k-fold variant)',
    )
    parser.add_argument('--candidates', type=Path, default=None, help='Override candidates.jsonl path')
    parser.add_argument('--k', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--run-name', default='three_way_comparison', help='Suffix for the run directory name')
    parser.add_argument(
        '--dedup-threshold', type=float, default=0.90,
        help='original_sql similarity at or above which two query families are merged into one '
             'split unit, so a near-copy of a test query cannot sit in train (default: 0.90). '
             'Both the raw and deduplicated splits are always reported.',
    )
    parser.add_argument(
        '--outputs-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs',
        help='Directory under which a new <timestamp>_<run-name> run directory is created',
    )
    return parser.parse_args(argv)


def _normalize_sql(sql: str) -> str:
    sql = re.sub(r'--.*?$', ' ', sql.lower(), flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', sql).strip()


def near_duplicate_family_map(
    records: t.Sequence[CandidateRecord], threshold: float
) -> t.Tuple[t.Dict[t.Tuple[str, str], str], t.List[dict]]:
    """Map each (workload, family) to a canonical family name, merging
    families whose representative original_sql is >= threshold similar.

    Without this, two near-identical queries that live in different
    query_family values can land in different folds, so the prior is
    trained on a near-copy of a test query. Merging them forces the whole
    cluster into one fold.
    """
    rep: t.Dict[t.Tuple[str, str], str] = {}
    for record in records:
        if record.usable:
            rep.setdefault((record.workload, record.query_family), record.original_sql)

    by_workload: t.Dict[str, t.List[t.Tuple[str, str]]] = defaultdict(list)
    for key in rep:
        by_workload[key[0]].append(key)

    parent: t.Dict[t.Tuple[str, str], t.Tuple[str, str]] = {k: k for k in rep}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: t.List[dict] = []
    for keys in by_workload.values():
        for a, b in itertools.combinations(sorted(keys), 2):
            sa, sb = _normalize_sql(rep[a]), _normalize_sql(rep[b])
            if not sa or not sb:
                continue
            if min(len(sa), len(sb)) / max(len(sa), len(sb)) < 0.7:
                continue
            ratio = difflib.SequenceMatcher(None, sa, sb).ratio()
            if ratio >= threshold:
                pairs.append({
                    'workload': a[0], 'family_a': a[1], 'family_b': b[1],
                    'similarity': round(ratio, 4),
                })
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    mapping = {k: find(k)[1] for k in rep}
    return mapping, pairs


def apply_family_map(
    records: t.Sequence[CandidateRecord], mapping: t.Dict[t.Tuple[str, str], str]
) -> t.List[CandidateRecord]:
    """Rewrite query_family to its cluster canonical. `query` is untouched,
    so groups stay separate -- only the split unit is merged."""
    out = []
    for record in records:
        canonical = mapping.get((record.workload, record.query_family))
        out.append(record if canonical is None or canonical == record.query_family
                   else replace(record, query_family=canonical))
    return out


def _is_correlate(record: CandidateRecord) -> bool:
    return CORRELATE_RULE_NAME in record.rules


def _has_latency(record: CandidateRecord) -> bool:
    return (
        isinstance(record.original_latency_ms, (int, float))
        and isinstance(record.rewritten_latency_ms, (int, float))
    )


def selectable_records(records: t.Sequence[CandidateRecord], include_ruleless: bool) -> t.List[CandidateRecord]:
    """Candidates any policy is allowed to execute.

    Always the usable set; optionally plus equivalent-but-rule-less rows,
    which the live pipeline can select even though the MVM dataset drops
    them for having no rule metadata.
    """
    out = [r for r in records if r.usable]
    if include_ruleless:
        out += [
            r for r in records
            if not r.usable
            and r.exclude_reason == RULELESS_EXCLUDE_REASON
            and r.equivalence_status == 'equivalent_full_scale'
            and _has_latency(r)
        ]
    missing = [r.id for r in out if not _has_latency(r)]
    if missing:
        raise ValueError(f'{len(missing)} selectable candidates lack measured latency, e.g. {missing[:3]}')
    return out


# ── the four policies ────────────────────────────────────────────────────────
# Each returns the record to execute, or None to keep the original query.

def pick_cost_only(
    group: t.Sequence[CandidateRecord], cost_policy: CostOnlyPolicy
) -> t.Optional[CandidateRecord]:
    best = min(group, key=lambda r: r.rewritten_cost)
    return best if cost_policy.decide(to_features(best)) == 'rewrite' else None


def pick_correlate(
    group: t.Sequence[CandidateRecord], cost_policy: CostOnlyPolicy, real_verify: bool
) -> t.Optional[CandidateRecord]:
    """real_verify=True  -> correlate pick is gated on measured latency.
    real_verify=False -> correlate pick is accepted unconditionally."""
    correlate = [r for r in group if _is_correlate(r)]
    if not correlate:
        return pick_cost_only(group, cost_policy)
    best = min(correlate, key=lambda r: r.rewritten_cost)
    if not real_verify:
        return best
    return best if best.rewritten_latency_ms < best.original_latency_ms else None


def pick_mvm(
    group: t.Sequence[CandidateRecord], policy: ConditionalMVMPolicy, min_support: int
) -> t.Optional[CandidateRecord]:
    scored: t.List[t.Tuple[CandidateRecord, t.Optional[float], t.Optional[float]]] = []
    for record in group:
        decision = policy.decide_detailed(to_features(record))
        if decision.decision == 'rewrite':
            scored.append((record, decision.median_latency_improvement_pct, decision.reliability))
    if not scored:
        return None
    return max(
        scored,
        key=lambda triple: (
            triple[1] is not None,
            triple[1] if triple[1] is not None else 0.0,
            triple[2] or 0.0,
            -triple[0].rewritten_cost,
        ),
    )[0]


def pick_hybrid(
    group: t.Sequence[CandidateRecord], cost_policy: CostOnlyPolicy, policy: ConditionalMVMPolicy, min_support: int
) -> t.Optional[CandidateRecord]:
    """Measure where the pipeline already pays to measure, predict elsewhere.

    POST-HOC: this combination was formed after seeing that correlate
    real-verify and the MVM prior win on disjoint workloads. It is reported
    for completeness, not as a validated policy.
    """
    if any(_is_correlate(r) for r in group):
        return pick_correlate(group, cost_policy, real_verify=True)
    return pick_mvm(group, policy, min_support)


def pick_oracle(group: t.Sequence[CandidateRecord]) -> t.Optional[CandidateRecord]:
    """Upper bound: the best possible choice given full knowledge."""
    best = min(group, key=lambda r: r.rewritten_latency_ms)
    return best if best.rewritten_latency_ms < best.original_latency_ms else None


@dataclass(frozen=True)
class GroupOutcome:
    workload: str
    query: str
    query_family: str
    fold_index: int
    candidate_count: int
    has_correlate: bool
    original_latency_ms: float
    selected: t.Dict[str, t.Optional[str]]
    latency_ms: t.Dict[str, float]

    def to_dict(self) -> dict:
        row = {
            'workload': self.workload,
            'query': self.query,
            'query_family': self.query_family,
            'fold_index': self.fold_index,
            'candidate_count': self.candidate_count,
            'has_correlate': self.has_correlate,
            'original_latency_ms': self.original_latency_ms,
        }
        for name in POLICIES:
            row[f'{name}_selected_candidate_id'] = self.selected[name]
            row[f'{name}_latency_ms'] = self.latency_ms[name]
        return row


def evaluate_pool(
    records: t.Sequence[CandidateRecord],
    config: ExperimentConfig,
    k: int,
    include_ruleless: bool,
) -> t.List[GroupOutcome]:
    pool = selectable_records(records, include_ruleless)
    train_source = [r for r in records if r.usable]  # priors are only ever built from usable rows
    cost_policy = CostOnlyPolicy(strict=config.cost_strict)
    splits: t.List[SplitInfo] = family_kfold_splits(records, k)

    by_group: t.Dict[t.Tuple[str, str], t.List[CandidateRecord]] = defaultdict(list)
    for record in pool:
        by_group[(record.workload, record.query)].append(record)

    outcomes: t.List[GroupOutcome] = []
    for fold_index, split in enumerate(splits):
        train_rows = [
            r for r in train_source
            if split.split_of(r.workload, r.query_family) == 'train'
        ]
        mvm_policy = ConditionalMVMPolicy.build(train_rows, config.mvm, cost_policy)

        for (workload, query), group in sorted(by_group.items()):
            if split.split_of(workload, group[0].query_family) != 'test':
                continue
            original_latency = group[0].original_latency_ms
            picks = {
                'cost_only': pick_cost_only(group, cost_policy),
                'correlate_realverify': pick_correlate(group, cost_policy, real_verify=True),
                'correlate_override': pick_correlate(group, cost_policy, real_verify=False),
                'mvm_conditional': pick_mvm(group, mvm_policy, config.mvm.min_support),
                'hybrid_measure_correlate': pick_hybrid(
                    group, cost_policy, mvm_policy, config.mvm.min_support),
                'oracle': pick_oracle(group),
            }
            outcomes.append(GroupOutcome(
                workload=workload,
                query=query,
                query_family=group[0].query_family,
                fold_index=fold_index,
                candidate_count=len(group),
                has_correlate=any(_is_correlate(r) for r in group),
                original_latency_ms=original_latency,
                selected={n: (p.id if p else None) for n, p in picks.items()},
                latency_ms={
                    n: (p.rewritten_latency_ms if p else original_latency)
                    for n, p in picks.items()
                },
            ))
    return outcomes


def summarize(outcomes: t.Sequence[GroupOutcome]) -> dict:
    def block(rows: t.Sequence[GroupOutcome]) -> dict:
        original_total = sum(r.original_latency_ms for r in rows)
        out = {
            'group_count': len(rows),
            'original_total_latency_ms': original_total,
            'policies': {},
        }
        for name in POLICIES:
            total = sum(r.latency_ms[name] for r in rows)
            rewrite_count = sum(1 for r in rows if r.selected[name] is not None)
            out['policies'][name] = {
                'total_latency_ms': total,
                'improvement_pct_vs_original': (
                    (original_total - total) / original_total * 100 if original_total else None
                ),
                'rewrite_accepted_groups': rewrite_count,
            }
        return out

    by_workload = defaultdict(list)
    for row in outcomes:
        by_workload[row.workload].append(row)
    return {
        'overall': block(outcomes),
        'by_workload': {w: block(rows) for w, rows in sorted(by_workload.items())},
    }


def _pct(value: t.Optional[float]) -> str:
    return '--' if value is None else f'{value:+.2f}%'


def render_report(snapshot: dict, results: dict) -> str:
    lines = [
        '# Three-way selection-policy comparison (measured latency replay)',
        '',
        f"candidates.jsonl sha256: `{snapshot['candidates_sha256']}`",
        f"config: `{snapshot['config_path']}` (min_support={snapshot['min_support']}, "
        f"fp_guard={snapshot['fp_guard_threshold']}, fn_recovery={snapshot['fn_recovery_threshold']})",
        f"folds: {snapshot['k']} (group-stratified by query family; every policy scored on test folds only)",
        f"git commit: {snapshot['git']['commit']} (dirty={snapshot['git']['dirty']}, branch={snapshot['git']['branch']})",
        '',
        'Policies:',
        '- **cost_only** -- min rewritten_cost, then the cost rejection gate.',
        '- **correlate_realverify** -- correlate candidates preferred, accept/reject decided by MEASURED latency '
        '(mirrors `correlate_real_verify_enabled`; costs 5 interleaved executions per correlate query at runtime).',
        '- **correlate_override** -- same selection, correlate pick accepted unconditionally '
        '(mirrors `correlate_override_enabled`).',
        '- **mvm_conditional** -- conditional prior accept/reject, then rank accepted set by train-only '
        'median latency improvement (costs nothing at runtime).',
        '- **hybrid_measure_correlate** -- POST-HOC combination: real-verify on groups containing a correlate '
        'candidate, MVM prior everywhere else. Formed after observing that the two win on disjoint workloads; '
        'not independently validated.',
        '- **oracle** -- upper bound, picks the lowest measured latency.',
        '',
        f"Near-duplicate audit (original_sql similarity >= {snapshot['dedup_threshold']}): "
        f"{len(snapshot['near_duplicate_pairs'])} family pairs found; "
        f"the 'near-duplicate-merged split' rows force each cluster into one fold so the prior "
        f"cannot train on a near-copy of a test query.",
        '',
    ]
    for pair in snapshot['near_duplicate_pairs']:
        lines.append(
            f"  - {pair['workload']}: `{pair['family_a']}` <-> `{pair['family_b']}` "
            f"(similarity {pair['similarity']})"
        )
    lines.append('')
    for pool_name, payload in results.items():
        lines += [f'## Candidate pool: {pool_name}', '']
        overall = payload['overall']
        lines += [
            f"Test groups: {overall['group_count']}  |  "
            f"original total: {overall['original_total_latency_ms'] / 1000:.1f}s",
            '',
            '| scope | groups | original (s) | ' + ' | '.join(POLICIES) + ' |',
            '|---|---:|---:|' + '---:|' * len(POLICIES),
        ]

        def row(scope: str, block: dict) -> str:
            cells = [
                _pct(block['policies'][name]['improvement_pct_vs_original'])
                for name in POLICIES
            ]
            return (
                f"| {scope} | {block['group_count']} | "
                f"{block['original_total_latency_ms'] / 1000:.1f} | " + ' | '.join(cells) + ' |'
            )

        for workload, block in payload['by_workload'].items():
            lines.append(row(workload, block))
        lines.append(row('**overall**', overall))
        lines += ['', '### Absolute selected latency (seconds)', '',
                  '| scope | original | ' + ' | '.join(POLICIES) + ' |',
                  '|---|---:|' + '---:|' * len(POLICIES)]
        for scope, block in list(payload['by_workload'].items()) + [('**overall**', overall)]:
            cells = [f"{block['policies'][n]['total_latency_ms'] / 1000:.1f}" for n in POLICIES]
            lines.append(
                f"| {scope} | {block['original_total_latency_ms'] / 1000:.1f} | " + ' | '.join(cells) + ' |'
            )
        lines += ['', '### Groups where a rewrite was accepted', '',
                  '| scope | groups | ' + ' | '.join(POLICIES) + ' |',
                  '|---|---:|' + '---:|' * len(POLICIES)]
        for scope, block in list(payload['by_workload'].items()) + [('**overall**', overall)]:
            cells = [str(block['policies'][n]['rewrite_accepted_groups']) for n in POLICIES]
            lines.append(f"| {scope} | {block['group_count']} | " + ' | '.join(cells) + ' |')
        lines.append('')

    lines += [
        '## Limitations',
        '- Replay over already-measured latency; no SQL/LLM/DB calls were made by this script.',
        '- `correlate_realverify` and `correlate_override` need no training, so their numbers are '
        'independent of the fold split; they are reported on the same test groups purely for comparability.',
        '- `correlate_realverify` consumes the candidate\'s own measured latency, which the live pipeline '
        'pays for with real executions. `mvm_conditional` uses train-only statistics and pays nothing. '
        'The two columns are therefore not equal-cost.',
        '- The live pipeline can early-stop candidate generation (`final_accept_ratio`), so a real run may '
        'present fewer candidates per group than this replay does.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_dict = json.loads(args.config.read_text())
    config = ExperimentConfig.from_dict(config_dict)

    candidates_path = args.candidates or (
        MY_REWRITER_DIR / 'query_sets' / 'mvm_v1' / 'candidates.jsonl'
    )
    records = load_candidates_jsonl(candidates_path)

    family_map, dup_pairs = near_duplicate_family_map(records, args.dedup_threshold)
    deduped = apply_family_map(records, family_map)
    merged_clusters = sorted({
        canonical for (wl, fam), canonical in family_map.items() if canonical != fam
    })

    results = {}
    all_outcomes = {}
    for split_name, split_records in (('raw family split', records),
                                      ('near-duplicate-merged split', deduped)):
        for pool_name, include_ruleless in (('usable', False), ('usable+ruleless', True)):
            label = f'{split_name} | pool={pool_name}'
            outcomes = evaluate_pool(split_records, config, args.k, include_ruleless)
            all_outcomes[label] = outcomes
            results[label] = summarize(outcomes)

    snapshot = {
        'candidates_path': str(candidates_path),
        'candidates_sha256': sha256_file(candidates_path),
        'candidate_count': len(records),
        'config_path': str(args.config),
        'k': args.k,
        'min_support': config.mvm.min_support,
        'fp_guard_threshold': config.mvm.fp_guard_threshold,
        'fn_recovery_threshold': config.mvm.fn_recovery_threshold,
        'alpha': config.mvm.alpha,
        'beta': config.mvm.beta,
        'dedup_threshold': args.dedup_threshold,
        'near_duplicate_pairs': dup_pairs,
        'merged_family_clusters': merged_clusters,
        'git': git_info(MY_REWRITER_DIR),
    }

    run_dir = make_run_dir(args.outputs_dir, args.run_name)
    atomic_write_json(run_dir / 'dataset_snapshot.json', snapshot)
    atomic_write_json(run_dir / 'summary.json', results)
    for pool_name, outcomes in all_outcomes.items():
        slug = re.sub(r'[^a-z0-9]+', '_', pool_name.lower()).strip('_')
        atomic_write_jsonl(run_dir / f'group_rows_{slug}.jsonl', [o.to_dict() for o in outcomes])
    report = render_report(snapshot, results)
    atomic_write_text(run_dir / 'report.md', report)
    print(report)
    print(f'[three_way] run written to {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
