"""Programmatic train/test SplitInfo construction for cross-validation.

query_sets/mvm_v1/split.json is a single fixed 80/20 family split and is
never touched by this module. Everything here builds SplitInfo objects
in memory, from the query_family values already present in
candidates.jsonl, so run_evaluation() (experiments/mvm_reliability/evaluate.py)
can be replayed once per fold without any change to the offline
evaluator itself.

Two designs, both family-grouped (never splitting a family's candidates
across train and test, so no family leakage):

  * family_kfold_splits(): per-workload, deterministic k-fold over all
    usable query families -- the "group-stratified cross-validation"
    design. Distribution across folds is stratified per workload (each
    workload's families are bucketed independently), matching the
    workload-stratified style already used by build_mvm_repository.py's
    family_split().
  * leave_one_family_out_splits(): one fold per requested family, that
    family alone as test, every other usable family (any workload) as
    train. Used for "does the prior generalize to a query family whose
    candidates were never in train" -- most useful for the 10 families
    that contain a FILTER_SUB_QUERY_TO_CORRELATE candidate, since general
    k-fold cannot guarantee a correlate family lands in every fold's test
    set.
"""

from __future__ import annotations

import hashlib
import typing as t
from collections import defaultdict

from .schemas import ALLOWED_WORKLOADS, CandidateRecord, SplitInfo

CV_HASH_NAMESPACE = 'mvm_v1_cv'
# Exact match only: the dataset also has JOIN_TO_CORRELATE,
# PROJECT_CORRELATE_TRANSPOSE and PROJECT_SUB_QUERY_TO_CORRELATE, which are
# ordinary Calcite rules unrelated to the special real-latency-verification
# path this experiment is about (see query_sets/mvm_v1/README.md).
CORRELATE_RULE_NAME = 'FILTER_SUB_QUERY_TO_CORRELATE'


def usable_families_by_workload(records: t.Sequence[CandidateRecord]) -> t.Dict[str, t.List[str]]:
    """Sorted, deduped query_family values that have at least one usable
    candidate, grouped by workload."""
    seen: t.Dict[str, t.Set[str]] = defaultdict(set)
    for record in records:
        if record.usable:
            seen[record.workload].add(record.query_family)
    return {workload: sorted(families) for workload, families in seen.items()}


def correlate_families(records: t.Sequence[CandidateRecord]) -> t.List[t.Tuple[str, str]]:
    """(workload, query_family) pairs that have at least one usable
    candidate whose rules include a FILTER_SUB_QUERY_TO_CORRELATE-style
    rule. Sorted for determinism."""
    found: t.Set[t.Tuple[str, str]] = set()
    for record in records:
        if record.usable and CORRELATE_RULE_NAME in record.rules:
            found.add((record.workload, record.query_family))
    return sorted(found)


def _fold_rank(workload: str, family: str) -> str:
    key = f'{CV_HASH_NAMESPACE}|{workload}|{family}'
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def family_kfold_splits(records: t.Sequence[CandidateRecord], k: int) -> t.List[SplitInfo]:
    """k SplitInfo folds. Within each workload, usable families are
    ranked by sha256(f'mvm_v1_cv|<workload>|<family>') and assigned to
    fold index (rank % k), round-robin -- deterministic, no reliance on
    the fixed split.json partition. A workload with fewer than k usable
    families leaves some folds with zero test families for that workload
    (that workload simply contributes nothing to that fold's test set).
    """
    if k < 2:
        raise ValueError(f'k must be >= 2, got {k}')

    families = usable_families_by_workload(records)
    fold_test: t.List[t.Dict[str, t.Set[str]]] = [defaultdict(set) for _ in range(k)]
    for workload, fam_list in families.items():
        ranked = sorted(fam_list, key=lambda fam: _fold_rank(workload, fam))
        for i, fam in enumerate(ranked):
            fold_test[i % k][workload].add(fam)

    splits: t.List[SplitInfo] = []
    for fold_idx in range(k):
        test_families = {wl: frozenset(fold_test[fold_idx].get(wl, set())) for wl in ALLOWED_WORKLOADS}
        train_families = {
            wl: frozenset(fam for fam in families.get(wl, []) if fam not in test_families[wl])
            for wl in ALLOWED_WORKLOADS
        }
        splits.append(
            SplitInfo(
                method=f'group-stratified {k}-fold cross-validation over usable query_family, '
                       f'fold {fold_idx + 1}/{k}',
                hash_spec=f'sha256("{CV_HASH_NAMESPACE}|<workload>|<query_family>") % {k}',
                train_families=train_families,
                test_families=test_families,
            )
        )
    return splits


def leave_one_family_out_splits(
    records: t.Sequence[CandidateRecord], target_families: t.Sequence[t.Tuple[str, str]]
) -> t.List[t.Tuple[t.Tuple[str, str], SplitInfo]]:
    """One (held_out_family, SplitInfo) pair per entry in target_families.
    train = every usable family other than the held-out one, pooled
    across all workloads (not just the held-out family's own workload) --
    prior computation already pools rule/sequence statistics across
    workloads (see candidate_selection/mvm/prior.py), so this maximizes
    train support for the held-out family's generalization test.
    """
    families = usable_families_by_workload(records)
    results: t.List[t.Tuple[t.Tuple[str, str], SplitInfo]] = []
    for held_workload, held_family in target_families:
        test_families = {wl: frozenset() for wl in ALLOWED_WORKLOADS}
        test_families[held_workload] = frozenset({held_family})
        train_families = {}
        for wl in ALLOWED_WORKLOADS:
            fam_list = families.get(wl, [])
            if wl == held_workload:
                train_families[wl] = frozenset(f for f in fam_list if f != held_family)
            else:
                train_families[wl] = frozenset(fam_list)
        split = SplitInfo(
            method=f'leave-one-family-out: held out ({held_workload}, {held_family})',
            hash_spec='n/a (explicit single-family holdout)',
            train_families=train_families,
            test_families=test_families,
        )
        results.append(((held_workload, held_family), split))
    return results
