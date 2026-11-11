"""Within-family, query-instance-level split -- the SECONDARY evaluation
axis from the original design review ("known-template generalization":
does the prior generalize to a new parameter/column variant of a query
template it has already partially seen), explicitly NOT meant to replace
the PRIMARY family-grouped CV/LOFO evaluation ("unseen-template
generalization").

Unlike experiments/mvm_reliability/cv_split.py (which keeps every
instance of a family together, entirely in train or entirely in test),
this module splits at the (workload, query) INSTANCE level: sibling
instances of the same family (e.g. dsb query001_0 and query001_1) can
land in different folds, so a family with >=2 usable instances can
contribute to BOTH train and test. Families with exactly one usable
instance cannot test within-family generalization by definition and are
always kept in train.

This is a DIFFERENT split.json entirely (built in memory, never touching
query_sets/mvm_v1/split.json), and query_sets/mvm_v1/candidates.jsonl is
only ever read, never written, from this module.
"""

from __future__ import annotations

import hashlib
import typing as t
from collections import defaultdict

from candidate_selection.cost_policy import CostOnlyPolicy
from candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
from candidate_selection.mvm.policy import MVMPolicy
from candidate_selection.mvm.prior import PriorTable

from .conditional_evaluate import MultiEvaluationResult, build_conditional_group_result
from .dataset import group_usable_by_query
from .evaluate import ExperimentConfig
from .schemas import CandidateRecord

INSTANCE_HASH_NAMESPACE = 'mvm_v1_instance_cv'


def usable_instances_by_family(
    records: t.Sequence[CandidateRecord],
) -> t.Dict[str, t.Dict[str, t.Set[str]]]:
    """{workload: {family: {query instance, ...}}} over usable candidates."""
    out: t.Dict[str, t.Dict[str, t.Set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        if r.usable:
            out[r.workload][r.query_family].add(r.query)
    return {wl: dict(fams) for wl, fams in out.items()}


def multi_instance_families(records: t.Sequence[CandidateRecord]) -> t.Dict[str, t.List[str]]:
    """{workload: [family, ...]} restricted to families with >= 2 usable
    query instances -- only these can participate in a within-family
    split; everything else stays in train for every fold."""
    by_family = usable_instances_by_family(records)
    return {
        wl: sorted(fam for fam, instances in fams.items() if len(instances) >= 2)
        for wl, fams in by_family.items()
    }


def _rank(workload: str, family: str, instance: str) -> str:
    key = f'{INSTANCE_HASH_NAMESPACE}|{workload}|{family}|{instance}'
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def instance_kfold_splits(records: t.Sequence[CandidateRecord], k: int) -> t.List[t.Dict[str, t.FrozenSet[str]]]:
    """k folds, each {workload: frozenset(query instances held out as
    test)}. Only instances belonging to a multi-instance family are ever
    eligible for test. Ranking (and the i % k fold assignment) is done
    once over ALL eligible instances of a workload together -- NOT
    per-family -- because most families here have exactly 2 usable
    instances; ranking within each family alone would only ever populate
    fold indices 0 and 1 (i in {0, 1}) and leave the rest of a k>2 split
    empty. A global per-workload ranking spreads instances from many
    different families evenly across all k folds instead."""
    if k < 2:
        raise ValueError(f'k must be >= 2, got {k}')

    by_family = usable_instances_by_family(records)
    eligible = multi_instance_families(records)

    fold_test: t.List[t.Dict[str, t.Set[str]]] = [defaultdict(set) for _ in range(k)]
    for workload, families in eligible.items():
        all_instances: t.List[t.Tuple[str, str]] = [
            (family, inst) for family in families for inst in by_family[workload][family]
        ]
        ranked = sorted(all_instances, key=lambda pair: _rank(workload, pair[0], pair[1]))
        for i, (family, inst) in enumerate(ranked):
            fold_test[i % k][workload].add(inst)

    return [{wl: frozenset(insts) for wl, insts in fold.items()} for fold in fold_test]


def filter_by_instance(
    records: t.Sequence[CandidateRecord], instance_set: t.Dict[str, t.FrozenSet[str]], want_test: bool
) -> t.List[CandidateRecord]:
    """Usable records whose (workload, query) is (want_test=True) or is
    not (want_test=False) in instance_set."""
    out = []
    for r in records:
        if not r.usable:
            continue
        in_test = r.query in instance_set.get(r.workload, frozenset())
        if in_test == want_test:
            out.append(r)
    return out


def run_instance_split_evaluation(
    records: t.Sequence[CandidateRecord],
    test_instances: t.Dict[str, t.FrozenSet[str]],
    config: ExperimentConfig,
) -> MultiEvaluationResult:
    """Same control flow as conditional_evaluate.run_multi_evaluation, but
    train/test membership is decided by (workload, query) instance rather
    than by (workload, query_family)."""
    train_rows = filter_by_instance(records, test_instances, want_test=False)
    global_prior_table = PriorTable.build(train_rows, alpha=config.mvm.alpha, beta=config.mvm.beta)

    cost_policy = CostOnlyPolicy(strict=config.cost_strict)
    global_mvm_policy = MVMPolicy(global_prior_table, config.mvm, cost_policy=cost_policy)
    cond_mvm_policy = ConditionalMVMPolicy.build(train_rows, config.mvm, cost_policy=cost_policy)

    test_records = filter_by_instance(records, test_instances, want_test=True)
    test_groups = group_usable_by_query(test_records)

    group_results = []
    for (workload, query), group_records in sorted(test_groups.items()):
        query_family = group_records[0].query_family
        group_results.append(
            build_conditional_group_result(
                workload, query, query_family, group_records, cost_policy, global_mvm_policy, cond_mvm_policy
            )
        )

    return MultiEvaluationResult(
        train_candidate_count=len(train_rows),
        accept_train_count=cond_mvm_policy.accept_train_count,
        reject_train_count=cond_mvm_policy.reject_train_count,
        groups=tuple(group_results),
    )
