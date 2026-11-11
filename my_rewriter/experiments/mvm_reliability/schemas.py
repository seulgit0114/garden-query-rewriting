"""Typed views of the mvm_v1 dataset files.

candidates.jsonl is treated as canonical; split.json is treated as the
authoritative train/test partition over query families. Neither file is
ever written by this package.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

ALLOWED_WORKLOADS = ('calcite', 'dsb', 'imdb', 'tpch')
ALLOWED_LABELS = ('TP', 'FP', 'FN', 'TN', 'UNVERIFIED')


@dataclass(frozen=True)
class CandidateRecord:
    id: str
    workload: str
    query: str
    query_family: str
    original_sql: str
    rewritten_sql: str
    rules: t.Tuple[str, ...]
    source_ablations: t.Tuple[str, ...]
    original_cost: float
    rewritten_cost: float
    original_latency_ms: float
    rewritten_latency_ms: float
    equivalent: bool
    equivalence_status: str
    label: str
    usable: bool
    exclude_reason: t.Optional[str] = None
    # Optional context for versioned histories. Legacy mvm_v2 rows omit it
    # and therefore remain usable in every distribution.
    distribution: t.Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'CandidateRecord':
        return cls(
            id=d['id'],
            workload=d['workload'],
            query=d['query'],
            query_family=d['query_family'],
            original_sql=d['original_sql'],
            rewritten_sql=d['rewritten_sql'],
            rules=tuple(d.get('rules') or []),
            source_ablations=tuple(d.get('source_ablations') or []),
            original_cost=d.get('original_cost'),
            rewritten_cost=d.get('rewritten_cost'),
            original_latency_ms=d.get('original_latency_ms'),
            rewritten_latency_ms=d.get('rewritten_latency_ms'),
            equivalent=d.get('equivalent'),
            equivalence_status=d.get('equivalence_status'),
            label=d['label'],
            usable=d['usable'],
            exclude_reason=d.get('exclude_reason'),
            distribution=d.get('distribution'),
        )

    @property
    def group_key(self) -> t.Tuple[str, str]:
        return (self.workload, self.query)


@dataclass(frozen=True)
class SplitInfo:
    method: str
    hash_spec: str
    train_families: t.Dict[str, t.FrozenSet[str]]
    test_families: t.Dict[str, t.FrozenSet[str]]

    @classmethod
    def from_dict(cls, d: dict) -> 'SplitInfo':
        train: t.Dict[str, t.FrozenSet[str]] = {}
        test: t.Dict[str, t.FrozenSet[str]] = {}
        for workload, v in d['workloads'].items():
            train[workload] = frozenset(v['train_families'])
            test[workload] = frozenset(v['test_families'])
        return cls(method=d['method'], hash_spec=d['hash'], train_families=train, test_families=test)

    def split_of(self, workload: str, query_family: str) -> t.Optional[str]:
        if query_family in self.train_families.get(workload, frozenset()):
            return 'train'
        if query_family in self.test_families.get(workload, frozenset()):
            return 'test'
        return None
