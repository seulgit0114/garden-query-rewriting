"""Read-only loader and integrity checks for the mvm_v1 dataset.

Every function here only opens ``candidates.jsonl`` / ``split.json`` for
reading. Nothing in this module writes to ``query_sets/mvm_v1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ALLOWED_WORKLOADS, CandidateRecord, SplitInfo


class DatasetIntegrityError(Exception):
    """Raised when candidates.jsonl / split.json fail a consistency check.

    Never swallowed or logged-and-continued -- callers are expected to let
    this propagate so a bad dataset snapshot cannot silently produce a
    misleading evaluation.
    """


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidates_jsonl(path: Path) -> t.List[CandidateRecord]:
    """Read the whole file in one call so concurrent atomic-replace writes
    by the upstream collector cannot produce a torn read (a partial read
    would either see the old complete file or the new complete file, never
    a mix, because ``build_mvm_repository.py`` writes via a temp file +
    ``os.replace``)."""
    text = path.read_text()
    records: t.List[CandidateRecord] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f'{path}:{lineno}: invalid JSON: {exc}') from exc
        records.append(CandidateRecord.from_dict(raw))
    return records


def load_split(path: Path) -> SplitInfo:
    raw = json.loads(path.read_text())
    return SplitInfo.from_dict(raw)


def validate_dataset(records: t.Sequence[CandidateRecord], split: SplitInfo) -> None:
    """Raise DatasetIntegrityError with every problem found (not just the
    first) if any of the required invariants do not hold:

      - no duplicate candidate ids
      - workload is one of the allowed values
      - query_family is non-empty
      - usable candidates have finite cost/latency values
      - train/test family sets do not intersect, per workload
      - every usable candidate's family is in exactly one split
    """
    errors: t.List[str] = []

    seen_ids: t.Set[str] = set()
    for record in records:
        if record.id in seen_ids:
            errors.append(f"duplicate candidate id: {record.id!r}")
        seen_ids.add(record.id)

        if record.workload not in ALLOWED_WORKLOADS:
            errors.append(
                f"candidate {record.id}: workload {record.workload!r} not in allowed set {ALLOWED_WORKLOADS}"
            )

        if not record.query_family or not record.query_family.strip():
            errors.append(f"candidate {record.id}: empty query_family")

        if record.usable:
            for field_name, value in (
                ('original_cost', record.original_cost),
                ('rewritten_cost', record.rewritten_cost),
                ('original_latency_ms', record.original_latency_ms),
                ('rewritten_latency_ms', record.rewritten_latency_ms),
            ):
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(
                        f"candidate {record.id}: usable=True but {field_name}={value!r} is not a finite number"
                    )

    for workload, train_fams in split.train_families.items():
        test_fams = split.test_families.get(workload, frozenset())
        overlap = train_fams & test_fams
        if overlap:
            errors.append(f"workload {workload!r}: train/test family overlap: {sorted(overlap)}")

    for record in records:
        if not record.usable:
            continue
        split_name = split.split_of(record.workload, record.query_family)
        if split_name is None:
            errors.append(
                f"candidate {record.id}: usable candidate's family {record.query_family!r} "
                f"(workload={record.workload!r}) is in neither train nor test split"
            )

    if errors:
        raise DatasetIntegrityError(
            f'Dataset integrity check failed ({len(errors)} problem(s)):\n'
            + '\n'.join(f'  - {e}' for e in errors)
        )


def group_usable_by_query(records: t.Sequence[CandidateRecord]) -> t.Dict[t.Tuple[str, str], t.List[CandidateRecord]]:
    """Group usable candidates by (workload, query) -- i.e. by original
    query -- so that all rewrite candidates logged for the same original
    SQL are evaluated together. See README.md "Grouping" section for the
    caveat that co-grouped candidates may originate from different
    ablation arms rather than one single run's candidate pool."""
    groups: t.Dict[t.Tuple[str, str], t.List[CandidateRecord]] = defaultdict(list)
    for record in records:
        if record.usable:
            groups[record.group_key].append(record)
    return dict(groups)


def filter_split(
    records: t.Sequence[CandidateRecord], split: SplitInfo, split_name: str
) -> t.List[CandidateRecord]:
    """Usable candidates whose query_family falls in `split_name` ('train'
    or 'test')."""
    if split_name not in ('train', 'test'):
        raise ValueError(f"split_name must be 'train' or 'test', got {split_name!r}")
    return [
        record
        for record in records
        if record.usable and split.split_of(record.workload, record.query_family) == split_name
    ]


@dataclass(frozen=True)
class DatasetSnapshot:
    candidates_path: str
    split_path: str
    candidates_sha256: str
    split_sha256: str
    candidate_count: int
    usable_count: int
    train_candidate_count: int
    test_candidate_count: int
    train_family_count: t.Dict[str, int]
    test_family_count: t.Dict[str, int]
    workload_counts: t.Dict[str, int]
    includes_correlate_rules: bool
    loaded_at: str

    def to_dict(self) -> dict:
        return {
            'candidates_path': self.candidates_path,
            'split_path': self.split_path,
            'candidates_sha256': self.candidates_sha256,
            'split_sha256': self.split_sha256,
            'candidate_count': self.candidate_count,
            'usable_count': self.usable_count,
            'train_candidate_count': self.train_candidate_count,
            'test_candidate_count': self.test_candidate_count,
            'train_family_count': self.train_family_count,
            'test_family_count': self.test_family_count,
            'workload_counts': self.workload_counts,
            'includes_correlate_rules': self.includes_correlate_rules,
            'loaded_at': self.loaded_at,
        }


def build_snapshot(
    candidates_path: Path, split_path: Path, records: t.Sequence[CandidateRecord], split: SplitInfo
) -> DatasetSnapshot:
    usable = [r for r in records if r.usable]
    train = filter_split(records, split, 'train')
    test = filter_split(records, split, 'test')
    includes_correlate = any(
        'CORRELATE' in rule for record in records for rule in record.rules
    )
    return DatasetSnapshot(
        candidates_path=str(candidates_path),
        split_path=str(split_path),
        candidates_sha256=sha256_file(candidates_path),
        split_sha256=sha256_file(split_path),
        candidate_count=len(records),
        usable_count=len(usable),
        train_candidate_count=len(train),
        test_candidate_count=len(test),
        train_family_count={wl: len(fams) for wl, fams in split.train_families.items()},
        test_family_count={wl: len(fams) for wl, fams in split.test_families.items()},
        workload_counts=dict(Counter(record.workload for record in records)),
        includes_correlate_rules=includes_correlate,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
