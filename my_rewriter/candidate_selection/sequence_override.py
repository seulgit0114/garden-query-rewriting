"""Declarative, workload-scoped overrides for exact rewrite sequences.

This wrapper is intentionally evaluated before the statistical HBDC policy.
It is suitable for narrow safety knowledge that must not be diluted by a
broader rule-level prior.  Only an *exact ordered sequence* matches, so an
override for a singleton rule never suppresses a multi-rule candidate that
contains the same rule.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

from .interfaces import CandidateFeatures, MVMDecision, SelectionPolicy


@dataclass(frozen=True)
class SequenceOverride:
    id: str
    workloads: t.Tuple[str, ...]
    exact_rules: t.Tuple[str, ...]
    decision: str
    reason: str

    @classmethod
    def from_dict(cls, row: dict) -> "SequenceOverride":
        decision = str(row["decision"])
        if decision not in ("original", "rewrite"):
            raise ValueError(f"unsupported sequence-override decision: {decision!r}")
        workloads = tuple(str(item) for item in row["workloads"])
        exact_rules = tuple(str(item) for item in row["exact_rules"])
        if not workloads or not exact_rules:
            raise ValueError("sequence override requires non-empty workloads and exact_rules")
        return cls(
            id=str(row["id"]),
            workloads=workloads,
            exact_rules=exact_rules,
            decision=decision,
            reason=str(row["reason"]),
        )

    def matches(self, candidate: CandidateFeatures) -> bool:
        return (
            candidate.workload in self.workloads
            and tuple(candidate.rules) == self.exact_rules
        )


@dataclass(frozen=True)
class SequenceOverridePolicy(SelectionPolicy):
    base_policy: SelectionPolicy
    overrides: t.Tuple[SequenceOverride, ...]
    knowledge_path: str

    @classmethod
    def from_json(
        cls, base_policy: SelectionPolicy, path: t.Union[str, Path]
    ) -> "SequenceOverridePolicy":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text())
        overrides = tuple(
            SequenceOverride.from_dict(row) for row in payload.get("overrides", [])
        )
        if not overrides:
            raise ValueError(f"no sequence overrides found in {resolved}")
        return cls(base_policy, overrides, str(resolved))

    @property
    def accept_train_count(self) -> int:
        return self.base_policy.accept_train_count

    @property
    def reject_train_count(self) -> int:
        return self.base_policy.reject_train_count

    def decide(self, candidate: CandidateFeatures) -> str:
        return self.decide_detailed(candidate).decision

    def decide_detailed(self, candidate: CandidateFeatures) -> MVMDecision:
        for override in self.overrides:
            if not override.matches(candidate):
                continue
            return MVMDecision(
                decision=override.decision,
                reliability=None,
                support=None,
                prior_source="sequence_override",
                prior_key=override.id,
                reason=(
                    f"explicit exact-sequence knowledge: {override.reason}; "
                    f"workload={candidate.workload}, rules={list(candidate.rules)!r}"
                ),
                mean_latency_improvement_pct=None,
                median_latency_improvement_pct=None,
            )
        return self.base_policy.decide_detailed(candidate)

