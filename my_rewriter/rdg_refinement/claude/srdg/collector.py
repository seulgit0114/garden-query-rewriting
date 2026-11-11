#!/usr/bin/env python3
"""Per-state matcher re-run and transition-row construction.

One `expand_state()` call == one `matchAllRules` call == the full one-step
expansion of a state (02 §0.1). The rows it emits are the schema in 02 §D.

THE INVARIANTS THIS MODULE RECORDS RATHER THAN ASSERTS
------------------------------------------------------
Because `matchAllRules` reports a rule only when applying it changed the plan
(Rewriter.java:87-89), two things should be impossible:

  * reported_by_matcher == True  and  application_outcome == 'noop'
  * reported_by_matcher == False and  application_outcome == 'changed'

They are not asserted. If they occur, that IS the measurement of matcher
nondeterminism the design asks for (02 §E.4, gate KF3), so the collector counts
them and lets the run continue.
"""
from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass, field

from . import canonical
from .calcite_adapter import AdapterError, RuleMatch


@dataclass
class StateNode:
    state_id: str
    parent_state_id: t.Optional[str]
    sql: str
    plan: str
    depth: int
    rule_prefix: t.Tuple[str, ...]

    @property
    def prefix_sha256(self) -> str:
        return canonical.sql_fingerprint('|'.join(self.rule_prefix))


@dataclass
class Expansion:
    """Result of expanding one state."""
    node: StateNode
    matches: t.List[RuleMatch] = field(default_factory=list)
    enabled: t.List[str] = field(default_factory=list)
    error: t.Optional[str] = None
    wall_ms: int = 0


def expand_state(adapter, node: StateNode, create_tables: t.Sequence[str]) -> Expansion:
    """Run the matcher once. Never raises — failures become `error`.

    A failing state must not abort the query, and a failing query must not abort
    the run (required behaviour: "실패 query가 전체 작업을 중단하지 않음").
    """
    t0 = time.time()
    try:
        matches = adapter.match(node.sql, create_tables)
    except AdapterError as exc:
        return Expansion(node=node, error=f'match_failed: {exc}',
                         wall_ms=int((time.time() - t0) * 1000))
    except Exception as exc:                                        # noqa: BLE001
        return Expansion(node=node, error=f'adapter_error: {type(exc).__name__}: {exc}',
                         wall_ms=int((time.time() - t0) * 1000))
    return Expansion(node=node, matches=matches,
                     enabled=sorted({m.name for m in matches}),
                     wall_ms=int((time.time() - t0) * 1000))


def child_node(parent: StateNode, m: RuleMatch) -> StateNode:
    """Materialise the child. No Calcite call — `sql_after`/`plan_after` are
    already in the match (02 §0.1)."""
    return StateNode(
        state_id=canonical.fingerprint(m.plan_after),
        parent_state_id=parent.state_id,
        sql=m.sql_after,
        plan=m.plan_after,
        depth=parent.depth + 1,
        rule_prefix=parent.rule_prefix + (m.name,),
    )


def transition_row(
    *,
    meta: dict,
    parent: StateNode,
    match: RuleMatch,
    child: StateNode,
    enabled_before: t.Sequence[str],
    enabled_after: t.Optional[t.Sequence[str]],
    application_outcome: str,
    termination_reason: t.Optional[str],
    expand_wall_ms: int,
    cost_before: t.Optional[float] = None,
    cost_after: t.Optional[float] = None,
    sql_after_valid: t.Optional[bool] = None,
) -> dict:
    """Build one 02 §D row.

    `enabled_after` is None when the child was not expanded (beam cut, depth
    limit, dedup). In that case `newly_enabled`/`disabled` are left null rather
    than guessed — a pruned child contributes to coverage of the PARENT's
    Enabled set but says nothing about its own successors.
    """
    nb = canonical.structural_features(parent.plan)
    na = canonical.structural_features(child.plan)
    row = {
        'workload': meta['workload'],
        'query_id': meta['query_id'],
        'query_family': meta['query_family'],
        'original_sql_sha256': meta['original_sql_sha256'],
        'collection_run_id': meta['collection_run_id'],

        'state_id': child.state_id,
        'parent_state_id': parent.state_id,
        'canonical_plan_sha256': child.state_id,
        'depth': child.depth,
        'rule_prefix': list(child.rule_prefix),
        'prefix_sha256': child.prefix_sha256,

        'attempted_rule': match.name,
        'rule_type': match.rule_type,
        'match_position_count': None,        # 02 §E.3 — not observable via this API
        'reported_by_matcher': True,
        'application_attempted': True,
        'application_outcome': application_outcome,
        'exception_class': None,

        'enabled_before': list(enabled_before),
        'enabled_after': list(enabled_after) if enabled_after is not None else None,
        'newly_enabled': (canonical.newly_enabled(enabled_before, enabled_after, match.name)
                          if enabled_after is not None else None),
        'disabled': (canonical.disabled(enabled_before, enabled_after, match.name)
                     if enabled_after is not None else None),

        'features_before': nb,
        'features_after': na,
        'ctx_before': canonical.ctx_dict(parent.plan),
        'ctx_after': canonical.ctx_dict(child.plan),

        'calcite_cost_before': cost_before,
        'calcite_cost_after': cost_after,
        # M4: distinguishes "Calcite re-parsed this state" from "the state had
        # no applicable rule". None means not probed (child was never expanded).
        'sql_after_valid': sql_after_valid,
        'roundtrip_failed': termination_reason == 'roundtrip_failed',
        'sql_after_sha256': canonical.sql_fingerprint(child.sql),
        'equivalence_status': 'unverified',
        'measured_latency_ms': None,          # 02 §H.3 — never an oracle label
        'latency_measurement_id': None,

        'expand_wall_ms': expand_wall_ms,
        'cost_estimation_wall_ms': None,
        'termination_reason': termination_reason,
    }
    return row


def invariant_violations(row: dict) -> t.List[str]:
    """02 §E.1. Returns violated invariant names; caller records, never raises."""
    out = []
    if row['reported_by_matcher'] and row['application_outcome'] == 'noop':
        out.append('reported_but_noop')
    if not row['reported_by_matcher'] and row['application_outcome'] == 'changed':
        out.append('unreported_but_changed')
    return out
