#!/usr/bin/env python3
"""Bounded BFS search controller: depth, beam, budget, exploration quota,
cycle/dedup, checkpoint/resume, deterministic seeding.

WHY BFS AND NOT SOMETHING CLEVERER
-----------------------------------
02 §G.1: depth 2 full expansion is the whole first question. MCTS, uncertainty
guidance and cost-guided pruning are all deferred because the single fact the
MVE must establish — do rules first appear at depth 2 at all — does not need
them, and cost-guided pruning is actively dangerous here: exp6 measured the
Calcite cost ratio's regression-prediction AUC at 0.513, CI [0.354, 0.679],
i.e. no information. Pruning coverage with an uninformative signal is how
query054-style false negatives get reintroduced.

COVERAGE IS DECOUPLED FROM THE BEAM
-----------------------------------
`Enabled(S)` is recorded in full for every expanded state regardless of the
beam. The beam only decides which children are expanded FURTHER. So G1
(coverage) measurements do not depend on the budget policy.
"""
from __future__ import annotations

import random
import time
import typing as t
from collections import deque
from dataclasses import dataclass, field

from . import canonical, config
from .collector import (Expansion, StateNode, child_node, expand_state,
                        invariant_violations, transition_row)


@dataclass
class SearchLimits:
    max_depth: int = config.DEFAULT_MAX_DEPTH
    beam: int = config.DEFAULT_BEAM                 # 0 = unlimited
    max_states: int = config.DEFAULT_MAX_STATES_PER_QUERY
    max_calls: int = config.DEFAULT_MAX_CALLS_PER_QUERY
    max_rules_per_state: int = config.DEFAULT_MAX_RULES_PER_STATE
    explore_quota: float = config.DEFAULT_EXPLORE_QUOTA
    literal_insensitive_dedup: bool = False
    # M5: recover round-trip-broken states via prefix replay. Costs ~70 extra
    # Calcite calls per broken state, so it is bounded per query rather than
    # left unlimited.
    replay_recovery: bool = True
    max_replays_per_query: int = 8


@dataclass
class SearchStats:
    calls: int = 0
    states: int = 0
    rows: int = 0
    dedup_hits: int = 0
    cycles: int = 0
    beam_cuts: int = 0
    errors: int = 0
    roundtrip_failures: int = 0      # M4: SQL that Calcite cannot re-parse
    true_dead_ends: int = 0          # parses fine, genuinely no applicable rule
    replay_recovered: int = 0        # M5: states whose Enabled set replay restored
    replay_rules_recovered: int = 0  # rules that would otherwise have been lost
    invariant_violations: t.Dict[str, int] = field(default_factory=dict)
    depth_first_seen: t.Dict[str, int] = field(default_factory=dict)
    termination: str = 'completed'

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d['invariant_violations'] = dict(self.invariant_violations)
        d['depth_first_seen'] = dict(self.depth_first_seen)
        return d


def _dedup_key(node: StateNode, limits: SearchLimits) -> str:
    if limits.literal_insensitive_dedup:
        return canonical.fingerprint(canonical.literal_masked_plan(node.plan))
    return node.state_id


def select_children(matches, rng: random.Random, limits: SearchLimits,
                    novelty: t.Optional[t.Dict[str, int]] = None):
    """Choose which children to expand further.

    beam == 0 -> everything (the depth<=2 default).
    Otherwise the beam is split: an exploitation part in the matcher's own
    order, and an exploration quota that prefers rules expanded least often so
    far (02 §G.2 novelty / coverage bonus). The split exists so a ranking
    signal can never starve coverage entirely.

    Ordering is made deterministic by sorting on rule name before any random
    use, so a given seed reproduces a run exactly.
    """
    ordered = sorted(matches, key=lambda m: m.name)
    if limits.beam <= 0 or len(ordered) <= limits.beam:
        return ordered, []
    n_explore = min(len(ordered), max(0, int(round(limits.beam * limits.explore_quota))))
    n_exploit = max(0, limits.beam - n_explore)
    exploit = ordered[:n_exploit]
    rest = [m for m in ordered if m not in exploit]
    if novelty:
        rest.sort(key=lambda m: (novelty.get(m.name, 0), m.name))
    else:
        rng.shuffle(rest)
    explore = rest[:n_explore]
    chosen = exploit + explore
    cut = [m for m in ordered if m not in chosen]
    return chosen, cut


def search_query(
    adapter,
    *,
    meta: dict,
    original_sql: str,
    create_tables: t.Sequence[str],
    limits: SearchLimits,
    seed: int = config.DEFAULT_SEED,
    novelty: t.Optional[t.Dict[str, int]] = None,
    on_row: t.Optional[t.Callable[[dict], None]] = None,
) -> t.Tuple[SearchStats, t.Optional[str]]:
    """Bounded BFS over one query. Returns (stats, error_or_None).

    Never raises on Calcite failure: a broken state is recorded and skipped.
    """
    rng = random.Random(seed)
    stats = SearchStats()
    emit = on_row or (lambda r: None)

    root_exp = expand_state(adapter, StateNode(
        state_id='ROOT-PENDING', parent_state_id=None, sql=original_sql,
        plan='', depth=0, rule_prefix=()), create_tables)
    stats.calls += 1
    if root_exp.error:
        stats.errors += 1
        stats.termination = 'exception'
        return stats, root_exp.error
    # The root plan is only observable through a child's `plan_before`; with no
    # children there is nothing to expand and nothing to fingerprint.
    if not root_exp.matches:
        stats.termination = 'no_children'
        return stats, None

    root = StateNode(
        state_id=canonical.fingerprint(root_exp.matches[0].plan_before),
        parent_state_id=None, sql=original_sql,
        plan=root_exp.matches[0].plan_before, depth=0, rule_prefix=())
    root_exp.node = root

    seen = {_dedup_key(root, limits)}
    stats.states = 1
    for r in root_exp.enabled:
        stats.depth_first_seen.setdefault(r, 0)

    # queue carries (node, its expansion, ancestor fingerprints)
    queue: deque = deque([(root, root_exp, frozenset({_dedup_key(root, limits)}))])

    while queue:
        node, exp, ancestors = queue.popleft()
        chosen, cut = select_children(exp.matches, rng, limits, novelty)
        stats.beam_cuts += len(cut)

        for m in list(chosen) + list(cut):
            will_expand = (m in chosen)
            child = child_node(node, m)
            key = _dedup_key(child, limits)

            term = None
            child_exp: t.Optional[Expansion] = None
            if not will_expand:
                term = 'beam_cut'
            elif key in ancestors:
                term = 'cycle'
                stats.cycles += 1
            elif key in seen:
                term = 'dedup_hit'
                stats.dedup_hits += 1
            elif child.depth >= limits.max_depth:
                term = 'depth_limit'
            elif stats.calls >= limits.max_calls:
                term = 'budget_exhausted'
            elif stats.states >= limits.max_states:
                term = 'state_limit'
            else:
                child_exp = expand_state(adapter, child, create_tables)
                stats.calls += 1
                if child_exp.error:
                    stats.errors += 1
                    term = 'exception'
                    child_exp = None
                else:
                    seen.add(key)
                    stats.states += 1
                    for r in child_exp.enabled:
                        stats.depth_first_seen.setdefault(r, child.depth)
                    if not child_exp.matches:
                        # M4: zero matches is ambiguous -- either nothing applies,
                        # or the un-parsed SQL is not valid input any more. Ask.
                        # Only reached on empty results, so the extra Calcite call
                        # is rare.
                        parses = True
                        probe = getattr(adapter, 'parses', None)
                        if probe is not None:
                            try:
                                parses = bool(probe(child.sql, create_tables))
                            except Exception:                      # noqa: BLE001
                                parses = True     # undecidable -> do not mislabel
                        if parses:
                            term = 'no_children'
                            stats.true_dead_ends += 1
                        else:
                            term = 'roundtrip_failed'
                            stats.roundtrip_failures += 1
                            # M5: the state is not really a dead end -- rebuild
                            # its Enabled set from the original SQL by replaying
                            # the prefix, so the round trip stops costing us
                            # coverage.
                            rp = getattr(adapter, 'enabled_by_replay', None)
                            if (limits.replay_recovery and rp is not None
                                    and stats.replay_recovered < limits.max_replays_per_query):
                                try:
                                    rec = rp(original_sql, create_tables, child.rule_prefix)
                                except Exception:                  # noqa: BLE001
                                    rec = []
                                if rec:
                                    stats.replay_recovered += 1
                                    stats.replay_rules_recovered += len(rec)
                                    child_exp.matches = rec
                                    child_exp.enabled = sorted({m.name for m in rec})
                                    term = 'roundtrip_recovered'
                                    for r_ in child_exp.enabled:
                                        stats.depth_first_seen.setdefault(r_, child.depth)
                    queue.append((child, child_exp,
                                  ancestors | {key}))

            row = transition_row(
                meta=meta, parent=node, match=m, child=child,
                enabled_before=exp.enabled,
                enabled_after=(child_exp.enabled if child_exp else None),
                application_outcome='changed',
                termination_reason=term,
                expand_wall_ms=(child_exp.wall_ms if child_exp else 0),
                sql_after_valid=bool(child.sql and child.sql != 'None'),
            )
            for v in invariant_violations(row):
                stats.invariant_violations[v] = stats.invariant_violations.get(v, 0) + 1
            emit(row)
            stats.rows += 1

            if novelty is not None:
                novelty[m.name] = novelty.get(m.name, 0) + 1

        if stats.calls >= limits.max_calls:
            stats.termination = 'budget_exhausted'
            break
        if stats.states >= limits.max_states:
            stats.termination = 'state_limit'
            break

    return stats, None
