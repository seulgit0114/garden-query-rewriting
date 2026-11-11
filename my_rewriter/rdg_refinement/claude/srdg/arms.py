#!/usr/bin/env python3
"""The seven comparison policies, run over one shared state space under one
shared budget.

WHY ALL ARMS ARE EXPRESSED AS SEARCH POLICIES
---------------------------------------------
R-Bot's own rule ordering comes from an LLM (`rag_rewrite.py:58,64`), and the
current pipeline's Gate2 injects a skeleton into an LLM prompt. Reproducing
those verbatim would make the arms differ in LLM sampling as well as in search,
which is exactly the confound 01 §3 identified. So every arm here is reduced to
the part that is comparable without an LLM: *which states does it reach*.

That makes arms 1 and 2 STRUCTURAL REPRODUCTIONS, not the full systems:
  `original_state_only`  = the candidate space reachable when rules are matched
                           once against the original query, which is what R-Bot
                           does (rag/gen_rewrites_from_rules.py:59, the single
                           matcher call in that pipeline).
  `existing_rdg_closure` = that space plus whatever the stored RDG says can
                           follow, i.e. `closure_expand`'s contribution.
They therefore bound the reachable set of those systems from ABOVE (no LLM
filtering), which is the conservative direction for the question being asked:
"what did the existing approach miss?" Anything unreachable even for this
generous version is genuinely out of reach.

BUDGET
------
One budget object is shared by every arm: expansion calls (= matchAllRules
invocations = rule application attempts, since one call applies every rule
once), visited states, max depth, wall clock. An arm that cannot use its budget
(e.g. original_state_only, which has only one state to expand) simply stops.
"""
from __future__ import annotations

import json
import random
import time
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from . import canonical
from .calcite_adapter import AdapterError


@dataclass
class Budget:
    max_calls: int = 12          # matchAllRules invocations per query
    max_states: int = 12
    max_depth: int = 3           # see srdg.config.DEFAULT_MAX_DEPTH for why 3
    wall_s: float = 240.0

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Cand:
    """A reachable rewrite candidate."""
    sql: str
    plan: str
    prefix: t.Tuple[str, ...]
    state_id: str
    depth: int


@dataclass
class ArmResult:
    arm: str
    calls: int = 0
    states: int = 0
    depth_reached: int = 0
    wall_s: float = 0.0
    stopped_by: str = 'completed'
    errors: int = 0
    candidates: t.Dict[str, Cand] = field(default_factory=dict)   # state_id -> Cand
    transitions: t.List[dict] = field(default_factory=list)
    enabled_seen: t.Set[str] = field(default_factory=set)
    dedup_hits: int = 0
    cycles: int = 0
    # M4: a state with zero matches is either a genuine dead end or SQL the
    # round trip broke. Kept apart here too, not only in srdg.search, or the
    # arm table would still report the two as one number.
    roundtrip_failures: int = 0
    true_dead_ends: int = 0
    replay_recovered: int = 0
    replay_rules_recovered: int = 0

    def summary(self) -> dict:
        return {
            'arm': self.arm, 'calls': self.calls, 'states_expanded': self.states,
            'depth_reached': self.depth_reached, 'wall_s': round(self.wall_s, 2),
            'stopped_by': self.stopped_by, 'errors': self.errors,
            'unique_candidates': len(self.candidates),
            'unique_rules_seen': len(self.enabled_seen),
            'transitions': len(self.transitions),
            'dedup_hits': self.dedup_hits, 'cycles': self.cycles,
            'roundtrip_failures': self.roundtrip_failures,
            'true_dead_ends': self.true_dead_ends,
            'replay_recovered': self.replay_recovered,
            'replay_rules_recovered': self.replay_rules_recovered,
        }


def load_existing_rdg(path: Path) -> t.Dict[str, t.Set[str]]:
    """Successor map from the CURRENT knowledge-base graph. Read-only."""
    g = json.loads(path.read_text())
    succ: t.Dict[str, t.Set[str]] = {}
    for e in g.get('edges', []):
        succ.setdefault(e['from'], set()).add(e['to'])
    return succ


# ── the generic engine every arm shares ──────────────────────────────────────
MAX_REPLAYS_PER_QUERY = 8      # see srdg.search.SearchLimits for the rationale


def _run(adapter, sql0: str, create_tables, budget: Budget, arm: str,
         pick: t.Callable[[list, int, t.Tuple[str, ...]], list],
         max_depth_override: t.Optional[int] = None) -> ArmResult:
    """BFS whose only arm-specific part is `pick(matches, depth, prefix)`.

    Holding everything else identical is what makes the arms comparable: any
    difference in coverage comes from the selection rule, not from a different
    traversal.
    """
    res = ArmResult(arm=arm)
    t0 = time.time()
    depth_cap = budget.max_depth if max_depth_override is None else max_depth_override

    try:
        root_ms = adapter.match(sql0, create_tables)
    except AdapterError as e:
        res.errors += 1
        res.stopped_by = f'root_error: {e}'
        res.wall_s = time.time() - t0
        return res
    res.calls += 1
    res.states += 1
    if not root_ms:
        res.stopped_by = 'no_children'
        res.wall_s = time.time() - t0
        return res

    root_plan = root_ms[0].plan_before
    root_id = canonical.fingerprint(root_plan)
    res.enabled_seen.update(m.name for m in root_ms)
    seen = {root_id}
    frontier = [(Cand(sql0, root_plan, (), root_id, 0), root_ms, frozenset({root_id}))]

    while frontier:
        node, ms, ancestors = frontier.pop(0)
        chosen = pick(ms, node.depth, node.prefix)
        for m in chosen:
            child_id = canonical.fingerprint(m.plan_after)
            child = Cand(m.sql_after, m.plan_after, node.prefix + (m.name,),
                         child_id, node.depth + 1)
            res.depth_reached = max(res.depth_reached, child.depth)
            res.transitions.append({
                'parent_state_id': node.state_id, 'state_id': child_id,
                'rule': m.name, 'depth': child.depth, 'prefix': list(child.prefix),
            })
            if child_id in ancestors:
                res.cycles += 1
                continue
            if child_id in seen:
                res.dedup_hits += 1
                continue
            seen.add(child_id)
            res.candidates[child_id] = child

            if child.depth >= depth_cap:
                continue
            if res.calls >= budget.max_calls or res.states >= budget.max_states:
                res.stopped_by = 'budget'
                continue
            if time.time() - t0 > budget.wall_s:
                res.stopped_by = 'wall_clock'
                continue
            try:
                cms = adapter.match(child.sql, create_tables)
            except AdapterError:
                res.errors += 1
                continue
            res.calls += 1
            res.states += 1
            res.enabled_seen.update(x.name for x in cms)
            if cms:
                frontier.append((child, cms, ancestors | {child_id}))
            else:
                # M4: ask Calcite whether this SQL is even parseable before
                # calling it a dead end. Only runs on empty results.
                probe = getattr(adapter, 'parses', None)
                ok = True
                if probe is not None:
                    try:
                        ok = bool(probe(child.sql, create_tables))
                    except Exception:                              # noqa: BLE001
                        ok = True        # undecidable -> do not mislabel
                if ok:
                    res.true_dead_ends += 1
                else:
                    res.roundtrip_failures += 1
                    # M5: rebuild the Enabled set by replaying the prefix from
                    # the original SQL, so a broken round trip no longer ends
                    # the branch. Same fallback as srdg.search.
                    rp = getattr(adapter, 'enabled_by_replay', None)
                    if rp is not None and res.replay_recovered < MAX_REPLAYS_PER_QUERY:
                        try:
                            rec = rp(sql0, create_tables, child.prefix)
                        except Exception:                          # noqa: BLE001
                            rec = []
                        if rec:
                            res.replay_recovered += 1
                            res.replay_rules_recovered += len(rec)
                            res.enabled_seen.update(x.name for x in rec)
                            frontier.append((child, rec, ancestors | {child_id}))
        if res.stopped_by in ('budget', 'wall_clock'):
            break
    res.wall_s = time.time() - t0
    return res


# ── the seven arms ───────────────────────────────────────────────────────────
def arm_original_state_only(adapter, sql0, create_tables, budget, **kw) -> ArmResult:
    """Depth 1. Everything matchable against the ORIGINAL query, nothing more.
    Structural upper bound on what R-Bot can reach."""
    return _run(adapter, sql0, create_tables, budget, 'original_state_only',
                pick=lambda ms, d, p: sorted(ms, key=lambda m: m.name),
                max_depth_override=1)


def arm_existing_rdg_closure(adapter, sql0, create_tables, budget, *, rdg_succ, **kw) -> ArmResult:
    """Depth 2, but the second step is restricted to rules the STORED RDG lists
    as successors of the rule just applied — exactly the information
    `closure_expand` contributes."""
    def pick(ms, depth, prefix):
        ordered = sorted(ms, key=lambda m: m.name)
        if depth == 0:
            return ordered
        allowed = rdg_succ.get(prefix[-1], set()) if prefix else set()
        return [m for m in ordered if m.name in allowed]
    return _run(adapter, sql0, create_tables, budget, 'existing_rdg_closure', pick=pick)


def arm_bounded_bfs(adapter, sql0, create_tables, budget, *, beam=None, **kw) -> ArmResult:
    """No guidance at all. The control that decides whether the graph earns its
    keep (02 §L' G5).

    [2026-09-14] `beam` is now honoured. Without it this arm expanded EVERY
    matched rule at every state and so always ran to the budget cap, while the
    guided arms stopped early at beam=3: measured over 76 dsb queries it used
    20/20 calls on 76 of 76 queries against the guided arms' median 11 and zero
    budget hits. Comparing "candidates per call" across that gap does not
    isolate the graph -- a shallower search simply revisits fewer states. Pass
    the same beam to both arms to make the contrast about WHICH children get
    expanded rather than how many.

    beam=None keeps the old exhaustive behaviour for callers that want the
    unbounded control.
    """
    if beam is None or beam <= 0:
        pick = lambda ms, d, p: sorted(ms, key=lambda m: m.name)          # noqa: E731
    else:
        pick = lambda ms, d, p: sorted(ms, key=lambda m: m.name)[:beam]   # noqa: E731
    return _run(adapter, sql0, create_tables, budget, 'bounded_bfs', pick=pick)


DEFAULT_ARM_BEAM = 3      # guided arms; RANDOM_ARM_BEAM matches it on purpose
RANDOM_ARM_BEAM = DEFAULT_ARM_BEAM


def arm_random_expansion(adapter, sql0, create_tables, budget, *, seed=0,
                         beam=None, **kw) -> ArmResult:
    """Uniformly random choice of which children to expand, under a FORCED beam.

    [2026-09-13] The beam is forced because without one this arm is not a
    control at all. With `beam=0` every child is expanded regardless of order,
    so shuffling changes nothing: in PILOT01 this arm reached a byte-identical
    state set to `bounded_bfs` on 24 of 24 queries (04 §2.1).

    The comparison it is meant to support is "does choosing by RDG beat
    choosing at random, given the same number of expansions" — which only
    becomes a question once the budget forces a choice. `DEFAULT_ARM_BEAM = 3      # guided arms; RANDOM_ARM_BEAM matches it on purpose
RANDOM_ARM_BEAM = DEFAULT_ARM_BEAM`
    matches the default beam of the guided arms so the arms differ in HOW they
    choose, not in how many they choose.
    """
    rng = random.Random(seed)
    k = RANDOM_ARM_BEAM if beam is None else max(1, beam)

    def pick(ms, depth, prefix):
        o = sorted(ms, key=lambda m: m.name)   # sort first -> seed reproduces exactly
        rng.shuffle(o)
        return o[:k]
    return _run(adapter, sql0, create_tables, budget, 'random_expansion', pick=pick)


def arm_cost_guided_beam(adapter, sql0, create_tables, budget, *, cost_fn=None,
                         beam=DEFAULT_ARM_BEAM, **kw) -> ArmResult:
    """Rank children by estimated cost, keep the cheapest `beam`.

    With no cost function available the arm degrades to plan SIZE (node count)
    as a stand-in and says so in `stopped_by`. That is reported, never hidden:
    a cost-guided arm evaluated without costs is not a cost-guided arm.
    """
    def key(m):
        if cost_fn is not None:
            c = cost_fn(m.sql_after)
            if c is not None:
                return (0, c)
        return (1, len(canonical.normalize_plan(m.plan_after)))
    def pick(ms, depth, prefix):
        return sorted(ms, key=lambda m: (key(m), m.name))[:beam]
    r = _run(adapter, sql0, create_tables, budget, 'cost_guided_beam', pick=pick)
    if cost_fn is None and r.stopped_by == 'completed':
        r.stopped_by = 'completed(plan_size_proxy)'
    return r


def arm_state_aware_rdg(adapter, sql0, create_tables, budget, *, layer_a=None,
                        layer_b=None, beam=DEFAULT_ARM_BEAM, **kw) -> ArmResult:
    """The proposed method: Layer A narrows, Layer B ranks, novelty protects
    coverage (02 §J).

    With no learned graph supplied it ranks by novelty alone, which is the
    honest cold-start behaviour — and the pilot runs it that way on purpose,
    because a graph learned on the pilot queries and evaluated on the same
    queries would be exactly the leakage 02 §I forbids.
    """
    seen_rules: t.Dict[str, int] = {}
    def pick(ms, depth, prefix):
        ordered = sorted(ms, key=lambda m: m.name)
        if depth > 0 and layer_a is not None and prefix:
            allowed = set(layer_a.successors(prefix[-1]))
            narrowed = [m for m in ordered if m.name in allowed]
            ordered = narrowed or ordered      # never let the index starve search
        def score(m):
            p = 0.0
            if layer_b is not None and prefix:
                r = layer_b.lookup(prefix[-1], canonical.ctx_dict(m.plan_before), m.name)
                p = r['p'] or 0.0
            return (-p, seen_rules.get(m.name, 0), m.name)
        chosen = sorted(ordered, key=score)[:beam]
        for m in chosen:
            seen_rules[m.name] = seen_rules.get(m.name, 0) + 1
        return chosen
    return _run(adapter, sql0, create_tables, budget, 'state_aware_rdg', pick=pick)


def arm_rbot_sequence(adapter, sql0, create_tables, budget, **kw) -> ArmResult:
    """R-Bot applies ONE arranged sequence, so it walks a PATH, not a tree.

    `execute_rewrite` hands the whole sequence to `Rewriter._rewrite`, which
    applies the rules one at a time onto the accumulated RelNode
    (Rewriter.java:133-135). The states R-Bot can therefore reach are
    S0 -> S1 -> S2 ... along a single ordering, and the ordering comes from an
    LLM (`rag_rewrite.py:58`).

    Without an LLM the faithful structural model is one deterministic path. An
    earlier version of this arm expanded every depth-0 child, which turned a
    path into a tree and credited R-Bot with exploration it does not perform.

    Note this arm therefore cannot consume the shared budget — that is the
    finding, not a defect of the comparison."""
    def pick(ms, depth, prefix):
        return sorted(ms, key=lambda m: m.name)[:1]
    return _run(adapter, sql0, create_tables, budget, 'rbot_sequence', pick=pick)


ARM_FUNCS = {
    'rbot_sequence': arm_rbot_sequence,
    'existing_rdg_closure': arm_existing_rdg_closure,
    'random_expansion': arm_random_expansion,
    'original_state_only': arm_original_state_only,
    'bounded_bfs': arm_bounded_bfs,
    'cost_guided_beam': arm_cost_guided_beam,
    'state_aware_rdg': arm_state_aware_rdg,
}
