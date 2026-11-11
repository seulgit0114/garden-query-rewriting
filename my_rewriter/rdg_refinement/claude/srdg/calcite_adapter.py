#!/usr/bin/env python3
"""Adapter over the existing Calcite bridge. WRAPPER ONLY — nothing is modified.

WHY AN ADAPTER AND NOT A DIRECT IMPORT
--------------------------------------
`my_rewriter/rewrite.py` starts a JVM at import time (rewrite.py:17-18). Importing
it makes every unit test require a working JVM, jars, and several seconds of
startup. The adapter defers that import to first use and offers a `FakeAdapter`
with the same surface, so the logic in `collector`/`search`/`builder` is testable
without Calcite at all.

WHY STATES ARE SQL STRINGS AND NOT RelNode
------------------------------------------
The design (02 §0.2) found that `Rewriter.traverseRewrite` already walks RelNode
states without a SQL round trip. It is NOT exposed in `my_rewriter/rewrite.py`
(grep: 0 hits), and exposing it would mean editing that file, which the
protection rules forbid. Reported as a blocker in 03 §Known limitations.

What makes the SQL-state approach viable anyway is 02 §0.1: `matchAllRules`
already applies every rule and returns `plan_before` / `plan_after` / `sql_after`
for each one that changes the plan. So ONE call expands a state completely —
the children come back with the call, and no extra Calcite work is needed to
materialise them. The cost is the RelNode -> SQL -> RelNode round trip between
levels (design P3), which is recorded per transition via `sql_after_valid`.
"""
from __future__ import annotations

import typing as t
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleMatch:  # noqa: D101 -- documented below
    """One rule that `matchAllRules` reported at a state.

    Reported == "applying it changed the plan" (Rewriter.java:87-89), so a match
    is already an applied, non-no-op transition. `sql_after` is the child state.
    """
    name: str
    rule_type: str                 # 'normal' | 'explore'
    sql_after: str                 # '' when the un-parse produced nothing usable
    plan_before: str
    plan_after: str
    via_replay: bool = False       # True when recovered by prefix replay


class AdapterError(RuntimeError):
    pass


class CalciteAdapter:
    """Real adapter. Imports `my_rewriter.rewrite` lazily on first use."""

    name = 'calcite'

    def __init__(self, database: str = 'PostgreSQL', verbose: bool = False):
        self.database = database
        self.verbose = verbose
        self._match = None
        self._rewrite = None
        self._explain = None
        self._all_rules: t.Optional[t.List[str]] = None
        self.last_dropped: t.List[str] = []
        self.replay_calls = 0

    def _ensure(self):
        if self._match is None:
            try:
                from my_rewriter.rewrite import (match_all_rules, rewrite,  # starts JVM
                                                 Rewriter, MyRules)
            except Exception as exc:                              # noqa: BLE001
                raise AdapterError(f'cannot import my_rewriter.rewrite: '
                                   f'{type(exc).__name__}: {exc}') from exc
            self._match = match_all_rules
            self._rewrite = rewrite
            self._explain = Rewriter.explain
            self._all_rules = sorted(str(k) for k in MyRules.USED_RULES.keySet())

    def all_rules(self) -> t.List[str]:
        self._ensure()
        return list(self._all_rules or [])

    def enabled_by_replay(self, original_sql: str, create_tables: t.Sequence[str],
                          prefix: t.Sequence[str]) -> t.List[RuleMatch]:
        """Enabled(S) for S = apply(prefix, S0), computed WITHOUT a SQL round trip.

        Why this exists
        ---------------
        The normal expansion path re-parses a child's `sql_after`. That SQL comes
        from `RelToSqlConverter`, and for some plans it un-parses to text Calcite
        cannot read back (`Expression 't2.x' is not being grouped`). When that
        happens `matchAllRules` returns an empty list, so the state looks like a
        dead end. In PILOT02 every single apparent dead end was this, not a real
        one: 40 round-trip failures, 0 genuine dead ends.

        How it avoids the round trip
        ----------------------------
        `Rewriter._rewrite` applies a rule list one at a time onto the ACCUMULATED
        RelNode (Rewriter.java:133-135) and records a rule only if it changed the
        plan. So `rewrite(S0, prefix + [B])` reconstructs the intermediate state
        from the original SQL inside Calcite, and `res.rules == prefix + [B]`
        means exactly "B applies at S, and changed the plan" -- the same condition
        `matchAllRules` uses. No SQL is parsed except the original.

        `Rewriter.explain(res.r2)` then gives the plan for fingerprinting.
        `res.sql` may still be None or invalid; that is recorded rather than
        hidden, because a candidate whose SQL cannot be produced cannot be
        executed either.

        Cost: one `rewrite` per candidate rule (70 in MyRules.USED_RULES), each
        applying |prefix|+1 rules. Measured ~0.44s per call, so ~30s per state.
        That is why callers use it only after the cheap path has failed.
        """
        self._ensure()
        pre = list(prefix)
        out: t.List[RuleMatch] = []
        try:
            base = self._rewrite(original_sql, list(create_tables), pre, 1, self.database)
            plan_before = str(self._explain(base.r2)) if pre else ''
        except Exception as exc:                                  # noqa: BLE001
            raise AdapterError(f'replay base failed: {type(exc).__name__}: {exc}') from exc
        for rule in (self._all_rules or []):
            try:
                res = self._rewrite(original_sql, list(create_tables), pre + [rule],
                                    1, self.database)
                self.replay_calls += 1
                used = [str(r) for r in res.rules]
            except Exception:                                     # noqa: BLE001
                self.replay_calls += 1
                continue
            if used != pre + [rule]:
                continue          # a prefix rule or `rule` itself did not apply
            try:
                plan_after = str(self._explain(res.r2))
            except Exception:                                     # noqa: BLE001
                continue
            sql_after = getattr(res, 'sql', None)
            sql_after = '' if sql_after is None or str(sql_after) == 'None' else str(sql_after)
            out.append(RuleMatch(name=rule, rule_type='replay', sql_after=sql_after,
                                 plan_before=plan_before, plan_after=plan_after,
                                 via_replay=True))
        return out

    def parses(self, sql: str, create_tables: t.Sequence[str]) -> bool:
        """Does Calcite accept this SQL?

        [2026-09-13] Added for M4. `matchAllRules` wraps everything in a
        try/catch and returns an EMPTY list both when a state genuinely has no
        applicable rule and when its SQL cannot be parsed at all
        (Rewriter.java:100-105). The two are indistinguishable from the match
        result, so the pilot counted 8 of 150 depth-1 children (5.3%) as dead
        ends that were really round-trip failures (04 §6.2).

        `Rewriter.rewrite(sql, tables, [], 1)` separates them: with no rules to
        apply it returns `sql=None` exactly when the input could not be turned
        into a RelNode, and a non-None SQL otherwise. Verified on a known-bad
        child (dsb/query001_1) and a known-good one (dsb/query054_1).

        Costs one extra Calcite call, so callers should only use it on states
        that already came back with zero matches.
        """
        self._ensure()
        try:
            res = self._rewrite(sql, list(create_tables), [], 1, self.database)
        except Exception:                                         # noqa: BLE001
            return False
        out = getattr(res, 'sql', None)
        return out is not None and str(out) not in ('None', '')

    def match(self, sql: str, create_tables: t.Sequence[str]) -> t.List[RuleMatch]:
        self._ensure()
        try:
            raw = self._match(sql, list(create_tables), self.database, self.verbose)
        except Exception as exc:                                  # noqa: BLE001
            raise AdapterError(f'{type(exc).__name__}: {exc}') from exc
        out, dropped = [], []
        for r in raw:
            # NOTE: `r` is a java.util.Map through JPype. It has Java's
            # single-argument get(), NOT Python's get(key, default) -- calling
            # the two-argument form raises TypeError. An earlier version did
            # exactly that inside a bare `except: continue`, which silently
            # discarded EVERY row and made a query with 7 real matches look
            # like it had none. Keys are therefore read by subscript, and a
            # row that still fails is recorded rather than vanishing.
            try:
                out.append(RuleMatch(
                    name=str(r['name']),
                    rule_type=str(r['type']),
                    sql_after=str(r['sql_after']),
                    plan_before=str(r['plan_before']),
                    plan_after=str(r['plan_after']),
                ))
            except Exception as exc:                              # noqa: BLE001
                dropped.append(f'{type(exc).__name__}: {exc}')
        self.last_dropped = dropped
        if dropped and not out:
            # Every row unparseable means the contract changed, not bad luck.
            raise AdapterError(f'all {len(raw)} matcher rows unparseable; '
                               f'first error: {dropped[0]}')
        return out


@dataclass
class FakeAdapter:
    """Deterministic in-memory Calcite stand-in for tests.

    `graph` maps a state's sql -> list of (rule_name, child_sql). It models every
    behaviour the collector has to cope with: no-ops (absent from the list, since
    a real matcher never reports them), cycles (a child that is an ancestor),
    convergence (two rules leading to the same child), and failures (a sql listed
    in `failing`).
    """
    name: str = 'fake'
    graph: t.Dict[str, t.List[t.Tuple[str, str]]] = field(default_factory=dict)
    failing: t.Set[str] = field(default_factory=set)
    unparseable: t.Set[str] = field(default_factory=set)
    calls: t.List[str] = field(default_factory=list)

    replay_graph: t.Dict[str, t.List[t.Tuple[str, str]]] = field(default_factory=dict)
    replay_calls: int = 0

    def parses(self, sql: str, create_tables: t.Sequence[str]) -> bool:
        return sql not in self.unparseable

    def all_rules(self):
        return sorted({r for v in self.graph.values() for r, _ in v})

    def enabled_by_replay(self, original_sql, create_tables, prefix):
        """Synthetic replay: `replay_graph` is keyed by the '|'-joined prefix."""
        self.replay_calls += 1
        key = '|'.join(prefix)
        return [RuleMatch(name=r, rule_type='replay', sql_after='',
                          plan_before=f'PLAN({key})', plan_after=f'PLAN({key}|{r})',
                          via_replay=True)
                for r, _ in self.replay_graph.get(key, [])]

    def match(self, sql: str, create_tables: t.Sequence[str]) -> t.List[RuleMatch]:
        self.calls.append(sql)
        if sql in self.failing:
            raise AdapterError('synthetic failure')
        return [
            RuleMatch(name=rule, rule_type='normal', sql_after=child,
                      plan_before=f'PLAN({sql})', plan_after=f'PLAN({child})')
            for rule, child in self.graph.get(sql, [])
        ]
