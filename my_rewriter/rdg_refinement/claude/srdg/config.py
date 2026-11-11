#!/usr/bin/env python3
"""Paths, safety limits, seeds, and the controlled vocabularies.

Every limit here is a HARD stop. The design (02 §M) could not bound the search
analytically because |Enabled(S0)| is unknown until measured, so the collector
refuses to exceed these rather than discovering the blow-up at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # .../rdg_refinement/claude
DATA = HERE / 'data'
# claude -> rdg_refinement -> my_rewriter -> repository root
PROJECT_ROOT = HERE.parents[2]
BASELINE_WORKTREE = Path(
    os.environ.get("GARDEN_BASELINE_WORKTREE", PROJECT_ROOT)
).resolve()

# ── safety limits (02 §G) ────────────────────────────────────────────────────
# [2026-09-13] 2 -> 3.
# `search_query` fills `enabled_after` only for children it actually EXPANDS, so
# a child sitting at the depth limit is recorded without its own Enabled set.
# That means max_depth=D yields enabling observations only up to depth D-1:
# the pilot ran at 2 and produced ZERO depth-2 edges (04 §4.4), not because
# 2-hop chains are absent but because depth-2 states were never expanded.
# 3 is the smallest value that can observe an A -> B -> C chain.
DEFAULT_MAX_DEPTH = 3
DEFAULT_BEAM = 0                 # 0 = no beam (full expansion) at depth<=2
# Depth 3 makes the unbounded frontier 1 + m + m^2 + m^3; with the pilot's
# measured median |Enabled(S0)| = 5-6 that is ~250 states per query. These caps
# are what keeps a high-|Enabled| query (max seen: 13) from exploding.
DEFAULT_MAX_STATES_PER_QUERY = 400
DEFAULT_MAX_CALLS_PER_QUERY = 400     # matchAllRules calls; == Calcite budget
# |MyRules.USED_RULES| measured at runtime = 70 (the source has 37+37 `entry(`
# lines but the merged map has 70 distinct keys).
DEFAULT_MAX_RULES_PER_STATE = 70
DEFAULT_EXPLORE_QUOTA = 0.25          # 02 §G.2: fraction of beam reserved for novelty
DEFAULT_SEED = 20260911
DEFAULT_MATCH_TIMEOUT_S = 300         # per matchAllRules call

# ── controlled vocabularies (02 §D) ──────────────────────────────────────────
APPLICATION_OUTCOMES = ('changed', 'noop', 'exception', 'timeout', 'nondeterministic')
TERMINATION_REASONS = (
    'depth_limit', 'beam_cut', 'dedup_hit', 'cycle', 'budget_exhausted',
    'state_limit', 'exception', 'timeout', 'no_children',
    # [2026-09-13] M4. A state whose SQL Calcite cannot re-parse is NOT a dead
    # end -- it is a measurement failure of the RelNode->SQL round trip, and
    # counting it as 'no_children' understates reachable coverage (04 §6.2).
    'roundtrip_failed',
    # [2026-09-14] M5. The round trip broke the SQL, but prefix replay rebuilt
    # the state's Enabled set from the original query, so coverage was NOT lost.
    'roundtrip_recovered',
)
FAILURE_REASONS = (
    'match_failed', 'parse_failed', 'empty_sql', 'adapter_error',
    'timeout', 'cost_failed',
)

# ── leakage control (02 §I) ──────────────────────────────────────────────────
# Context features are the ONLY thing the builder may condition on. They are
# deliberately free of table/column names: `canonical_plan` identifies the
# query family almost perfectly, so conditioning on it would leak the label.
CTX_FEATURE_NAMES = (
    'join_bin', 'agg_bin', 'filter_bin', 'depth_bin', 'has_correlate', 'has_subquery',
)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default
