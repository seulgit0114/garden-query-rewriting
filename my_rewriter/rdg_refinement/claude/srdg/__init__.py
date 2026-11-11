"""State-aware RDG (srdg) — collector / builder / evaluator.

Implements `02_state_aware_rdg_design.md`. Self-contained: nothing outside
`my_rewriter/rdg_refinement/claude/` is written, and existing pipeline modules
are only ever imported, never modified.
"""
__all__ = ['config', 'calcite_adapter', 'canonical', 'collector', 'search',
           'builder', 'evaluator', 'manifest']
