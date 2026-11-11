"""Cost-only baseline selection policy.

This is a reference re-implementation of the "pick the candidate with the
lower estimated cost" rule used throughout the analysis. It does not call
into ``portfolio_rewrite.py`` and does not change it; it exists so the
offline evaluator has a precise, independently testable baseline to
compare MVM against.
"""

from __future__ import annotations

from .interfaces import CandidateFeatures, SelectionPolicy


class CostOnlyPolicy(SelectionPolicy):
    """Recommend 'rewrite' iff the rewritten cost beats the original cost.

    strict=True (default): rewritten_cost < original_cost -> 'rewrite'.
                            A tie (rewritten_cost == original_cost) -> 'original'.
    strict=False:           rewritten_cost <= original_cost -> 'rewrite'.

    The live pipeline's own tie-break behavior (in
    ``portfolio_rewrite.py``) may differ from either of these -- this
    class only defines the baseline used for offline comparison, and the
    strict/non-strict choice is exposed so the offline config can be
    changed without touching this class.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def decide(self, candidate: CandidateFeatures) -> str:
        if self.strict:
            is_rewrite = candidate.rewritten_cost < candidate.original_cost
        else:
            is_rewrite = candidate.rewritten_cost <= candidate.original_cost
        return 'rewrite' if is_rewrite else 'original'
