"""Future online-adapter glue -- NOT imported by portfolio_rewrite.py.

This module converts the ``all_valid`` list built inside
``portfolio_rewrite.py`` (each entry: ``{'family', 'cost', 'result'}``,
where ``result['used_rules']`` is either a rule-name string or a list of
them) into the shared :class:`CandidateFeatures` type, and shows how a
group-level decision would be composed on top of the same
:class:`MVMPolicy` used by the offline evaluator.

Nothing in this file is called from the live pipeline today. The intended
future attachment point is ``_select_best_candidate()`` in
``portfolio_rewrite.py`` (the function that currently does
``min(all_valid, key=lambda x: x['cost'])``), which is not modified as
part of this change. Wiring this in is a separate, explicitly-approved
step.
"""

from __future__ import annotations

import typing as t

from ..interfaces import CandidateFeatures
from .policy import MVMPolicy


def normalize_used_rules(used_rules: t.Union[str, t.List[str], None]) -> t.Tuple[str, ...]:
    """Reproduces the str-or-list normalization that
    ``_matches_correlate_rule()`` in portfolio_rewrite.py applies to
    ``result['used_rules']``, independently (this module does not import
    from portfolio_rewrite.py)."""
    if isinstance(used_rules, list):
        return tuple(used_rules)
    if used_rules:
        return (used_rules,)
    return ()


def all_valid_entry_to_features(
    entry: dict,
    original_cost: float,
    workload: str,
    query: str,
    query_family: str,
    candidate_id: str,
) -> CandidateFeatures:
    """Convert one ``all_valid`` entry into a CandidateFeatures.

    ``original_cost`` is the ``input_cost`` local variable already
    computed in ``portfolio_rewrite.py`` before ``all_valid`` is built; it
    is not part of the entry itself.
    """
    rules = normalize_used_rules(entry['result'].get('used_rules'))
    return CandidateFeatures(
        candidate_id=candidate_id,
        workload=workload,
        query=query,
        query_family=query_family,
        rules=rules,
        original_cost=original_cost,
        rewritten_cost=entry['cost'],
    )


def select_group_with_mvm(
    all_valid: t.List[dict],
    original_cost: float,
    workload: str,
    query: str,
    query_family: str,
    policy: MVMPolicy,
) -> t.Optional[dict]:
    """Illustrative composition, not wired in: among the ``all_valid``
    entries MVM decides 'rewrite' for, return the one with the highest
    prior reliability (cost as tiebreaker), mirroring the group-selection
    rule used by the offline evaluator
    (``experiments/mvm_reliability/evaluate.py::select_group_mvm``).
    Returns ``None`` if MVM decides 'original' for every entry, meaning
    the caller should keep the original query unmodified.
    """
    scored = []
    for i, entry in enumerate(all_valid):
        features = all_valid_entry_to_features(
            entry, original_cost, workload, query, query_family, candidate_id=str(i)
        )
        decision = policy.decide_detailed(features)
        if decision.decision == 'rewrite':
            scored.append((decision.reliability or 0.0, -entry['cost'], entry))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]
