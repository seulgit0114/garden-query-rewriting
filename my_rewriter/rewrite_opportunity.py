"""
rewrite_opportunity.py

Deterministic SQL structural analyzer for Gated Rewrite Escalation.
Called after C_rule runs; decides which NL rewrite families to try.

No LLM, no DB calls.  Uses sqlglot AST only.
Conservative policy: uncertain → do not activate.
C_nl is never returned here; caller checks broad_fallback_allowed instead.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


# ─── Public dataclasses ───────────────────────────────────────────────────────

@dataclass
class RewriteOpportunity:
    family: str          # 'C_pred' | 'C_cte' | 'C_join' | 'C_intent'
    activated: bool
    confidence: float    # 0.0–1.0
    reasons: List[str] = field(default_factory=list)
    priority: int = 5    # 1=highest priority (run first), 5=lowest


@dataclass
class OpportunityResult:
    features: Dict[str, Any]
    opportunities: List[RewriteOpportunity]
    broad_fallback_allowed: bool  # caller may try C_nl as last resort


# ─── Activation thresholds (conservative: only activate if confident) ─────────

_ACTIVATE_MIN: Dict[str, float] = {
    'C_pred':   0.60,
    'C_cte':    0.60,
    'C_join':   0.60,
    'C_intent': 0.55,   # slightly lower: more speculative family
}


# ─── AST helpers ─────────────────────────────────────────────────────────────

def _max_subquery_depth(node: exp.Expression, depth: int = 0) -> int:
    """Return the maximum Subquery nesting depth found under `node`."""
    best = depth
    if hasattr(node, 'args'):
        for val in node.args.values():
            items = val if isinstance(val, list) else [val]
            for item in items:
                if not isinstance(item, exp.Expression):
                    continue
                next_d = depth + 1 if isinstance(item, exp.Subquery) else depth
                best = max(best, _max_subquery_depth(item, next_d))
    return best


def _inner_from_tables(sel: exp.Select) -> set:
    """
    Return top-level FROM aliases/names from a SELECT without recursing
    into nested subqueries.  Used for correlated-ref detection.
    """
    tables: set = set()
    from_node = sel.args.get('from')
    if from_node:
        main = from_node.this
        if isinstance(main, exp.Table):
            tables.add((main.alias or main.name).lower())
        elif isinstance(main, exp.Subquery) and main.alias:
            tables.add(main.alias.lower())
    for join in (sel.args.get('joins') or []):
        j_this = join.this
        if isinstance(j_this, exp.Table):
            tables.add((j_this.alias or j_this.name).lower())
        elif isinstance(j_this, exp.Subquery) and j_this.alias:
            tables.add(j_this.alias.lower())
    return tables


def _is_correlated(inner_sel: exp.Select) -> bool:
    """
    Return True if inner_sel references a column whose explicit table prefix
    does not appear in the inner SELECT's own FROM clause.
    Only detects correlations via explicit table-prefixed column references.
    """
    local = _inner_from_tables(inner_sel)
    # CTE names defined inside the subquery itself
    for w in inner_sel.find_all(exp.With):
        for cte_expr in (w.args.get('expressions') or []):
            local.add(cte_expr.alias_or_name.lower())
    for col in inner_sel.find_all(exp.Column):
        tbl = col.table   # '' if no explicit prefix; table name/alias otherwise
        if tbl and tbl.lower() not in local:
            return True
    return False


def _detect_correlated_subquery(tree: exp.Expression) -> bool:
    """
    Detect correlated subqueries via explicit outer table references.
    Covers IN/comparison subqueries in WHERE/HAVING and EXISTS patterns.
    """
    # WHERE / HAVING scalar/semi-join subqueries
    for context_node in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
        for subq in context_node.find_all(exp.Subquery):
            inner_sel = subq.this if isinstance(subq.this, exp.Select) else None
            if inner_sel and _is_correlated(inner_sel):
                return True

    # EXISTS / NOT EXISTS — the inner is a Select directly, not wrapped in Subquery
    for exists_node in tree.find_all(exp.Exists):
        inner = exists_node.this
        if isinstance(inner, exp.Select):
            inner_sel = inner
        elif isinstance(inner, exp.Subquery):
            inner_sel = inner.this if isinstance(inner.this, exp.Select) else None
        else:
            inner_sel = None
        if inner_sel and _is_correlated(inner_sel):
            return True
    return False


def _detect_repeated_subquery(tree: exp.Expression) -> bool:
    """
    Return True if the same base table is scanned in 2+ separate
    scalar/semi-join subqueries (candidate for CTE materialization).
    """
    subqueries = list(tree.find_all(exp.Subquery))
    if len(subqueries) < 2:
        return False

    scalar_table_sets: List[frozenset] = []
    for sq in subqueries:
        parent = sq.parent
        # Skip FROM-clause derived tables
        if parent is None or isinstance(parent, (exp.From, exp.Join)):
            continue
        tables = frozenset(t.name.lower() for t in sq.find_all(exp.Table) if t.name)
        if tables:
            scalar_table_sets.append(tables)

    for i in range(len(scalar_table_sets)):
        for j in range(i + 1, len(scalar_table_sets)):
            if scalar_table_sets[i] & scalar_table_sets[j]:
                return True
    return False


def _extract_features(tree: exp.Expression) -> Dict[str, Any]:
    """Extract structural signals from the sqlglot AST."""

    # ── Join counts ────────────────────────────────────────────────────────
    all_joins = list(tree.find_all(exp.Join))
    n_implicit = 0
    n_explicit = 0
    for j in all_joins:
        kind = j.kind    # 'INNER', 'LEFT', 'CROSS', '' (plain / comma-style)
        side = j.side    # 'LEFT', 'RIGHT', 'FULL', ''
        has_cond = bool(j.args.get('on') or j.args.get('using'))
        # Implicit (comma-style) join: no ON/USING, no explicit side
        if kind in ('', 'CROSS') and not side and not has_cond:
            n_implicit += 1
        else:
            n_explicit += 1

    # ── Table references ──────────────────────────────────────────────────
    table_names = [t.name.lower() for t in tree.find_all(exp.Table) if t.name]
    counts = Counter(table_names)
    # Self-join: same table used ≥2 times AND at least one explicit JOIN
    has_self_join = any(v > 1 for v in counts.values()) and n_explicit >= 1

    # ── Predicates ────────────────────────────────────────────────────────
    has_or = bool(list(tree.find_all(exp.Or)))

    has_exists = False
    has_not_exists = False
    for ex_node in tree.find_all(exp.Exists):
        if isinstance(ex_node.parent, exp.Not):
            has_not_exists = True
        else:
            has_exists = True

    has_in_subquery = any(
        node.args.get('query') is not None
        for node in tree.find_all(exp.In)
    )
    has_not_in = any(
        isinstance(node.this, exp.In)
        for node in tree.find_all(exp.Not)
    )

    # ── Subquery structure ────────────────────────────────────────────────
    n_subqueries = len(list(tree.find_all(exp.Subquery)))
    nesting_depth = _max_subquery_depth(tree)

    has_orderby_in_subquery = False
    for order_node in tree.find_all(exp.Order):
        p = order_node.parent
        while p is not None:
            if isinstance(p, exp.Subquery):
                has_orderby_in_subquery = True
                break
            p = getattr(p, 'parent', None)

    n_ctes = sum(
        len(w.args.get('expressions') or [])
        for w in tree.find_all(exp.With)
    )

    return {
        'parse_ok': True,
        # joins
        'n_joins': n_explicit + n_implicit,
        'n_explicit_joins': n_explicit,
        'n_implicit_joins': n_implicit,
        'n_tables': len(set(table_names)),
        'has_self_join': has_self_join,
        # predicates
        'has_or_predicate': has_or,
        'has_exists': has_exists,
        'has_not_exists': has_not_exists,
        'has_in_subquery': has_in_subquery,
        'has_not_in': has_not_in,
        # structure
        'has_distinct': bool(tree.find(exp.Distinct)),
        'has_groupby': bool(tree.find(exp.Group)),
        'has_correlated_subquery': _detect_correlated_subquery(tree),
        'has_repeated_subquery': _detect_repeated_subquery(tree),
        'nesting_depth': nesting_depth,
        'n_subqueries': n_subqueries,
        'n_ctes': n_ctes,
        'has_orderby_in_subquery': has_orderby_in_subquery,
    }


def _conservative_features() -> Dict[str, Any]:
    """All-False features used as parse-failure fallback."""
    return {
        'parse_ok': False,
        'n_joins': 0, 'n_explicit_joins': 0, 'n_implicit_joins': 0,
        'n_tables': 0, 'has_self_join': False,
        'has_or_predicate': False, 'has_exists': False, 'has_not_exists': False,
        'has_in_subquery': False, 'has_not_in': False,
        'has_distinct': False, 'has_groupby': False,
        'has_correlated_subquery': False, 'has_repeated_subquery': False,
        'nesting_depth': 0, 'n_subqueries': 0, 'n_ctes': 0,
        'has_orderby_in_subquery': False,
    }


# ─── Skew / Calcite signals ───────────────────────────────────────────────────

def _join_key_skew(uniformity_result) -> float:
    """Max skew_score among JOIN_KEY columns in uniformity_result.per_column."""
    if uniformity_result is None or not hasattr(uniformity_result, 'per_column'):
        return 0.0
    return max(
        (info.get('skew_score', 0.0)
         for info in uniformity_result.per_column.values()
         if 'JOIN_KEY' in info.get('roles', set())),
        default=0.0,
    )


def _calcite_improvement(calcite_result: Optional[dict]) -> Optional[float]:
    """
    Improvement ratio = (input_cost - output_cost) / input_cost.
    Returns None if either cost is unavailable/invalid.
    """
    if calcite_result is None:
        return None
    ic = calcite_result.get('input_cost', -1)
    oc = calcite_result.get('output_cost', -1)
    if ic > 0 and oc > 0:
        return max(0.0, (ic - oc) / ic)
    return None


# ─── Per-family opportunity builders ─────────────────────────────────────────

def _build_c_pred(features: Dict, calcite_imp: Optional[float]) -> RewriteOpportunity:
    reasons: List[str] = []
    conf = 0.0

    if features.get('has_or_predicate'):
        conf = max(conf, 0.75)
        reasons.append('OR predicate (candidate: OR-to-UNION or filter simplification)')
    if features.get('has_not_in'):
        conf = max(conf, 0.80)
        reasons.append('NOT IN (candidate: NOT EXISTS or anti-join rewrite)')
    if features.get('has_in_subquery'):
        conf = max(conf, 0.75)
        reasons.append('IN (subquery) (candidate: semi-join or EXISTS rewrite)')
    if features.get('has_exists') or features.get('has_not_exists'):
        conf = max(conf, 0.80)
        reasons.append('EXISTS / NOT EXISTS (candidate: join or semi-join rewrite)')

    if reasons and calcite_imp is not None and calcite_imp >= 0.20:
        conf *= 0.80
        reasons.append(f'Calcite already improved {calcite_imp:.0%} → reduced urgency')

    return RewriteOpportunity(
        family='C_pred',
        activated=bool(reasons) and conf >= _ACTIVATE_MIN['C_pred'],
        confidence=round(conf, 3),
        reasons=reasons,
        priority=1,
    )


def _build_c_cte(features: Dict, calcite_imp: Optional[float]) -> RewriteOpportunity:
    reasons: List[str] = []
    conf = 0.0

    if features.get('has_correlated_subquery'):
        conf = max(conf, 0.80)
        reasons.append('Correlated subquery (candidate: CTE decorrelation or derived table)')
    if features.get('has_repeated_subquery'):
        conf = max(conf, 0.75)
        reasons.append('Repeated scalar subquery on same table(s) (candidate: CTE materialization)')
    if not reasons and features.get('n_subqueries', 0) >= 3:
        # Weak signal: many subqueries without obvious repetition
        conf = max(conf, 0.55)
        reasons.append(
            f"High subquery count ({features['n_subqueries']}) — CTE may reduce duplication"
        )

    if reasons and calcite_imp is not None and calcite_imp >= 0.20:
        conf *= 0.80
        reasons.append(f'Calcite already improved {calcite_imp:.0%} → reduced urgency')

    return RewriteOpportunity(
        family='C_cte',
        activated=bool(reasons) and conf >= _ACTIVATE_MIN['C_cte'],
        confidence=round(conf, 3),
        reasons=reasons,
        priority=2,
    )


def _build_c_join(
    features: Dict,
    calcite_imp: Optional[float],
    join_skew: float,
) -> RewriteOpportunity:
    reasons: List[str] = []
    conf = 0.0

    if features.get('n_implicit_joins', 0) > 0:
        conf = max(conf, 0.85)
        reasons.append(
            f"{features['n_implicit_joins']} implicit (comma-style) join(s) "
            f"(candidate: explicit join order optimization)"
        )
    if features.get('n_explicit_joins', 0) >= 2:
        conf = max(conf, 0.75)
        reasons.append(
            f"{features['n_explicit_joins']} explicit joins "
            f"(candidate: join order / selective pre-filter)"
        )
    if join_skew > 0.5 and reasons:
        bonus = min(0.15, join_skew * 0.20)
        conf = min(0.95, conf + bonus)
        reasons.append(
            f'Skewed JOIN key(s) detected (max skew={join_skew:.2f}) '
            f'→ distribution-aware reorder more valuable'
        )

    if reasons and calcite_imp is not None and calcite_imp >= 0.20:
        conf *= 0.80
        reasons.append(f'Calcite already improved {calcite_imp:.0%} → reduced urgency')

    return RewriteOpportunity(
        family='C_join',
        activated=bool(reasons) and conf >= _ACTIVATE_MIN['C_join'],
        confidence=round(conf, 3),
        reasons=reasons,
        priority=3,
    )


def _build_c_intent(features: Dict, calcite_imp: Optional[float]) -> RewriteOpportunity:
    reasons: List[str] = []
    conf = 0.0

    if features.get('nesting_depth', 0) >= 2:
        conf = max(conf, 0.65)
        reasons.append(
            f"Deep subquery nesting (depth={features['nesting_depth']}) "
            f"(candidate: flatten to join or simplify)"
        )
    if features.get('has_self_join'):
        conf = max(conf, 0.55)
        reasons.append('Self-join (candidate: window function or CTE simplification)')
    if features.get('has_orderby_in_subquery'):
        conf = max(conf, 0.60)
        reasons.append('ORDER BY inside subquery (likely redundant; candidate: remove)')
    if features.get('has_distinct') and features.get('n_subqueries', 0) >= 1:
        conf = max(conf, 0.55)
        reasons.append('DISTINCT with nested subquery (candidate: intent-preserving simplification)')

    if reasons and calcite_imp is not None and calcite_imp >= 0.20:
        conf *= 0.75
        reasons.append(f'Calcite already improved {calcite_imp:.0%} → reduced urgency')

    return RewriteOpportunity(
        family='C_intent',
        activated=bool(reasons) and conf >= _ACTIVATE_MIN['C_intent'],
        confidence=round(conf, 3),
        reasons=reasons,
        priority=4,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_rewrite_opportunities(
    query: str,
    schema: str = '',
    uniformity_result=None,
    calcite_result: Optional[dict] = None,
) -> OpportunityResult:
    """
    Deterministic SQL structure analysis.  No LLM, no DB calls.

    Args:
        query:             SQL query string.
        schema:            CREATE TABLE DDL (unused currently; reserved for future
                           schema-aware analysis).
        uniformity_result: Optional UniformityResult from uniformity_score.py.
                           Provides per-column skew scores to boost C_join confidence.
        calcite_result:    Optional dict with 'input_cost' and 'output_cost' keys
                           (output of execute_rewrite).  Used to downweight families
                           when Calcite already improved significantly.

    Returns:
        OpportunityResult with per-family activation decisions and raw features.
        C_nl is never activated here; check broad_fallback_allowed instead.
    """
    # ── Parse ──────────────────────────────────────────────────────────────
    try:
        stmts = sqlglot.parse(query, read='postgres')
        if not stmts or stmts[0] is None:
            raise ValueError('empty parse result')
        tree = stmts[0]
        features = _extract_features(tree)
    except Exception as exc:
        logger.warning(
            f'[rewrite_opportunity] parse failed ({exc!r}) → conservative fallback'
        )
        features = _conservative_features()

    # ── External signals ──────────────────────────────────────────────────
    join_skew = _join_key_skew(uniformity_result)
    calcite_imp = _calcite_improvement(calcite_result)

    # ── Per-family detection ──────────────────────────────────────────────
    opportunities = [
        _build_c_pred(features, calcite_imp),
        _build_c_cte(features, calcite_imp),
        _build_c_join(features, calcite_imp, join_skew),
        _build_c_intent(features, calcite_imp),
    ]

    any_activated = any(op.activated for op in opportunities)

    # broad_fallback_allowed: caller may try C_nl when no structural family is found,
    # when Calcite failed to improve, or when parsing failed.
    broad_fallback_allowed = (
        not any_activated
        or not features.get('parse_ok', True)
        or (calcite_imp is not None and calcite_imp <= 0.0)
    )

    return OpportunityResult(
        features=features,
        opportunities=opportunities,
        broad_fallback_allowed=broad_fallback_allowed,
    )
