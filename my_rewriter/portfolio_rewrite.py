"""
portfolio_rewrite.py

"Logical NL Rewrite Portfolio" 파이프라인.

기존 distaware_rewrite.py / distribution_guidance.py / join_reorder.py 는
전혀 수정하지 않는다.  이 파일은 독립적인 새 실행 경로(별도 CLI / 별도 logdir)를
제공한다.

파이프라인 개요:
  1) Gate1  : Uniformity Score + Distribution Guidance  (distaware_rewrite 재사용)
  2) Gate2  : RDG Closure + Skeleton 주입              (distaware_rewrite 재사용)
  3) stat_prior: 선택적 rule scoring                   (distaware_rewrite 재사용)
  4) 후보군 동시 생성:
       C_rule  — Calcite rule sequence 기반 (arrange → execute)
       C_nl    — 범용 LLM holistic rewrite
       C_cte   — repeated/correlated subquery → CTE/derived table
       C_join  — distribution-guided join order / selective pre-filter
       C_pred  — predicate pushdown / OR-UNION / EXISTS-IN / simplification
       C_intent— query intent 보존 단순화 (불필요 join/subquery/agg 제거)
  5) syntax 검증 (EXPLAIN 기반)
  6) semantic equivalence 검증 (결과 비교, 타임아웃 보호)
  7) cost 기반 best 선택 + cost rejection gate
  8) 모든 후보의 검증/비용/실패 원인을 로그에 저장

로그 키워드:
  [PORTFOLIO] : 이 파이프라인 전용 체크포인트
  Rewrite Execution Results: ... : analyze.py 호환 포맷 (기존과 동일)
"""
import logging
import os
import itertools
import functools
import hashlib
import sqlglot
from sqlglot import exp
import re
import time
import typing as t

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.schema import NodeWithScore

from my_rewriter.database import DBArgs, Database
from my_rewriter.my_utils import MyModel, chat
from my_rewriter.db_utils import execute_rewrite, actual_time_paired
from my_rewriter.equivalence_hybrid import verify_equivalence_hybrid, sqlsolver_verdict, EQ, NEQ, UNKNOWN
from my_rewriter.evaluation_equivalence import compare_query_results
from my_rewriter.uniformity_score import compute_uniformity_score
from my_rewriter.distribution_guidance import (
    EMPTY_GUIDANCE,
    build_distribution_evidence,
    gen_distribution_guidance,
    render_guidance_for_prompt,
    plan_rules_with_guidance_dist,
    select_rules_dist,
    arrange_rules_dist,
    rearrange_rules_dist,
)
from my_rewriter.rule_dependency import (
    load_rdg,
    rdg_gate,
    build_skeleton,
    build_skeleton_scc,
    load_opportunities,
    render_skeleton_for_prompt,
    closure_expand,
    linearize_skeleton,
    load_generic_rule_descriptions,
)
from my_rewriter.rule_stat_prior import (
    load_rule_stat_matrix,
    build_query_stat_vector,
    score_rules_by_statistics,
    render_rule_stat_priors_for_prompt,
)
from my_rewriter.join_reorder import (
    _structural_check,
    _alias_consistency_check,
)
from my_rewriter.rewrite_opportunity import detect_rewrite_opportunities
from my_rewriter.rewrite import match_all_rules

PORTFOLIO_LOG_PREFIX = '[PORTFOLIO]'
GATED_LOG_PREFIX = '[GATED-PORTFOLIO]'

# ── High-table blacklist (distaware_rewrite.py와 동일 기준 유지) ──────────────
_HIGH_TABLE_BLACKLIST: t.FrozenSet[str] = frozenset({
    'FILTER_INTO_JOIN',
    'AGGREGATE_PROJECT_MERGE',
    'FILTER_REDUCE_EXPRESSIONS',
})
_HIGH_TABLE_THRESHOLD: int = 12


# ═══════════════════════════════════════════════════════════════════════════════
#  Candidate-family prompts
# ═══════════════════════════════════════════════════════════════════════════════

# ── C_nl : generic holistic rewrite ──────────────────────────────────────────
_NL_SYS = """\
You are a PostgreSQL query optimization expert.
Rewrite the SQL query to improve execution efficiency while preserving exact semantics.

Apply any combination of:
- Eliminating redundant subqueries or joins
- Simplifying aggregation/projection
- Rewriting correlated subqueries to joins
- Pushing filters closer to base tables
- Flattening unnecessary nesting

CRITICAL: The rewritten SQL must return the exact same result set as the original.
Output ONLY the rewritten SQL wrapped in ```sql ... ```. No explanation."""

_NL_USER = """\
Schema:
{schema}

{distribution_context}

Original SQL:
```sql
{sql}
```

Rewrite Suggestions (for context only):
{suggestions}

Output the optimized SQL in ```sql ... ```."""

# ── C_cte : CTE extraction ────────────────────────────────────────────────────
_CTE_SYS = """\
You are a PostgreSQL expert specializing in CTE-based query restructuring.

Your task: Rewrite the SQL using WITH ... AS (...) CTEs to:
1. Extract repeated subqueries or correlated subqueries evaluated multiple times
2. Compute shared aggregate expressions once and reference the CTE
3. Use LATERAL or derived tables where CTEs would cause unnecessary materialization

CRITICAL CONSTRAINTS:
- Every table, column, and condition from the original must be preserved
- Result set must be identical to the original
- Do NOT use WITH ... AS MATERIALIZED unless the CTE is referenced more than once
- If no CTE opportunity exists, return the original SQL unchanged
- After moving a table into a CTE, update EVERY reference to that table in the
  rest of the query to use the CTE's alias/output columns instead. Never leave
  the table's original alias referenced outside the CTE — that produces a
  "missing FROM-clause entry" error (undeclared alias).

Output ONLY the SQL in ```sql ... ```. No explanation."""

_CTE_USER = """\
Schema:
{schema}

SQL to restructure with CTEs:
```sql
{sql}
```

Output in ```sql ... ```."""

# ── C_join : distribution-guided join reordering ──────────────────────────────
_JOIN_SYS = """\
You are a PostgreSQL query optimization expert focused on join order and selective pre-filtering.

Your task: Rewrite the SQL so that the most selective operations happen first, reducing
intermediate result sizes.  Use the provided data distribution context to guide decisions.

Strategies:
1. Apply highly selective WHERE filters on a table by wrapping it in a subquery/CTE before joining
2. Reorder explicit JOINs so smaller/more-selective results appear earlier in the join tree
3. Use LATERAL subqueries for correlated selective lookups
4. Convert implicit comma-joins (FROM a, b, c WHERE ...) to explicit JOIN ... ON syntax
   to expose join order to the optimizer

CRITICAL:
- Preserve all original tables, columns, conditions, and aggregations exactly
- After wrapping a table in a subquery/CTE, update EVERY reference to that
  table in the rest of the query to use the subquery/CTE's alias/output
  columns instead. Never leave the table's original alias referenced outside
  it — that produces a "missing FROM-clause entry" error (undeclared alias).
- Output ONLY the rewritten SQL in ```sql ... ```. No explanation."""

_JOIN_USER = """\
Schema:
{schema}

Data Distribution Context:
{distribution_context}

Original SQL:
```sql
{sql}
```

Output the join-optimized SQL in ```sql ... ```."""

# ── C_pred : predicate transformations ───────────────────────────────────────
_PRED_SYS = """\
You are a PostgreSQL expert specializing in predicate optimization.

Apply any relevant transformations:
1. Predicate pushdown: move WHERE conditions as close to base tables as possible
2. OR-to-UNION: convert OR predicates on indexed columns to UNION ALL branches
3. EXISTS/IN rewrite: convert correlated EXISTS/IN subqueries to (anti-)joins
4. Constant folding: simplify constant expressions (e.g., 1=1, x IN (single_value))
5. NOT IN to NOT EXISTS: when the subquery may return NULLs, convert for correct semantics

CRITICAL: Preserve exact semantics. If no transformation improves the query, return it unchanged.
If a transformation moves a table into a subquery/CTE, update EVERY reference
to that table elsewhere in the query to use the subquery/CTE's alias/output
columns instead — never leave the table's original alias referenced outside it.
Output ONLY the SQL in ```sql ... ```. No explanation."""

_PRED_USER = """\
Schema:
{schema}

SQL:
```sql
{sql}
```

Output the predicate-optimized SQL in ```sql ... ```."""

# ── C_intent : query intent simplification ────────────────────────────────────
_INTENT_SYS = """\
You are a PostgreSQL expert. Identify the true computational intent of the SQL query
and simplify it by removing structural overhead that does not affect the result.

Look for:
1. Joins whose output columns are never used in SELECT, WHERE, or aggregation
2. Subqueries whose results are equivalent to a simpler expression
3. GROUP BY + aggregation that can be simplified when the grouping key is unique
4. Self-joins that can be collapsed
5. Redundant DISTINCT or unnecessary ORDER BY in subqueries

CRITICAL:
- The simplified query must return the identical result set
- When in doubt, keep the original structure
- Output ONLY the SQL in ```sql ... ```. No explanation."""

_INTENT_USER = """\
Schema:
{schema}

Rewrite Suggestions (intent context):
{suggestions}

SQL:
```sql
{sql}
```

Output the intent-simplified SQL in ```sql ... ```."""

# ── Corrective retry (ported from QUITE §5.2 "Equivalence Error Correction":
#    iteratively feed the rejected candidate + rejection reason back to the
#    LLM until it fixes the issue or the retry budget is exhausted).
#    Originally equivalence-only; also drives alias_fail retries (see
#    _generate_and_verify_with_retry) since _alias_consistency_check already
#    computes the exact undeclared alias set, which is a very concrete,
#    actionable feedback signal — same mechanism, different rejection reason.
#    ─────────────────────────────────────────────────────────────────────────
_EQUIV_RETRY_SYS = """\
You are a PostgreSQL expert correcting a SQL rewrite that was REJECTED by an
automated verification check (semantic equivalence or alias/reference
consistency). Your previous rewrite has a specific, concrete problem — see
the rejection reason below.

Fix the specific issue identified below while preserving the optimization
intent (do not simply revert every change back to the original). If you
cannot find a rewrite that is both correct and beneficial, return the
ORIGINAL query unchanged.

CRITICAL: The corrected SQL must (1) be semantically equivalent to the
original — same result set for any valid database instance, and (2) reference
only aliases/tables actually declared in its own FROM/JOIN/WITH clauses. If
any table is wrapped in a subquery/CTE, update EVERY reference to that table
elsewhere in the query to use the subquery/CTE's alias/output columns — never
leave the table's original alias referenced outside it.
Output ONLY the SQL in ```sql ... ```. No explanation."""

_EQUIV_RETRY_USER = """\
Schema:
{schema}

Original SQL:
```sql
{original_sql}
```

Your previous rewrite ({family}), which was REJECTED:
```sql
{rejected_sql}
```

Rejection reason: {eq_reason}

Output a corrected SQL in ```sql ... ```."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper utilities(portfolio_rewrite.py::portfolio_rag_rewrite())
# ═══════════════════════════════════════════════════════════════════════════════

def _count_from_tables(query: str) -> int:
    """FROM 절의 최상위 테이블 수 근사 (distaware_rewrite.py 동일 로직)."""
    m = re.search(
        r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
        query, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return 0
    from_text = m.group(1)
    depth, flat = 0, []
    for ch in from_text:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        elif depth == 0: flat.append(ch)
    flat_text = ''.join(flat)
    return flat_text.count(',') + len(re.findall(r'\bJOIN\b', flat_text, re.IGNORECASE)) + 1


def _extract_sql_block(response: str) -> t.Optional[str]:
    """LLM 응답에서 ```sql ... ``` 블록 안의 SQL을 추출한다."""
    m = re.search(r'```sql\s*(.+?)\s*```', response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # 백틱 없이 SELECT로 시작하는 경우 fallback
    m2 = re.search(r'(SELECT\s.+)', response, re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return None


def _strip_semi(sql: str) -> str:
    return sql.rstrip().rstrip(';')


def _canonical_sql(sql: t.Optional[str]) -> str:
    """Return a conservative, stable representation for exact-candidate dedup.

    SQLGlot formatting removes irrelevant whitespace/casing differences while
    preserving literals and the parsed tree.  PostgreSQL syntax that SQLGlot
    cannot parse falls back to semicolon trimming only; the fallback deliberately
    avoids lower-casing or whitespace rewriting inside string literals.
    """
    if not sql:
        return ''
    stripped = _strip_semi(sql.strip())
    try:
        return sqlglot.parse_one(stripped, read='postgres').sql(
            dialect='postgres', pretty=False)
    except Exception:
        return stripped


def _candidate_key(original_sql: str, rewritten_sql: t.Optional[str]) -> str:
    payload = _canonical_sql(original_sql) + '\0' + _canonical_sql(rewritten_sql)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _schema_for_referenced_tables(query: str, schema: str) -> str:
    """Reduce schema only for LLM prompts; validators retain the full schema."""
    try:
        tree = sqlglot.parse_one(query, read='postgres')
        referenced = {table.name.lower() for table in tree.find_all(exp.Table) if table.name}
        if not referenced:
            return schema
        selected = []
        for statement in (part.strip() for part in schema.split(';') if part.strip()):
            match = re.search(
                r'\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:"?[\w]+"?\.)?"?([\w]+)"?',
                statement, re.IGNORECASE)
            if match and match.group(1).lower() in referenced:
                selected.append(statement + ';')
        return '\n\n'.join(selected) if selected else schema
    except Exception:
        return schema


# ── Syntax verification ───────────────────────────────────────────────────────

def _verify_syntax(sql: str, db_args: DBArgs) -> t.Tuple[bool, str]:
    """
    EXPLAIN (FORMAT JSON) 으로 SQL 문법을 검증한다.
    성공 시 (True, cost_str), 실패 시 (False, error_msg).
    """
    ok, reason, _cost = _verify_syntax_with_cost(sql, db_args)
    return ok, reason


def _verify_syntax_with_cost(
    sql: str, db_args: DBArgs
) -> t.Tuple[bool, str, float]:
    """Validate with one EXPLAIN and return its cost for immediate reuse."""
    db = None
    try:
        db = Database(db_args)
        cost = db.cost_estimation(sql)
        if cost == -1:
            return False, 'EXPLAIN failed (cost=-1)', -1
        return True, f'cost={cost:.2f}', cost
    except Exception as e:
        return False, f'syntax_error: {str(e)[:120]}', -1
    finally:
        if db is not None and hasattr(db, 'close'):
            db.close()


# ── Semantic equivalence verification ────────────────────────────────────────

def _verify_equivalence(
    original_sql: str,
    rewritten_sql: str,
    db_args: DBArgs,
    timeout_s: int = 60,
    original_result_cache: t.Optional[dict] = None,
) -> t.Tuple[t.Optional[bool], str]:
    """Backward-compatible alias for the shared evaluation comparator."""
    return compare_query_results(
        original_sql, rewritten_sql, db_args, timeout_s,
        original_result_cache=original_result_cache)


# ── Per-family candidate generation ──────────────────────────────────────────

def _gen_nl_candidate(
    model: MyModel,
    query: str,
    schema: str,
    distribution_guidance: str,
    suggestions_str: str,
) -> t.Optional[str]:
    dist_ctx = f'Distribution Guidance:\n{distribution_guidance}' if distribution_guidance else ''
    msgs = [
        {'role': 'system', 'content': _NL_SYS},
        {'role': 'user', 'content': _NL_USER.format(
            schema=schema, sql=query,
            distribution_context=dist_ctx,
            suggestions=suggestions_str or '(none)',
        )},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_nl attempt={attempt} SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_nl attempt={attempt} extraction failed')
    return None


def _gen_cte_candidate(model: MyModel, query: str, schema: str) -> t.Optional[str]:
    msgs = [
        {'role': 'system', 'content': _CTE_SYS},
        {'role': 'user', 'content': _CTE_USER.format(schema=schema, sql=query)},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_cte attempt={attempt} SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_cte attempt={attempt} extraction failed')
    return None


def _gen_join_candidate(
    model: MyModel,
    query: str,
    schema: str,
    distribution_guidance: str,
) -> t.Optional[str]:
    dist_ctx = distribution_guidance if distribution_guidance else '(no distribution data available)'
    msgs = [
        {'role': 'system', 'content': _JOIN_SYS},
        {'role': 'user', 'content': _JOIN_USER.format(
            schema=schema, sql=query, distribution_context=dist_ctx,
        )},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_join attempt={attempt} SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_join attempt={attempt} extraction failed')
    return None


def _gen_pred_candidate(model: MyModel, query: str, schema: str) -> t.Optional[str]:
    msgs = [
        {'role': 'system', 'content': _PRED_SYS},
        {'role': 'user', 'content': _PRED_USER.format(schema=schema, sql=query)},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_pred attempt={attempt} SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_pred attempt={attempt} extraction failed')
    return None


def _gen_intent_candidate(
    model: MyModel,
    query: str,
    schema: str,
    suggestions_str: str,
) -> t.Optional[str]:
    msgs = [
        {'role': 'system', 'content': _INTENT_SYS},
        {'role': 'user', 'content': _INTENT_USER.format(
            schema=schema, sql=query,
            suggestions=suggestions_str or '(none)',
        )},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_intent attempt={attempt} SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_intent attempt={attempt} extraction failed')
    return None


def _gen_corrective_retry(
    family: str,
    query: str,
    schema: str,
    rejected_sql: str,
    eq_reason: str,
) -> t.Optional[str]:
    """
    Regenerate a candidate after an equivalence-check failure, feeding the
    rejected SQL and the rejection reason back to the LLM so it can correct
    the specific issue — ported from QUITE's iterative equivalence correction
    loop (paper §5.2: "the LLM repeatedly compares the original query with the
    rewrite candidate, systematically identifying discrepancies and applying
    refinements ... until confidently equivalent or the budget is exhausted").
    """
    msgs = [
        {'role': 'system', 'content': _EQUIV_RETRY_SYS},
        {'role': 'user', 'content': _EQUIV_RETRY_USER.format(
            schema=schema, original_sql=query, family=family,
            rejected_sql=rejected_sql, eq_reason=eq_reason,
        )},
    ]
    for attempt in range(1, 3):
        resp = chat(msgs)
        sql = _extract_sql_block(resp)
        if sql:
            logging.info(f'{GATED_LOG_PREFIX} {family}: corrective retry SQL extracted (len={len(sql)})')
            return sql
        logging.info(f'{GATED_LOG_PREFIX} {family}: corrective retry extraction failed (attempt={attempt})')
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def portfolio_rag_rewrite(
    retriever_res: t.List[NodeWithScore],
    rewrites: t.List[t.Dict],
    query: str,
    schema: str,
    db_args: DBArgs,
    model_args: t.Dict[str, str],
    pool: t.Optional[dict] = None,
    tau: float = 0.2,
    alpha: float = 0.5,
    CASE_BATCH: int = 5,
    RULE_BATCH: int = 10,
    REWRITE_ROUNDS: int = 1,
    gate1_enabled: bool = True,
    gate2_enabled: bool = True,
    stat_prior_enabled: bool = True,
    stat_prior_hard_filter: bool = False,
    portfolio_enabled: bool = True,
    verify_enabled: bool = True,
    max_candidates_per_family: int = 1,
    cost_rejection_threshold: float = 1.05,
    input_cost: float = -1.0,
    equiv_timeout_s: int = 300,
) -> None:
    """
    Logical NL Rewrite Portfolio 파이프라인.

    - portfolio_enabled=False 이면 C_rule 단독 실행 (distaware와 동일 수준 baseline).
    - verify_enabled=False 이면 syntax/equiv 검증 없이 cost만 비교 (실험용).
    """
    model = MyModel(model_args)
    logging.info(
        f'{PORTFOLIO_LOG_PREFIX} portfolio_rag_rewrite 시작: '
        f'gate1={gate1_enabled}, gate2={gate2_enabled}, '
        f'stat_prior={stat_prior_enabled}, portfolio={portfolio_enabled}, '
        f'verify={verify_enabled}'
    )

    # ── Step 1: Gate1 — Uniformity Score + Distribution Guidance ─────────────
    guidance_activated = False
    uresult = None
    closure_added: list = []
    rule_stat_priors_str = ''
    rule_priors: list = []

    if gate1_enabled:
        uresult = compute_uniformity_score(query, schema, db_args, tau=tau, alpha=alpha)
        logging.info(f'{PORTFOLIO_LOG_PREFIX} Uniformity Score: '
                     f'score={uresult.score:.4f}, activate={uresult.activate}')
        if uresult.activate:
            evidence = build_distribution_evidence(uresult, pool=pool)
            guidance = gen_distribution_guidance(model, query, evidence)
            guidance_activated = True
        else:
            guidance = dict(EMPTY_GUIDANCE)
    else:
        guidance = dict(EMPTY_GUIDANCE)
        logging.info(f'{PORTFOLIO_LOG_PREFIX} Gate1 비활성화')

    distribution_guidance_str = render_guidance_for_prompt(guidance)
    logging.info(f'{PORTFOLIO_LOG_PREFIX} Distribution Guidance Result: '
                 f'{{"activated": {guidance_activated}, "hints": {len(guidance.get("rewrite_hints", []))}}}')

    # ── Step 2: Recipe 생성 (RAG → rule selection) ───────────────────────────
    nl_suggestions = [obj['rewrite'] for obj in rewrites['nl']]
    normal_rules = [r for r in rewrites['calcite'] if r['type'] == 'normal']
    explore_rules = [r for r in rewrites['calcite'] if r['type'] == 'explore']
    normal_suggestions = [obj['rewrite'] for obj in normal_rules]
    strategies = normal_suggestions + nl_suggestions

    tasks = [
        model.gen_summarize_strategies(query, retriever_res, strategies, case_batch=CASE_BATCH),
        model.select_rules_from_cases(
            retriever_res, normal_rules=normal_rules, explore_rules=explore_rules
        ),
    ]
    task_results = run_async_tasks(tasks)
    summarized_strategies = task_results[0]
    selected_rules: t.List[t.List[t.Dict[str, str]]] = task_results[1]

    suggestions_str = '\n\n'.join(
        [f'### Suggestion {i+1}:\n"""{s}"""' for i, s in enumerate(summarized_strategies)]
    )
    logging.info('Intermediate Results: ' + str(
        {'suggestions_str': suggestions_str, 'selected_rules': selected_rules}
    ))

    relevant_rules: t.List[t.Dict[str, str]] = []
    selected_rules_lst = list(itertools.chain.from_iterable(selected_rules))

    _n_tables = _count_from_tables(query)
    if _n_tables >= _HIGH_TABLE_THRESHOLD:
        before_n = len(selected_rules_lst)
        selected_rules_lst = [r for r in selected_rules_lst
                               if r['name'] not in _HIGH_TABLE_BLACKLIST]
        removed = before_n - len(selected_rules_lst)
        if removed:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} High-table blacklist: {removed}개 제거 '
                         f'(n_tables={_n_tables})')

    n_batches = max(1, (len(selected_rules_lst) - 1) // RULE_BATCH + 1)
    for i in range(n_batches):
        start_idx = i * RULE_BATCH
        end_idx = min((i + 1) * RULE_BATCH, len(selected_rules_lst))
        relevant_rules = select_rules_dist(
            model, query, suggestions_str,
            relevant_rules + selected_rules_lst[start_idx:end_idx],
            distribution_guidance_str,
            rule_stat_priors=rule_stat_priors_str,
        )
        logging.info(f'Rules After the {i+1}th Selection: '
                     f'{[r["name"] for r in relevant_rules]}')

    # ── Step 3: Gate2 — RDG Closure + Skeleton ───────────────────────────────
    rdg_edges: list = []
    rule_dependency_constraints_str = ''
    if gate2_enabled:
        rdg_graph = load_rdg()
        full_rule_descriptions = load_generic_rule_descriptions()
        relevant_rules, closure_added = closure_expand(
            relevant_rules, selected_rules_lst, rdg_graph, full_rule_descriptions
        )
        if closure_added:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} RDG Closure: +{len(closure_added)}개 → '
                         f'{[r["name"] for r in relevant_rules]}')

        rdg_active = rdg_gate(relevant_rules)
        if rdg_active:
            _sk = build_skeleton_scc(
                [r['name'] for r in relevant_rules], rdg_graph,
                enabled_at_s0=_enabled_at_s0(rewrites),
                opportunities=load_opportunities(),
            )
            layers, rdg_edges, unresolved = _sk['layers'], _sk['edges'], _sk['unresolved']
            rule_dependency_constraints_str = render_skeleton_for_prompt(
                layers, rdg_edges, unresolved, _sk['cyclic_groups']
            )
            logging.info(f'{PORTFOLIO_LOG_PREFIX} RDG Gate: activate=True, '
                         f'edges={len(rdg_edges)}, layers={len(layers)}')
        else:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} RDG Gate: activate=False')
    else:
        logging.info(f'{PORTFOLIO_LOG_PREFIX} Gate2 비활성화')

    # ── Step 4: Stat Prior ────────────────────────────────────────────────────
    if stat_prior_enabled:
        try:
            stat_matrix = load_rule_stat_matrix()
            _uresult = uresult if gate1_enabled and uresult and uresult.activate else None
            query_stat_vec = build_query_stat_vector(
                query, schema, db_args, pool=pool, uniformity_result=_uresult
            )
            rdg_added_set = {r_name for r_name, _, _ in closure_added} if closure_added else set()
            rule_priors = score_rules_by_statistics(
                relevant_rules, query_stat_vec, stat_matrix,
                rdg_added_names=rdg_added_set,
            )
            if stat_prior_hard_filter:
                suppressed = [p['name'] for p in rule_priors
                              if p['decision'] == 'suppress' and p['is_rdg_added']]
                if suppressed:
                    relevant_rules = [r for r in relevant_rules
                                      if r['name'] not in suppressed]
                    rule_priors = [p for p in rule_priors
                                   if p['name'] not in suppressed]
                    logging.info(f'{PORTFOLIO_LOG_PREFIX} Stat Prior hard-filter: '
                                 f'{suppressed} 제거')
            rule_stat_priors_str = render_rule_stat_priors_for_prompt(rule_priors)
            logging.info(f'{PORTFOLIO_LOG_PREFIX} Stat Prior scoring 완료: '
                         f'{len(rule_priors)}개 규칙')
        except Exception as e:
            logging.warning(f'{PORTFOLIO_LOG_PREFIX} Stat Prior 실패 (무시): {e}')

    # ── Step 5: C_rule — Calcite arrange + execute ───────────────────────────
    arranged_rule_seq = arrange_rules_dist(
        model, query, suggestions_str, relevant_rules, distribution_guidance_str,
        rule_dependency_constraints=rule_dependency_constraints_str,
        rdg_edges=rdg_edges,
        rule_stat_priors=rule_stat_priors_str,
    )
    logging.info(f'Arranged Rule Sequence: {arranged_rule_seq}')

    c_rule_res = execute_rewrite(query, schema, db_args, arranged_rule_seq, REWRITE_ROUNDS)
    logging.info(f'{PORTFOLIO_LOG_PREFIX} C_rule execute 완료: '
                 f'used={c_rule_res["used_rules"]}, cost={c_rule_res["output_cost"]}')

    # ── Step 5b: C_rule rearrange (2nd attempt) ──────────────────────────────
    # 1차 결과가 cost 개선 없으면 rearrange 시도 (기존 distaware 동일 동작 유지)
    _crule_cost = c_rule_res.get('output_cost', -1)
    _improved_by_rule = input_cost > 0 and _crule_cost > 0 and _crule_cost < input_cost

    if not _improved_by_rule:
        rearranged_seq = rearrange_rules_dist(
            model, query, suggestions_str, relevant_rules, arranged_rule_seq,
            c_rule_res['used_rules'], distribution_guidance_str,
            rule_dependency_constraints=rule_dependency_constraints_str,
            rdg_edges=rdg_edges,
            rule_stat_priors=rule_stat_priors_str,
        )
        logging.info(f'Rearranged Rule Sequence: {rearranged_seq}')
        c_rule_res2 = execute_rewrite(query, schema, db_args, rearranged_seq, REWRITE_ROUNDS)
        logging.info(f'{PORTFOLIO_LOG_PREFIX} C_rule rearrange 완료: '
                     f'used={c_rule_res2["used_rules"]}, cost={c_rule_res2["output_cost"]}')
        # 두 Calcite 결과 중 더 좋은 쪽을 C_rule 대표 결과로 채택
        cost1 = c_rule_res.get('output_cost', -1)
        cost2 = c_rule_res2.get('output_cost', -1)
        if cost2 > 0 and (cost1 <= 0 or cost2 < cost1):
            c_rule_res = c_rule_res2
            logging.info(f'{PORTFOLIO_LOG_PREFIX} C_rule: rearrange 결과 채택 (cost {cost1:.2f}→{cost2:.2f})')

    # C_rule을 analyze.py 호환 포맷으로 로깅 (기존 파이프라인과 동일 키워드)
    logging.info(f'Rewrite Execution Results: {_tag_line(c_rule_res, "LINE1")}')

    # ── Step 6: NL 후보군 생성 (portfolio_enabled=True 시) ──────────────────
    if not portfolio_enabled:
        logging.info(f'{PORTFOLIO_LOG_PREFIX} portfolio_enabled=False: NL 후보 생성 스킵')
        _apply_cost_rejection_gate(c_rule_res, input_cost, cost_rejection_threshold, query)
        logging.info(f'{PORTFOLIO_LOG_PREFIX} portfolio_rag_rewrite 종료 (rule-only mode)')
        return

    nl_candidates: t.List[t.Dict] = []  # {'family', 'sql', 'raw_cost', 'syntax_ok', 'equiv_ok', 'fail_reason'}

    families = [
        ('C_nl',     _gen_nl_candidate(model, query, schema, distribution_guidance_str, suggestions_str)),
        ('C_cte',    _gen_cte_candidate(model, query, schema)),
        ('C_join',   _gen_join_candidate(model, query, schema, distribution_guidance_str)),
        ('C_pred',   _gen_pred_candidate(model, query, schema)),
        ('C_intent', _gen_intent_candidate(model, query, schema, suggestions_str)),
    ]

    for family, sql in families:
        if sql is None:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: LLM 생성 실패')
            nl_candidates.append({
                'family': family, 'sql': None,
                'raw_cost': -1, 'syntax_ok': False, 'equiv_ok': False,
                'fail_reason': 'llm_gen_failed',
            })
            continue

        # 원본과 동일한 SQL인 경우 skip (LLM이 변경하지 않은 경우)
        if sql.strip().rstrip(';') == query.strip().rstrip(';'):
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: 원본과 동일 → skip')
            nl_candidates.append({
                'family': family, 'sql': sql,
                'raw_cost': -1, 'syntax_ok': False, 'equiv_ok': False,
                'fail_reason': 'identical_to_original',
            })
            continue

        entry: dict = {'family': family, 'sql': sql, 'raw_cost': -1,
                       'syntax_ok': False, 'equiv_ok': False, 'fail_reason': ''}

        # Step 6a: 구조/alias 사전검사 — WARN-ONLY(비차단). syntax(EXPLAIN)+동치검증이
        # 진짜 게이트이며, 이 휴리스틱은 거짓양성으로 정상 rewrite를 죽이던 문제가 있어
        # 이제 로그만 남기고 통과시킨다. (gated 경로와 동일)
        struct_ok, struct_reason = _structural_check(query, sql)
        if not struct_ok:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: [warn] 구조 사전검사 지적(비차단) — {struct_reason}')

        alias_ok, alias_reason = _alias_consistency_check(sql)
        if not alias_ok:
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: [warn] alias 사전검사 지적(비차단) — {alias_reason}')

        # Step 6b: Syntax 검증 (EXPLAIN)
        if verify_enabled:
            syn_ok, syn_reason, syntax_cost = _verify_syntax_with_cost(sql, db_args)
            entry['syntax_ok'] = syn_ok
            if not syn_ok:
                logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: 문법 검증 실패 — {syn_reason}')
                entry['fail_reason'] = f'syntax_fail: {syn_reason}'
                nl_candidates.append(entry)
                continue
            entry['raw_cost'] = syntax_cost
        else:
            entry['syntax_ok'] = True
            db_tmp = None
            try:
                db_tmp = Database(db_args)
                entry['raw_cost'] = db_tmp.cost_estimation(sql)
            except Exception:
                entry['raw_cost'] = -1
            finally:
                if db_tmp is not None and hasattr(db_tmp, 'close'):
                    db_tmp.close()

        # Step 6c: Semantic equivalence 검증
        if verify_enabled:
            eq_ok, eq_reason = _verify_equivalence(
                query, sql, db_args, timeout_s=equiv_timeout_s
            )
            entry['equiv_ok'] = eq_ok
            if not eq_ok:
                logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: 동치 검증 실패 — {eq_reason}')
                entry['fail_reason'] = f'equiv_fail: {eq_reason}'
                nl_candidates.append(entry)
                continue
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: 검증 통과 '
                         f'(cost={entry["raw_cost"]:.2f}, equiv={eq_reason})')
        else:
            entry['equiv_ok'] = True
            logging.info(f'{PORTFOLIO_LOG_PREFIX} {family}: verify_enabled=False, '
                         f'cost={entry["raw_cost"]:.2f}')

        nl_candidates.append(entry)

    # ── Step 7: Portfolio 요약 로그 ──────────────────────────────────────────
    for c in nl_candidates:
        logging.info(
            f'{PORTFOLIO_LOG_PREFIX} Portfolio Candidate: '
            f'family={c["family"]}, syntax={c["syntax_ok"]}, equiv={c["equiv_ok"]}, '
            f'cost={c["raw_cost"]:.2f if c["raw_cost"] > 0 else -1}, '
            f'fail={c["fail_reason"] or "none"}'
        )

    # ── Step 8: Best selection ───────────────────────────────────────────────
    # valid_nl: 검증 통과 (syntax + equiv) & cost > 0
    valid_nl = [c for c in nl_candidates if c['syntax_ok'] and c['equiv_ok'] and c['raw_cost'] > 0]

    # C_rule 결과도 후보에 포함
    c_rule_cost = c_rule_res.get('output_cost', -1)
    all_valid: t.List[t.Dict] = []
    if c_rule_cost > 0:
        all_valid.append({'family': 'C_rule', 'cost': c_rule_cost, 'result': c_rule_res})
    for c in valid_nl:
        _res = {
            'used_rules': [c['family']],
            'output_sql': c['sql'],
            'output_cost': c['raw_cost'],
            'time': 0,
            '_equiv_ok': c.get('equiv_ok'),
            '_equiv_source': c.get('equiv_source', 'unknown'),
            '_equiv_reason': c.get('equiv_reason', ''),
        }
        all_valid.append({'family': c['family'], 'cost': c['raw_cost'], 'result': _res})

    if not all_valid:
        logging.warning(f'{PORTFOLIO_LOG_PREFIX} 유효한 후보 없음 → 원본 SQL 유지')
        _fallback_res = {
            'used_rules': ['PORTFOLIO_NO_VALID_CANDIDATE'],
            'output_sql': query,
            'output_cost': input_cost if input_cost > 0 else -1,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(_fallback_res, "LINE2")}')
        return

    best = min(all_valid, key=lambda x: x['cost'])
    logging.info(
        f'{PORTFOLIO_LOG_PREFIX} Best candidate: family={best["family"]}, '
        f'cost={best["cost"]:.2f} '
        f'(from {len(all_valid)} valid candidates)'
    )

    # ── Step 9: Cost Rejection Gate ──────────────────────────────────────────
    _apply_cost_rejection_gate(best['result'], input_cost, cost_rejection_threshold, query)

    logging.info(f'{PORTFOLIO_LOG_PREFIX} portfolio_rag_rewrite 종료')


# ── Cost Rejection Gate helper ────────────────────────────────────────────────

def _enabled_at_s0(rewrites: dict) -> t.Set[str]:
    """Enabled(S0) — Calcite 가 원본 쿼리에 매칭시킨 규칙 전체.

    B1 의 stale-edge 필터가 쓰는 기준이다. LLM 이 걸러낸 뒤(selected_rules_lst)가
    아니라 matcher 원본 출력을 써야 한다: "이 쿼리에서 그 규칙을 지금 쓸 수 있는가"가
    기준이고, LLM 의 선택 여부와는 무관하기 때문이다.
    """
    try:
        return {str(r['name']) for r in (rewrites or {}).get('calcite', [])}
    except Exception:          # noqa: BLE001 - 진단용 보조값이므로 실패해도 진행
        return set()

def _tag_line(result: dict, seq: str) -> dict:
    """
    'Rewrite Execution Results:' 로그 한 줄에 '_seq' 마커를 붙인 사본을 반환한다.
    execute_rewrite()(db_utils.py)가 매 호출마다 내부적으로 남기는 동일 문구의
    로그와, 이 파일이 analyze.py 호환을 위해 명시적으로 남기는 LINE1(첫 C_rule
    결과)/LINE2(최종 결정)를 구분하기 위한 것 — analyze.py는 이 마커가 있으면
    마커 없는(=execute_rewrite() 내부 로깅) 줄을 전부 무시한다. 원본 dict는
    다른 곳에서 계속 쓰이므로 사본에만 태그를 붙인다.
    """
    return {**result, '_seq': seq}


CORRELATE_OVERRIDE_RULE_NAME = 'FILTER_SUB_QUERY_TO_CORRELATE'


def _matches_correlate_rule(used_rules) -> bool:
    """
    used_rules(str 또는 list)에 'FILTER_SUB_QUERY_TO_CORRELATE' 규칙이
    포함되는지 검사한다.

    이 규칙이 적용된 rewrite는 cost 추정(Postgres EXPLAIN)이 DSB류의 편향된
    데이터 분포에서 outer row count를 체계적으로 과대추정해 실제로는 더 빠른데도
    fallback 처리되는 경우가 많다는 게 로그 재분석으로 확인됨 — correlate_override
    실험은 이 경우 cost 추정과 무관하게 무조건 채택한다.

    처음엔 '*_TO_CORRELATE'(JOIN_TO_CORRELATE 포함) 전체를 대상으로 했으나,
    실측 결과 JOIN_TO_CORRELATE는 FILTER_SUB_QUERY_TO_CORRELATE보다 승률이
    낮고 극단적 역효과(타임아웃 등) 비중도 높아 FILTER_SUB_QUERY_TO_CORRELATE
    하나로 범위를 좁혔다.
    """
    rules = used_rules if isinstance(used_rules, list) else [used_rules]
    return any(str(r) == CORRELATE_OVERRIDE_RULE_NAME for r in rules)


def _select_best_candidate(
    all_valid: t.List[t.Dict],
    correlate_override_enabled: bool = False,
    correlate_real_verify_enabled: bool = False,
    dcrp_policy=None,
    dcrp_context: t.Optional[t.Dict[str, str]] = None,
    input_cost: float = -1.0,
) -> t.Tuple[t.Optional[t.Dict], bool]:
    """
    all_valid(각 원소: {'family', 'cost', 'result'}, C_rule + 각 NL family 후보)에서
    최종 후보 하나를 고른다.

    기본은 cost 최소인 후보. correlate_override_enabled 또는
    correlate_real_verify_enabled가 True면 '*_TO_CORRELATE' 규칙이 포함된
    후보가 하나라도 있는 경우, 전체 cost 최소가 아니더라도 그 correlate 후보들
    중 cost가 가장 낮은 것을 선택한다 — "채택 여부"는 cost 추정과 무관하지만,
    correlate 후보끼리 경쟁할 땐 여전히 cost를 타이브레이커로 쓴다(로그
    재분석에서 확인된 tie-break 기준과 동일). 두 스위치는 "무엇을 선택할지"는
    동일하게 공유하고, "선택 이후 무조건 채택할지 vs 실측 검증할지"만 호출부에서
    갈린다.

    Returns: (best, correlate_matched) — correlate_matched=True면 반환된 best가
    correlate 후보 중에서 뽑힌 것(override/real_verify 어느 쪽이든 호출부가
    correlate_matched으로 이후 행동을 결정).
    """
    best = min(all_valid, key=lambda x: x['cost'])
    if correlate_override_enabled or correlate_real_verify_enabled:
        correlate_candidates = [c for c in all_valid if _matches_correlate_rule(c['result'].get('used_rules'))]
        if correlate_candidates:
            best = min(correlate_candidates, key=lambda x: x['cost'])
            return best, True
    if dcrp_policy is not None:
        from my_rewriter.candidate_selection.interfaces import CandidateFeatures

        context = dcrp_context or {}
        accepted = []
        for index, candidate in enumerate(all_valid):
            used_rules = candidate['result'].get('used_rules')
            rules = tuple(used_rules) if isinstance(used_rules, list) else (
                (str(used_rules),) if used_rules else ()
            )
            features = CandidateFeatures(
                candidate_id=f"{context.get('query_family', 'query')}:{index}",
                workload=context.get('workload', 'calcite'),
                query=context.get('query', ''),
                query_family=context.get('query_family', ''),
                rules=rules,
                original_cost=float(input_cost),
                rewritten_cost=float(candidate['cost']),
            )
            decision = dcrp_policy.decide_detailed(features)
            logging.info(
                f'{GATED_LOG_PREFIX} DCRP candidate: family={candidate["family"]}, '
                f'decision={decision.decision}, reliability={decision.reliability}, '
                f'support={decision.support}, source={decision.prior_source}, '
                f'key={decision.prior_key}, reason={decision.reason}'
            )
            if decision.decision == 'rewrite':
                accepted.append((candidate, decision))
        if not accepted:
            logging.info(f'{GATED_LOG_PREFIX} DCRP: 모든 후보 reject → 원본 유지')
            return None, False
        best, best_decision = max(
            accepted,
            key=lambda pair: (
                pair[1].median_latency_improvement_pct is not None,
                pair[1].median_latency_improvement_pct or 0.0,
                pair[1].reliability or 0.0,
                -pair[0]['cost'],
            ),
        )
        logging.info(
            f'{GATED_LOG_PREFIX} DCRP selected: family={best["family"]}, '
            f'median_history_improvement={best_decision.median_latency_improvement_pct}, '
            f'reliability={best_decision.reliability}'
        )
        return best, False
    return best, False


def _select_non_correlate_fallback(
    all_valid: t.List[t.Dict], dcrp_policy=None,
    dcrp_context: t.Optional[t.Dict[str, str]] = None, input_cost: float = -1.0,
) -> t.Optional[t.Dict]:
    """Return the cheapest still-valid candidate not requiring real-verify."""
    alternatives = [
        candidate for candidate in all_valid
        if not _matches_correlate_rule(candidate['result'].get('used_rules'))
    ]
    if not alternatives:
        return None
    selected, _ = _select_best_candidate(
        alternatives, dcrp_policy=dcrp_policy,
        dcrp_context=dcrp_context, input_cost=input_cost,
    )
    return selected


def _dcrp_accept_single(candidate: t.Dict, dcrp_policy, dcrp_context,
                        input_cost: float) -> bool:
    """Apply the same online DCRP decision to an early-stop candidate."""
    if dcrp_policy is None:
        return True
    selected, _ = _select_best_candidate(
        [candidate], dcrp_policy=dcrp_policy,
        dcrp_context=dcrp_context, input_cost=input_cost,
    )
    return selected is not None


def _dcrp_rule_pruning_repair(
    candidate: t.Dict,
    arranged_sequence: t.Sequence[str],
    dcrp_policy,
    dcrp_context: t.Optional[t.Dict[str, str]],
    input_cost: float,
    query: str,
    schema: str,
    db_args: DBArgs,
    rewrite_rounds: int,
    *,
    max_cost_ratio: float = 0.9,
    min_support: int = 5,
) -> t.Optional[dict]:
    """Repair a Gate1 multi-rule false negative without using latency.

    If DCRP rejects a cost-improving C_rule candidate because its selected
    individual-rule prior is unreliable, remove only that rule from the
    arranged sequence and execute Calcite once more.  Return the new candidate
    only when a supported rule/sequence prior accepts it and the estimated
    cost remains below ``max_cost_ratio`` of the input cost.

    Semantic verification remains in the caller so the repaired SQL passes
    through exactly the same verifier as every other C_rule candidate.
    """
    if dcrp_policy is None or candidate.get('family') != 'C_rule' or input_cost <= 0:
        return None
    result = candidate.get('result') or {}
    used_rules = result.get('used_rules')
    rules = tuple(used_rules) if isinstance(used_rules, list) else (
        (str(used_rules),) if used_rules else ()
    )
    candidate_cost = float(candidate.get('cost', -1.0))
    if len(rules) < 2 or candidate_cost <= 0 or candidate_cost / input_cost >= max_cost_ratio:
        return None

    from my_rewriter.candidate_selection.interfaces import CandidateFeatures

    context = dcrp_context or {}
    initial_features = CandidateFeatures(
        candidate_id=f"{context.get('query_family', 'query')}:dcrp-repair-initial",
        workload=context.get('workload', 'calcite'),
        query=context.get('query', query),
        query_family=context.get('query_family', ''),
        rules=rules,
        original_cost=float(input_cost),
        rewritten_cost=candidate_cost,
    )
    initial = dcrp_policy.decide_detailed(initial_features)
    rejected_rule = initial.prior_key
    if (
        initial.decision != 'original'
        or initial.prior_source != 'rule'
        or not rejected_rule
        or rejected_rule not in arranged_sequence
    ):
        return None

    repaired_sequence = [rule for rule in arranged_sequence if rule != rejected_rule]
    if not repaired_sequence or len(repaired_sequence) == len(arranged_sequence):
        return None
    logging.info(
        f'{GATED_LOG_PREFIX} DCRP rule-pruning repair: '
        f'rejected_rule={rejected_rule}, reliability={initial.reliability}, '
        f'support={initial.support}, cost_ratio={candidate_cost / input_cost:.6f}, '
        f'sequence={repaired_sequence}'
    )

    repaired = execute_rewrite(query, schema, db_args, repaired_sequence, rewrite_rounds)
    repaired_cost = float(repaired.get('output_cost', -1.0))
    repaired_sql = repaired.get('output_sql')
    if repaired_cost <= 0 or repaired_sql in (None, 'None', query):
        logging.info(f'{GATED_LOG_PREFIX} DCRP repair reject: Calcite produced no usable SQL')
        return None
    repaired_ratio = repaired_cost / input_cost
    if repaired_ratio >= max_cost_ratio:
        logging.info(
            f'{GATED_LOG_PREFIX} DCRP repair reject: '
            f'cost_ratio={repaired_ratio:.6f} >= {max_cost_ratio:.6f}'
        )
        return None

    repaired_used = repaired.get('used_rules')
    repaired_rules = tuple(repaired_used) if isinstance(repaired_used, list) else (
        (str(repaired_used),) if repaired_used else ()
    )
    repaired_features = CandidateFeatures(
        candidate_id=f"{context.get('query_family', 'query')}:dcrp-repair",
        workload=context.get('workload', 'calcite'),
        query=context.get('query', query),
        query_family=context.get('query_family', ''),
        rules=repaired_rules,
        original_cost=float(input_cost),
        rewritten_cost=repaired_cost,
    )
    decision = dcrp_policy.decide_detailed(repaired_features)
    accepted = bool(
        decision.decision == 'rewrite'
        and decision.prior_source in {'rule', 'rule_sequence'}
        and decision.support >= min_support
    )
    logging.info(
        f'{GATED_LOG_PREFIX} DCRP repair recheck: decision={decision.decision}, '
        f'source={decision.prior_source}, key={decision.prior_key}, '
        f'reliability={decision.reliability}, support={decision.support}, '
        f'cost_ratio={repaired_ratio:.6f}, accepted={accepted}'
    )
    if not accepted:
        return None

    repaired['_dcrp_rule_pruning_repair'] = {
        'removed_rule': rejected_rule,
        'initial_reliability': initial.reliability,
        'initial_support': initial.support,
        'repaired_prior_source': decision.prior_source,
        'repaired_prior_key': decision.prior_key,
        'repaired_reliability': decision.reliability,
        'repaired_support': decision.support,
        'repaired_cost_ratio': repaired_ratio,
        'selection_uses_observed_latency': False,
    }
    return repaired


def _apply_cost_rejection_gate(
    result: dict,
    input_cost: float,
    threshold: float,
    original_query: str,
    restore_on_regression: bool = True,
    triage_real_verify: bool = False,
    triage_timeout_s: int = 60,
    db_args: t.Optional[DBArgs] = None,
    uresult: t.Optional['UniformityResult'] = None,
    gate1_enabled: bool = False,
    precomputed_effective: t.Optional[t.Tuple[float, float]] = None,
    correlate_override_enabled: bool = False,
    correlate_real_verify_enabled: bool = False,
    correlate_real_verify_timeout_s: int = 300,
) -> None:
    """
    best 결과에 cost rejection gate를 적용해 최종 Rewrite Execution Results를 로깅한다.
    input_cost ≤ 0이면 gate를 적용하지 않는다 (비용 미지).

    correlate_override_enabled=True면, result의 used_rules에 '*_TO_CORRELATE'
    규칙이 하나라도 포함된 경우 cost 추정(및 triage 실측)과 무관하게 무조건
    채택한다 — DSB류 편향 데이터에서 이 규칙군의 cost 추정이 체계적으로
    과대평가되는 것으로 확인되어(로그 재분석), LLM 재호출 없이 이 규칙 하나만
    보고 accept/reject를 판단하는 실험용 스위치.

    restore_on_regression=False면 이 게이트 자체를 끈다 — cost가 원본보다 나빠도
    무조건 채택된 후보를 그대로 내보낸다 (R-Bot류 baseline처럼 시스템 자체에는
    회귀 방지 장치가 없는 상태를 재현하기 위한 실험용 스위치). 이 경우 회귀 판정/
    원복은 evaluation 단계(analyze.py류)에서만 사후에 이뤄진다.

    triage-aware: 이 게이트를 거치는 모든 지점(1차 C_rule early-stop, rearrange
    early-stop, portfolio_enabled=False, NL 탐색 후 최종 accept)에서 공통으로
    동작한다 — 예전엔 Step 10(NL 탐색 후 최종 accept)에서만 실측 triage를 했는데,
    조기종료 경로 3곳은 이 게이트만 거치고 바로 return 해버려서 estimate가
    틀려도 검증할 기회 자체가 없었다. gate1이 이미 계산해 둔 uresult(uniformity
    score, 재계산 없이 재활용)가 활성화된 쿼리에 한해, 여기서 실측
    (actual_time_paired)으로 cost 대신 latency 기준 accept/reject를 판단한다.

    precomputed_effective=(effective_input, effective_output)이 주어지면 호출부가
    이미 실측을 마쳤다는 뜻이므로 그 값을 그대로 쓰고 여기서 다시 측정하지 않는다
    (Step 10의 final_accept_ratio 체크와 이 게이트가 같은 후보를 중복 실측하는
    것을 방지).

    correlate_real_verify_enabled=True면, result의 used_rules에
    'FILTER_SUB_QUERY_TO_CORRELATE'가 포함된 경우 cost 추정을 아예 안 믿고
    actual_time_paired()로 원본/재작성본을 실측(interleave 5회, trimmed mean)해
    그 결과로 accept/reject를 판단한다. correlate_override_enabled(무조건 채택)와
    달리 실측 latency가 threshold를 넘으면 여전히 원본으로 폴백한다 — 이 규칙의
    cost 추정이 구조적으로 못 믿을 수준(Step1: 파생 서브쿼리 무통계, Step3: 파생값
    비교)임을 EXPLAIN ANALYZE로 직접 확인했기 때문에, gate1 uniformity score
    (uresult.activate) 여부와 무관하게 이 규칙 하나만 보고 트리거한다(gate1
    활성화 쿼리와 이 문제 쿼리 집합이 겹치지 않는다는 게 로그 분석으로 확인됨).
    """
    out_cost = result.get('output_cost', -1)
    effective_input_cost, effective_out_cost = input_cost, out_cost
    triage_fired = False

    if correlate_override_enabled and _matches_correlate_rule(result.get('used_rules')):
        logging.info(
            f'{PORTFOLIO_LOG_PREFIX} Correlate Override 발동(cost_rejection_gate): '
            f'used_rules={result.get("used_rules")} → cost 추정과 무관하게 무조건 채택'
        )
        logging.info(f'Rewrite Execution Results: {_tag_line(result, "LINE2")}')
        return

    if precomputed_effective is not None:
        effective_input_cost, effective_out_cost = precomputed_effective
        triage_fired = True
    elif (correlate_real_verify_enabled and _matches_correlate_rule(result.get('used_rules'))
            and db_args is not None and result.get('output_sql') not in (None, 'None')):
        logging.info(
            f'{PORTFOLIO_LOG_PREFIX} Correlate Real-Verify 발동(cost_rejection_gate): '
            f'used_rules={result.get("used_rules")} → 실측 검증'
        )
        try:
            (real_input, _), (real_output, _) = actual_time_paired(
                original_query, result['output_sql'], db_args, correlate_real_verify_timeout_s
            )
            if (real_input not in (float('inf'), None) and real_output not in (float('inf'), None)
                    and real_input > 0):
                logging.info(
                    f'{PORTFOLIO_LOG_PREFIX} Correlate Real-Verify 실측 결과: input={real_input:.2f}ms, '
                    f'output={real_output:.2f}ms (estimate는 input={input_cost:.2f}, '
                    f'output={out_cost:.2f}이었음)'
                )
                effective_input_cost, effective_out_cost = real_input, real_output
                triage_fired = True
            else:
                logging.warning(
                    f'{PORTFOLIO_LOG_PREFIX} Correlate Real-Verify 실측 실패(inf/None) '
                    f'→ 검증 실패로 원본 SQL 폴백'
                )
                fallback = {
                    'used_rules': ['CORRELATE_REAL_VERIFY_FAILED_FALLBACK'],
                    'output_sql': original_query,
                    'output_cost': input_cost,
                    'time': 0,
                }
                logging.info(f'Rewrite Execution Results: {_tag_line(fallback, "LINE2")}')
                return
        except Exception as e:
            logging.warning(
                f'{PORTFOLIO_LOG_PREFIX} Correlate Real-Verify 실측 예외 '
                f'→ 검증 실패로 원본 SQL 폴백: {e}'
            )
            fallback = {
                'used_rules': ['CORRELATE_REAL_VERIFY_FAILED_FALLBACK'],
                'output_sql': original_query,
                'output_cost': input_cost,
                'time': 0,
            }
            logging.info(f'Rewrite Execution Results: {_tag_line(fallback, "LINE2")}')
            return
    elif (triage_real_verify and gate1_enabled and uresult is not None and uresult.activate
            and db_args is not None and result.get('output_sql') not in (None, 'None')):
        logging.info(
            f'{PORTFOLIO_LOG_PREFIX} Triage 발동(cost_rejection_gate): '
            f'uniformity_score={uresult.score:.4f} → 실측 검증'
        )
        try:
            (real_input, _), (real_output, _) = actual_time_paired(
                original_query, result['output_sql'], db_args, triage_timeout_s
            )
            if (real_input not in (float('inf'), None) and real_output not in (float('inf'), None)
                    and real_input > 0):
                logging.info(
                    f'{PORTFOLIO_LOG_PREFIX} Triage 실측 결과: input={real_input:.2f}ms, '
                    f'output={real_output:.2f}ms (estimate는 input={input_cost:.2f}, '
                    f'output={out_cost:.2f}이었음)'
                )
                effective_input_cost, effective_out_cost = real_input, real_output
                triage_fired = True
            else:
                logging.warning(
                    f'{PORTFOLIO_LOG_PREFIX} Triage 실측 실패(timeout/inf) → estimate 기준으로 폴백'
                )
        except Exception as e:
            logging.warning(f'{PORTFOLIO_LOG_PREFIX} Triage 실측 예외 (무시, estimate로 폴백): {e}')

    basis = '실측(latency)' if triage_fired else 'estimate(cost)'
    if (restore_on_regression and effective_input_cost > 0 and effective_out_cost > 0
            and effective_out_cost > effective_input_cost * threshold):
        pct = (effective_out_cost / effective_input_cost - 1) * 100
        logging.warning(
            f'{PORTFOLIO_LOG_PREFIX} Cost Rejection Gate 발동 ({basis} 기준): '
            f'output={effective_out_cost:.2f} > input({effective_input_cost:.2f}) × {threshold} '
            f'(+{pct:.1f}%) → 원본 SQL로 롤백'
        )
        fallback = {
            'used_rules': ['COST_REJECTION_FALLBACK'],
            'output_sql': original_query,
            'output_cost': input_cost,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(fallback, "LINE2")}')
    else:
        logging.info(f'Rewrite Execution Results: {_tag_line(result, "LINE2")}')


def _measure_correlate_candidate(
    result: dict,
    original_query: str,
    db_args: DBArgs,
    timeout_s: int,
    acceptance_ratio: float,
) -> t.Tuple[bool, t.Optional[t.Tuple[float, float]]]:
    """Measure a verified correlate C_rule before any NL escalation.

    Returns ``(accepted, measurements)``.  ``measurements`` is ``None`` when
    execution failed or timed out.  A finite measurement that does not meet
    ``acceptance_ratio`` is a normal rejection, not an operational failure.
    """
    logging.info(
        f'{GATED_LOG_PREFIX} Correlate C_rule 즉시 Real-Verify 발동: '
        f'used_rules={result.get("used_rules")} → NL 진입 전 실측 검증'
    )
    try:
        (real_input, _), (real_output, _) = actual_time_paired(
            original_query, result['output_sql'], db_args, timeout_s
        )
    except Exception as exc:
        logging.warning(
            f'{GATED_LOG_PREFIX} Correlate C_rule 즉시 Real-Verify 예외: {exc} '
            f'→ 해당 C_rule 제외 후 다음 후보 탐색'
        )
        return False, None

    if (real_input in (float('inf'), None) or real_output in (float('inf'), None)
            or real_input <= 0 or real_output <= 0):
        logging.warning(
            f'{GATED_LOG_PREFIX} Correlate C_rule 즉시 Real-Verify 실패'
            f'(input={real_input}, output={real_output}) '
            f'→ 해당 C_rule 제외 후 다음 후보 탐색'
        )
        return False, None

    # Correlate is adopted only on a measured strict improvement.  Equality is
    # not evidence of a speedup and therefore continues to the next candidate.
    accepted = real_output < real_input * acceptance_ratio
    logging.info(
        f'{GATED_LOG_PREFIX} Correlate C_rule 즉시 Real-Verify 결과: '
        f'input={real_input:.2f}ms, output={real_output:.2f}ms, '
        f'acceptance_ratio={acceptance_ratio}, accepted={accepted}'
    )
    return accepted, (real_input, real_output)


# ═══════════════════════════════════════════════════════════════════════════════
#  Gated Rewrite Escalation — helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_calcite_sufficient(new_cost: float, base_cost: float, ratio: float) -> bool:
    """True if new_cost ≤ base_cost × ratio (improvement ≥ 1-ratio)."""
    return base_cost > 0 and new_cost > 0 and new_cost <= base_cost * ratio


def _cost_improvement_pct(new_cost: float, base_cost: float) -> float:
    """Return improvement percentage (positive = better). 0.0 if costs invalid."""
    if base_cost <= 0 or new_cost <= 0:
        return 0.0
    return (base_cost - new_cost) / base_cost * 100.0


def _equivalence_source(reason: str) -> str:
    if reason.startswith('solver_'):
        return 'solver'
    if '_llm_' in reason:
        return 'llm'
    return 'unknown'


def _verify_c_rule_result(
    label: str,
    c_rule_res: t.Dict,
    query: str,
    input_cost: float,
    db_args: DBArgs,
    schema: str,
    verify_enabled: bool,
    equiv_use_solver: bool,
    equiv_use_llm_judge: bool,
    equiv_solver_timeout: int,
    equiv_timeout_s: int,
    original_result_cache: t.Optional[dict] = None,
    equivalence_cache: t.Optional[dict] = None,
) -> t.Dict:
    """
    Calcite 규칙 카탈로그는 이론상 각 규칙이 동치 보존이지만, 규칙 조합이나
    재작성 엔진 자체의 버그로 실제로는 비동치인 rewrite가 나오는 경우가 있어
    C_rule 결과에도 검증을 건다.

    검증은 SQLSolver(formal, sound) 단독이다 — LLM judge는 호출하지 않는다.
    DSB 재검증 로그 실측 결과(results/dsb_sf10/deepseek/03_2026-08-20_.../
    evaluation/{strict_reverify,c_rule_retroactive_verify}.jsonl), 실제 실행
    결과 비교(strict_reverify)는 C_rule 채택 37건 전부를 동치로 확인했는데,
    같은 37건을 LLM judge 단독으로 사후 재검증하면 37건 전부 'unknown_llm_reject'로
    오탈락했다 — 오늘 gate1_only/gate2_only 실행 로그에서도 C_rule 1차 후보의
    30~40%가 같은 'unknown_llm_reject' 사유로 거부되는 동일 패턴이 재현됨. LLM
    judge가 문법적으로 많이 달라진(별칭 변경, JOIN 방식 변경 등) 재작성을
    실제로는 맞아도 과도하게 reject하는 경향이 있다는 뜻이므로, C_rule
    경로에서는 이 신뢰할 수 없는 veto를 빼고 SQLSolver의 sound한 판정만 쓴다:
    EQ면 채택, NEQ(SQLSolver가 확정적으로 비동치라고 증명한 경우)면 원본
    폴백, UNKNOWN(증명 실패/timeout/quoted identifier)이면 여기서 막지 않고
    통과시킨다 — 실제 동치 여부의 최종 판정은 evaluation 단계
    (run_ablation_latency.py/analyze.py → evaluation_equivalence.evaluate_query_pair,
    실제 DB 실행 결과 비교)에서 한 번 더 걸러진다. equiv_use_llm_judge 인자는
    _verify_nl_candidate와 시그니처를 맞추기 위해 남아 있을 뿐 여기서는 쓰이지
    않는다.

    output_sql이 없거나 원본과 동일하면(= 실질적으로 rewrite가 없었던 경우)
    검증할 것이 없으므로 그대로 통과시킨다.
    """
    output_sql = c_rule_res.get('output_sql')
    if output_sql in (None, 'None') or output_sql == query:
        c_rule_res.update({
            '_equiv_ok': True, '_equiv_source': 'no_op',
            '_equiv_reason': 'same_as_original',
        })
        return c_rule_res
    if not verify_enabled:
        c_rule_res.update({
            '_equiv_ok': None, '_equiv_source': 'disabled',
            '_equiv_reason': 'verification_disabled',
        })
        return c_rule_res

    cache_key = _candidate_key(query, output_sql)
    cached = equivalence_cache.get(cache_key) if equivalence_cache is not None else None
    if cached is not None:
        eq_ok, eq_reason = cached
        logging.info(
            f'{GATED_LOG_PREFIX} {label}: canonical duplicate '
            f'(hash={cache_key[:16]}) → equivalence result reuse'
        )
    else:
        verdict = sqlsolver_verdict(query, output_sql, schema, timeout=equiv_solver_timeout) \
            if equiv_use_solver else UNKNOWN
        if verdict == EQ:
            eq_ok, eq_reason = True, 'solver_eq'
        elif verdict == NEQ:
            eq_ok, eq_reason = False, 'solver_neq'
        else:
            eq_ok, eq_reason = True, 'solver_unknown_deferred'
        if equivalence_cache is not None:
            equivalence_cache[cache_key] = (eq_ok, eq_reason)
    if eq_ok:
        c_rule_res.update({
            '_equiv_ok': True,
            '_equiv_source': _equivalence_source(eq_reason),
            '_equiv_reason': eq_reason,
        })
        logging.info(f'{GATED_LOG_PREFIX} {label}: 동치 검증 통과 (equiv={eq_reason})')
        return c_rule_res

    logging.info(f'{GATED_LOG_PREFIX} {label}: 동치 검증 실패 — {eq_reason} → 원본 폴백')
    return {
        'used_rules': ['C_RULE_EQUIV_CHECK_FAILED_FALLBACK'],
        'output_sql': query,
        'output_cost': input_cost,
        'time': c_rule_res.get('time', 0),
        '_equiv_ok': False,
        '_equiv_source': _equivalence_source(eq_reason),
        '_equiv_reason': eq_reason,
    }


def _verify_nl_candidate(
    family: str,
    sql: t.Optional[str],
    query: str,
    db_args: DBArgs,
    verify_enabled: bool,
    equiv_timeout_s: int,
    schema: str = '',
    equiv_use_solver: bool = True,
    equiv_use_llm_judge: bool = True,
    equiv_solver_timeout: int = 10,
    original_result_cache: t.Optional[dict] = None,
    equivalence_cache: t.Optional[dict] = None,
    max_acceptable_cost: t.Optional[float] = None,
) -> t.Dict:
    """
    Run struct / alias / syntax / equivalence checks for one NL rewrite candidate.
    Returns a candidate entry dict (same schema as portfolio_rag_rewrite's nl_candidates).

    Equivalence uses the rewrite-time hybrid verifier: SQLSolver first, then a
    single LLM judge on every non-EQ verdict. The QUITE-style execution-result
    comparison runs later, once for final LINE2, in run_ablation_latency.py.
    Set equiv_use_solver / equiv_use_llm_judge=False to disable a layer.
    """
    entry: t.Dict = {
        'family': family, 'sql': sql, 'raw_cost': -1,
        'syntax_ok': False, 'equiv_ok': False, 'fail_reason': '',
        'equiv_source': 'not_called', 'equiv_reason': '',
    }

    if sql is None:
        entry['fail_reason'] = 'llm_gen_failed'
        logging.info(f'{GATED_LOG_PREFIX} {family}: LLM 생성 실패')
        return entry

    if sql.strip().rstrip(';') == query.strip().rstrip(';'):
        entry['fail_reason'] = 'identical_to_original'
        logging.info(f'{GATED_LOG_PREFIX} {family}: 원본과 동일 → skip')
        return entry

    # Structural substring checks remain warning-only because they are noisy.
    # Rationale: a genuinely dropped table or an undeclared alias produces a DB
    # error that the syntax (EXPLAIN) check + hybrid equivalence gate already
    # catch soundly. These cheap heuristics (substring / regex) instead produced
    # false positives — e.g. schema-qualified names, subquery-scoped aliases —
    # that killed VALID NL rewrites before they ever reached those sound gates.
    # We now only log their findings and let syntax/equivalence decide.
    struct_ok, struct_reason = _structural_check(query, sql)
    if not struct_ok:
        logging.info(f'{GATED_LOG_PREFIX} {family}: [warn] 구조 사전검사 지적(비차단) — {struct_reason}')

    alias_ok, alias_reason = _alias_consistency_check(sql)
    if not alias_ok:
        # User-added extension over QUITE: preserve the concrete alias failure as
        # corrective feedback and regenerate instead of silently discarding it.
        entry['fail_reason'] = f'alias_fail: {alias_reason}'
        logging.info(f'{GATED_LOG_PREFIX} {family}: alias 검증 실패 — {alias_reason}')
        return entry

    if verify_enabled:
        syn_ok, syn_reason, syntax_cost = _verify_syntax_with_cost(sql, db_args)
        entry['syntax_ok'] = syn_ok
        if not syn_ok:
            entry['fail_reason'] = f'syntax_fail: {syn_reason}'
            logging.info(f'{GATED_LOG_PREFIX} {family}: 문법 검증 실패 — {syn_reason}')
            return entry
    else:
        entry['syntax_ok'] = True
        syntax_cost = -1

    if verify_enabled:
        # _verify_syntax_with_cost() already ran the exact EXPLAIN needed here.
        entry['raw_cost'] = syntax_cost
    else:
        db = None
        try:
            db = Database(db_args)
            entry['raw_cost'] = db.cost_estimation(sql)
        except Exception:
            entry['raw_cost'] = -1
        finally:
            if db is not None and hasattr(db, 'close'):
                db.close()

    if verify_enabled:
        if (max_acceptable_cost is not None and entry['raw_cost'] > 0
                and entry['raw_cost'] > max_acceptable_cost):
            entry['fail_reason'] = (
                f'cost_ineligible: {entry["raw_cost"]:.2f} > {max_acceptable_cost:.2f}'
            )
            logging.info(
                f'{GATED_LOG_PREFIX} {family}: cost 사전필터 탈락 '
                f'({entry["raw_cost"]:.2f} > {max_acceptable_cost:.2f}) '
                f'→ equivalence verification skip'
            )
            return entry

        cache_key = _candidate_key(query, sql)
        cached = equivalence_cache.get(cache_key) if equivalence_cache is not None else None
        if cached is not None:
            eq_ok, eq_reason = cached
            logging.info(
                f'{GATED_LOG_PREFIX} {family}: canonical duplicate '
                f'(hash={cache_key[:16]}) → equivalence result reuse'
            )
        else:
            eq_ok, eq_reason = verify_equivalence_hybrid(
                query, sql, db_args, schema,
                use_solver=equiv_use_solver,
                use_llm_judge=equiv_use_llm_judge,
                solver_timeout=equiv_solver_timeout,
                equiv_timeout_s=equiv_timeout_s,
            )
            if equivalence_cache is not None:
                equivalence_cache[cache_key] = (eq_ok, eq_reason)
        entry['equiv_ok'] = eq_ok
        entry['equiv_source'] = _equivalence_source(eq_reason)
        entry['equiv_reason'] = eq_reason
        if not eq_ok:
            entry['fail_reason'] = f'equiv_fail: {eq_reason}'
            logging.info(f'{GATED_LOG_PREFIX} {family}: 동치 검증 실패 — {eq_reason}')
        else:
            cost_str = f'{entry["raw_cost"]:.2f}' if entry['raw_cost'] > 0 else '-1'
            logging.info(
                f'{GATED_LOG_PREFIX} {family}: 검증 통과 '
                f'(cost={cost_str}, equiv={eq_reason})'
            )
    else:
        entry['equiv_ok'] = True
        entry['equiv_source'] = 'disabled'
        entry['equiv_reason'] = 'verification_disabled'
        cost_str = f'{entry["raw_cost"]:.2f}' if entry['raw_cost'] > 0 else '-1'
        logging.info(f'{GATED_LOG_PREFIX} {family}: verify=False, cost={cost_str}')

    return entry


def _generate_and_verify_with_retry(
    family: str,
    gen_fn: t.Callable[[], t.Optional[str]],
    query: str,
    db_args: DBArgs,
    verify_enabled: bool,
    equiv_timeout_s: int,
    schema: str,
    equiv_use_solver: bool,
    equiv_use_llm_judge: bool,
    equiv_solver_timeout: int,
    max_equiv_retries: int,
    original_result_cache: t.Optional[dict] = None,
    verification_schema: t.Optional[str] = None,
    equivalence_cache: t.Optional[dict] = None,
    max_acceptable_cost: t.Optional[float] = None,
) -> t.Tuple[t.Dict, int]:
    """
    Generate one candidate and verify it (struct / alias / syntax / equivalence).
    If — and only if — it is rejected by the equivalence check (SQLSolver + LLM
    judge, see equivalence_hybrid.py) OR by the alias consistency check
    (_alias_consistency_check in join_reorder.py), retry up to max_equiv_retries
    times with corrective feedback (_gen_corrective_retry), QUITE §5.2-style,
    instead of discarding the candidate after one attempt.

    alias_fail is retried because _alias_consistency_check's rejection reason
    already names the exact undeclared alias(es) (e.g. "선언되지 않은 테이블
    별칭 참조: ['t1']") — concrete, actionable feedback the LLM can act on
    directly, unlike a generic reminder. struct_fail/syntax_fail are NOT
    retried here — they are unrelated to a specific, nameable defect and a
    blind LLM re-sample is unlikely to fix them.

    Returns (final candidate entry, number of LLM generation calls made).
    """
    sql = gen_fn()
    solver_schema = verification_schema if verification_schema is not None else schema
    calls = 1
    entry = _verify_nl_candidate(
        family, sql, query, db_args, verify_enabled, equiv_timeout_s,
        schema=solver_schema, equiv_use_solver=equiv_use_solver,
        equiv_use_llm_judge=equiv_use_llm_judge, equiv_solver_timeout=equiv_solver_timeout,
        original_result_cache=original_result_cache,
        equivalence_cache=equivalence_cache,
        max_acceptable_cost=max_acceptable_cost,
    )

    retry = 0
    # User-added extension retained intentionally: unlike QUITE's generic
    # correction loop, feed the concrete alias/equivalence failure reason into
    # the corrective rewrite prompt.
    def _retryable(reason: str) -> bool:
        if reason.startswith('alias_fail'):
            return True
        if not reason.startswith('equiv_fail'):
            return False
        lowered = reason.lower()
        non_correctable = (
            'original execution failed', 'equiv_timeout', 'equiv_error',
            'connection', 'could not connect', 'server closed the connection',
        )
        return not any(marker in lowered for marker in non_correctable)

    while _retryable(entry['fail_reason']) and retry < max_equiv_retries:
        retry += 1
        logging.info(
            f'{GATED_LOG_PREFIX} {family}: 검증 실패 → feedback retry {retry}/{max_equiv_retries} '
            f'({entry["fail_reason"]})'
        )
        sql = _gen_corrective_retry(family, query, schema, entry['sql'], entry['fail_reason'])
        calls += 1
        entry = _verify_nl_candidate(
            family, sql, query, db_args, verify_enabled, equiv_timeout_s,
            schema=solver_schema, equiv_use_solver=equiv_use_solver,
            equiv_use_llm_judge=equiv_use_llm_judge, equiv_solver_timeout=equiv_solver_timeout,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
            max_acceptable_cost=max_acceptable_cost,
        )

    if retry > 0:
        logging.info(
            f'{GATED_LOG_PREFIX} {family}: feedback retry 종료 '
            f'(총 {retry}회 재시도, 최종 fail_reason={entry["fail_reason"] or "none"}, '
            f'equiv_ok={entry["equiv_ok"]})'
        )
    return entry, calls


# ═══════════════════════════════════════════════════════════════════════════════
#  Gated Rewrite Escalation — main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def gated_portfolio_rag_rewrite(
    retriever_res: t.List[NodeWithScore],
    rewrites: t.List[t.Dict],
    query: str,
    schema: str,
    db_args: DBArgs,
    model_args: t.Dict[str, str],
    pool: t.Optional[dict] = None,
    tau: float = 0.2,
    alpha: float = 0.5,
    CASE_BATCH: int = 5,
    RULE_BATCH: int = 10,
    REWRITE_ROUNDS: int = 1,
    gate1_enabled: bool = True,
    gate2_enabled: bool = True,
    stat_prior_enabled: bool = False,
    stat_prior_hard_filter: bool = False,
    portfolio_enabled: bool = True,
    verify_enabled: bool = True,
    cost_rejection_threshold: float = 1.05,
    input_cost: float = -1.0,
    equiv_timeout_s: int = 300,
    # ── Gated escalation parameters ─────────────────────────────────────────
    calcite_success_ratio: float = 0.97,
    final_accept_ratio: float = 1.0,   # accept a rewrite iff cost does not worsen (≤ input_cost); was 0.99 (required ≥1% gain)
    max_nl_families: int = 2,
    broad_fallback_enabled: bool = False,
    # ── Hybrid equivalence verification (SQLSolver + LLM judge) ──────────────
    equiv_use_solver: bool = True,
    equiv_use_llm_judge: bool = True,
    equiv_solver_timeout: int = 10,
    # ── Equivalence-failure corrective retry (QUITE §5.2 style) ──────────────
    max_equiv_retries: int = 1,
    # ── 실험용: 회귀 방지 게이트 자체를 끄는 스위치 (기본 True = 기존 동작 그대로) ──
    restore_on_regression: bool = True,
    # ── 실험용: distribution evidence 기반 cost-estimate 신뢰도 triage(gate1 활성화 시, 실제 latency로 대소관계 측정) ─────────
    triage_real_verify: bool = False,
    triage_timeout_s: int = 60,
    # ── 실험용: '*_TO_CORRELATE' 규칙 포함시 cost 추정과 무관하게 무조건 채택 ──
    correlate_override_enabled: bool = False,
    # ── 실험용: FILTER_SUB_QUERY_TO_CORRELATE 포함시 cost 추정 대신 실측으로 판단 ──
    correlate_real_verify_enabled: bool = True,
    correlate_real_verify_timeout_s: int = 300,
    # Decision-Conditioned Reliability Prior. None preserves legacy behavior.
    dcrp_policy=None,
    dcrp_workload: str = 'calcite',
    dcrp_query_family: str = '',
    dcrp_rule_pruning_repair_enabled: bool = True,
    dcrp_repair_max_cost_ratio: float = 0.9,
    dcrp_repair_min_support: int = 5,
) -> None:
    """
    Gated Rewrite Escalation 파이프라인.

    기존 portfolio_rag_rewrite()와 달리:
      - C_rule이 calcite_success_ratio 이내로 개선하면 NL을 건너뛴다.
      - rearrange는 첫 C_rule이 충분히 개선 못했을 때만 실행한다.
      - NL family는 detect_rewrite_opportunities()가 활성화한 것만, confidence 순으로 실행한다.
      - 최대 max_nl_families 개 family만 호출한다.
      - 한 family가 final_accept_ratio를 만족하면 즉시 종료한다.
      - C_nl은 broad_fallback_enabled=True일 때만 마지막에 한 번 시도한다.
      - NL 후보가 동치 검증(SQLSolver + LLM judge)에서 실패하면, 실패 이유를
        LLM에 피드백으로 넣어 최대 max_equiv_retries회 재시도한다 (QUITE §5.2
        "Equivalence Error Correction" 이식 — _generate_and_verify_with_retry).
        struct/alias/syntax 실패는 재시도 대상이 아니다.

    기존 portfolio_rag_rewrite() 함수는 그대로 유지된다 (성능 비교용).

    analyze.py 호환:
      - 'Rewrite Execution Results:' 로그 라인을 정확히 2개 출력한다.
        LINE 1: 첫 C_rule 결과 (early exit 여부와 무관하게 항상)
        LINE 2: _apply_cost_rejection_gate() 를 통한 최종 결과

    로그 prefix: [GATED-PORTFOLIO]
    """
    model = MyModel(model_args)
    dcrp_context = {
        'workload': dcrp_workload,
        'query': query,
        'query_family': dcrp_query_family,
    }
    # Per-query caches only: never leak a verdict across prompt/model/config
    # changes in separate experiment runs.
    original_result_cache: dict = {}  # retained for legacy helper compatibility
    equivalence_cache: dict = {}
    candidate_cache: t.Dict[str, t.Dict] = {}

    def _try_verified_dcrp_repair(
        candidate: t.Dict, sequence: t.Sequence[str], label: str,
    ) -> t.Optional[dict]:
        """Run the deterministic repair and then the normal semantic verifier."""
        if (
            not dcrp_rule_pruning_repair_enabled
            or not gate1_enabled
            or uresult is None
            or not uresult.activate
        ):
            return None
        repaired = _dcrp_rule_pruning_repair(
            candidate, sequence, dcrp_policy, dcrp_context,
            input_cost, query, schema, db_args, REWRITE_ROUNDS,
            max_cost_ratio=dcrp_repair_max_cost_ratio,
            min_support=dcrp_repair_min_support,
        )
        if repaired is None:
            return None
        verified = _verify_c_rule_result(
            label, repaired, query, input_cost, db_args, schema,
            verify_enabled, equiv_use_solver, equiv_use_llm_judge,
            equiv_solver_timeout, equiv_timeout_s,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
        )
        if verified.get('output_sql') in (None, 'None', query):
            logging.info(f'{GATED_LOG_PREFIX} DCRP repair semantic verification failed')
            return None
        logging.info(
            f'{GATED_LOG_PREFIX} DCRP repair accepted after semantic verification: '
            f'used_rules={verified.get("used_rules")}, cost={verified.get("output_cost")}'
        )
        return verified
    prompt_schema = _schema_for_referenced_tables(query, schema)
    logging.info(
        f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 시작: '
        f'gate1={gate1_enabled}, gate2={gate2_enabled}, '
        f'stat_prior={stat_prior_enabled}, portfolio={portfolio_enabled}, '
        f'verify={verify_enabled}, calcite_success_ratio={calcite_success_ratio}, '
        f'final_accept_ratio={final_accept_ratio}, max_nl_families={max_nl_families}, '
        f'broad_fallback_enabled={broad_fallback_enabled}'
    )

    # ── Step 1: Gate1 — Uniformity Score + Distribution Guidance (for C_rule) ─
    # Distribution Guidance는 C_rule arrange에도 쓰이므로 gate1 활성 시 eager 생성.
    # NL family들은 아래에서 이 guidance_str을 재사용한다 (추가 LLM 호출 없음).
    guidance_activated = False
    uresult = None
    closure_added: list = []
    rule_stat_priors_str = ''
    rule_priors: list = []
    distribution_evidence: dict = {}

    if gate1_enabled:
        uresult = compute_uniformity_score(query, schema, db_args, tau=tau, alpha=alpha)
        logging.info(
            f'{GATED_LOG_PREFIX} Uniformity Score: '
            f'score={uresult.score:.4f}, activate={uresult.activate}'
        )
        if uresult.activate:
            distribution_evidence = build_distribution_evidence(uresult, pool=pool)
            # Generated later by the integrated guidance/selection/order planner.
            guidance = dict(EMPTY_GUIDANCE)
            guidance_activated = True
        else:
            guidance = dict(EMPTY_GUIDANCE)
    else:
        guidance = dict(EMPTY_GUIDANCE)
        logging.info(f'{GATED_LOG_PREFIX} Gate1 비활성화')

    distribution_guidance_str = ''

    # ── Step 2: Recipe 생성 (RAG → rule selection) ───────────────────────────
    nl_suggestions = [obj['rewrite'] for obj in rewrites['nl']]
    normal_rules = [r for r in rewrites['calcite'] if r['type'] == 'normal']
    explore_rules = [r for r in rewrites['calcite'] if r['type'] == 'explore']
    normal_suggestions = [obj['rewrite'] for obj in normal_rules]
    strategies = normal_suggestions + nl_suggestions

    tasks = [
        model.gen_consolidated_strategies(query, retriever_res, strategies),
        model.select_rules_from_cases(
            retriever_res, normal_rules=normal_rules, explore_rules=explore_rules
        ),
    ]
    task_results = run_async_tasks(tasks)
    summarized_strategies = task_results[0]
    selected_rules: t.List[t.List[t.Dict[str, str]]] = task_results[1]

    suggestions_str = '\n\n'.join(
        [f'### Suggestion {i+1}:\n"""{s}"""' for i, s in enumerate(summarized_strategies)]
    )
    logging.info('Intermediate Results: ' + str(
        {'suggestions_str': suggestions_str, 'selected_rules': selected_rules}
    ))

    selected_rules_lst = list(itertools.chain.from_iterable(selected_rules))

    _n_tables = _count_from_tables(query)
    if _n_tables >= _HIGH_TABLE_THRESHOLD:
        before_n = len(selected_rules_lst)
        selected_rules_lst = [r for r in selected_rules_lst
                               if r['name'] not in _HIGH_TABLE_BLACKLIST]
        removed = before_n - len(selected_rules_lst)
        if removed:
            logging.info(
                f'{GATED_LOG_PREFIX} High-table blacklist: {removed}개 제거 '
                f'(n_tables={_n_tables})'
            )

    # Give the one-pass planner the same RDG ordering evidence that the former
    # separate arrangement call received. Build it over the candidate pool;
    # the selected induced subgraph is recomputed below after closure.
    rdg_graph = None
    planner_dependency_constraints = ''
    if gate2_enabled and selected_rules_lst:
        rdg_graph = load_rdg()
        if rdg_gate(selected_rules_lst):
            _sk0 = build_skeleton_scc(
                [r['name'] for r in selected_rules_lst], rdg_graph,
                enabled_at_s0=_enabled_at_s0(rewrites),
                opportunities=load_opportunities())
            planner_dependency_constraints = render_skeleton_for_prompt(
                _sk0['layers'], _sk0['edges'], _sk0['unresolved'], _sk0['cyclic_groups'])

    guidance, relevant_rules, arranged_rule_seq = plan_rules_with_guidance_dist(
        model, query, suggestions_str, selected_rules_lst,
        distribution_evidence=distribution_evidence,
        rule_dependency_constraints=planner_dependency_constraints,
    )
    distribution_guidance_str = render_guidance_for_prompt(guidance)
    logging.info(
        f'{GATED_LOG_PREFIX} Integrated planner: '
        f'guidance_activated={guidance_activated}, '
        f'hints={len(guidance.get("rewrite_hints", []))}, '
        f'selected={len(relevant_rules)}/{len(selected_rules_lst)}'
    )

    # ── Step 3: Gate2 — RDG Closure + Skeleton ───────────────────────────────
    # full_catalog(=이 쿼리에 구조적으로 매칭된 적 없는 규칙을 RDG 통계만으로 추정
    # 편입)는 여기서 쓰지 않는다 — full_rule_descriptions=None이므로 closure_expand는
    # matched_pool(원래 Calcite가 이 쿼리에 매칭시켰던 규칙 중 LLM이 걸렀던 것) 소스만
    # 편입한다. "1차 rewrite로 구조가 바뀐 뒤에야 새로 매칭되는 규칙"을 잡는 역할은
    # full_catalog 통계 추정 대신, 아래 Step 5a(1차 execute_rewrite 이후)에서
    # match_all_rules()로 변환된 쿼리를 실제 재매칭해 검증하는 방식으로 대체한다
    # (근거: 2026-07-02 analysis_report_baseline_vs_bothgates.md — full_catalog가
    # 검증 없이 주입한 JOIN_ADD_REDUNDANT_SEMI_JOIN 등이 HepPlanner에서는 성공
    # 적용되지만 downstream SQL 생성에서 깨지는 사례 다수 확인).
    rdg_edges: list = []
    rule_dependency_constraints_str = ''
    if gate2_enabled:
        rdg_graph = rdg_graph if rdg_graph is not None else load_rdg()
        relevant_rules, closure_added = closure_expand(
            relevant_rules, selected_rules_lst, rdg_graph, full_rule_descriptions=None,
            enabled_at_s0=_enabled_at_s0(rewrites),
        )
        if closure_added:
            logging.info(
                f'{GATED_LOG_PREFIX} RDG Closure: +{len(closure_added)}개 → '
                f'{[r["name"] for r in relevant_rules]}'
            )
        # Preserve the planner's order and append only rules introduced by
        # deterministic RDG closure. Rearrangement remains available after the
        # first Calcite execution if this initial order underperforms.
        selected_name_set = {r['name'] for r in relevant_rules}
        arranged_rule_seq = [r for r in arranged_rule_seq if r in selected_name_set]
        arranged_rule_seq.extend(
            r['name'] for r in relevant_rules if r['name'] not in arranged_rule_seq
        )
        rdg_active = rdg_gate(relevant_rules)
        if rdg_active:
            _sk = build_skeleton_scc(
                [r['name'] for r in relevant_rules], rdg_graph,
                enabled_at_s0=_enabled_at_s0(rewrites),
                opportunities=load_opportunities(),
            )
            layers, rdg_edges, unresolved = _sk['layers'], _sk['edges'], _sk['unresolved']
            rule_dependency_constraints_str = render_skeleton_for_prompt(
                layers, rdg_edges, unresolved, _sk['cyclic_groups']
            )
            if os.environ.get('RDG_FINAL_SCC_ORDER') == '1':
                arranged_rule_seq = linearize_skeleton(
                    arranged_rule_seq, layers, _sk['cyclic_groups'])
                logging.info(
                    f'{GATED_LOG_PREFIX} Final SCC order applied to C_rule 1st: '
                    f'{arranged_rule_seq}'
                )
            logging.info(
                f'{GATED_LOG_PREFIX} RDG Gate: activate=True, '
                f'edges={len(rdg_edges)}, layers={len(layers)}'
            )
        else:
            logging.info(f'{GATED_LOG_PREFIX} RDG Gate: activate=False')
    else:
        logging.info(f'{GATED_LOG_PREFIX} Gate2 비활성화')

    # ── Step 4: Stat Prior ────────────────────────────────────────────────────
    if stat_prior_enabled:
        try:
            stat_matrix = load_rule_stat_matrix()
            _uresult = uresult if gate1_enabled and uresult and uresult.activate else None
            query_stat_vec = build_query_stat_vector(
                query, schema, db_args, pool=pool, uniformity_result=_uresult
            )
            rdg_added_set = {r_name for r_name, _, _ in closure_added} if closure_added else set()
            rule_priors = score_rules_by_statistics(
                relevant_rules, query_stat_vec, stat_matrix,
                rdg_added_names=rdg_added_set,
            )
            if stat_prior_hard_filter:
                suppressed = [p['name'] for p in rule_priors
                              if p['decision'] == 'suppress' and p['is_rdg_added']]
                if suppressed:
                    relevant_rules = [r for r in relevant_rules
                                      if r['name'] not in suppressed]
                    rule_priors = [p for p in rule_priors
                                   if p['name'] not in suppressed]
                    logging.info(f'{GATED_LOG_PREFIX} Stat Prior hard-filter: {suppressed} 제거')
                    arranged_rule_seq = [r for r in arranged_rule_seq if r not in suppressed]
            rule_stat_priors_str = render_rule_stat_priors_for_prompt(rule_priors)
            logging.info(
                f'{GATED_LOG_PREFIX} Stat Prior scoring 완료: {len(rule_priors)}개 규칙'
            )
        except Exception as e:
            logging.warning(f'{GATED_LOG_PREFIX} Stat Prior 실패 (무시): {e}')

    # ── Step 5: C_rule 1st attempt ────────────────────────────────────────────
    logging.info(f'Arranged Rule Sequence: {arranged_rule_seq}')

    c_rule_res_first = execute_rewrite(query, schema, db_args, arranged_rule_seq, REWRITE_ROUNDS)
    c_rule_cost_first = c_rule_res_first.get('output_cost', -1)
    logging.info(
        f'{GATED_LOG_PREFIX} C_rule 1st: '
        f'used={c_rule_res_first["used_rules"]}, cost={c_rule_cost_first}'
    )

    # LINE 1 — analyze.py는 이 라인을 항상 기대함
    logging.info(f'Rewrite Execution Results: {_tag_line(c_rule_res_first, "LINE1")}')
    first_key = _candidate_key(query, c_rule_res_first.get('output_sql'))
    candidate_cache[first_key] = c_rule_res_first

    # Correlate rewrites are the exception to cost-based escalation.  Once the
    # Measure Correlate before semantic verification: its estimate is explicitly
    # untrusted, so a latency-rejected candidate must not spend an LLM call.
    # until Step 10 would waste the complete NL portfolio on a candidate whose
    # estimated cost is known to be unreliable.
    rejected_correlate_sqls: t.Set[str] = set()
    if (correlate_real_verify_enabled and not correlate_override_enabled
            and _matches_correlate_rule(c_rule_res_first.get('used_rules'))
            and c_rule_res_first.get('output_sql') not in (None, 'None', query)):
        accepted, measured = _measure_correlate_candidate(
            c_rule_res_first, query, db_args,
            correlate_real_verify_timeout_s, final_accept_ratio,
        )
        if accepted and measured is not None:
            c_rule_res_first = _verify_c_rule_result(
                'C_rule 1st', c_rule_res_first,
                query, input_cost, db_args, schema, verify_enabled,
                equiv_use_solver, equiv_use_llm_judge, equiv_solver_timeout,
                equiv_timeout_s, original_result_cache=original_result_cache,
                equivalence_cache=equivalence_cache,
            )
            if c_rule_res_first.get('output_sql') in (None, 'None', query):
                c_rule_cost_first = -1
            else:
                c_rule_cost_first = c_rule_res_first.get('output_cost', -1)
        if accepted and measured is not None and c_rule_cost_first > 0:
            _apply_cost_rejection_gate(
                c_rule_res_first, input_cost, cost_rejection_threshold, query,
                restore_on_regression=restore_on_regression,
                db_args=db_args,
                precomputed_effective=measured,
            )
            logging.info(
                f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 '
                f'(C_rule correlate immediate real-verify accepted; NL 전체 스킵)'
            )
            return
        rejected_correlate_sqls.add(c_rule_res_first['output_sql'])
        c_rule_cost_first = -1
        logging.info(
            f'{GATED_LOG_PREFIX} Correlate 후보 제외 후 재선택: '
            f'1st C_rule을 제외하고 rearrange/NL 후보 탐색 계속'
        )

    # Escalation gate: 1st C_rule이 충분히 개선했으면 NL 전체 스킵
    imp_pct_1 = _cost_improvement_pct(c_rule_cost_first, input_cost)
    sufficient_1 = _is_calcite_sufficient(c_rule_cost_first, input_cost, calcite_success_ratio)
    logging.info(
        f'{GATED_LOG_PREFIX} C_rule 1st 개선율: {imp_pct_1:.1f}% '
        f'(threshold≥{(1-calcite_success_ratio)*100:.1f}%), early_stop={sufficient_1}'
    )

    if sufficient_1 and dcrp_policy is not None:
        sufficient_1 = _dcrp_accept_single(
            {'family': 'C_rule', 'cost': c_rule_cost_first, 'result': c_rule_res_first},
            dcrp_policy, dcrp_context, input_cost,
        )
        if not sufficient_1:
            repaired = _try_verified_dcrp_repair(
                {'family': 'C_rule', 'cost': c_rule_cost_first, 'result': c_rule_res_first},
                arranged_rule_seq,
                'C_rule 1st DCRP rule-pruning repair',
            )
            if repaired is not None:
                _apply_cost_rejection_gate(
                    repaired, input_cost, cost_rejection_threshold, query,
                    restore_on_regression=restore_on_regression,
                    db_args=db_args,
                )
                logging.info(
                    f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 '
                    f'(C_rule 1st DCRP repair accepted)'
                )
                return
            logging.info(
                f'{GATED_LOG_PREFIX} Calcite early-stop (1st) DCRP reject; '
                f'repair 불가 → 탐색 계속'
            )
    if sufficient_1:
        c_rule_res_first = _verify_c_rule_result(
            'C_rule 1st', c_rule_res_first, query, input_cost,
            db_args, schema, verify_enabled, equiv_use_solver,
            equiv_use_llm_judge, equiv_solver_timeout, equiv_timeout_s,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
        )
        c_rule_cost_first = c_rule_res_first.get('output_cost', -1)
        if c_rule_res_first.get('output_sql') in (None, 'None', query):
            sufficient_1 = False
    if sufficient_1:
        logging.info(f'{GATED_LOG_PREFIX} Calcite early-stop (1st): NL 전체 스킵')
        _apply_cost_rejection_gate(c_rule_res_first, input_cost, cost_rejection_threshold, query,
                                    restore_on_regression=restore_on_regression,
                                    triage_real_verify=triage_real_verify, triage_timeout_s=triage_timeout_s,
                                    db_args=db_args, uresult=uresult, gate1_enabled=gate1_enabled,
                                    correlate_override_enabled=correlate_override_enabled,
                                    correlate_real_verify_enabled=correlate_real_verify_enabled,
                                    correlate_real_verify_timeout_s=correlate_real_verify_timeout_s)
        logging.info(f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 (calcite 1st sufficient)')
        return

    # ── Step 5a: Post-transform re-match ─────────────────────────────────────
    # full_catalog가 하던 "구조 변화로 새로 열리는 규칙 찾기"를 여기서 검증된
    # 방식으로 한다: 1차 rewrite로 실제 변환된 SQL(c_rule_res_first['output_sql'])에
    # match_all_rules()를 다시 돌려서, "원래 이 쿼리엔 안 맞았지만 1차 변환 후에는
    # Calcite가 실제로 매칭시키는" 규칙만 검증해서 골라낸다 — RDG 통계 추정이 아니라
    # Calcite 자신에게 직접 재질의하는 것이므로 오탐이 구조적으로 불가능하다.
    # build_rdg_empirical.py::discover_enables()의 오프라인 그래프 구축 로직과
    # 동일한 원리를 런타임에 그대로 적용한 것.
    # Post-transform re-match is part of Gate2's exploration expansion: it
    # discovers rules that become applicable only after the first structural
    # rewrite.  A Gate2 ablation must disable this path as well as RDG closure
    # and SCC ordering; otherwise the OFF arm still receives Gate2's expanded
    # search space.
    if gate2_enabled and c_rule_res_first['output_sql'] not in (None, 'None'):
        try:
            create_tables = [x for x in schema.split(';') if x.strip() != '']
            baseline_names = {r['name'] for r in selected_rules_lst}
            post_matches = match_all_rules(c_rule_res_first['output_sql'], create_tables, verbose=False)
            post_names = {str(r['name']) for r in post_matches}
            new_names = post_names - baseline_names - set(c_rule_res_first['used_rules'])
            # Parameter-study-only bounded post-rematch expansion.  The historical
            # default remains unlimited (budget=0, require_edge=0), so existing and
            # already queued campaigns are behaviorally unchanged.
            #
            # A large structural rewrite can make nearly the entire Calcite catalog
            # match.  Adding all of it turns the RDG from a search guide into a dense
            # prompt.  When enabled, rank newly applicable rules by the strongest
            # incoming edge from a rule that the first Calcite pass actually used,
            # then retain only the configured top-K.  The strict arm also removes
            # post-matches unsupported by such an edge.
            post_budget = int(os.environ.get('RDG_POST_REMATCH_MAX_ADDITIONS', '0'))
            post_require_edge = os.environ.get('RDG_POST_REMATCH_REQUIRE_EDGE', '0') == '1'
            if new_names and (post_budget > 0 or post_require_edge):
                source_names = set(c_rule_res_first.get('used_rules') or [])
                if not source_names:
                    source_names = set(baseline_names)
                target_scores = {}
                target_sources = {}
                for edge in (rdg_graph or {}).get('edges', []):
                    source = str(edge.get('from', ''))
                    target = str(edge.get('to', ''))
                    if source not in source_names or target not in new_names:
                        continue
                    score = (
                        float(edge.get('structural_strength', 0.0) or 0.0),
                        int(edge.get('family_support', 0) or 0),
                        int(edge.get('observed_count', 0) or 0),
                    )
                    if score > target_scores.get(target, (-1.0, -1, -1)):
                        target_scores[target] = score
                        target_sources[target] = source
                discovered_count = len(new_names)
                if post_require_edge:
                    new_names = {name for name in new_names if name in target_scores}
                ranked_names = sorted(
                    new_names,
                    key=lambda name: (
                        -target_scores.get(name, (0.0, 0, 0))[0],
                        -target_scores.get(name, (0.0, 0, 0))[1],
                        -target_scores.get(name, (0.0, 0, 0))[2],
                        name,
                    ),
                )
                if post_budget > 0:
                    ranked_names = ranked_names[:post_budget]
                new_names = set(ranked_names)
                logging.info(
                    f'{GATED_LOG_PREFIX} Post-rematch budget: '
                    f'discovered={discovered_count}, edge_supported={len(target_scores)}, '
                    f'kept={len(new_names)}, budget={post_budget}, '
                    f'require_edge={post_require_edge}, '
                    f'sources={sorted(source_names)}, '
                    f'kept_edges={[(target_sources.get(name), name) for name in ranked_names]}'
                )
            # [2026-09-13] "새로 열린 규칙이 baseline 의 절반을 넘으면 전부 스킵"
            # 필터를 제거했다. build_rdg_empirical.py 의 동일 필터와 같은 근거다
            # (rdg_refinement/claude/04_pilot_evaluation.md §4.3): 24개 쿼리에서
            # 관측한 enabling 엣지 821개 중 776개(94.5%)를 폐기했고, 폐기 대상에
            # FILTER_INTO_JOIN -> PROJECT_REMOVE (24개 중 18개 family) 같은 최고
            # 지지도 관계가 포함돼 있었다. 많이 여는 규칙일수록 걸리는데 그런
            # 규칙이야말로 선행 의존이 가장 강한 대상이다.
            #
            # 이 지점에서 잘못된 추가의 비용은 특히 낮다. 여기서 편입되는 규칙은
            # match_all_rules() 가 변환된 SQL 에 대해 "실제로 적용하면 plan 이
            # 바뀐다"고 확인한 것들이고(Rewriter.java:87-89), 이후 rearrange 와
            # execute_rewrite() 가 한 번 더 검증한다. 반대로 누락하면 그 규칙은
            # 이 쿼리에서 영원히 후보에 들어가지 못한다.
            #
            # 이전 동작으로 되돌리려면 git 으로 이 파일을 복원하면 된다.
            if new_names:
                # match_all_rules()는 규칙을 실제 적용해 plan 변화가 있는 것만
                # 돌려주지만 {'name': ...}만 주고 자연어 설명('rewrite')은 없다 —
                # arrange/rearrange 프롬프트가 r['rewrite']를 요구하므로, 기존
                # full_catalog와 동일하게 generic 설명 카탈로그에서 채운다.
                full_rule_descriptions = load_generic_rule_descriptions()
                existing_names = {r['name'] for r in relevant_rules}
                newly_unlocked = [
                    {'name': n, 'rewrite': full_rule_descriptions.get(n, n)}
                    for n in sorted(new_names) if n not in existing_names
                ]
                if newly_unlocked:
                    relevant_rules = relevant_rules + newly_unlocked
                    logging.info(
                        f'{GATED_LOG_PREFIX} Post-transform re-match: '
                        f'+{len(newly_unlocked)}개 실제 검증된 규칙 → '
                        f'{[r["name"] for r in newly_unlocked]}'
                    )
                    # The candidate set changed, so the old skeleton is stale.
                    # Rebuild before the rearrangement prompt in the opt-in RDG
                    # experiment; the default pipeline remains unchanged.
                    if gate2_enabled and os.environ.get('RDG_FINAL_SCC_ORDER') == '1':
                        _sk_post = build_skeleton_scc(
                            [r['name'] for r in relevant_rules], rdg_graph,
                            enabled_at_s0=_enabled_at_s0(rewrites),
                            opportunities=load_opportunities())
                        rdg_edges = _sk_post['edges']
                        rule_dependency_constraints_str = render_skeleton_for_prompt(
                            _sk_post['layers'], _sk_post['edges'],
                            _sk_post['unresolved'], _sk_post['cyclic_groups'])
                        logging.info(
                            f'{GATED_LOG_PREFIX} Post-rematch SCC rebuilt: '
                            f'edges={len(rdg_edges)}, layers={len(_sk_post["layers"])}')
        except Exception as e:
            logging.warning(f'{GATED_LOG_PREFIX} Post-transform re-match 실패 (무시): {e}')

    # ── Step 5b: Rearrangement (1st가 충분하지 않은 경우만) ─────────────────
    best_c_rule = c_rule_res_first
    best_c_rule_cost = c_rule_cost_first
    best_c_rule_sequence = list(arranged_rule_seq)

    rearranged_seq = rearrange_rules_dist(
        model, query, suggestions_str, relevant_rules, arranged_rule_seq,
        c_rule_res_first['used_rules'], distribution_guidance_str,
        rule_dependency_constraints=rule_dependency_constraints_str,
        rdg_edges=rdg_edges,
        rule_stat_priors=rule_stat_priors_str,
    )
    logging.info(f'Rearranged Rule Sequence: {rearranged_seq}')
    c_rule_res2 = execute_rewrite(query, schema, db_args, rearranged_seq, REWRITE_ROUNDS)
    c_rule_cost2 = c_rule_res2.get('output_cost', -1)
    logging.info(
        f'{GATED_LOG_PREFIX} Rearrange: '
        f'used={c_rule_res2["used_rules"]}, cost={c_rule_cost2}'
    )
    rearrange_key = _candidate_key(query, c_rule_res2.get('output_sql'))
    rearrange_is_duplicate = rearrange_key in candidate_cache
    if rearrange_is_duplicate:
        logging.info(
            f'{GATED_LOG_PREFIX} Rearrange: canonical duplicate '
            f'(hash={rearrange_key[:16]}) → duplicate candidate 제거'
        )
        c_rule_res2 = candidate_cache[rearrange_key]
        c_rule_cost2 = c_rule_res2.get('output_cost', -1)
    else:
        candidate_cache[rearrange_key] = c_rule_res2

    if (correlate_real_verify_enabled and not correlate_override_enabled
            and _matches_correlate_rule(c_rule_res2.get('used_rules'))
            and c_rule_res2.get('output_sql') not in (None, 'None', query)):
        if c_rule_res2['output_sql'] in rejected_correlate_sqls:
            logging.info(
                f'{GATED_LOG_PREFIX} Rearrange Correlate C_rule은 이미 실측 거절된 동일 SQL '
                f'→ 중복 실측 없이 제외'
            )
            c_rule_cost2 = -1
        else:
            accepted, measured = _measure_correlate_candidate(
                c_rule_res2, query, db_args,
                correlate_real_verify_timeout_s, final_accept_ratio,
            )
            if accepted and measured is not None:
                c_rule_res2 = _verify_c_rule_result(
                    'Rearrange (correlate latency-qualified)', c_rule_res2,
                    query, input_cost, db_args, schema, verify_enabled,
                    equiv_use_solver, equiv_use_llm_judge,
                    equiv_solver_timeout, equiv_timeout_s,
                    original_result_cache=original_result_cache,
                    equivalence_cache=equivalence_cache,
                )
                if c_rule_res2.get('output_sql') in (None, 'None', query):
                    c_rule_cost2 = -1
                else:
                    c_rule_cost2 = c_rule_res2.get('output_cost', -1)
            if accepted and measured is not None and c_rule_cost2 > 0:
                _apply_cost_rejection_gate(
                    c_rule_res2, input_cost, cost_rejection_threshold, query,
                    restore_on_regression=restore_on_regression,
                    db_args=db_args,
                    precomputed_effective=measured,
                )
                logging.info(
                    f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 '
                    f'(rearranged C_rule correlate immediate real-verify accepted; NL 전체 스킵)'
                )
                return
            rejected_correlate_sqls.add(c_rule_res2['output_sql'])
            c_rule_cost2 = -1
            logging.info(
                f'{GATED_LOG_PREFIX} Correlate 후보 제외 후 재선택: '
                f'rearranged C_rule을 제외하고 NL 후보 탐색 계속'
            )

    # Rearrange 결과가 더 좋으면 best_c_rule 갱신
    if c_rule_cost2 > 0 and (best_c_rule_cost <= 0 or c_rule_cost2 < best_c_rule_cost):
        best_c_rule = c_rule_res2
        best_c_rule_cost = c_rule_cost2
        best_c_rule_sequence = list(rearranged_seq)
        logging.info(
            f'{GATED_LOG_PREFIX} Rearrange 채택: '
            f'cost {c_rule_cost_first:.2f}→{c_rule_cost2:.2f}'
        )

    imp_pct_2 = _cost_improvement_pct(best_c_rule_cost, input_cost)
    sufficient_2 = _is_calcite_sufficient(best_c_rule_cost, input_cost, calcite_success_ratio)
    logging.info(
        f'{GATED_LOG_PREFIX} Rearrange 개선율: {imp_pct_2:.1f}% '
        f'(threshold≥{(1-calcite_success_ratio)*100:.1f}%), early_stop={sufficient_2}'
    )

    if sufficient_2 and dcrp_policy is not None:
        sufficient_2 = _dcrp_accept_single(
            {'family': 'C_rule', 'cost': best_c_rule_cost, 'result': best_c_rule},
            dcrp_policy, dcrp_context, input_cost,
        )
        if not sufficient_2:
            repaired = _try_verified_dcrp_repair(
                {'family': 'C_rule', 'cost': best_c_rule_cost, 'result': best_c_rule},
                best_c_rule_sequence,
                'Best C_rule DCRP rule-pruning repair',
            )
            if repaired is not None:
                _apply_cost_rejection_gate(
                    repaired, input_cost, cost_rejection_threshold, query,
                    restore_on_regression=restore_on_regression,
                    db_args=db_args,
                )
                logging.info(
                    f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 '
                    f'(rearranged C_rule DCRP repair accepted)'
                )
                return
            logging.info(
                f'{GATED_LOG_PREFIX} Calcite early-stop (rearrange) DCRP reject; '
                f'repair 불가 → 탐색 계속'
            )
    if sufficient_2:
        best_c_rule = _verify_c_rule_result(
            'Best C_rule (cost-qualified)', best_c_rule, query, input_cost,
            db_args, schema, verify_enabled, equiv_use_solver,
            equiv_use_llm_judge, equiv_solver_timeout, equiv_timeout_s,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
        )
        best_c_rule_cost = best_c_rule.get('output_cost', -1)
        if best_c_rule.get('output_sql') in (None, 'None', query):
            sufficient_2 = False
    if sufficient_2:
        logging.info(f'{GATED_LOG_PREFIX} Calcite early-stop (rearrange): NL 전체 스킵')
        _apply_cost_rejection_gate(best_c_rule, input_cost, cost_rejection_threshold, query,
                                    restore_on_regression=restore_on_regression,
                                    triage_real_verify=triage_real_verify, triage_timeout_s=triage_timeout_s,
                                    db_args=db_args, uresult=uresult, gate1_enabled=gate1_enabled,
                                    correlate_override_enabled=correlate_override_enabled,
                                    correlate_real_verify_enabled=correlate_real_verify_enabled,
                                    correlate_real_verify_timeout_s=correlate_real_verify_timeout_s)
        logging.info(f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 (rearrange sufficient)')
        return

    # ── Step 6: NL phase (portfolio_enabled=True, Calcite 둘 다 부족한 경우) ─
    if not portfolio_enabled:
        logging.info(f'{GATED_LOG_PREFIX} portfolio_enabled=False: NL 스킵')
        best_c_rule = _verify_c_rule_result(
            'Best C_rule (rule-only final candidate)', best_c_rule, query,
            input_cost, db_args, schema, verify_enabled, equiv_use_solver,
            equiv_use_llm_judge, equiv_solver_timeout, equiv_timeout_s,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
        )
        _apply_cost_rejection_gate(best_c_rule, input_cost, cost_rejection_threshold, query,
                                    restore_on_regression=restore_on_regression,
                                    triage_real_verify=triage_real_verify, triage_timeout_s=triage_timeout_s,
                                    db_args=db_args, uresult=uresult, gate1_enabled=gate1_enabled,
                                    correlate_override_enabled=correlate_override_enabled,
                                    correlate_real_verify_enabled=correlate_real_verify_enabled,
                                    correlate_real_verify_timeout_s=correlate_real_verify_timeout_s)
        logging.info(f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 (rule-only)')
        return

    # ── Step 6a: detect_rewrite_opportunities() ──────────────────────────────
    opp_result = detect_rewrite_opportunities(
        query, schema,
        uniformity_result=uresult,
        calcite_result={'input_cost': input_cost, 'output_cost': best_c_rule_cost},
    )
    logging.info(f'{GATED_LOG_PREFIX} Detector features: {opp_result.features}')
    for op in opp_result.opportunities:
        logging.info(
            f'{GATED_LOG_PREFIX} Opportunity: family={op.family}, '
            f'activated={op.activated}, confidence={op.confidence:.3f}, '
            f'priority={op.priority}, reasons={op.reasons}'
        )

    # ── Step 6b: 활성화된 family를 confidence 내림차순, priority 오름차순 정렬 ─
    active_opps = sorted(
        [op for op in opp_result.opportunities if op.activated],
        key=lambda op: (-op.confidence, op.priority),
    )
    logging.info(
        f'{GATED_LOG_PREFIX} Active families (sorted): {[op.family for op in active_opps]}'
    )

    # ── Step 6c: Lazy family generators (callable, 선택된 것만 실제 호출됨) ──
    # distribution_guidance_str은 이미 Step 1에서 생성된 것을 재사용 (추가 LLM 호출 없음)
    _FAMILIES_USING_GUIDANCE: t.FrozenSet[str] = frozenset({'C_join', 'C_nl'})

    family_generators: t.Dict[str, t.Callable[[], t.Optional[str]]] = {
        'C_pred':   lambda: _gen_pred_candidate(model, query, prompt_schema),
        'C_cte':    lambda: _gen_cte_candidate(model, query, prompt_schema),
        'C_join':   lambda: _gen_join_candidate(model, query, prompt_schema, distribution_guidance_str),
        'C_intent': lambda: _gen_intent_candidate(model, query, prompt_schema, suggestions_str),
    }

    # ── Step 7: 선택된 family 순차 호출 (max_nl_families 상한 준수) ───────────
    nl_candidates: t.List[t.Dict] = []
    nl_family_count = 0
    nl_llm_call_count = 0
    early_accept = False
    max_acceptable_cost = (
        input_cost * final_accept_ratio
        if restore_on_regression and input_cost > 0 and dcrp_policy is None else None
    )

    for opp in active_opps:
        if nl_family_count >= max_nl_families:
            logging.info(
                f'{GATED_LOG_PREFIX} {opp.family}: skip '
                f'(max_nl_families={max_nl_families} 도달)'
            )
            continue

        if early_accept:
            logging.info(
                f'{GATED_LOG_PREFIX} {opp.family}: skip '
                f'(이전 family early-accept 발동)'
            )
            continue

        gen_fn = family_generators.get(opp.family)
        if gen_fn is None:
            logging.warning(f'{GATED_LOG_PREFIX} {opp.family}: generator 없음 → skip')
            continue

        logging.info(
            f'{GATED_LOG_PREFIX} {opp.family}: LLM 호출 시작 '
            f'(confidence={opp.confidence:.3f}, family_count={nl_family_count+1})'
        )
        entry, calls = _generate_and_verify_with_retry(
            opp.family, gen_fn, query, db_args, verify_enabled, equiv_timeout_s,
            schema=prompt_schema,
            equiv_use_solver=equiv_use_solver,
            equiv_use_llm_judge=equiv_use_llm_judge,
            equiv_solver_timeout=equiv_solver_timeout,
            max_equiv_retries=max_equiv_retries,
            original_result_cache=original_result_cache,
            verification_schema=schema,
            equivalence_cache=equivalence_cache,
            max_acceptable_cost=max_acceptable_cost,
        )
        nl_family_count += 1
        nl_llm_call_count += calls
        nl_candidates.append(entry)

        # Early accept: 첫 성공 후보가 final_accept_ratio를 만족하면 즉시 중단
        if (entry['syntax_ok'] and entry['equiv_ok'] and entry['raw_cost'] > 0
                and _is_calcite_sufficient(entry['raw_cost'], input_cost, final_accept_ratio)):
            logging.info(
                f'{GATED_LOG_PREFIX} {opp.family}: final_accept early-stop '
                f'(cost={entry["raw_cost"]:.2f} ≤ {input_cost:.2f}×{final_accept_ratio})'
            )
            early_accept = True

    # ── Step 8: C_nl broad fallback ──────────────────────────────────────────
    if not broad_fallback_enabled:
        logging.info(f'{GATED_LOG_PREFIX} C_nl: broad_fallback_enabled=False → skip')
    elif early_accept:
        logging.info(f'{GATED_LOG_PREFIX} C_nl: early_accept 발동됨 → broad fallback 불필요')
    elif nl_family_count >= max_nl_families:
        logging.info(
            f'{GATED_LOG_PREFIX} C_nl: broad fallback 시도하려 했으나 '
            f'max_nl_families={max_nl_families} 이미 도달 → skip'
        )
    elif not opp_result.broad_fallback_allowed:
        logging.info(f'{GATED_LOG_PREFIX} C_nl: broad_fallback_allowed=False (detector 판정) → skip')
    else:
        logging.info(f'{GATED_LOG_PREFIX} C_nl: broad fallback 시도')
        gen_fn_nl = lambda: _gen_nl_candidate(model, query, prompt_schema, distribution_guidance_str, suggestions_str)
        entry_nl, calls = _generate_and_verify_with_retry(
            'C_nl', gen_fn_nl, query, db_args, verify_enabled, equiv_timeout_s,
            schema=prompt_schema,
            equiv_use_solver=equiv_use_solver,
            equiv_use_llm_judge=equiv_use_llm_judge,
            equiv_solver_timeout=equiv_solver_timeout,
            max_equiv_retries=max_equiv_retries,
            original_result_cache=original_result_cache,
            verification_schema=schema,
            equivalence_cache=equivalence_cache,
            max_acceptable_cost=max_acceptable_cost,
        )
        nl_family_count += 1
        nl_llm_call_count += calls
        nl_candidates.append(entry_nl)
        if (entry_nl['syntax_ok'] and entry_nl['equiv_ok'] and entry_nl['raw_cost'] > 0
                and _is_calcite_sufficient(entry_nl['raw_cost'], input_cost, final_accept_ratio)):
            logging.info(f'{GATED_LOG_PREFIX} C_nl broad fallback: final_accept')
            early_accept = True

    logging.info(
        f'{GATED_LOG_PREFIX} NL 실행 family 수: {nl_family_count}, '
        f'NL 생성/corrective 호출 수: {nl_llm_call_count}'
    )

    # ── Step 9: 후보 요약 로그 ───────────────────────────────────────────────
    for c in nl_candidates:
        cost_str = f'{c["raw_cost"]:.2f}' if c['raw_cost'] > 0 else '-1'
        logging.info(
            f'{GATED_LOG_PREFIX} Candidate: '
            f'family={c["family"]}, syntax={c["syntax_ok"]}, equiv={c["equiv_ok"]}, '
            f'cost={cost_str}, fail={c["fail_reason"] or "none"}'
        )

    # ── Step 10: Best selection + Cost Rejection Gate ────────────────────────
    valid_nl = [
        c for c in nl_candidates
        if c['syntax_ok'] and c['equiv_ok'] and c['raw_cost'] > 0
    ]
    all_valid: t.List[t.Dict] = []
    best_nl_cost = min((c['raw_cost'] for c in valid_nl), default=float('inf'))
    # A 0–3% C_rule rewrite is kept unverified as a fallback while NL candidates
    # are explored. Verify it only when it can actually win the final cost rank.
    if best_c_rule_cost > 0 and (dcrp_policy is not None or best_c_rule_cost <= best_nl_cost):
        best_c_rule = _verify_c_rule_result(
            'Best C_rule (lazy final candidate)', best_c_rule, query, input_cost,
            db_args, schema, verify_enabled, equiv_use_solver,
            equiv_use_llm_judge, equiv_solver_timeout, equiv_timeout_s,
            original_result_cache=original_result_cache,
            equivalence_cache=equivalence_cache,
        )
        best_c_rule_cost = best_c_rule.get('output_cost', -1)
        if best_c_rule.get('output_sql') not in (None, 'None', query) and best_c_rule_cost > 0:
            all_valid.append({'family': 'C_rule', 'cost': best_c_rule_cost, 'result': best_c_rule})
    elif best_c_rule_cost > 0:
        logging.info(
            f'{GATED_LOG_PREFIX} C_rule lazy verification skip: '
            f'cost={best_c_rule_cost:.2f} > verified NL best={best_nl_cost:.2f}'
        )
    for c in valid_nl:
        _res = {
            'used_rules': [c['family']],
            'output_sql': c['sql'],
            'output_cost': c['raw_cost'],
            'time': 0,
            '_equiv_ok': c.get('equiv_ok'),
            '_equiv_source': c.get('equiv_source', 'unknown'),
            '_equiv_reason': c.get('equiv_reason', ''),
        }
        all_valid.append({'family': c['family'], 'cost': c['raw_cost'], 'result': _res})

    if not all_valid:
        reason = 'early_accept' if early_accept else 'no_valid_candidate'
        logging.warning(f'{GATED_LOG_PREFIX} 유효한 후보 없음 → 원본 SQL 유지 (reason={reason})')
        _fallback_res = {
            'used_rules': [
                'CORRELATE_REAL_VERIFY_FAILED_FALLBACK'
                if rejected_correlate_sqls else 'GATED_NO_VALID_CANDIDATE'
            ],
            'output_sql': query,
            'output_cost': input_cost if input_cost > 0 else -1,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(_fallback_res, "LINE2")}')  # LINE 2
        return

    best, correlate_matched = _select_best_candidate(
        all_valid, correlate_override_enabled, correlate_real_verify_enabled,
        dcrp_policy=dcrp_policy, dcrp_context=dcrp_context, input_cost=input_cost,
    )
    if best is None:
        c_rule_candidate = next(
            (candidate for candidate in all_valid if candidate.get('family') == 'C_rule'),
            None,
        )
        repaired = (
            _try_verified_dcrp_repair(
                c_rule_candidate,
                best_c_rule_sequence,
                'Final C_rule DCRP rule-pruning repair',
            )
            if c_rule_candidate is not None else None
        )
        if repaired is not None:
            best = {
                'family': 'C_rule_dcrp_repair',
                'cost': repaired.get('output_cost', -1),
                'result': repaired,
            }
            correlate_matched = False
    if best is None:
        _fallback_res = {
            'used_rules': ['DCRP_REJECT_FALLBACK'],
            'output_sql': query,
            'output_cost': input_cost,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(_fallback_res, "LINE2")}')
        logging.info(f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료 (DCRP fallback)')
        return
    # override가 켜져 있으면 override가 우선(무조건 채택); 아니면 real_verify 대상으로 취급.
    correlate_override_fired = correlate_matched and correlate_override_enabled
    correlate_verify_pending = correlate_matched and correlate_real_verify_enabled and not correlate_override_enabled
    logging.info(
        f'{GATED_LOG_PREFIX} Best: family={best["family"]}, cost={best["cost"]:.2f} '
        f'(from {len(all_valid)} valid candidates)'
    )
    if correlate_override_fired:
        logging.info(
            f'{GATED_LOG_PREFIX} Correlate Override 발동: family={best["family"]}, '
            f'used_rules={best["result"].get("used_rules")} → cost 추정과 무관하게 무조건 채택'
        )
    elif correlate_verify_pending:
        logging.info(
            f'{GATED_LOG_PREFIX} Correlate 후보 선택됨(실측 검증 예정): family={best["family"]}, '
            f'used_rules={best["result"].get("used_rules")}'
        )

    # ── Step 10a: Estimate-reliability triage (실험용) ────────────────────────
    # cost 추정치(EXPLAIN, 미실행)는 스큐/상관관계가 심한 컬럼에서 크게 틀릴 수
    # 있다 (예: 추정으론 개선처럼 보이는데 실측하면 대폭 악화). Step 1(gate1)에서
    # 이미 계산해 둔 uniformity_score(pg_stats 기반 skew 지표)를 추가 DB 조회나
    # LLM 호출 없이 그대로 재활용해서, "이 쿼리는 estimate를 못 믿을 만큼 스큐가
    # 심하다"고 이미 판단된 경우(uresult.activate)에만 최종 후보 하나를
    # actual_time_paired()로 실측(EXPLAIN ANALYZE, timeout=triage_timeout_s)해
    # 검증한다. 대부분의(스큐 안 심한) 쿼리는 기존처럼 estimate만으로 판단해
    # 오버헤드를 최소화한다.
    effective_input_cost, effective_best_cost = input_cost, best['cost']
    triage_fired = False
    correlate_verify_failed = False
    if correlate_verify_pending and best['result'].get('output_sql') not in (None, 'None'):
        logging.info(
            f'{GATED_LOG_PREFIX} Correlate Real-Verify 발동: used_rules={best["result"].get("used_rules")} '
            f'→ 최종 후보 실측 검증'
        )
        try:
            (real_input, _), (real_output, _) = actual_time_paired(
                query, best['result']['output_sql'], db_args, correlate_real_verify_timeout_s
            )
            if (real_input not in (float('inf'), None) and real_output not in (float('inf'), None)
                    and real_input > 0):
                logging.info(
                    f'{GATED_LOG_PREFIX} Correlate Real-Verify 실측 결과: input={real_input:.2f}ms, '
                    f'output={real_output:.2f}ms (estimate cost는 input={input_cost:.2f}, '
                    f'best={best["cost"]:.2f}이었음)'
                )
                effective_input_cost, effective_best_cost = real_input, real_output
                triage_fired = True
            else:
                logging.warning(
                    f'{GATED_LOG_PREFIX} Correlate Real-Verify 실측 실패(inf/None) '
                    f'→ correlate 후보 검증 실패'
                )
                correlate_verify_failed = True
        except Exception as e:
            logging.warning(
                f'{GATED_LOG_PREFIX} Correlate Real-Verify 실측 예외 '
                f'→ correlate 후보 검증 실패: {e}'
            )
            correlate_verify_failed = True
    elif (not correlate_override_fired and triage_real_verify and gate1_enabled
            and uresult is not None and uresult.activate
            and best['result'].get('output_sql') not in (None, 'None')):
        logging.info(
            f'{GATED_LOG_PREFIX} Triage 발동: uniformity_score={uresult.score:.4f} '
            f'(gate1이 이미 스큐 심하다고 판단) → 최종 후보 실측 검증'
        )
        try:
            (real_input, _), (real_output, _) = actual_time_paired(
                query, best['result']['output_sql'], db_args, triage_timeout_s
            )
            if (real_input not in (float('inf'), None) and real_output not in (float('inf'), None)
                    and real_input > 0):
                logging.info(
                    f'{GATED_LOG_PREFIX} Triage 실측 결과: input={real_input:.2f}ms, '
                    f'output={real_output:.2f}ms (estimate cost는 input={input_cost:.2f}, '
                    f'best={best["cost"]:.2f}이었음)'
                )
                effective_input_cost, effective_best_cost = real_input, real_output
                triage_fired = True
            else:
                logging.warning(
                    f'{GATED_LOG_PREFIX} Triage 실측 실패(timeout/inf) → estimate 기준으로 폴백'
                )
        except Exception as e:
            logging.warning(f'{GATED_LOG_PREFIX} Triage 실측 예외 (무시, estimate로 폴백): {e}')

    # A correlate candidate is intentionally prioritized before measurement,
    # but a failed or regressing measurement must not discard an independently
    # valid NL/Calcite alternative. Re-enter the normal estimate-based final
    # gate with the cheapest non-correlate candidate. Only fall back directly
    # to the original when no such candidate exists.
    correlate_measured_regression = (
        correlate_verify_pending and triage_fired
        and effective_input_cost > 0 and effective_best_cost > 0
        and effective_best_cost > effective_input_cost * final_accept_ratio
    )
    if correlate_verify_failed or correlate_measured_regression:
        alternative = _select_non_correlate_fallback(
            all_valid, dcrp_policy=dcrp_policy,
            dcrp_context=dcrp_context, input_cost=input_cost,
        )
        if alternative is not None:
            reason = 'measurement_failed' if correlate_verify_failed else 'measured_regression'
            logging.info(
                f'{GATED_LOG_PREFIX} Correlate 후보 제외 후 재선택: reason={reason}, '
                f'family={alternative["family"]}, cost={alternative["cost"]:.2f}'
            )
            best = alternative
            correlate_verify_failed = False
            correlate_verify_pending = False
            correlate_override_fired = False
            triage_fired = False
            effective_input_cost, effective_best_cost = input_cost, best['cost']

    if correlate_verify_failed:
        _fallback_res = {
            'used_rules': ['CORRELATE_REAL_VERIFY_FAILED_FALLBACK'],
            'output_sql': query,
            'output_cost': input_cost,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(_fallback_res, "LINE2")}')
        return

    # ── Final accept gate: best 후보도 final_accept_ratio를 만족해야 채택 ─────
    # restore_on_regression=False면 이 게이트를 끄고 best를 무조건 채택한다 —
    # R-Bot류 baseline처럼 시스템 자체에 회귀 방지 장치가 없는 상태를 재현하는
    # 실험용 스위치 (_not_restore_regression_query 실험).
    # triage_fired=True면 estimate 대신 실측(ms) 기준으로 이 게이트를 판단한다.
    # correlate_override_fired=True면 이 게이트 자체를 건너뛴다(무조건 채택).
    if (not correlate_override_fired and restore_on_regression
            and dcrp_policy is None
            and effective_input_cost > 0 and effective_best_cost > 0
            and effective_best_cost > effective_input_cost * final_accept_ratio):
        pct_gap = (effective_best_cost / effective_input_cost - 1) * 100
        basis = '실측(latency)' if triage_fired else 'estimate(cost)'
        logging.warning(
            f'{GATED_LOG_PREFIX} Final accept gate 미달 ({basis} 기준): '
            f'best={effective_best_cost:.2f} > input({effective_input_cost:.2f})×{final_accept_ratio} '
            f'(gap={pct_gap:+.2f}%) → 원본 SQL fallback'
        )
        _fallback_res = {
            'used_rules': ['FINAL_ACCEPT_FALLBACK'],
            'output_sql': query,
            'output_cost': input_cost,
            'time': 0,
        }
        logging.info(f'Rewrite Execution Results: {_tag_line(_fallback_res, "LINE2")}')  # LINE 2
        return

    if triage_fired:
        # 이미 위에서 실측했으므로, cost_rejection_gate에는 그 값을 그대로
        # 넘겨서 중복 실측(actual_time_paired 재호출)을 피한다 — 게이트 자체는
        # 여전히 통과하되(같은 실측값이 cost_rejection_threshold도 만족하는 게
        # 보통이다), LINE2 로깅 형식은 일관되게 _apply_cost_rejection_gate를 통한다.
        _apply_cost_rejection_gate(best['result'], input_cost, cost_rejection_threshold, query,
                                    restore_on_regression=restore_on_regression,
                                    precomputed_effective=(effective_input_cost, effective_best_cost),
                                    correlate_override_enabled=correlate_override_enabled,
                                    correlate_real_verify_enabled=correlate_real_verify_enabled)
    else:
        _apply_cost_rejection_gate(best['result'], input_cost, cost_rejection_threshold, query,
                                    restore_on_regression=(
                                        restore_on_regression if dcrp_policy is None else False
                                    ),
                                    triage_real_verify=triage_real_verify, triage_timeout_s=triage_timeout_s,
                                    db_args=db_args, uresult=uresult, gate1_enabled=gate1_enabled,
                                    correlate_override_enabled=correlate_override_enabled,
                                    correlate_real_verify_enabled=correlate_real_verify_enabled,
                                    correlate_real_verify_timeout_s=correlate_real_verify_timeout_s)
    # ↑ _apply_cost_rejection_gate 가 LINE 2를 출력함

    logging.info(f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 종료')
