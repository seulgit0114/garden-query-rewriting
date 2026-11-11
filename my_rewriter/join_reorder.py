"""
join_reorder.py

NL 방식의 Join Reordering 모듈 (CTE MATERIALIZED 전략).

Gate1이 활성화된 쿼리 중 implicit cross-join 3개 이상 테이블 패턴이 감지되면,
LLM에게 CTE MATERIALIZED로 선택도 높은 테이블을 먼저 평가하도록 재구성된
SQL을 요청한다. 생성된 SQL은 2단계 검증(구조 제약 + 결과 비교)을 통과해야만
'Rewrite Execution Results'로 로깅된다.

반환 dict 형식은 db_utils.execute_rewrite()와 동일:
  {'used_rules': [...], 'output_sql': str, 'output_cost': float, 'time': int}
"""
import re
import logging
import time
import typing as t

import sqlglot
from sqlglot.optimizer.scope import build_scope

from my_rewriter.database import Database, DBArgs
from my_rewriter.my_utils import MyModel, chat

RULE_NAME = 'NL_JOIN_REORDER_CTE'
PILOT_LOG_PREFIX = '[PILOT-CHECK]'

# ── Prompts ──────────────────────────────────────────────────────────────────

_JR_SYS_PROMPT = """\
You are a PostgreSQL query optimization expert specializing in join reordering via CTE materialization.

Your task: Rewrite a multi-table SQL query so that highly selective tables are evaluated first
using `WITH <name> AS MATERIALIZED (...)` CTEs. This forces PostgreSQL to compute those
filtered result sets before executing the main join, effectively controlling join order.

Strategy:
1. Identify which tables have the most restrictive WHERE conditions (e.g., equality on small lookup tables, LIKE patterns, enum values).
2. Wrap those tables in MATERIALIZED CTEs.
3. Build the remaining query by joining from the smallest CTE outward.
4. Express all remaining cross-joins as explicit JOIN ... ON conditions.

CRITICAL CONSTRAINTS — all must be satisfied exactly:
- Every table from the original FROM clause must appear in the rewrite (no additions or removals).
- Every condition from the original WHERE clause must be preserved (move conditions into CTEs or JOIN ON, never drop them).
- The SELECT list and all aggregate functions must be identical to the original.
- No new filter values, no changed column references, no additional predicates.

Output: only the rewritten SQL wrapped in ```sql ... ``` — no explanation."""

_JR_USER_PROMPT = """\
Rewrite the following SQL using CTE MATERIALIZED to improve join ordering.

Schema (for reference):
```sql
{schema}
```

Original SQL:
```sql
{sql}
```

Identify the most selective tables first (those whose WHERE conditions reduce row count the most),
wrap them as MATERIALIZED CTEs, then build the query joining from the smallest result set outward.

Output the rewritten SQL wrapped in ```sql ... ``` only."""


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_candidate(query: str) -> bool:
    """FROM절에 쉼표로 구분된 테이블이 3개 이상 있는 implicit cross-join 쿼리를 감지한다."""
    m = re.search(
        r'\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|;|$)',
        query, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return False
    from_clause = m.group(1)
    depth = commas = 0
    for ch in from_clause:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            commas += 1
    return commas >= 2  # 쉼표 2개 → 테이블 3개 이상


# ── Structural validation (regex, no sqlglot) ────────────────────────────────

def _extract_from_tables(query: str) -> t.Set[str]:
    """
    FROM절 comma-separated 항목에서 테이블 이름(alias 앞 부분)을 소문자로 추출한다.
    CTE 이름(WITH ... AS)은 제외한다.
    """
    cte_names: t.Set[str] = set()
    for m in re.finditer(r'\bWITH\b\s+(\w+)\s+AS\s*(?:MATERIALIZED\s*)?\s*\(', query, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    for m in re.finditer(r',\s*(\w+)\s+AS\s*(?:MATERIALIZED\s*)?\s*\(', query, re.IGNORECASE):
        cte_names.add(m.group(1).lower())

    m = re.search(
        r'\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|;|$)',
        query, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return set()

    tables: t.Set[str] = set()
    depth = 0
    current: t.List[str] = []
    for ch in m.group(1):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            entry = ''.join(current).strip()
            if entry:
                parts = entry.split()
                if parts:
                    tbl = parts[0].lower()
                    if tbl not in cte_names:
                        tables.add(tbl)
            current = []
            continue
        current.append(ch)
    # last entry
    entry = ''.join(current).strip()
    if entry:
        parts = entry.split()
        if parts:
            tbl = parts[0].lower()
            if tbl not in cte_names:
                tables.add(tbl)
    return tables


def _structural_check(original_sql: str, rewritten_sql: str) -> t.Tuple[bool, str]:
    """
    원본 FROM 테이블이 rewrite에 모두 참조되는지 확인한다.
    - original의 base table 집합이 rewritten SQL 텍스트에 모두 등장해야 함.

    (과거에는 최상위 SELECT 절 텍스트가 원본과 글자 그대로 일치해야 한다는 체크도
    있었으나 제거했다: C_join/C_cte처럼 FROM 절 구조를 재배치하는 rewrite는 결과가
    동일해도 컬럼 별칭이 정상적으로 바뀌므로(a.x → t1.x 등) 거의 항상 이 텍스트
    비교에서 걸렸다 — dsb50 76쿼리 실측에서 C_join 63/63, C_cte 8/8이 전부 이걸로
    거부되어 equivalence check까지 아예 도달하지 못했다. semantic equivalence는
    verify_equivalence_hybrid()(SQLSolver Z3 증명 + LLM judge + 결과비교)가 이미
    훨씬 엄밀하게 검증하므로 이 텍스트 매칭은 중복이면서 더 거칠기만 했다.)
    """
    orig_tables = _extract_from_tables(original_sql)
    if not orig_tables:
        return True, 'no tables extracted from original, skipping table check'

    rew_lower = rewritten_sql.lower()
    missing = [t for t in orig_tables if t not in rew_lower]
    if missing:
        return False, f'tables missing from rewrite: {missing}'

    return True, 'OK'


# ── Alias consistency check (B) ──────────────────────────────────────────────

def _alias_consistency_check(rewritten_sql: str) -> t.Tuple[bool, str]:
    """
    rewritten SQL 내부 일관성 검사 (DB 실행 없음).
    <alias>.column 형태로 참조된 이름이 그 컬럼이 속한 스코프(및 상위 스코프,
    correlated subquery 대응)에서 보이는 FROM/JOIN/WITH/subquery 소스 중
    어디에도 없으면 'missing FROM-clause entry' DB 오류가 발생함을 조기 감지한다.

    LLM이 테이블 X를 CTE로 이동하고 메인 쿼리에서 여전히 원본 별칭으로
    X.column을 참조하는 패턴을 잡는다.

    sqlglot AST + Scope 기반 (구 버전은 정규식 기반이었으나, `FROM t alias`
    처럼 AS 없이 쓰는 암묵적 별칭과 `FROM a, b, c` 콤마 조인의 두 번째 항목부터를
    선언으로 인식하지 못해 정상 SQL을 대량으로 오탐 처리했다 — 실 로그 기준
    dsb10sf10_aliasfix_0809에서 alias_fail로 3회 재시도 끝에 포기한 쿼리들이
    실제로는 전부 이 오탐이었다). Scope 단위로 검사하되 부모 스코프까지 거슬러
    올라가며 봐야 하는 이유는 correlated subquery(WHERE EXISTS (... outer_alias.col
    ...))가 외부 스코프의 별칭을 참조하는 게 정상 SQL이기 때문이다.
    """
    try:
        tree = sqlglot.parse_one(rewritten_sql, read='postgres')
        root_scope = build_scope(tree)
    except Exception as e:
        # 파싱 실패는 alias 체크가 아니라 뒤의 syntax(EXPLAIN) 체크가 판단할 문제.
        return True, f'parse_skip: {str(e)[:120]}'

    if root_scope is None:
        return True, 'parse_skip: no_scope'

    undeclared: t.Set[str] = set()
    for scope in root_scope.traverse():
        visible: t.Set[str] = set()
        s = scope
        while s is not None:
            visible.update(name.lower() for name in s.sources.keys())
            s = s.parent
        for column in scope.columns:
            if column.table and column.table.lower() not in visible:
                undeclared.add(column.table.lower())

    if undeclared:
        return False, (
            f'선언되지 않은 테이블 별칭 참조: {sorted(undeclared)} '
            f'— CTE로 이동된 테이블의 원본 별칭이 메인 쿼리에서 여전히 사용됨'
        )
    return True, 'OK'


# ── Result equivalence ───────────────────────────────────────────────────────

def _result_check(original_sql: str, rewritten_sql: str, db: Database) -> t.Tuple[bool, str]:
    """
    두 쿼리를 실제로 실행해 결과를 비교한다.
    집계 쿼리(단일 행 반환)는 결과 직접 비교, 그 외는 행 수 비교.
    """
    def _strip_semi(sql: str) -> str:
        return sql.rstrip().rstrip(';')

    orig_success, orig_res, orig_logs = db.execute_sql(_strip_semi(original_sql))
    if orig_success != 1:
        return False, f'original query execution failed: {orig_logs}'

    rew_success, rew_res, rew_logs = db.execute_sql(_strip_semi(rewritten_sql))
    if rew_success != 1:
        return False, f'rewritten query execution failed: {rew_logs}'

    # 단일 행 결과 (JOB MIN/MAX 집계) → 결과 직접 비교
    if orig_res is not None and len(orig_res) <= 5:
        if orig_res != rew_res:
            return False, (f'result mismatch: orig={str(orig_res)[:120]}, '
                           f'rew={str(rew_res)[:120]}')
        return True, f'result match ({len(orig_res)} rows)'

    # 다중 행 → 행 수 비교
    orig_cnt = len(orig_res) if orig_res else 0
    rew_cnt = len(rew_res) if rew_res else 0
    if orig_cnt != rew_cnt:
        return False, f'row count mismatch: orig={orig_cnt}, rew={rew_cnt}'
    return True, f'row count match ({orig_cnt} rows)'


# ── Main entry point ─────────────────────────────────────────────────────────

def execute_join_reorder(
    query: str,
    schema: str,
    db_args: DBArgs,
    model: MyModel,
) -> t.Optional[t.Dict]:
    """
    NL Join Reordering을 시도한다.
    성공 시 execute_rewrite()와 동일한 형식의 dict를 반환한다.
    감지 실패 또는 검증 실패 시 None을 반환한다 (caller가 결과를 로깅하지 않음).
    """
    if not detect_candidate(query):
        logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering 스킵: '
                     f'implicit cross-join 3+ 테이블 패턴 미감지')
        return None

    logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering 시작: CTE MATERIALIZED 전략 시도')
    start = time.time()

    # Step 1: LLM이 CTE 재구성 SQL 생성 (최대 2회 시도)
    messages = [
        {'role': 'system', 'content': _JR_SYS_PROMPT},
        {'role': 'user', 'content': _JR_USER_PROMPT.format(schema=schema, sql=query)},
    ]
    rewritten_sql: t.Optional[str] = None
    response: str = ''
    for attempt in range(1, 3):
        response = chat(messages)
        m = re.search(r'```sql\s*(.+?)\s*```', response, re.IGNORECASE | re.DOTALL)
        if m:
            rewritten_sql = m.group(1).strip()
            logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering LLM attempt={attempt}: '
                         f'SQL 추출 성공 (len={len(rewritten_sql)})')
            break
        logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering LLM attempt={attempt}: SQL 추출 실패')

    if not rewritten_sql:
        elapsed = int((time.time() - start) * 1000)
        logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: LLM SQL 생성 실패 ({elapsed}ms)')
        return None

    # Step 2: 구조 제약 검사 (zero DB cost)
    struct_ok, struct_reason = _structural_check(query, rewritten_sql)
    if not struct_ok:
        elapsed = int((time.time() - start) * 1000)
        logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: 구조 검증 실패 — {struct_reason} ({elapsed}ms)')
        return None

    # Step 2b: alias 일관성 검사 [B] — DB 실행 전 missing FROM-clause 조기 감지
    alias_ok, alias_reason = _alias_consistency_check(rewritten_sql)
    if not alias_ok:
        elapsed_so_far = int((time.time() - start) * 1000)
        logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 불일치 감지 — {alias_reason} ({elapsed_so_far}ms)')

        # [C] 에러 피드백과 함께 LLM 재시도
        fix_prompt = (
            f'Your SQL has an error: {alias_reason}\n\n'
            f'When you move a table into a CTE, you must replace ALL dot-references '
            f'(e.g. "mc.company_id") to that table in the main query with the CTE alias. '
            f'Go through every alias.column reference in your main SELECT/FROM/JOIN/WHERE and '
            f'make sure each alias is actually declared in the main FROM/JOIN list or as a CTE name.\n\n'
            f'Output only the corrected SQL wrapped in ```sql ... ```.'
        )
        retry_messages = [
            messages[0],
            messages[1],
            {'role': 'assistant', 'content': response},
            {'role': 'user', 'content': fix_prompt},
        ]
        retry_response = chat(retry_messages)
        m_retry = re.search(r'```sql\s*(.+?)\s*```', retry_response, re.IGNORECASE | re.DOTALL)
        if not m_retry:
            elapsed = int((time.time() - start) * 1000)
            logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 재시도 SQL 추출 실패 ({elapsed}ms)')
            return None

        rewritten_sql = m_retry.group(1).strip()
        logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 재시도 SQL 추출 성공 (len={len(rewritten_sql)})')

        # 재시도 후 재검증
        struct_ok2, struct_reason2 = _structural_check(query, rewritten_sql)
        if not struct_ok2:
            elapsed = int((time.time() - start) * 1000)
            logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 재시도 후 구조 검증 실패 — {struct_reason2} ({elapsed}ms)')
            return None
        alias_ok2, alias_reason2 = _alias_consistency_check(rewritten_sql)
        if not alias_ok2:
            elapsed = int((time.time() - start) * 1000)
            logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 재시도 후에도 불일치 — {alias_reason2} ({elapsed}ms)')
            return None
        logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering: alias 재시도 검증 통과')

    # Step 3: 실제 결과 비교 (DB 실행)
    db = Database(db_args)
    result_ok, result_reason = _result_check(query, rewritten_sql, db)
    if not result_ok:
        elapsed = int((time.time() - start) * 1000)
        logging.warning(f'{PILOT_LOG_PREFIX} NL Join Reordering: 결과 동치 검증 실패 — {result_reason} ({elapsed}ms)')
        return None

    # Step 4: Cost estimation
    output_cost = db.cost_estimation(rewritten_sql)
    elapsed = int((time.time() - start) * 1000)

    logging.info(f'{PILOT_LOG_PREFIX} NL Join Reordering 완료: '
                 f'output_cost={output_cost}, validation={result_reason}, time={elapsed}ms')

    return {
        'used_rules': [RULE_NAME],
        'output_sql': rewritten_sql,
        'output_cost': output_cost,
        'time': elapsed,
    }
