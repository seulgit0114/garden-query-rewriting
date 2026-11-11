"""
uniformity_score.py

Method 섹션의 Algorithm 1/2/3을 그대로 구현한다. 각 함수 본문의 주석은
어느 알고리즘의 몇 번째 줄에 대응하는지를 표시한다 (예: "# Alg1 L10").

  Algorithm 1  compute_uniformity_score(query, schema, pg_args, tau, alpha)
  Algorithm 2  _extract_occurrences(tree)            -- Stage 1: 쿼리 구조 파싱 & 역할 부여
  Algorithm 3  _column_skew_score(db, table, col, a)  -- Stage 2: 컬럼 단위 skew score S(c)
  (Stage 3, 4는 별도 알고리즘 없이 Algorithm 1 본문에 직접 인라인됨)

핵심 공식:
  NUS_i = W_type(role_i) * S(c_i)
  US    = sum(W_i) / sum( W_i / (1 - NUS_i + eps) )      <- 균등도(1-NUS)에 대한 가중 조화평균
  Activate <=> (1 - US) * ln(1 + Complexity) > tau
"""

import re
import math
import logging
import typing as t
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from my_rewriter.database import DBArgs, Database
from my_rewriter.distribution_stats import _parse_pg_array

EPSILON = 1e-5

ROLE_WEIGHTS = {
    'JOIN_KEY':      1.0,
    'EQ_FILTER':     0.8,
    'GROUPBY_KEY':   0.6,
    'RANGE_FILTER':  0.4,
    'ORDERBY_KEY':   0.3,
    'AGG_ARG':       0.2,
}

_IDENT_RE = re.compile(r'^[a-z_][a-z0-9_]*$')


@dataclass
class UniformityResult:
    score: float
    activate: bool
    query_complexity: int
    per_column: t.Dict[t.Tuple[str, str], dict] = field(default_factory=dict)
    skipped: t.List[str] = field(default_factory=list)
    notes: t.List[str] = field(default_factory=list)


# ── 쿼리 구조 파싱 보조 (Algorithm 1, L1-L3에서 사용) ───────────────────────

def _build_alias_map(tree: exp.Expression) -> t.Tuple[t.Dict[str, str], t.Set[str]]:
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    alias_map = {}
    for tbl in tree.find_all(exp.Table):
        real_name = tbl.name.lower()
        alias = (tbl.alias or tbl.name).lower()
        alias_map[alias] = real_name
    return alias_map, cte_names


def _parse_schema_columns(schema_sql: str) -> t.Dict[str, t.Set[str]]:
    table_cols = {}
    try:
        stmts = sqlglot.parse(schema_sql, read='postgres')
    except Exception as e:
        logging.warning(f"[uniformity_score] schema parse 실패: {e}")
        return table_cols
    for stmt in stmts:
        if not isinstance(stmt, exp.Create):
            continue
        body = stmt.this
        if isinstance(body, exp.Schema):
            name = body.this.name.lower()
            cols = {c.name.lower() for c in body.expressions if isinstance(c, exp.ColumnDef)}
            table_cols[name] = cols
    return table_cols


def _resolve_table(table_hint: t.Optional[str], col_name: str,
                    alias_map: t.Dict[str, str], cte_names: t.Set[str],
                    schema_cols: t.Dict[str, t.Set[str]],
                    query_tables: t.Set[str]) -> t.Optional[str]:
    # Alg1 L10 (ResolveTable)
    if table_hint:
        real = alias_map.get(table_hint, table_hint)
        if real in cte_names or real not in schema_cols:
            return None
        return real
    candidates = [tb for tb in query_tables if col_name in schema_cols.get(tb, set())]
    if len(candidates) == 1:
        return candidates[0]
    return None


# ── Algorithm 2: ExtractRoleOccurrences(T) ──────────────────────────────────

def _extract_occurrences(tree: exp.Expression) -> t.List[t.Tuple[t.Optional[str], str, str]]:
    occ: t.List[t.Tuple[t.Optional[str], str, str]] = []

    def col_id(c: exp.Column):
        return (c.table.lower() if c.table else None, c.name.lower())

    # Alg2 L2-L13: comparison nodes (WHERE/HAVING/JOIN-ON 전부 포함, comma-join도 잡힘)
    cmp_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    for node in tree.find_all(cmp_types):
        left, right = node.left, node.right
        is_eq = isinstance(node, exp.EQ)
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            lt, ln = col_id(left)
            rt, rn = col_id(right)
            if lt != rt:                                    # Alg2 L5-L6
                occ.append((lt, ln, 'JOIN_KEY'))
                occ.append((rt, rn, 'JOIN_KEY'))
        elif isinstance(left, exp.Column) and not isinstance(right, exp.Column):
            tb, cn = col_id(left)                           # Alg2 L7-L9
            occ.append((tb, cn, 'EQ_FILTER' if is_eq else 'RANGE_FILTER'))
        elif isinstance(right, exp.Column) and not isinstance(left, exp.Column):
            tb, cn = col_id(right)                          # Alg2 L10-L12
            occ.append((tb, cn, 'EQ_FILTER' if is_eq else 'RANGE_FILTER'))

    # Alg2 L14-L15
    for node in tree.find_all(exp.In):
        if isinstance(node.this, exp.Column):
            tb, cn = col_id(node.this)
            occ.append((tb, cn, 'EQ_FILTER'))

    for node in tree.find_all(exp.Like, exp.ILike):
        if isinstance(node.this, exp.Column):
            tb, cn = col_id(node.this)
            occ.append((tb, cn, 'EQ_FILTER'))

    for node in tree.find_all(exp.Is):
        if isinstance(node.this, exp.Column):
            tb, cn = col_id(node.this)
            occ.append((tb, cn, 'EQ_FILTER'))

    # Alg2 L17-L18
    for node in tree.find_all(exp.Between):
        if isinstance(node.this, exp.Column):
            tb, cn = col_id(node.this)
            occ.append((tb, cn, 'RANGE_FILTER'))

    # Alg2 L20-L21
    for grp in tree.find_all(exp.Group):
        for e in grp.expressions:
            if isinstance(e, exp.Column):
                tb, cn = col_id(e)
                occ.append((tb, cn, 'GROUPBY_KEY'))

    # Alg2 L23-L24
    for order in tree.find_all(exp.Order):
        for e in order.expressions:
            target = e.this if isinstance(e, exp.Ordered) else e
            if isinstance(target, exp.Column):
                tb, cn = col_id(target)
                occ.append((tb, cn, 'ORDERBY_KEY'))

    # Alg2 L26-L27
    for node in tree.find_all(exp.Sum, exp.Avg, exp.Max, exp.Min):
        if isinstance(node.this, exp.Column):
            tb, cn = col_id(node.this)
            occ.append((tb, cn, 'AGG_ARG'))

    return occ  # Alg2 L29


def _query_complexity(tree: exp.Expression) -> int:
    # Alg1 L5
    n_tables = len({t.name.lower() for t in tree.find_all(exp.Table)})
    n_joins = len(list(tree.find_all(exp.Join)))
    n_selects = len(list(tree.find_all(exp.Select)))
    n_subqueries = max(0, n_selects - 1)
    return n_tables + n_joins + n_subqueries


# ── Algorithm 3: ColumnSkewScore(table, col, alpha) ─────────────────────────

def _entropy_skew(null_frac: float, mcv_freqs: t.List[float], n_hist_buckets: int) -> float:
    # Alg3 L5-L17 (S_conc)
    probs = []
    if null_frac and null_frac > 0:
        probs.append(null_frac)
    probs.extend(f for f in mcv_freqs if f > 0)
    residual = max(0.0, 1.0 - sum(probs))
    if residual > 0:
        if n_hist_buckets > 0:
            probs.extend([residual / n_hist_buckets] * n_hist_buckets)
        else:
            probs.append(residual)

    if len(probs) <= 1:
        return 1.0
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    h_max = math.log2(len(probs))
    h_norm = (h / h_max) if h_max > 0 else 1.0
    return max(0.0, min(1.0, 1.0 - h_norm))


def _histogram_cv_skew(bounds: t.List[float]) -> t.Optional[float]:
    # Alg3 L19-L22 (S_hist)
    if len(bounds) < 3:
        return None
    widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    widths = [w for w in widths if w >= 0]
    if not widths:
        return None
    mean_w = sum(widths) / len(widths)
    if mean_w <= 0:
        return None
    var = sum((w - mean_w) ** 2 for w in widths) / len(widths)
    cv = math.sqrt(var) / mean_w
    return math.tanh(cv)


def _column_skew_score(db: Database, table: str, col: str,
                        alpha: float = 0.5) -> t.Tuple[t.Optional[float], dict]:
    # Alg3 L1: QueryPgStats(table, col)
    try:
        row = db.exec_fetch(
            f"""SELECT n_distinct, null_frac, most_common_vals::text,
                       most_common_freqs::text, histogram_bounds::text
                FROM pg_stats
                WHERE schemaname = 'public' AND tablename = '{table}' AND attname = '{col}'"""
        )
    except Exception as e:
        logging.warning(f"[uniformity_score] pg_stats 조회 실패 ({table}.{col}): {e}")
        row = None

    if row is None or all(v is None for v in row):           # Alg3 L2
        return None, {}

    n_distinct, null_frac, mcv_str, mcf_str, hist_str = row  # Alg3 L3
    mcv_freqs = [float(f) for f in _parse_pg_array(mcf_str)] if mcf_str else []
    hist_vals = _parse_pg_array(hist_str) if hist_str else []
    n_hist_buckets = max(0, len(hist_vals) - 1)

    s_conc = _entropy_skew(null_frac or 0.0, mcv_freqs, n_hist_buckets)

    numeric_bounds = []
    for v in hist_vals:
        try:
            numeric_bounds.append(float(v))
        except ValueError:
            numeric_bounds = []
            break
    s_hist = _histogram_cv_skew(numeric_bounds) if numeric_bounds else None

    if s_hist is None:                                        # Alg3 L24
        score = s_conc
    else:                                                      # Alg3 L25
        score = alpha * s_conc + (1 - alpha) * s_hist

    return score, {'n_distinct': n_distinct, 'null_frac': null_frac,
                    's_conc': round(s_conc, 4),
                    's_hist': round(s_hist, 4) if s_hist is not None else None}


# ── 운영 보조 신호 (Algorithm 1~3 범위 밖, informational only) ──────────────

def _check_staleness(db: Database, tables: t.Set[str]) -> t.Dict[str, float]:
    stale = {}
    for tb in tables:
        if not _IDENT_RE.match(tb):
            continue
        try:
            row = db.exec_fetch(
                f"SELECT n_mod_since_analyze, n_live_tup FROM pg_stat_user_tables "
                f"WHERE relname = '{tb}'"
            )
            if row and row[1]:
                stale[tb] = row[0] / max(row[1], 1)
        except Exception:
            pass
    return stale


def _check_extended_stats(db: Database, tables: t.Set[str]) -> t.Dict[str, t.List[str]]:
    found = {}
    for tb in tables:
        if not _IDENT_RE.match(tb):
            continue
        try:
            rows = db.exec_fetch(
                f"SELECT se.stxname FROM pg_statistic_ext se "
                f"WHERE se.stxrelid = '{tb}'::regclass", one=False
            )
            if rows:
                found[tb] = [r[0] for r in rows]
        except Exception:
            pass
    return found


# ── Algorithm 1: ComputeUniformityScore(Q, Sigma, tau, alpha) ───────────────

def compute_uniformity_score(
    query: str,
    schema: str,
    pg_args: DBArgs,
    # [2026-09-16] 0.3 -> 0.2. gate1_tau_sweep 의 민감도 분석 결과,
    # zipf(비균등) 대비 uniform(균등) 활성화 격차가 tau=0.20 에서 25.0%p
    # (12건 vs 1건), tau=0.30 에서 13.6%p (7건 vs 1건)였다. 0.20 은 활성 표본을
    # 7 -> 12 건으로 71% 늘리면서 uniform false positive 는 1건(2.3%)으로 유지한다.
    # 0.10 은 격차가 43.2%p 로 가장 크지만 uniform 에서도 3건이 켜져 "불필요할 때
    # 개입하지 않는다"는 성질이 약해지므로 택하지 않았다.
    tau: float = 0.2,
    alpha: float = 0.5,
) -> UniformityResult:
    # Alg1 L1
    try:
        tree = sqlglot.parse_one(query, read='postgres')
    except Exception as e:
        return UniformityResult(score=1.0, activate=False, query_complexity=0,
                                 notes=[f"parse 실패, 안전하게 비활성화: {e}"])

    # Alg1 L2-L3
    alias_map, cte_names = _build_alias_map(tree)
    schema_cols = _parse_schema_columns(schema)
    query_tables = {v for v in alias_map.values() if v not in cte_names and v in schema_cols}

    # Alg1 L4 (= Algorithm 2)
    occurrences = _extract_occurrences(tree)
    # Alg1 L5
    complexity = _query_complexity(tree)

    # Alg1 L7-L8
    pairs: t.List[t.Tuple[float, float]] = []
    cache: t.Dict[t.Tuple[str, str], t.Tuple[t.Optional[float], dict]] = {}
    skipped: t.List[str] = []
    per_column: t.Dict[t.Tuple[str, str], dict] = {}

    try:
        db = Database(pg_args)
    except Exception as e:
        return UniformityResult(score=1.0, activate=False, query_complexity=complexity,
                                 notes=[f"DB 연결 실패, 안전하게 비활성화: {e}"])

    # Alg1 L9-L22
    for table_hint, col, role in occurrences:
        table = _resolve_table(table_hint, col, alias_map, cte_names, schema_cols, query_tables)
        if table is None or not _IDENT_RE.match(table) or not _IDENT_RE.match(col):
            skipped.append(f"{table_hint or '?'}.{col} ({role}) - 미해석/CTE파생 컬럼")  # L11-L13
            continue

        key = (table, col)
        if key not in cache:                                  # L14
            cache[key] = _column_skew_score(db, table, col, alpha)  # L15 (= Algorithm 3)
        score, debug = cache[key]                              # L16

        if score is None:                                      # L17-L19
            skipped.append(f"{table}.{col} - pg_stats 없음 (ANALYZE 필요)")
            continue

        weight = ROLE_WEIGHTS[role]                             # L20
        nus = weight * score                                   # L21
        pairs.append((weight, nus))                             # L22
        entry = per_column.setdefault(key, {'roles': [], 'skew_score': score, **debug})
        entry['roles'].append(role)

    # Alg1 L24-L25
    if not pairs:
        db.close()
        return UniformityResult(score=1.0, activate=False, query_complexity=complexity,
                                 per_column=per_column, skipped=skipped,
                                 notes=["판단 가능한 컬럼 통계가 없어 균등으로 간주, 비활성화"])

    # Alg1 L27-L29: US = HM(1 - NUS_i ; W_i)
    numerator = sum(w for w, _ in pairs)
    denominator = sum(w / (1 - nus + EPSILON) for w, nus in pairs)
    us = numerator / denominator if denominator > 0 else 1.0
    us = max(0.0, min(1.0, us))

    # Alg1 L31
    activate = (1 - us) * math.log(1 + complexity) > tau

    result = UniformityResult(score=us, activate=activate, query_complexity=complexity,
                               per_column=per_column, skipped=skipped)

    # 운영 보조 신호 (점수/활성화 결정에는 영향 없음, notes로만 노출)
    stale = _check_staleness(db, {tbl for tbl, _ in per_column})
    if any(r > 0.2 for r in stale.values()):
        result.notes.append(f"통계 신선도 낮은 테이블 존재 (ANALYZE 재실행 권장): "
                             f"{[t for t, r in stale.items() if r > 0.2]}")

    ext = _check_extended_stats(db, query_tables)
    if ext:
        result.notes.append(f"확장통계(extended statistics) 존재 — 현재 점수에는 미반영: {ext}")

    db.close()
    return result  # Alg1 L32
