"""
distribution_stats.py

쿼리의 WHERE 절에 등장하는 컬럼들에 대해 PostgreSQL pg_stats에서
데이터 분포 정보를 추출하고 자연어로 변환한다.
"""
import re
import typing as t
import logging

from my_rewriter.database import DBArgs, Database


def _extract_where_columns(query: str) -> t.List[t.Tuple[t.Optional[str], str]]:
    """
    쿼리 전체에서 비교 연산자(<, >, =, LIKE, IN ...) 왼쪽의 컬럼 참조를 추출한다.
    서브쿼리가 있어도 모든 WHERE 조건을 커버하기 위해 WHERE 절 제한 없이 전체 스캔.
    """
    # 주석 및 문자열 리터럴 제거 (false match 방지)
    q = re.sub(r"--[^\n]*", " ", query)
    q = re.sub(r"'[^']*'", " ? ", q)
    q = re.sub(r'"[^"]*"', " ? ", q)

    # table.column 또는 column 단독으로 비교 연산자 앞에 오는 패턴
    col_pattern = re.compile(
        r'\b([a-z_][a-z0-9_]*)(?:\.([a-z_][a-z0-9_]*))?\s*'
        r'(?:=|!=|<>|<=|>=|<|>|\bLIKE\b|\bIN\b|\bNOT\s+IN\b|\bIS\b)',
        re.IGNORECASE
    )

    results = []
    for m in col_pattern.finditer(q):
        part1, part2 = m.group(1).lower(), m.group(2)
        if part2:
            results.append((part1, part2.lower()))   # (table_or_alias, column)
        else:
            results.append((None, part1))            # (None, column)
    return results


def _parse_pg_array(arr_str: str) -> t.List[str]:
    """'{val1,val2,...}' 형태의 PostgreSQL 배열 문자열을 리스트로 파싱한다."""
    if not arr_str or arr_str in ('{}', ''):
        return []
    arr_str = arr_str.strip('{}')
    elements, current, in_quote = [], [], False
    for c in arr_str:
        if c == '"':
            in_quote = not in_quote
        elif c == ',' and not in_quote:
            elements.append(''.join(current).strip('"'))
            current = []
        else:
            current.append(c)
    if current:
        elements.append(''.join(current).strip('"'))
    return elements


def get_distribution_info(query: str, pg_args: DBArgs, top_k: int = 5) -> str:
    """
    쿼리 WHERE 절 컬럼들의 분포 정보를 자연어 문자열로 반환한다.

    반환 예시:
        [Data Distribution Statistics]
        - orders.o_clerk: 2 distinct values
          * value='Clerk#000000001': 45.0% of rows
          * value='Clerk#000000002': 45.0% of rows
    """
    col_refs = _extract_where_columns(query)
    if not col_refs:
        return ""

    try:
        db = Database(pg_args)
    except Exception as e:
        logging.warning(f"[distribution_stats] DB 연결 실패: {e}")
        return ""

    lines = ["[Data Distribution Statistics]"]
    seen: t.Set[str] = set()

    for (table_hint, col_name) in col_refs:
        if col_name in seen:
            continue
        seen.add(col_name)

        # table_hint는 alias일 수 있으므로 컬럼명만으로 조회
        # (alias 해소 없이 tablename으로 필터하면 결과가 0건이 됨)
        where = f"attname = '{col_name}'"

        try:
            rows = db.exec_fetch(
                f"""SELECT tablename, attname, n_distinct,
                           most_common_vals::text, most_common_freqs::text
                    FROM pg_stats
                    WHERE schemaname = 'public' AND {where}
                    ORDER BY tablename""",
                one=False
            )
        except Exception as e:
            logging.warning(f"[distribution_stats] pg_stats 조회 실패 ({col_name}): {e}")
            continue

        if not rows:
            continue

        for row in rows:
            tname, cname, n_distinct, mcv_str, mcf_str = row

            # n_distinct 해석: 양수=정확한 개수, 음수=전체 행 대비 비율
            if n_distinct is None:
                continue
            elif n_distinct >= 0:
                distinct_desc = f"{int(n_distinct)} distinct values"
            else:
                distinct_desc = f"~{int(abs(n_distinct) * 100)}% rows are unique (high cardinality)"

            lines.append(f"- {tname}.{cname}: {distinct_desc}")

            if mcv_str and mcf_str:
                vals = _parse_pg_array(mcv_str)[:top_k]
                freqs = _parse_pg_array(mcf_str)[:top_k]
                for v, f in zip(vals, freqs):
                    try:
                        pct = float(f) * 100
                        lines.append(f"  * value='{v}': {pct:.1f}% of rows")
                    except ValueError:
                        pass

    if len(lines) == 1:   # 헤더만 있고 실제 정보 없음
        db.close()
        return ""

    db.close()
    return "\n".join(lines)
