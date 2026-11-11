"""Shared QUITE-style final-result comparison for evaluation entry points."""
import collections
import re
import typing as t

from my_rewriter.database import DBArgs, Database


POLICY_VERSION = 'full_scale_then_smaller_scale_v2'


def _strip_semi(sql: str) -> str:
    return sql.rstrip().rstrip(';')


def _has_outer_order_by(sql: str) -> bool:
    sql_no_strings = re.sub(r"'[^']*'", '', sql)
    sql_no_strings = re.sub(r'"[^"]*"', '', sql_no_strings)
    sql_no_comments = re.sub(r'--.*?$', '', sql_no_strings, flags=re.MULTILINE)
    sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
    previous = None
    outer_sql = sql_no_comments
    while previous != outer_sql:
        previous = outer_sql
        outer_sql = re.sub(r'\([^()]*\)', '', outer_sql)
    return bool(re.search(r'\bORDER\s+BY\b', outer_sql, re.IGNORECASE))


def _make_hashable(obj):
    if isinstance(obj, dict):
        return frozenset((key, _make_hashable(value))
                         for key, value in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(_make_hashable(item) for item in obj)
    if isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    return obj


def compare_query_results(
    original_sql: str,
    rewritten_sql: str,
    db_args: DBArgs,
    timeout_s: int = 300,
    original_result_cache: t.Optional[dict] = None,
) -> t.Tuple[t.Optional[bool], str]:
    """Return True/False, or None when only the original timed out.

    None is deliberately not proof of equivalence. It lets evaluation include
    a normally completed rewrite's latency while labeling the pair unverified.
    """
    db = None
    try:
        db = Database(db_args, timeout=timeout_s)
        cache_key = _strip_semi(original_sql)
        cached = original_result_cache.get(cache_key) if original_result_cache is not None else None
        original_timed_out = isinstance(cached, dict) and cached.get('status') == 'timeout'
        if isinstance(cached, dict) and cached.get('status') == 'failed':
            return False, cached['reason']
        if isinstance(cached, dict) and cached.get('status') == 'ok':
            original_rows = cached['rows']
        elif original_timed_out:
            original_rows = None
        else:
            original_ok, original_rows, original_logs = db.execute_sql(cache_key)
            if original_ok != 1:
                reason = f'original execution failed: {str(original_logs)[:80]}'
                timeout_failure = 'timeout' in str(original_logs).lower()
                if original_result_cache is not None:
                    original_result_cache[cache_key] = {
                        'status': 'timeout' if timeout_failure else 'failed',
                        'reason': reason,
                    }
                if not timeout_failure:
                    return False, reason
                original_timed_out = True
            elif original_result_cache is not None:
                original_result_cache[cache_key] = {'status': 'ok', 'rows': original_rows}

        rewrite_ok, rewritten_rows, rewrite_logs = db.execute_sql(_strip_semi(rewritten_sql))
        if rewrite_ok != 1:
            rewrite_timed_out = 'timeout' in str(rewrite_logs).lower()
            if original_timed_out and rewrite_timed_out:
                return False, f'both_queries_timeout>{timeout_s}s'
            return False, f'rewritten execution failed: {str(rewrite_logs)[:80]}'

        if original_timed_out:
            return None, 'unverified_original_timeout'

        original_converted = [_make_hashable(row) for row in (original_rows or [])]
        rewritten_converted = [_make_hashable(row) for row in (rewritten_rows or [])]
        if _has_outer_order_by(original_sql):
            equivalent = original_converted == rewritten_converted
            mode = 'ordered'
        else:
            equivalent = (collections.Counter(original_converted)
                          == collections.Counter(rewritten_converted))
            mode = 'multiset'
        if not equivalent:
            return False, f'quite_result_mismatch[{mode}]'
        return True, f'quite_result_match[{mode},rows={len(original_converted)}]'
    except Exception as exc:
        error = str(exc)
        if 'canceling statement due to statement timeout' in error or 'timeout' in error.lower():
            return False, f'equiv_timeout>{timeout_s}s'
        return False, f'equiv_error: {error[:100]}'
    finally:
        if db is not None and hasattr(db, 'close'):
            db.close()


def evaluate_query_pair(
    original_sql: str,
    rewritten_sql: str,
    full_scale_db_args: DBArgs,
    timeout_s: int = 300,
    original_result_cache: t.Optional[dict] = None,
    smaller_scale_db_args: t.Optional[DBArgs] = None,
    smaller_scale_timeout_s: int = 300,
    smaller_scale_original_cache: t.Optional[dict] = None,
) -> dict:
    """Evaluate a final pair, using a fixed smaller DB only after timeout.

    ``latency_eligible`` is intentionally separate from
    ``execution_equivalent``: a smaller-scale match permits reporting the
    full-scale rewrite latency, but is not mislabeled as a full-scale result
    match.
    """
    full_eq, full_reason = compare_query_results(
        original_sql, rewritten_sql, full_scale_db_args, timeout_s,
        original_result_cache=original_result_cache)
    timeout_case = (
        full_eq is None
        or 'timeout' in full_reason.lower()
    )
    if not timeout_case:
        return {
            'execution_equivalent': full_eq,
            'smaller_scale_equivalent': None,
            'latency_eligible': full_eq is True,
            'equivalence_status': (
                'equivalent_full_scale' if full_eq
                else 'non_equivalent_full_scale'),
            'reason': full_reason,
        }

    if smaller_scale_db_args is None:
        return {
            'execution_equivalent': None,
            'smaller_scale_equivalent': None,
            'latency_eligible': False,
            'equivalence_status': 'smaller_scale_database_not_configured',
            'reason': full_reason,
        }

    small_eq, small_reason = compare_query_results(
        original_sql, rewritten_sql, smaller_scale_db_args,
        smaller_scale_timeout_s,
        original_result_cache=smaller_scale_original_cache)
    if small_eq is True:
        return {
            'execution_equivalent': None,
            'smaller_scale_equivalent': True,
            'latency_eligible': True,
            'equivalence_status': 'verified_on_smaller_scale',
            'reason': small_reason,
        }
    if small_eq is False and 'result_mismatch' in small_reason:
        return {
            'execution_equivalent': False,
            'smaller_scale_equivalent': False,
            'latency_eligible': False,
            'equivalence_status': 'non_equivalent_on_smaller_scale',
            'reason': small_reason,
        }
    return {
        'execution_equivalent': None,
        'smaller_scale_equivalent': small_eq,
        'latency_eligible': False,
        'equivalence_status': 'smaller_scale_verification_failed',
        'reason': small_reason,
    }
