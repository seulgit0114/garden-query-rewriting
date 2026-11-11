import typing as t
import logging
from scipy import stats
import json
import os

from my_rewriter.config import CACHE_PATH
from my_rewriter.database import Database, DBArgs
from my_rewriter.rewrite import rewrite

def execute_rewrite(query: str, schema: str, db_args: DBArgs, rule_seq: t.List[str], rounds: int) -> t.Dict:
    create_tables = [x for x in schema.split(';') if x.strip() != '']
    res = rewrite(query, create_tables, rule_seq, rounds)

    used_rules = [str(r) for r in res.rules]
    output_sql = str(res.sql)
    rewrite_time = int(res.time)
    output_cost = -1
    if output_sql != 'None':
        with Database(db_args) as db:
            output_cost = db.cost_estimation(output_sql)
    res_dict = {'used_rules': used_rules, 'output_sql': output_sql, 'output_cost': output_cost, 'time': rewrite_time}
    logging.info(f'Rewrite Execution Results: {res_dict}')
    return res_dict

def compare(a: t.List[float], b: t.List[float], alternative: str = 'greater', threshold: float = 0.1) -> bool:
    _, p_value = stats.ttest_ind(a, b, alternative=alternative)
    return p_value < threshold

# DISABLE_QUERY_CACHE=1 이면 cache/{db}.jsonl에 있는(혹은 이번 실행 중 이미 측정된)
# 값을 재사용하지 않고 매번 새로 EXPLAIN ANALYZE를 실행한다. latency 측정치가
# 이전 실행(다른 코드/데이터 상태)에서 남은 캐시로 오염되지 않았음을 보장해야 할 때 사용.
_DISABLE_QUERY_CACHE = os.environ.get('DISABLE_QUERY_CACHE') == '1'


def actual_time(sql: str, db_args: DBArgs, timeout: int) -> t.Tuple[float, t.List[float]]:
    entry = db_args.cache.get(sql)
    has_samples = (isinstance(entry, dict)
                   and isinstance(entry.get('times'), list)
                   and len(entry['times']) == 3)
    if _DISABLE_QUERY_CACHE or not has_samples:
        times = []
        for _ in range(3):
            with Database(db_args, timeout) as db:
                latency = db.pgsql_actual_time(sql)
            if latency == -1:
                return float("inf"), [float("inf")] * 3
            # A timeout is a censored observation at the configured cap, not
            # a sample that should trigger up to four more 300 s executions.
            # Stop regardless of which repetition timed out and record the
            # SQL's latency as the cap for the whole three-sample result.
            if latency >= timeout * 1000:
                times = [float(timeout * 1000)] * 3
                break
            times.append(latency)
        # The first execution is a warm-up. Report the chronological second
        # and third executions' mean (do not sort the observations).
        actual_time = sum(times[1:]) / 2
        db_args.cache[sql] = {'time': actual_time, 'times': times}
        os.makedirs(os.path.dirname(os.path.abspath(db_args.cache_path)), exist_ok=True)
        with open(db_args.cache_path, 'a') as f:
            f.write(json.dumps({'sql': sql, 'time': actual_time, 'times': times}) + '\n')
    return db_args.cache[sql]['time'], db_args.cache[sql]['times']

def actual_time_paired(
    sql_a: str, sql_b: str, db_args: DBArgs, timeout: int
) -> t.Tuple[t.Tuple[float, t.List[float]], t.Tuple[float, t.List[float]]]:
    """
    두 SQL(보통 원본 input vs 재작성 output)의 latency를 A,B,A,B,... 순서로
    번갈아 3회씩 실행해 측정한다. 첫 실행은 warm-up으로 제외하고 각 SQL의
    두 번째와 세 번째 실행 latency를 평균한다.

    actual_time()을 두 번 순차 호출(A를 3회 다 돌리고 나서 B를 3회 돌림)하면,
    나중에 측정하는 쪽이 앞선 실행들 때문에 이미 데워진 PostgreSQL 버퍼/OS 페이지
    캐시 위에서 측정되어 실제보다 유리하게 나오는 편향이 생긴다 (동일 base table을
    스캔하는 쿼리끼리 특히 심함). 두 SQL을 교차 실행하면 캐시 워밍 효과가 A/B
    어느 한쪽에만 쏠리지 않고 고르게 나뉜다.

    sql_a == sql_b(예: 재작성이 채택되지 않아 output이 원본과 동일한 경우)이면
    측정을 반으로 낭비할 필요 없이 한 번만 재고 같은 값을 양쪽에 반환한다.
    """
    if sql_a == sql_b:
        result = actual_time(sql_a, db_args, timeout)
        return result, result

    cached: t.Dict[str, t.Tuple[float, t.List[float]]] = {}
    for sql in (sql_a, sql_b):
        entry = db_args.cache.get(sql)
        # Old five-sample entries and actual_time_once() entries cannot satisfy
        # the current API, which promises exactly three samples.
        if (not _DISABLE_QUERY_CACHE and isinstance(entry, dict)
                and isinstance(entry.get('times'), list) and len(entry['times']) == 3):
            cached[sql] = (db_args.cache[sql]['time'], db_args.cache[sql]['times'])

    pending = [sql for sql in (sql_a, sql_b) if sql not in cached]
    times: t.Dict[str, t.List[float]] = {sql: [] for sql in pending}
    done: t.Dict[str, bool] = {sql: False for sql in pending}
    failed: t.Dict[str, bool] = {sql: False for sql in pending}

    for _ in range(3):
        if all(done.get(sql, True) for sql in pending):
            break
        for sql in pending:
            if done[sql]:
                continue
            with Database(db_args, timeout) as db:
                latency = db.pgsql_actual_time(sql)
            if latency == -1:
                failed[sql] = True
                done[sql] = True
            elif latency >= timeout * 1000:
                # Stop measuring only the SQL that timed out.  Its paired SQL
                # may still finish its remaining repetitions normally.
                times[sql] = [float(timeout * 1000)] * 3
                done[sql] = True
            else:
                times[sql].append(latency)

    results: t.Dict[str, t.Tuple[float, t.List[float]]] = dict(cached)
    for sql in pending:
        if failed[sql]:
            results[sql] = (float("inf"), [float("inf")] * 3)
            continue
        mean_time = sum(times[sql][1:]) / 2
        db_args.cache[sql] = {'time': mean_time, 'times': times[sql]}
        os.makedirs(os.path.dirname(os.path.abspath(db_args.cache_path)), exist_ok=True)
        with open(db_args.cache_path, 'a') as f:
            f.write(json.dumps({'sql': sql, 'time': mean_time, 'times': times[sql]}) + '\n')
        results[sql] = (mean_time, times[sql])

    return results[sql_a], results[sql_b]


def actual_time_once(sql: str, db_args: DBArgs, timeout: int) -> float:
    if _DISABLE_QUERY_CACHE or sql not in db_args.cache:
        with Database(db_args, timeout) as db:
            latency = db.pgsql_actual_time(sql)
        if latency == -1:
            return float("inf")
        db_args.cache[sql] = {'time': latency}
        os.makedirs(os.path.dirname(os.path.abspath(db_args.cache_path)), exist_ok=True)
        with open(db_args.cache_path, 'a') as f:
            f.write(json.dumps({'sql': sql, 'time': latency}) + '\n')
    return db_args.cache[sql]['time']
