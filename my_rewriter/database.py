import psycopg2
import json
import logging
import typing as t
import os

from my_rewriter.config import CACHE_PATH

class DBArgs(object):

    def __init__(self, config: t.Dict[str, str]):
        self.dbtype = config['db']
        if self.dbtype == 'postgresql':
            self.host = config['host']
            self.port = config['port']
            self.user = config['user']
            self.password = config['password']
            self.dbname = config['dbname']
            self.driver = 'org.postgresql.Driver'
            self.jdbc = 'jdbc:postgresql://'
        else:
            raise NotImplementedError

        self.cache = {}
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_path = os.environ.get(
            'QUERY_LATENCY_CACHE',
            os.path.join(project_root, CACHE_PATH, f'{self.dbname}.jsonl'),
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        # Long-running benchmark campaigns may reuse a database name after the
        # data or PostgreSQL state has changed.  In that case, start this
        # process with an empty in-memory cache while still allowing fresh
        # measurements to be reused within the job and appended for auditing.
        ignore_persisted = os.environ.get('IGNORE_PERSISTED_QUERY_CACHE') == '1'
        if os.path.exists(self.cache_path) and not ignore_persisted:
            with open(self.cache_path, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    self.cache[obj['sql']] = obj

class Database():

    def __init__(self, args: DBArgs, timeout: int = -1, enable_indexscan: bool = False):
        self.args = args
        self.conn = None
        self.timeout = timeout
        self.resetConn(timeout)
        self.enable_indexscan = enable_indexscan

    def resetConn(self, timeout: int = -1):
        self.close()
        if timeout > 0:
            self.conn = psycopg2.connect(database=self.args.dbname,
                                        user=self.args.user,
                                        password=self.args.password,
                                        host=self.args.host,
                                        port=self.args.port,
                                        options=f'-c statement_timeout={timeout}s')
        elif timeout == 0:
            # An omitted option inherits a database/role/server default, which
            # may still impose a timeout.  The SF=1 exact-result validation
            # explicitly requests timeout=0 and must therefore force
            # PostgreSQL's unlimited setting on this connection.
            self.conn = psycopg2.connect(database=self.args.dbname,
                                        user=self.args.user,
                                        password=self.args.password,
                                        host=self.args.host,
                                        port=self.args.port,
                                        options='-c statement_timeout=0')
        else:
            self.conn = psycopg2.connect(database=self.args.dbname,
                                        user=self.args.user,
                                        password=self.args.password,
                                        host=self.args.host,
                                        port=self.args.port)

    def close(self):
        """Close the PostgreSQL connection owned by this instance."""
        conn = getattr(self, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def exec_fetch(self, statement: str, one: bool = True):
        with self.conn.cursor() as cur:
            cur.execute(statement)
            if one:
                return cur.fetchone()
            return cur.fetchall()

    def execute_sql(self, sql: str):
        success = 0
        i = 0
        cnt = 3
        res = None
        logs = []
        with self.conn.cursor() as cur:
            while success == 0 and i < cnt:
                try:
                    if not self.enable_indexscan:
                        cur.execute("SET enable_indexscan = off;")
                    cur.execute(sql)
                    success = 1
                except Exception as e:
                    logs.append(e)
                    if 'canceling statement due to statement timeout' in str(e):
                        try:
                            self.conn.rollback()
                        except Exception:
                            pass
                        return success, res, logs
                    # PostgreSQL aborts the transaction after an error. Roll it
                    # back before the next retry so the real statement can run.
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                if success == 1:
                    res = cur.fetchall()
                i = i + 1
        return success, res, logs

    def pgsql_cost_estimation(self, sql: str):
        success, res, logs = self.execute_sql('explain (FORMAT JSON) ' + sql)
        if success == 1:
            cost = res[0][0][0]['Plan']['Total Cost']
            return cost
        else:
            logging.error(f'Failed to execute pgsql_cost_estimation {sql[:300]}{"...(truncated)" if len(sql) > 300 else ""}\n' + str(logs))
            return -1

    def pgsql_actual_time(self, sql: str):
        success, res, logs = self.execute_sql('explain (FORMAT JSON, analyze) ' + sql)
        if success == 1:
            return res[0][0][0]['Plan']['Actual Total Time']
        else:
            if 'canceling statement due to statement timeout' in str(logs):
                # self.timeout (초) 에서 끊겼다고 가정하고 ms 로 환산한다.
                # 이 인스턴스가 statement_timeout 을 걸지 않은 경우
                # (self.timeout <= 0 -> resetConn 이 옵션을 생략) 타임아웃은
                # 서버/DB/롤 레벨 설정에서 온 것이므로 임계값을 알 수 없다.
                # 그 상태에서 self.timeout * 1000 을 돌려주면 음수 latency 가
                # 측정값으로 섞여 들어가므로, 실패 sentinel 을 반환한다.
                if self.timeout > 0:
                    return self.timeout * 1000
                logging.error(
                    'Statement timeout fired on a connection created without an '
                    'explicit timeout (self.timeout=%r); the threshold is unknown, '
                    'so no latency can be reported for %s%s',
                    self.timeout, sql[:300],
                    '...(truncated)' if len(sql) > 300 else '')
                return -1
            logging.error(f'Failed to execute pgsql_actual_time {sql[:300]}{"...(truncated)" if len(sql) > 300 else ""}\n' + str(logs))
            return -1

    # query cost estimated by the optimizer
    def cost_estimation(self, sql: str):
        if self.args.dbtype == 'postgresql':
            return self.pgsql_cost_estimation(sql)
        else:
            raise NotImplementedError
