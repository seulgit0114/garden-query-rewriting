"""
equivalence_hybrid.py

Hybrid semantic-equivalence verification for LLM-generated rewrite candidates,
ported from QUITE's Hybrid_SQL_Corrector (SQLSolver + LLM).

Design (QUITE-compatible equivalence decision):

    ① SQLSolver (formal, *sound*)   sqlsolver-v1.1.0.jar (z3-backed)
         EQ -> accept immediately
         every non-EQ outcome (NEQ / UNKNOWN / timeout / quoted identifier) -> ②
    ② LLM equivalence judge
         tagged content containing "true"/"True" -> accept
         otherwise -> reject and let the caller's correction loop retry

Judge-only in this module: the LLM returns a True/False verdict here —
verify_equivalence_hybrid() itself never rewrites the candidate or retries.
The correction loop lives one level up, in portfolio_rewrite.py's
_generate_and_verify_with_retry() / _gen_corrective_retry(): on an
equiv_fail verdict from this module, the caller feeds the rejected SQL + the
rejection reason back to the LLM and regenerates, up to max_equiv_retries
times (current gated default 1) — QUITE §5.2 "Equivalence Error Correction", ported at the
generation layer rather than into this (still single-shot, sound-first)
verifier.

The caller retains the user-added extension of feeding concrete equivalence/alias
failure reasons into the corrective rewrite prompt.  This is intentionally kept
on top of QUITE's equivalence decision policy.

Both C_rule and LLM-generated NL candidates are routed here by the gated
pipeline. Expensive execution-result comparison is intentionally deferred to
the final LINE2 evaluation step.

The LLM is called through my_rewriter.my_utils.chat() (sync, Settings.llm =
ollama gpt-oss:20b), so no model object needs to be threaded in.
"""
import os
import re
import hashlib
import tempfile
import subprocess
import logging
import typing as t

# ── Asset locations ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOLVER_DIR = os.path.join(_HERE, 'sqlsolver')
_JAR_PATH = os.path.join(_SOLVER_DIR, 'sqlsolver-v1.1.0.jar')

# Verdict constants
EQ, NEQ, UNKNOWN = 'EQ', 'NEQ', 'UNKNOWN'

# Cache: schema-content hash -> temp schema file path (avoid rewriting per call)
_schema_file_cache: t.Dict[str, str] = {}


def _java_bin() -> str:
    """Resolve the java binary (prefer $JAVA_HOME, as set by the Slurm job)."""
    jh = os.environ.get('JAVA_HOME')
    if jh:
        cand = os.path.join(jh, 'bin', 'java')
        if os.path.exists(cand):
            return cand
    return 'java'


def _schema_to_path(schema: str) -> str:
    """Write the schema DDL string to a (cached) temp file for sqlsolver -schema=."""
    h = hashlib.md5(schema.encode('utf-8')).hexdigest()
    path = _schema_file_cache.get(h)
    if path and os.path.exists(path):
        return path
    fd, path = tempfile.mkstemp(suffix='.sql', prefix='eqschema_')
    with os.fdopen(fd, 'w') as f:
        f.write(schema)
    _schema_file_cache[h] = path
    return path


# ── ① SQLSolver (formal) ──────────────────────────────────────────────────────
def sqlsolver_verdict(original_sql: str, rewritten_sql: str, schema: str,
                      timeout: int = 10, verbose: bool = False) -> str:
    """
    Run sqlsolver-v1.1.0.jar and return one of EQ / NEQ / UNKNOWN.

    Ported from QUITE's Equivalence_Check_Tool (middleware.py). Double-quoted
    identifiers are conservatively routed to UNKNOWN (SQLSolver would verify a
    different query than the one executed). Timeout / crash -> UNKNOWN.
    """
    if not os.path.exists(_JAR_PATH):
        logging.warning(f'[EQUIV] sqlsolver jar not found at {_JAR_PATH} -> UNKNOWN')
        return UNKNOWN

    # Quoted identifiers carry semantics SQLSolver would strip -> route to LLM/backstop.
    if '"' in original_sql or '"' in rewritten_sql:
        return UNKNOWN

    # libz3.so / libz3java.so live next to the jar.
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = _SOLVER_DIR + os.pathsep + env.get('LD_LIBRARY_PATH', '')

    schema_path = _schema_to_path(schema)
    s1 = original_sql.replace('\n', ' ')
    s2 = rewritten_sql.replace('\n', ' ')

    sql1_path = sql2_path = out_path = None
    proc = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f1:
            f1.write(s1); sql1_path = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f2:
            f2.write(s2); sql2_path = f2.name
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as fo:
            out_path = fo.name

        cmd = [
            _java_bin(), '-jar', _JAR_PATH,
            f'-sql1={sql1_path}', f'-sql2={sql2_path}',
            f'-schema={schema_path}', f'-output={out_path}',
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)
        try:
            _stdout, _stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if verbose:
                logging.info(f'[EQUIV] sqlsolver timeout>{timeout}s -> UNKNOWN')
            return UNKNOWN

        if proc.returncode != 0:
            if verbose:
                logging.info(f'[EQUIV] sqlsolver rc={proc.returncode} -> UNKNOWN')
            return UNKNOWN

        with open(out_path, 'r') as f:
            result = f.read()

        # Strict token parse: "EQ" / "NEQ" as standalone tokens. A substring test
        # would misclassify "NEQ" as EQ.
        tokens = re.findall(r'[A-Z_]+', (result or '').upper())
        if NEQ in tokens:
            return NEQ
        if EQ in tokens:
            return EQ
        return UNKNOWN
    except Exception as e:
        logging.warning(f'[EQUIV] sqlsolver error: {str(e)[:120]} -> UNKNOWN')
        return UNKNOWN
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        for p in (sql1_path, sql2_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ── ② LLM equivalence judge (judge-only; single yes/no, no correction loop) ────
def _llm_equiv_judge(original_sql: str, rewritten_sql: str) -> t.Tuple[bool, str]:
    """
    Ask the LLM (Settings.llm) whether the two SQLs are semantically equivalent.
    Judge-only: a single True/False verdict — no SQL rewriting, no retry loop.

    Ported (judge portion only) from QUITE's DecisionAgent.check_equivalence.

    Returns (is_equivalent, analysis). The judge prompt already asks for a
    brief <analysis> of *why* the queries diverge; previously that text was
    parsed out and thrown away, so a rejection collapsed to the bare label
    "unknown_llm_reject" with no information for the corrective retry prompt
    to act on. Surfacing it here lets the caller feed the judge's actual
    reasoning back into _gen_corrective_retry instead of a content-free tag.
    """
    from my_rewriter.my_utils import chat
    from my_rewriter.prompts import EQUIV_CHECK_SYS_PROMPT, EQUIV_CHECK_USER_PROMPT
    msgs = [
        {'role': 'system', 'content': EQUIV_CHECK_SYS_PROMPT},
        {'role': 'user', 'content': EQUIV_CHECK_USER_PROMPT.format(
            original_sql=original_sql,
            rewritten_sql=rewritten_sql,
        )},
    ]
    resp = chat(msgs) or ''
    # Prefer the normal XML-style opening tag, while retaining compatibility
    # with QUITE/legacy DeepSeek responses that used a closing tag as the
    # opener: </equivalence>True</equivalence>.
    m = re.search(
        r'<(?:/)?equivalence>\s*(true|false)\s*</equivalence>',
        resp,
        re.IGNORECASE | re.DOTALL,
    )
    verdict_txt = m.group(1).strip() if m else ''
    is_equiv = verdict_txt.lower() == 'true'

    a = re.search(
        r'<(?:/)?analysis>\s*(.*?)\s*</analysis>',
        resp,
        re.IGNORECASE | re.DOTALL,
    )
    analysis = a.group(1).strip()[:500] if a else ''
    return is_equiv, analysis


# ── ③ Orchestration ───────────────────────────────────────────────────────────
def verify_equivalence_hybrid(
    original_sql: str,
    rewritten_sql: str,
    db_args,
    schema: str,
    result_compare_fn: t.Optional[t.Callable[..., t.Tuple[bool, str]]] = None,
    *,
    use_solver: bool = True,
    use_llm_judge: bool = True,
    solver_timeout: int = 10,
    equiv_timeout_s: int = 300,
) -> t.Tuple[bool, str]:
    """
    Layered equivalence verdict for one (original, rewritten) pair. Returns
    (is_equivalent, reason). Judge-only — the candidate SQL is never modified.

        ① SQLSolver: EQ -> accept immediately
        ② every non-EQ outcome -> LLM judgment
        ③ LLM True -> accept for the rewrite phase

    ``result_compare_fn`` is retained only for source compatibility with older
    callers.  Runtime result comparison intentionally does *not* happen here:
    it is performed once, for the final LINE2 candidate, by the evaluation
    pipeline.  This keeps expensive benchmark queries out of candidate search.
    """
    # ① SQLSolver (sound)
    verdict = sqlsolver_verdict(original_sql, rewritten_sql, schema, timeout=solver_timeout) \
        if use_solver else UNKNOWN
    if verdict == EQ:
        return True, 'solver_eq'

    # QUITE does not distinguish NEQ from UNKNOWN here: every verdict other than
    # EQ enters the DecisionAgent equivalence/correction loop.
    if use_llm_judge:
        try:
            is_equiv, analysis = _llm_equiv_judge(original_sql, rewritten_sql)
        except Exception as e:
            logging.warning(f'[EQUIV] LLM judge error: {str(e)[:120]}')
            is_equiv, analysis = False, ''
        if is_equiv:
            return True, f'{verdict.lower()}_llm_accept'
        reason = f'{verdict.lower()}_llm_reject'
        if analysis:
            reason += f': {analysis}'
        return False, reason

    # QUITE always has the DecisionAgent layer.  If it is explicitly disabled in
    # this project's ablation CLI, a non-EQ result cannot be approved.
    return False, f'{verdict.lower()}_llm_disabled'
