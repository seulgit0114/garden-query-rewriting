"""
test_utils_portfolio.py

test_utils_distaware.py의 Portfolio 버전.
기존 파일(test_utils.py / test_utils_distaware.py)은 수정하지 않는다.

logdir 기본값은 호출부(test_portfolio.py)에서 'logs_portfolio_...' 형태로 넘긴다.
"""
import logging
import os

from llama_index.core.storage.docstore import SimpleDocumentStore

from my_rewriter.database import DBArgs, Database
from my_rewriter.rag_retrieve import rag_retrieve, rag_semantics_retrieve, rag_structure_retrieve
from my_rewriter.portfolio_rewrite import (
    portfolio_rag_rewrite,
    gated_portfolio_rag_rewrite,
    PORTFOLIO_LOG_PREFIX,
    GATED_LOG_PREFIX,
)

# 쿼리별 실행 관리
def test_portfolio(
    name: str,
    query: str,
    schema: str,
    pg_args: DBArgs,
    model_args: dict,
    docstore: SimpleDocumentStore,
    LOG_DIR: str,
    pool: dict = None,
    tau: float = 0.2,
    alpha: float = 0.5,
    RETRIEVER_TOP_K: int = 5,
    CASE_BATCH: int = 5,
    RULE_BATCH: int = 10,
    REWRITE_ROUNDS: int = 1,
    index: str = 'hybrid',
    gate1_enabled: bool = True,
    gate2_enabled: bool = True,
    stat_prior_enabled: bool = True,
    stat_prior_hard_filter: bool = False,
    portfolio_enabled: bool = True,
    verify_enabled: bool = True,
    max_candidates_per_family: int = 1,
    cost_rejection_threshold: float = 1.05,
    equiv_timeout_s: int = 300,
) -> None:
    myrewriter_root = os.path.dirname(os.path.abspath(__file__))
    log_dir_path = os.path.join(myrewriter_root, LOG_DIR)
    log_filename = os.path.join(log_dir_path, f'{name}.log')

    # 이미 완료된 쿼리는 건너뜀
    if os.path.exists(log_filename):
        return

    os.makedirs(log_dir_path, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
    )

    db = Database(pg_args)
    try:
        input_cost = db.cost_estimation(query)
    finally:
        if hasattr(db, 'close'):
            db.close()
    logging.info(f'Input Cost: {input_cost}')
    logging.info(f'{PORTFOLIO_LOG_PREFIX} 쿼리 시작: name={name}, tau={tau}, '
                 f'alpha={alpha}, index={index}')

    logging.info(f'{PORTFOLIO_LOG_PREFIX} RAG retrieval 시작 (index={index})')
    if index == 'hybrid':
        res = rag_retrieve(query, schema, docstore,
                           embed_dim=model_args['EMBED_DIM'],
                           RETRIEVER_TOP_K=RETRIEVER_TOP_K,
                           fast_llm=model_args.get('FAST_LLM'))
    elif index == 'semantics':
        res = rag_semantics_retrieve(query, schema, docstore,
                                     RETRIEVER_TOP_K=RETRIEVER_TOP_K)
    elif index == 'structure':
        res = rag_structure_retrieve(query, schema, docstore,
                                     embed_dim=model_args['EMBED_DIM'],
                                     RETRIEVER_TOP_K=RETRIEVER_TOP_K)
    else:
        raise ValueError(f'Invalid index type: {index}')
    logging.info(f'{PORTFOLIO_LOG_PREFIX} RAG retrieval 완료: '
                 f'retriever_res={len(res["retriever_res"])}개, '
                 f'calcite_rules={len(res["rewrites"]["calcite"])}개, '
                 f'nl_suggestions={len(res["rewrites"]["nl"])}개')

    try:
        portfolio_rag_rewrite(
            res['retriever_res'], res['rewrites'],
            query, schema, pg_args, model_args,
            pool=pool, tau=tau, alpha=alpha,
            CASE_BATCH=CASE_BATCH, RULE_BATCH=RULE_BATCH,
            REWRITE_ROUNDS=REWRITE_ROUNDS,
            gate1_enabled=gate1_enabled,
            gate2_enabled=gate2_enabled,
            stat_prior_enabled=stat_prior_enabled,
            stat_prior_hard_filter=stat_prior_hard_filter,
            portfolio_enabled=portfolio_enabled,
            verify_enabled=verify_enabled,
            max_candidates_per_family=max_candidates_per_family,
            cost_rejection_threshold=cost_rejection_threshold,
            input_cost=input_cost,
            equiv_timeout_s=equiv_timeout_s,
        )
    except Exception as _e:
        logging.error(
            f'{PORTFOLIO_LOG_PREFIX} portfolio_rag_rewrite 예외 발생 — 쿼리 스킵: '
            f'{type(_e).__name__}: {_e}'
        )

    logging.info(f'{PORTFOLIO_LOG_PREFIX} 쿼리 종료: name={name}')
    for handler in logging.root.handlers:
        handler.flush()
    with open(log_filename, 'r') as f:
        content = f.read()
        if '[DOC_TRUNCATED]' in content:
            truncation_log = os.path.join(log_dir_path, 'truncated_queries.txt')
            with open(truncation_log, 'a') as tf:
                tf.write(f'{name}\n')


# ─── Gated Rewrite Escalation 버전 ───────────────────────────────────────────

def _has_completed_line2(log_filename: str) -> bool:
    """A query is resumable unless its log contains the final tagged result."""
    if not os.path.exists(log_filename):
        return False
    try:
        with open(log_filename, 'r') as f:
            return any(
                'Rewrite Execution Results:' in line
                and ("'_seq': 'LINE2'" in line or '"_seq": "LINE2"' in line)
                for line in f
            )
    except OSError:
        return False


def _log_original_fallback(query: str, input_cost: float, marker: str) -> None:
    """Always leave analyzable LINE2 output on an operational failure."""
    fallback = {
        'used_rules': [marker],
        'output_sql': query,
        'output_cost': input_cost if input_cost > 0 else -1,
        'time': 0,
        '_seq': 'LINE2',
    }
    logging.info(f'Rewrite Execution Results: {fallback}')


def _close_root_handlers() -> None:
    """Flush and close per-query file handlers before returning."""
    for handler in logging.root.handlers[:]:
        try:
            handler.flush()
        finally:
            handler.close()
            logging.root.removeHandler(handler)


def test_gated_portfolio(
    name: str,
    query: str,
    schema: str,
    pg_args: DBArgs,
    model_args: dict,
    docstore,
    LOG_DIR: str,
    pool: dict = None,
    tau: float = 0.2,
    alpha: float = 0.5,
    RETRIEVER_TOP_K: int = 5,
    CASE_BATCH: int = 5,
    RULE_BATCH: int = 10,
    REWRITE_ROUNDS: int = 1,
    index: str = 'hybrid',
    gate1_enabled: bool = True,
    gate2_enabled: bool = True,
    stat_prior_enabled: bool = False,
    stat_prior_hard_filter: bool = False,
    portfolio_enabled: bool = True,
    verify_enabled: bool = True,
    cost_rejection_threshold: float = 1.05,
    equiv_timeout_s: int = 300,
    # Gated escalation parameters
    calcite_success_ratio: float = 0.97,
    final_accept_ratio: float = 1.0,   # accept iff cost does not worsen (was 0.99)
    max_nl_families: int = 2,
    broad_fallback_enabled: bool = False,
    # Hybrid equivalence verification (SQLSolver + LLM judge, judge-only)
    equiv_use_solver: bool = True,
    equiv_use_llm_judge: bool = True,
    equiv_solver_timeout: int = 10,
    restore_on_regression: bool = True,
    triage_real_verify: bool = False,
    triage_timeout_s: int = 60,
    correlate_override_enabled: bool = False,
    correlate_real_verify_enabled: bool = True,
    correlate_real_verify_timeout_s: int = 300,
    dcrp_policy=None,
    dcrp_workload: str = 'calcite',
    dcrp_query_family: str = '',
    dcrp_rule_pruning_repair_enabled: bool = True,
    dcrp_repair_max_cost_ratio: float = 0.9,
    dcrp_repair_min_support: int = 5,
) -> None:
    """
    gated_portfolio_rag_rewrite() 를 구동하는 테스트 드라이버.
    test_portfolio() 와 동일한 skip-if-exists / logging 구조를 유지한다.
    """
    from my_rewriter.rag_retrieve import rag_retrieve, rag_semantics_retrieve, rag_structure_retrieve

    myrewriter_root = os.path.dirname(os.path.abspath(__file__))
    log_dir_path = os.path.join(myrewriter_root, LOG_DIR)
    log_filename = os.path.join(log_dir_path, f'{name}.log')

    if _has_completed_line2(log_filename):
        return

    os.makedirs(log_dir_path, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
    )

    db = None
    try:
        db = Database(pg_args)
        input_cost = db.cost_estimation(query)
    except Exception as _cost_e:
        input_cost = -1
        logging.error(
            f'{GATED_LOG_PREFIX} 원본 cost 측정 예외: '
            f'{type(_cost_e).__name__}: {_cost_e}'
        )
    finally:
        if db is not None and hasattr(db, 'close'):
            db.close()
    logging.info(f'Input Cost: {input_cost}')
    logging.info(
        f'{GATED_LOG_PREFIX} 쿼리 시작: name={name}, tau={tau}, alpha={alpha}, index={index}'
    )

    if input_cost <= 0:
        logging.error(f'{GATED_LOG_PREFIX} 원본 cost가 유효하지 않음 → 원본 SQL fallback')
        _log_original_fallback(query, input_cost, 'INPUT_COST_FAILED_FALLBACK')
        logging.info(f'{GATED_LOG_PREFIX} 쿼리 종료: name={name}')
        _close_root_handlers()
        return

    logging.info(f'{GATED_LOG_PREFIX} RAG retrieval 시작 (index={index})')
    try:
        if index == 'hybrid':
            res = rag_retrieve(query, schema, docstore,
                               embed_dim=model_args['EMBED_DIM'],
                               RETRIEVER_TOP_K=RETRIEVER_TOP_K,
                               fast_llm=model_args.get('FAST_LLM'))
        elif index == 'semantics':
            res = rag_semantics_retrieve(query, schema, docstore,
                                         RETRIEVER_TOP_K=RETRIEVER_TOP_K)
        elif index == 'structure':
            res = rag_structure_retrieve(query, schema, docstore,
                                         embed_dim=model_args['EMBED_DIM'],
                                         RETRIEVER_TOP_K=RETRIEVER_TOP_K)
        else:
            raise ValueError(f'Invalid index type: {index}')
    except Exception as _rag_e:
        logging.error(
            f'{GATED_LOG_PREFIX} RAG retrieval 예외 — 쿼리 스킵: '
            f'{type(_rag_e).__name__}: {_rag_e}'
        )
        _log_original_fallback(query, input_cost, 'RAG_ERROR_FALLBACK')
        logging.info(f'{GATED_LOG_PREFIX} 쿼리 종료: name={name}')
        _close_root_handlers()
        return
    logging.info(
        f'{GATED_LOG_PREFIX} RAG retrieval 완료: '
        f'retriever_res={len(res["retriever_res"])}개, '
        f'calcite_rules={len(res["rewrites"]["calcite"])}개, '
        f'nl_suggestions={len(res["rewrites"]["nl"])}개'
    )

    try:
        gated_portfolio_rag_rewrite(
            res['retriever_res'], res['rewrites'],
            query, schema, pg_args, model_args,
            pool=pool, tau=tau, alpha=alpha,
            CASE_BATCH=CASE_BATCH, RULE_BATCH=RULE_BATCH,
            REWRITE_ROUNDS=REWRITE_ROUNDS,
            gate1_enabled=gate1_enabled,
            gate2_enabled=gate2_enabled,
            stat_prior_enabled=stat_prior_enabled,
            stat_prior_hard_filter=stat_prior_hard_filter,
            portfolio_enabled=portfolio_enabled,
            verify_enabled=verify_enabled,
            cost_rejection_threshold=cost_rejection_threshold,
            input_cost=input_cost,
            equiv_timeout_s=equiv_timeout_s,
            calcite_success_ratio=calcite_success_ratio,
            final_accept_ratio=final_accept_ratio,
            max_nl_families=max_nl_families,
            broad_fallback_enabled=broad_fallback_enabled,
            equiv_use_solver=equiv_use_solver,
            equiv_use_llm_judge=equiv_use_llm_judge,
            equiv_solver_timeout=equiv_solver_timeout,
            restore_on_regression=restore_on_regression,
            triage_real_verify=triage_real_verify,
            triage_timeout_s=triage_timeout_s,
            correlate_override_enabled=correlate_override_enabled,
            correlate_real_verify_enabled=correlate_real_verify_enabled,
            correlate_real_verify_timeout_s=correlate_real_verify_timeout_s,
            dcrp_policy=dcrp_policy,
            dcrp_workload=dcrp_workload,
            dcrp_query_family=dcrp_query_family or name,
            dcrp_rule_pruning_repair_enabled=dcrp_rule_pruning_repair_enabled,
            dcrp_repair_max_cost_ratio=dcrp_repair_max_cost_ratio,
            dcrp_repair_min_support=dcrp_repair_min_support,
        )
    except Exception as _e:
        logging.error(
            f'{GATED_LOG_PREFIX} gated_portfolio_rag_rewrite 예외 — 쿼리 스킵: '
            f'{type(_e).__name__}: {_e}'
        )
        _log_original_fallback(query, input_cost, 'PIPELINE_ERROR_FALLBACK')

    logging.info(f'{GATED_LOG_PREFIX} 쿼리 종료: name={name}')
    _close_root_handlers()
    with open(log_filename, 'r') as f:
        content = f.read()
        if '[DOC_TRUNCATED]' in content:
            truncation_log = os.path.join(log_dir_path, 'truncated_queries.txt')
            with open(truncation_log, 'a') as tf:
                tf.write(f'{name}\n')
