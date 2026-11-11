#!/usr/bin/env python3
"""Run one current-GARDEN arm over a frozen Calcite manifest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage


ROOT = Path(
    os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])
).resolve()
HERE = ROOT / "my_rewriter/experiments/garden_gate1_ablation_20261028"
LATEST_HBDC = ROOT / "knowledge/hbdc/calcite_log_mined_v2"
LATEST_SEQUENCE_OVERRIDES = (
    ROOT / "knowledge/hbdc/hbdc_calcite_sequence_guard_v1.json"
)
load_dotenv(ROOT / ".env", override=False)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalized_sql_hash(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gate1", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument(
        "--database", default="calcite10zipf",
        choices=("calcite10", "calcite10zipf"),
        help="Physical Calcite database used by Gate1 and candidate validation.",
    )
    parser.add_argument(
        "--correlate-selection",
        choices=("realverify", "current_hbdc", "correlate_aware_hbdc"),
        default="realverify",
        help="Selection policy for FILTER_SUB_QUERY_TO_CORRELATE candidates.",
    )
    parser.add_argument(
        "--exclude-query-family-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude the runtime Calcite (workload, query_family) from frozen "
            "HBDC history (default: enabled). Use "
            "--no-exclude-query-family-history only for explicit ablations."
        ),
    )
    parser.add_argument(
        "--model-backend",
        choices=("claude46", "gpt4o"),
        default="claude46",
        help="LLM backend. The default preserves all historical Calcite campaigns.",
    )
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help="Distribution evidence pool. The default preserves prior campaign behavior.",
    )
    parser.add_argument(
        "--hbdc-delta-candidates", type=Path,
        default=LATEST_HBDC / "delta_candidates.jsonl",
        help="Versioned history delta (default: latest strict Calcite log-mined HBDC).",
    )
    parser.add_argument(
        "--hbdc-split", type=Path, default=LATEST_HBDC / "split.json",
        help="Split covering canonical history plus the selected HBDC delta.",
    )
    parser.add_argument(
        "--hbdc-sequence-overrides", type=Path,
        default=LATEST_SEQUENCE_OVERRIDES,
        help=(
            "Workload-scoped exact-sequence HBDC knowledge. The default guards "
            "the singleton AGGREGATE_PROJECT_MERGE sequence on Calcite only."
        ),
    )
    args = parser.parse_args()
    arm = args.arm.resolve()
    inputs = args.inputs.resolve()
    rows = read_jsonl(inputs)
    if args.max_queries is not None:
        rows = rows[:args.max_queries]

    os.chdir(ROOT / "my_rewriter")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_rewriter/experiments/rbot_multillm_20261017"))

    # Freeze a mutable case-rule cache independently for each arm before
    # case_rules.py imports CASE_RULES_PATH by value.
    from my_rewriter import config

    case_cache = arm / "state/case_rules.jsonl"
    case_cache.parent.mkdir(parents=True, exist_ok=True)
    if not case_cache.exists():
        shutil.copy2(ROOT / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl", case_cache)
    config.CASE_RULES_PATH = str(case_cache)

    model_args = config.init_llms("gpt4o")  # prompts + OpenAI retrieval embedding
    from llama_index.core import Settings
    if args.model_backend == "claude46":
        from llama_index.llms.anthropic import Anthropic

        class Claude46Anthropic(Anthropic):
            @property
            def _model_kwargs(self):
                kwargs = super()._model_kwargs
                kwargs.pop("temperature", None)
                return kwargs

        selected_llm = Claude46Anthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=8192,
            timeout=600,
        )
        Settings.llm = selected_llm
        model_args["FAST_LLM"] = selected_llm
        model_name = selected_llm.model
    else:
        selected_llm = Settings.llm
        model_name = getattr(selected_llm, "model", "gpt-4o")

    # Authentication/model failures used to be swallowed by the portfolio
    # fallback and produced an all-original run that looked successful.  Make
    # one direct call before touching any query output so such a campaign
    # fails visibly and cannot advance to latency measurement.
    try:
        preflight = selected_llm.chat([
            ChatMessage(
                role="user",
                content='Return exactly this JSON object: {"garden_preflight":"ok"}',
            )
        ])
    except Exception as exc:
        raise RuntimeError(
            f"LLM preflight failed for {model_name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if "garden_preflight" not in (preflight.message.content or ""):
        raise RuntimeError("Anthropic preflight returned an unexpected response")

    token_log = arm / "01_rewrite/token_usage.jsonl"
    os.environ.update({
        "TOKEN_USAGE_LOG": str(token_log),
        "TOKEN_USAGE_SYSTEM": "garden",
        "TOKEN_USAGE_MODEL": model_name,
        "TOKEN_USAGE_WORKLOAD": f"{args.database}_{len(rows)}",
    })
    from token_usage_logger import record_response
    from my_rewriter import my_utils

    def chat_with_usage(messages, model=None):
        target = model or Settings.llm
        started = time.time()
        response = target.chat([ChatMessage(**message) for message in messages])
        record_response(response, call_kind="chat")
        logging.debug({"llm_elapsed_s": time.time() - started})
        return response.message.content

    async def achat_with_usage(messages, model=None):
        target = model or Settings.llm
        started = time.time()
        try:
            response = await target.achat([ChatMessage(**message) for message in messages])
        except (ConnectionResetError, openai.APIConnectionError):
            await asyncio.sleep(5)
            response = await target.achat([ChatMessage(**message) for message in messages])
        record_response(response, call_kind="achat")
        logging.debug({"llm_elapsed_s": time.time() - started})
        return response.message.content

    my_utils.chat = chat_with_usage
    my_utils.achat = achat_with_usage

    from my_rewriter.candidate_selection.mvm.conditional_prior import ConditionalMVMPolicy
    from my_rewriter.candidate_selection.mvm.policy import MVMConfig
    from my_rewriter.config import init_db_config
    from my_rewriter.database import DBArgs
    from my_rewriter.experiments.mvm_reliability.dataset import (
        filter_split, load_candidates_jsonl, load_split, validate_dataset,
    )
    from my_rewriter.rag_retrieve import init_docstore
    from my_rewriter.test_utils_portfolio import test_gated_portfolio

    candidates_path = ROOT / "knowledge/hbdc/mvm_v2/candidates.jsonl"
    delta_path = args.hbdc_delta_candidates.resolve() if args.hbdc_delta_candidates else None
    split_path = (
        args.hbdc_split.resolve()
        if args.hbdc_split
        else ROOT / "knowledge/hbdc/mvm_v2/split.json"
    )
    history = load_candidates_jsonl(candidates_path)
    if delta_path is not None:
        history += load_candidates_jsonl(delta_path)
    split = load_split(split_path)
    validate_dataset(history, split)
    train = filter_split(history, split, "train")
    runtime_distribution = "uniform" if args.database == "calcite10" else "zipf"
    train = [
        history_row for history_row in train
        if history_row.distribution in (None, runtime_distribution)
    ]
    if delta_path is None and len(train) != 381:
        raise RuntimeError(f"DCRP frozen train denominator mismatch: {len(train)}")
    dcrp_config = MVMConfig(
        alpha=1.0, beta=1.0, min_support=5,
        fp_guard_threshold=0.4, fn_recovery_threshold=0.6, cost_strict=True,
    )
    if args.correlate_selection == "correlate_aware_hbdc":
        from my_rewriter.experiments.garden_correlate_dcrp_dsb_20261029.correlate_aware_policy import (
            CorrelateAwareDCRPPolicy,
        )

        dcrp_policy = CorrelateAwareDCRPPolicy.build(train, dcrp_config)
    else:
        dcrp_policy = ConditionalMVMPolicy.build(train, dcrp_config)
    sequence_override_path = args.hbdc_sequence_overrides.resolve()
    from my_rewriter.candidate_selection.sequence_override import SequenceOverridePolicy
    dcrp_policy = SequenceOverridePolicy.from_json(
        dcrp_policy, sequence_override_path
    )
    dcrp_snapshot = {
        "candidate_sha256": sha256(candidates_path),
        "delta_candidate_path": str(delta_path) if delta_path else None,
        "delta_candidate_sha256": sha256(delta_path) if delta_path else None,
        "split_sha256": sha256(split_path),
        "train_count": len(train),
        "accept_train_count": dcrp_policy.accept_train_count,
        "reject_train_count": dcrp_policy.reject_train_count,
        "config": dcrp_config.__dict__,
        "runtime_workload": "calcite",
        "policy_class": type(dcrp_policy).__name__,
        "correlate_selection": args.correlate_selection,
        "exclude_query_family_history": args.exclude_query_family_history,
        "runtime_distribution": runtime_distribution,
        "distribution_context_filter": True,
        "normalized_sql_history_exclusion": True,
        "sequence_override_path": str(sequence_override_path),
        "sequence_override_sha256": sha256(sequence_override_path),
    }
    (arm / "state/dcrp_snapshot.json").write_text(
        json.dumps(dcrp_snapshot, indent=2) + "\n"
    )

    pg_args = DBArgs(init_db_config(args.database))
    schema = (ROOT / "workloads/calcite/create_tables.sql").read_text()
    pool_path = (
        args.pool_path
        if args.pool_path is not None
        else ROOT / f"knowledge/distribution/{args.database}_pool.json"
    ).resolve()
    pool = json.loads(pool_path.read_text())
    docstore = init_docstore()
    log_dir = arm / "01_rewrite/query_logs"
    timing_path = arm / "01_rewrite/query_timing.jsonl"
    prior_timings = {
        row["query_id"]: row for row in (read_jsonl(timing_path) if timing_path.exists() else [])
    }

    run_config = {
        "workload": "calcite_uniform" if args.database == "calcite10" else "calcite_zipf",
        "database": args.database,
        "gate1_enabled": args.gate1,
        "gate2_enabled": True,
        "tau": 0.2,
        "alpha": 0.5,
        "rdg_final_scc_order": os.environ.get("RDG_FINAL_SCC_ORDER"),
        "rdg_path": os.environ.get("RDG_PATH"),
        "rdg_sha256": (
            sha256(Path(os.environ["RDG_PATH"]))
            if os.environ.get("RDG_PATH")
            else None
        ),
        "rdg_max_closure_additions": os.environ.get("RDG_MAX_CLOSURE_ADDITIONS"),
        "rdg_post_rematch_max_additions": os.environ.get("RDG_POST_REMATCH_MAX_ADDITIONS"),
        "rdg_post_rematch_require_edge": os.environ.get("RDG_POST_REMATCH_REQUIRE_EDGE"),
        "rdg_strength_ranking": os.environ.get("RDG_STRENGTH_RANKING"),
        "dcrp_enabled": True,
        "hbdc_enabled": True,
        "hbdc_policy": type(dcrp_policy).__name__,
        "hbdc_base_candidate_path": str(candidates_path),
        "hbdc_base_candidate_sha256": sha256(candidates_path),
        "hbdc_delta_candidate_path": str(delta_path) if delta_path else None,
        "hbdc_delta_candidate_sha256": sha256(delta_path) if delta_path else None,
        "hbdc_split_path": str(split_path),
        "hbdc_split_sha256": sha256(split_path),
        "hbdc_sequence_override_path": str(sequence_override_path),
        "hbdc_sequence_override_sha256": sha256(sequence_override_path),
        "hbdc_train_count": len(train),
        "correlate_selection": args.correlate_selection,
        "exclude_query_family_history": args.exclude_query_family_history,
        "dcrp_rule_pruning_repair_enabled": True,
        "dcrp_repair_max_cost_ratio": 0.9,
        "dcrp_repair_min_support": 5,
        "correlate_real_verify_enabled": args.correlate_selection == "realverify",
        "correlate_real_verify_timeout_s": 300,
        "model": model_name,
        "model_backend": args.model_backend,
        "query_count": len(rows),
        "pool_path": str(pool_path),
        "pool_sha256": sha256(pool_path),
    }
    (arm / "state/rewrite_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

    family_policy_cache = {}

    def policy_for_family(query_family: str, original_sql: str):
        if not args.exclude_query_family_history:
            return dcrp_policy
        original_hash = normalized_sql_hash(original_sql)
        cache_key = (query_family, original_hash)
        if cache_key in family_policy_cache:
            return family_policy_cache[cache_key]
        family_train = [
            history_row for history_row in train
            if not (
                history_row.workload == "calcite"
                and (
                    history_row.query_family == query_family
                    or normalized_sql_hash(history_row.original_sql) == original_hash
                )
            )
        ]
        if args.correlate_selection == "correlate_aware_hbdc":
            from my_rewriter.experiments.garden_correlate_dcrp_dsb_20261029.correlate_aware_policy import (
                CorrelateAwareDCRPPolicy,
            )

            policy = CorrelateAwareDCRPPolicy.build(family_train, dcrp_config)
        else:
            policy = ConditionalMVMPolicy.build(family_train, dcrp_config)
        policy = SequenceOverridePolicy.from_json(
            policy, sequence_override_path
        )
        family_policy_cache[cache_key] = policy
        append_jsonl(arm / "state/hbdc_family_exclusions.jsonl", {
            "query_family": query_family,
            "original_sql_sha256": original_hash,
            "excluded_train_rows": len(train) - len(family_train),
            "remaining_train_rows": len(family_train),
            "policy_class": type(policy).__name__,
        })
        return policy

    for position, row in enumerate(rows, 1):
        query_id = row["query_id"]
        existing_log = log_dir / f"{query_id}.log"
        if query_id in prior_timings and existing_log.exists() and "'_seq': 'LINE2'" in existing_log.read_text(errors="replace"):
            print(f"[QUERY {position}/{len(rows)}] {query_id} RESUME_SKIP", flush=True)
            continue
        os.environ["TOKEN_USAGE_QUERY_ID"] = query_id
        os.environ["TOKEN_USAGE_STAGE"] = "rewrite"
        started = time.monotonic()
        query_dcrp_policy = policy_for_family(
            row["query_family"], row["original_query"]
        )
        print(f"[QUERY {position}/{len(rows)}] {query_id} START", flush=True)
        test_gated_portfolio(
            query_id, row["original_query"], schema, pg_args, model_args, docstore,
            str(log_dir), pool=pool, tau=0.2, alpha=0.5,
            RETRIEVER_TOP_K=5, CASE_BATCH=5, RULE_BATCH=10, REWRITE_ROUNDS=1,
            index="hybrid", gate1_enabled=args.gate1, gate2_enabled=True,
            stat_prior_enabled=False, stat_prior_hard_filter=False,
            portfolio_enabled=True, verify_enabled=True,
            cost_rejection_threshold=1.05, equiv_timeout_s=300,
            calcite_success_ratio=0.97, final_accept_ratio=1.0,
            max_nl_families=2, broad_fallback_enabled=False,
            equiv_use_solver=True, equiv_use_llm_judge=True,
            equiv_solver_timeout=10, restore_on_regression=True,
            triage_real_verify=False, triage_timeout_s=60,
            correlate_override_enabled=False,
            correlate_real_verify_enabled=args.correlate_selection == "realverify",
            correlate_real_verify_timeout_s=300,
            dcrp_policy=query_dcrp_policy, dcrp_workload="calcite",
            dcrp_query_family=row["query_family"],
            dcrp_rule_pruning_repair_enabled=True,
            dcrp_repair_max_cost_ratio=0.9,
            dcrp_repair_min_support=5,
        )
        query_log = log_dir / f"{query_id}.log"
        fatal_fallbacks = ("RAG_ERROR_FALLBACK", "PIPELINE_ERROR_FALLBACK")
        log_text = query_log.read_text(errors="replace") if query_log.exists() else ""
        detected_fallback = next(
            (marker for marker in fatal_fallbacks if marker in log_text), None
        )
        if detected_fallback:
            raise RuntimeError(
                f"{detected_fallback} detected for {query_id}; "
                "refusing to record an all-original rewrite result"
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        append_jsonl(timing_path, {
            "query_id": query_id, "position": position, "elapsed_ms": elapsed_ms,
            "gate1_enabled": args.gate1,
            "correlate_selection": args.correlate_selection,
            "exclude_query_family_history": args.exclude_query_family_history,
        })
        print(f"[QUERY {position}/{len(rows)}] {query_id} DONE elapsed_ms={elapsed_ms:.1f}", flush=True)
    print("REWRITE_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
