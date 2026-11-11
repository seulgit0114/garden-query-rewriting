#!/usr/bin/env python3
"""Mine strict, measured Calcite executions into a versioned HBDC delta.

The active HBDC history is never overwritten.  Repeated executions of the
same (original SQL, rewritten SQL, ordered rule sequence, distribution) are
collapsed to a single median observation so repeated campaigns cannot inflate
support.  Micro-equivalence evidence is rejected explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
DEFAULT_RESULTS = ROOT / "results/llm-agent"
DEFAULT_OUTPUT = ROOT / "knowledge/hbdc/calcite_log_mined_v2"
BASE_CANDIDATES = ROOT / "knowledge/hbdc/mvm_v2/candidates.jsonl"
BASE_SPLIT = ROOT / "knowledge/hbdc/mvm_v2/split.json"
CALCITE_QUERIES = ROOT / "workloads/calcite/calcite.jsonl"
CROSSOVER_RESULTS = (
    ROOT
    / "my_rewriter/results/calcite_distribution"
    / "02_2026-10-13_exhaustive_rule_crossover/raw_results.json"
)

STRICT_EQUIVALENCE = {
    "formally_verified",
    "verified_on_full_scale",
    "verified_on_sf1",
    "equivalent_full_scale",
    "equivalent_sf1",
}
MIN_ABS_IMPROVEMENT_PCT = 2.0
MIN_SIGN_AGREEMENT = 2.0 / 3.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).lower()


def sql_hash(sql: str) -> str:
    return hashlib.sha256(normalize_sql(sql).encode()).hexdigest()


def distribution_from_path(path: Path) -> str:
    text = str(path).lower()
    if "zipf" in text:
        return "zipf"
    if "uniform" in text:
        return "uniform"
    return "unspecified"


def is_calcite_row(row: dict[str, Any]) -> bool:
    query_id = str(row.get("query_id") or "")
    family = str(row.get("query_family") or "")
    source = str(row.get("source_group") or "").lower()
    return (
        query_id.startswith(("base_", "added_"))
        or family.startswith("test")
        or "calcite" in source
    )


def parse_input_cost(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    match = re.search(r"Input Cost: ([0-9.eE+-]+)", log_path.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def label(original_cost: float, rewritten_cost: float,
          original_latency: float, rewritten_latency: float) -> str:
    cost_accepts = rewritten_cost < original_cost
    latency_improves = rewritten_latency < original_latency
    return {
        (True, True): "TP",
        (True, False): "FP",
        (False, True): "FN",
        (False, False): "TN",
    }[(cost_accepts, latency_improves)]


def base_calcite_hashes() -> set[str]:
    return {sql_hash(row["input_sql"]) for row in read_jsonl(CALCITE_QUERIES)}


def discover_observations(results_root: Path) -> tuple[list[dict[str, Any]], Counter]:
    observations: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for rewrite_path in sorted(results_root.glob("**/01_rewrite/rewrites.jsonl")):
        arm = rewrite_path.parent.parent
        latency_path = arm / "03_latency/latency_per_query.jsonl"
        equivalence_path = arm / "02_equivalence/equivalence.jsonl"
        if not latency_path.exists() or not equivalence_path.exists():
            skipped["campaign_missing_latency_or_equivalence"] += 1
            continue
        if "micro" in str(arm).lower():
            skipped["micro_campaign"] += 1
            continue

        rewrites = {row.get("query_id"): row for row in read_jsonl(rewrite_path)}
        latencies = {row.get("query_id"): row for row in read_jsonl(latency_path)}
        equivalences = {row.get("query_id"): row for row in read_jsonl(equivalence_path)}
        for query_id, rewrite in rewrites.items():
            if not is_calcite_row(rewrite):
                continue
            latency = latencies.get(query_id)
            equivalence = equivalences.get(query_id)
            if latency is None or equivalence is None:
                skipped["missing_joined_row"] += 1
                continue
            eq_status = str(
                equivalence.get("equivalence_status")
                or latency.get("equivalence_status")
                or ""
            )
            if "micro" in eq_status.lower():
                skipped["micro_equivalence"] += 1
                continue
            if eq_status not in STRICT_EQUIVALENCE:
                skipped[f"equivalence:{eq_status or 'missing'}"] += 1
                continue
            if not equivalence.get("adopted") or not latency.get("adopted"):
                skipped["not_adopted"] += 1
                continue
            rules = tuple(rewrite.get("used_rules") or ())
            if not rules or "DCRP_REJECT_FALLBACK" in rules:
                skipped["no_rules_or_policy_fallback"] += 1
                continue
            original_sql = str(rewrite.get("original_query") or "")
            rewritten_sql = str(rewrite.get("rewritten_query") or "")
            if (
                not original_sql
                or not rewritten_sql
                or rewrite.get("rewrite_same_as_original")
                or normalize_sql(original_sql) == normalize_sql(rewritten_sql)
            ):
                skipped["noop_or_missing_sql"] += 1
                continue
            try:
                original_latency = float(latency["original_latency_ms"])
                rewritten_latency = float(latency["rewrite_latency_ms"])
                rewritten_cost = float(rewrite["estimated_output_cost"])
            except (KeyError, TypeError, ValueError):
                skipped["missing_numeric_measurement"] += 1
                continue
            original_cost = parse_input_cost(
                rewrite_path.parent / "query_logs" / f"{query_id}.log"
            )
            if original_cost is None:
                skipped["missing_input_cost"] += 1
                continue
            if original_latency <= 0 or rewritten_latency < 0:
                skipped["invalid_latency"] += 1
                continue
            observations.append({
                "query_id": query_id,
                "query_family": str(rewrite.get("query_family") or query_id),
                "source_group": str(rewrite.get("source_group") or "unknown"),
                "source_arm": str(arm.relative_to(ROOT)),
                "distribution": distribution_from_path(arm),
                "original_sql": original_sql,
                "rewritten_sql": rewritten_sql,
                "rules": list(rules),
                "original_cost": original_cost,
                "rewritten_cost": rewritten_cost,
                "original_latency_ms": original_latency,
                "rewritten_latency_ms": rewritten_latency,
                "equivalence_status": eq_status,
                "original_timeout": bool(latency.get("original_timeout")),
                "rewrite_timeout": bool(latency.get("rewrite_timeout")),
                "evidence_partition": "modern_campaign",
            })
    return observations, skipped


def choose_rule_sequence(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Choose the most frequently attributed non-fallback rule sequence."""
    forbidden = {
        "RAG_ERROR_FALLBACK", "GATED_NO_VALID_CANDIDATE",
        "C_RULE_EQUIV_CHECK_FAILED_FALLBACK", "DCRP_REJECT_FALLBACK",
    }
    counts: Counter = Counter()
    for sequence in (candidate.get("source_rule_sequences") or {}).values():
        sequence = tuple(sequence or ())
        if sequence and not any(rule in forbidden for rule in sequence):
            counts[sequence] += 1
    if not counts:
        for sequence in candidate.get("rule_sequences") or ():
            sequence = tuple(sequence or ())
            if sequence and not any(rule in forbidden for rule in sequence):
                counts[sequence] += 1
    if not counts:
        return ()
    return max(counts, key=lambda sequence: (counts[sequence], len(sequence), sequence))


def discover_crossover_observations(path: Path) -> tuple[list[dict[str, Any]], Counter]:
    """Import the completed uniform/Zipf exhaustive rule crossover."""
    skipped: Counter = Counter()
    if not path.exists():
        skipped["crossover_file_missing"] += 1
        return [], skipped
    raw = json.loads(path.read_text())
    candidates = raw.get("historical_candidates") or {}
    measurements = raw.get("measurements") or []
    estimated_plans = raw.get("estimated_plans") or {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in measurements:
        if row.get("status") in {"ok", "timeout"}:
            grouped[(row["query"], row["candidate_id"], row["database"])].append(row)

    def selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        confirm = [row for row in rows if row.get("phase") == "historical_confirm"]
        return confirm or [row for row in rows if row.get("phase") == "historical_screen"]

    output: list[dict[str, Any]] = []
    for query, query_candidates in candidates.items():
        by_id = {candidate["id"]: candidate for candidate in query_candidates}
        original = by_id.get("original")
        if original is None:
            skipped["crossover_missing_original"] += 1
            continue
        for candidate_id, candidate in by_id.items():
            if candidate_id == "original" or candidate.get("rule_kind") != "calcite_rule":
                continue
            rules = choose_rule_sequence(candidate)
            if not rules:
                skipped["crossover_rule_ambiguous_or_empty"] += 1
                continue
            for distribution in ("uniform", "zipf"):
                original_rows = selected(grouped.get((query, "original", distribution), []))
                candidate_rows = selected(grouped.get((query, candidate_id, distribution), []))
                if not original_rows or not candidate_rows:
                    skipped["crossover_missing_measurement"] += 1
                    continue
                original_latency = statistics.median(row["latency_ms"] for row in original_rows)
                original_cost_values = [
                    row["total_cost"]
                    for row in grouped.get((query, "original", distribution), [])
                    if row.get("total_cost") is not None
                ]
                original_plan = estimated_plans.get(f"{query}::original", {}).get(distribution, {})
                original_cost = statistics.median(original_cost_values) if original_cost_values else original_plan.get("total_cost")
                candidate_cost_values = [
                    row["total_cost"]
                    for row in grouped.get((query, candidate_id, distribution), [])
                    if row.get("total_cost") is not None
                ]
                candidate_plan = estimated_plans.get(
                    f"{query}::{candidate_id}", {}
                ).get(distribution, {})
                candidate_cost = (
                    statistics.median(candidate_cost_values)
                    if candidate_cost_values else candidate_plan.get("total_cost")
                )
                if original_cost is None or candidate_cost is None:
                    skipped["crossover_missing_cost"] += 1
                    continue
                for repetition, measured in enumerate(candidate_rows, 1):
                    output.append({
                        "query_id": f"crossover_{query}_{candidate_id}_{distribution}",
                        "query_family": query,
                        "source_group": "calcite_exhaustive_rule_crossover",
                        "source_arm": str(path.relative_to(ROOT)),
                        "distribution": distribution,
                        "original_sql": original["sql"],
                        "rewritten_sql": candidate["sql"],
                        "rules": list(rules),
                        "original_cost": original_cost,
                        "rewritten_cost": float(candidate_cost),
                        "original_latency_ms": original_latency,
                        "rewritten_latency_ms": float(measured["latency_ms"]),
                        "equivalence_status": "historical_full_scale_equivalent",
                        "original_timeout": all(
                            row.get("status") == "timeout" for row in original_rows
                        ),
                        "rewrite_timeout": measured.get("status") == "timeout",
                        "evidence_partition": "historical_crossover",
                        "measurement_repetition": repetition,
                    })
    return output, skipped


def aggregate(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        key = (
            sql_hash(row["original_sql"]),
            sql_hash(row["rewritten_sql"]),
            tuple(row["rules"]),
            row["distribution"],
            row.get("evidence_partition", "modern_campaign"),
        )
        groups[key].append(row)

    output = []
    for key, rows in groups.items():
        original_latency = statistics.median(row["original_latency_ms"] for row in rows)
        rewritten_latency = statistics.median(row["rewritten_latency_ms"] for row in rows)
        original_cost = statistics.median(row["original_cost"] for row in rows)
        rewritten_cost = statistics.median(row["rewritten_cost"] for row in rows)
        improvements = [
            (row["original_latency_ms"] - row["rewritten_latency_ms"])
            / row["original_latency_ms"] * 100.0
            for row in rows
        ]
        median_improvement = statistics.median(improvements)
        median_sign = median_improvement > 0
        sign_agreement = sum((value > 0) == median_sign for value in improvements) / len(improvements)
        representative = sorted(rows, key=lambda row: row["source_arm"])[0]
        identity = "|".join((key[0], key[1], ">".join(key[2]), key[3], key[4]))
        output.append({
            "id": hashlib.sha256(("hbdc_calcite_log_mined_v2|" + identity).encode()).hexdigest(),
            "workload": "calcite",
            "query": representative["query_id"],
            "query_family": representative["query_family"],
            "original_sql": representative["original_sql"],
            "rewritten_sql": representative["rewritten_sql"],
            "rules": list(key[2]),
            "source_ablations": sorted({row["source_arm"] for row in rows}),
            "original_cost": original_cost,
            "rewritten_cost": rewritten_cost,
            "original_latency_ms": original_latency,
            "rewritten_latency_ms": rewritten_latency,
            "equivalent": True,
            "equivalence_status": "strict_logged_equivalence",
            "label": label(original_cost, rewritten_cost, original_latency, rewritten_latency),
            "usable": True,
            "exclude_reason": None,
            "distribution": key[3],
            "evidence_partition": key[4],
            "replicate_count": len(rows),
            "sign_agreement": sign_agreement,
            "median_improvement_pct": median_improvement,
            "source_groups": sorted({row["source_group"] for row in rows}),
            "source_query_ids": sorted({row["query_id"] for row in rows}),
            "source_equivalence_statuses": sorted({row["equivalence_status"] for row in rows}),
            "has_timeout_observation": any(
                row["original_timeout"] or row["rewrite_timeout"] for row in rows
            ),
            "original_sql_sha256": key[0],
            "rewritten_sql_sha256": key[1],
        })
    return sorted(output, key=lambda row: (row["query_family"], row["distribution"], row["id"]))


def pattern_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        result = {}
        for key, items in groups.items():
            labels = Counter(row["label"] for row in items)
            errors = labels["FP"] + labels["FN"]
            result[key] = {
                "count": len(items),
                "cost_error_count": errors,
                "cost_error_rate": errors / len(items),
                "labels": dict(labels),
                "median_improvement_pct": statistics.median(
                    row["median_improvement_pct"] for row in items
                ),
            }
        return dict(sorted(
            result.items(),
            key=lambda item: (-item[1]["cost_error_count"], -item[1]["count"], item[0]),
        ))

    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[" > ".join(row["rules"])].append(row)
        for rule in dict.fromkeys(row["rules"]):
            by_rule[rule].append(row)
    return {"rules": summarize(by_rule), "rule_sequences": summarize(by_sequence)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    observations, skipped = discover_observations(args.results_root)
    crossover_observations, crossover_skipped = discover_crossover_observations(
        CROSSOVER_RESULTS
    )
    observations += crossover_observations
    skipped.update(crossover_skipped)
    aggregated = aggregate(observations)
    base_hashes = base_calcite_hashes()

    for row in aggregated:
        if row["evidence_partition"] == "historical_crossover":
            row["history_role"] = "cross_family_history"
        else:
            row["history_role"] = (
                "heldout_eval" if row["original_sql_sha256"] in base_hashes else "development"
            )
        row["quality_eligible"] = (
            row["sign_agreement"] >= MIN_SIGN_AGREEMENT
            and abs(row["median_improvement_pct"]) >= MIN_ABS_IMPROVEMENT_PCT
        )

    base_rows = read_jsonl(BASE_CANDIDATES)
    base_identities = {
        (sql_hash(row["original_sql"]), sql_hash(row["rewritten_sql"]), tuple(row.get("rules") or ()))
        for row in base_rows
        if row.get("workload") == "calcite"
    }
    split = json.loads(BASE_SPLIT.read_text())
    calcite_split = split["workloads"]["calcite"]
    test_families = set(calcite_split["test_families"])
    delta = []
    for row in aggregated:
        identity = (
            row["original_sql_sha256"], row["rewritten_sql_sha256"], tuple(row["rules"])
        )
        if (
            row["history_role"] not in {"development", "cross_family_history"}
            or not row["quality_eligible"]
        ):
            continue
        if (
            row["history_role"] == "development"
            and (row["query_family"] in test_families or identity in base_identities)
        ):
            continue
        delta.append(row)

    delta_families = {row["query_family"] for row in delta}
    calcite_split["train_families"] = sorted(
        set(calcite_split["train_families"])
        | set(calcite_split["test_families"])
        | delta_families
    )
    calcite_split["test_families"] = []
    split["method"] += (
        "; strict logged Calcite candidates, distribution-deduplicated, online leave-family-out"
    )

    audit_path = args.output / "audit_all_candidates.jsonl"
    delta_path = args.output / "delta_candidates.jsonl"
    split_path = args.output / "split.json"
    patterns_path = args.output / "cost_error_patterns.json"
    write_jsonl(audit_path, aggregated)
    write_jsonl(delta_path, delta)
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    patterns_path.write_text(json.dumps(pattern_report(delta), indent=2) + "\n")

    manifest = {
        "name": "hbdc_calcite_log_mined_v2",
        "created_from": str(args.results_root),
        "base_candidates": str(BASE_CANDIDATES),
        "base_candidates_sha256": sha256(BASE_CANDIDATES),
        "base_split_sha256": sha256(BASE_SPLIT),
        "filters": {
            "strict_equivalence": sorted(STRICT_EQUIVALENCE),
            "micro_equivalence_allowed": False,
            "minimum_absolute_improvement_pct": MIN_ABS_IMPROVEMENT_PCT,
            "minimum_replicate_sign_agreement": MIN_SIGN_AGREEMENT,
            "base_calcite44_original_sql_allowed_in_delta": (
                "only from the independent exhaustive crossover; runtime leave-family-out required"
            ),
            "dedup_key": (
                "original_sql+rewritten_sql+ordered_rules+distribution+evidence_partition"
            ),
        },
        "counts": {
            "raw_observations": len(observations),
            "deduplicated_candidates": len(aggregated),
            "heldout_eval_candidates": sum(
                row["history_role"] == "heldout_eval" for row in aggregated
            ),
            "cross_family_history_candidates": sum(
                row["history_role"] == "cross_family_history" for row in aggregated
            ),
            "development_candidates_before_quality_filter": sum(
                row["history_role"] == "development" for row in aggregated
            ),
            "delta_candidates": len(delta),
            "delta_families": len(delta_families),
            "delta_labels": dict(Counter(row["label"] for row in delta)),
            "skipped": dict(skipped),
        },
        "files": {
            "audit_all_candidates": str(audit_path),
            "delta_candidates": str(delta_path),
            "split": str(split_path),
            "cost_error_patterns": str(patterns_path),
        },
        "sha256": {
            "audit_all_candidates": sha256(audit_path),
            "delta_candidates": sha256(delta_path),
            "split": sha256(split_path),
            "cost_error_patterns": sha256(patterns_path),
        },
        "paper_safety": (
            "Modern Calcite44 campaigns remain evaluation-only. Older exhaustive-crossover rows "
            "are cross-family history and require family plus normalized-SQL exclusion at runtime."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
