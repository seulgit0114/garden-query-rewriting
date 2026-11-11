#!/usr/bin/env python3
"""Summarize the development grid and freeze a parameter choice."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
HERE = ROOT / "my_rewriter/experiments/garden_gate2_parameter_study_20261104"
RESULT = ROOT / "results/llm-agent/261104/garden_gate2_parameter_study_dsb_claude46"
MANIFEST = RESULT / "campaign_manifest.json"
POST = re.compile(
    r"Post-rematch budget: discovered=(\d+), edge_supported=(\d+), kept=(\d+)"
)
REPLAY_HIT = re.compile(r"PAIRED_LLM_REPLAY_HIT key=([0-9a-f]{16})")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_arm(arm_config: dict) -> dict | None:
    arm = RESULT / "arms" / arm_config["name"]
    latency = arm / "03_latency/latency_per_query.jsonl"
    if not latency.exists():
        return None
    rows = read_jsonl(latency)
    if len(rows) != 14:
        return None
    original = [float(row["original_latency_ms"]) for row in rows]
    effective = [float(row["effective_latency_ms"]) for row in rows]
    stats = {
        key: function(effective)
        for key, function in {
            "mean_ms": statistics.mean,
            "median_ms": statistics.median,
            "p75_ms": lambda x: percentile(x, 0.75),
            "p90_ms": lambda x: percentile(x, 0.90),
            "p95_ms": lambda x: percentile(x, 0.95),
        }.items()
    }
    generated = [row for row in rows if row.get("rewrite_generated")]
    adopted = [row for row in rows if row.get("adopted")]
    regressions = [row for row in rows if float(row["effective_latency_ms"]) > float(row["original_latency_ms"])]
    improvements = [row for row in rows if float(row["effective_latency_ms"]) < float(row["original_latency_ms"])]
    expansion = []
    replay_hits = 0
    for log in sorted((arm / "01_rewrite/query_logs").glob("*.log")):
        log_text = log.read_text(errors="replace")
        matches = POST.findall(log_text)
        if matches:
            expansion.append(tuple(map(int, matches[-1])))
        replay_hits += len(REPLAY_HIT.findall(log_text))
    rewrite_summary = json.loads((arm / "01_rewrite/rewrite_summary.json").read_text())
    return {
        **arm_config,
        **stats,
        "effective_sum_ms": sum(effective),
        "original_sum_ms": sum(original),
        "effective_over_original": sum(effective) / sum(original),
        "original_improvement_pct": (sum(original) - sum(effective)) / sum(original) * 100.0,
        "generated_count": len(generated),
        "adopted_count": len(adopted),
        "equivalence_ratio": len(adopted) / len(rows),
        "regression_count": len(regressions),
        "regression_ratio": len(regressions) / len(rows),
        "improvement_count": len(improvements),
        "mean_kept_post_rules": statistics.mean(x[2] for x in expansion) if expansion else 0.0,
        "paired_llm_replay_hits": replay_hits,
        "rewrite_time_sum_ms": rewrite_summary["rewrite_time_ms"]["sum"],
        "latency_sha256": sha256(latency),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    summaries = [summarize_arm(config) for config in manifest["arms"]]
    summaries = [row for row in summaries if row is not None]
    if not summaries:
        raise RuntimeError("no completed development latency arms")
    by_name = {row["name"]: row for row in summaries}
    baseline = by_name.get("dev_gate2_off")
    if baseline is None:
        raise RuntimeError("Gate2 OFF development baseline is missing")

    for row in summaries:
        row["vs_gate2_off_mean_reduction_pct"] = (
            (baseline["mean_ms"] - row["mean_ms"]) / baseline["mean_ms"] * 100.0
        )
        row["equivalence_guard"] = (
            row["adopted_count"] >= baseline["adopted_count"] - 1
        )
        row["regression_guard"] = (
            row["regression_count"] <= baseline["regression_count"] + 1
        )
        row["eligible"] = bool(
            row.get("gate2") and row["equivalence_guard"] and row["regression_guard"]
        )

    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no Gate2 ON configuration passed the preregistered equivalence/regression guards"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["effective_over_original"],
            row["mean_kept_post_rules"],
            row["name"],
        ),
    )
    selected_record = {
        "selection_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_partition": "development_only",
        "held_out_gate2_outcomes_observed": False,
        "held_out_original_runtime_audit_observed": True,
        "selection_rule": (
            "equivalence count >= OFF-1 query; regression count <= OFF+1 query; then minimum "
            "effective/original workload sum ratio; deterministic structural-complexity tie-break"
        ),
        "baseline": baseline,
        "selected": selected,
        "development_arms_completed": len(summaries),
        "all_development_results_path": str(RESULT / "development_summary.json"),
    }
    (RESULT / "development_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    with (RESULT / "development_summary.csv").open("w", newline="") as handle:
        fields = sorted({key for row in summaries for key in row if not isinstance(row[key], (dict, list))})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    selected_path = RESULT / "selected_configuration.json"
    selected_path.write_text(json.dumps(selected_record, ensure_ascii=False, indent=2) + "\n")

    with (HERE / "PROGRESS.md").open("a") as handle:
        handle.write(
            "\n## Development selection — " + time.strftime("%Y-%m-%d %H:%M KST") + "\n\n"
            f"- Completed latency arms: {len(summaries)}\n"
            f"- Selected: `{selected['name']}`\n"
            f"- Development mean: {selected['mean_ms']/1000:.3f}s\n"
            f"- Gate2-OFF mean: {baseline['mean_ms']/1000:.3f}s\n"
            f"- Mean reduction vs OFF: {selected['vs_gate2_off_mean_reduction_pct']:.3f}%\n"
            "- Held-out latency remained sealed during selection.\n"
        )
    print(json.dumps(selected_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
