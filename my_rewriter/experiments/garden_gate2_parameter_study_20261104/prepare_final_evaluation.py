#!/usr/bin/env python3
"""Freeze held-out and full-76 final evaluation after development selection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
HERE = ROOT / "my_rewriter/experiments/garden_gate2_parameter_study_20261104"
RESULT = ROOT / "results/llm-agent/261104/garden_gate2_parameter_study_dsb_claude46"
SELECTED = RESULT / "selected_configuration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_partition(name: str, inputs: Path, expected: int) -> dict:
    root = RESULT / "final_evaluation" / name
    rows = read_jsonl(inputs)
    if len(rows) != expected or len({row["query_id"] for row in rows}) != expected:
        raise RuntimeError(f"{name} denominator mismatch: {len(rows)}")
    for arm_name in ("gate2_off", "gate2_on_selected"):
        arm = root / "arms" / arm_name
        for subdir in (
            "01_rewrite/query_logs", "02_equivalence", "03_latency",
            "04_summary", "logs", "state",
        ):
            (arm / subdir).mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "root": str(root),
        "inputs": str(inputs),
        "inputs_sha256": sha256(inputs),
        "query_count": expected,
        "arms": ["gate2_off", "gate2_on_selected"],
    }


def main() -> None:
    if not SELECTED.exists():
        raise RuntimeError("development selection is not complete")
    selection = json.loads(SELECTED.read_text())
    chosen = selection["selected"]
    if not chosen.get("gate2"):
        raise RuntimeError("selected configuration is not Gate2 ON")
    graph_path = Path(chosen["graph_path"])
    if sha256(graph_path) != chosen["graph_sha256"]:
        raise RuntimeError("selected RDG graph hash changed")

    partitions = [
        prepare_partition(
            "held_out",
            RESULT / "inputs/held_out_queries.jsonl",
            16,
        ),
        prepare_partition(
            "full76",
            RESULT / "inputs/full_queries.jsonl",
            76,
        ),
    ]
    manifest = {
        "experiment": "Frozen Gate2 selected-parameter evaluation",
        "selection_path": str(SELECTED),
        "selection_sha256": sha256(SELECTED),
        "selection_partition": selection["selection_partition"],
        "held_out_gate2_outcomes_observed_at_freeze": False,
        "selected_configuration": chosen,
        "fixed_pipeline": {
            "model": "claude-sonnet-4-6",
            "gate1": True,
            "tau": 0.2,
            "alpha": 0.5,
            "hbdc": "refined correlate-aware with runtime family exclusion",
            "correlate_runtime_latency_selection": False,
            "rdg_strength_ranking": True,
            "rdg_final_scc_order": True,
            "closure_budget": chosen["closure_budget"],
            "post_rematch_budget": chosen["post_rematch_budget"],
            "post_rematch_require_edge": chosen["post_rematch_require_edge"],
        },
        "latency_protocol": {
            "passes": 3,
            "pass1": "warmup forward crossover",
            "pass2": "measured forward crossover",
            "pass3": "measured reverse crossover",
            "full_timeout_s": 300,
            "sf1_equivalence_timeout_s": 1000,
            "postgresql_server_jobs": "strictly sequential",
        },
        "partitions": partitions,
    }
    path = RESULT / "final_evaluation/final_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("refusing to alter frozen final-evaluation manifest")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
