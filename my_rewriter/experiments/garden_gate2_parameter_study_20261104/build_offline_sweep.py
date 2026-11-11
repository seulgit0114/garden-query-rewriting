#!/usr/bin/env python3
"""Freeze a family split and build latency-blind RDG parameter candidates.

Only graph-train transitions are used to estimate edges.  Development transitions
are used for structural recall/false-expansion diagnostics.  Held-out transitions
are deliberately not scored by this script.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("GARDEN_ROOT", Path(__file__).resolve().parents[3])).resolve()
HERE = ROOT / "my_rewriter/experiments/garden_gate2_parameter_study_20261104"
INPUTS = (
    ROOT
    / "results/llm-agent/261102/garden_gate2_off_hbdc_dsb_claude46_76"
    / "inputs/queries.jsonl"
)
TRANSITIONS = str(
    ROOT / "my_rewriter/rdg_refinement/claude/data/dsb_kfold/transitions_*.jsonl"
)
SALT = "garden-gate2-parameter-study-20261104-v1"
SUPPORT_GRID = (2, 3, 5, 7)
STRENGTH_GRID = (0.20, 0.30, 0.40)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_order(family: str) -> str:
    return hashlib.sha256(f"{SALT}:{family}".encode()).hexdigest()


def freeze_split(rows: list[dict]) -> dict:
    families = sorted({row["query_family"] for row in rows}, key=family_order)
    if len(families) != 37:
        raise RuntimeError(f"expected 37 DSB families, got {len(families)}")
    partitions = {
        "graph_train": families[:22],
        "development": families[22:29],
        "held_out_test": families[29:],
    }
    query_counts = {
        name: sum(row["query_family"] in selected for row in rows)
        for name, selected in partitions.items()
    }
    return {
        "salt": SALT,
        "assignment_method": "sort families by SHA256(salt:family); slices 22/7/8",
        "family_counts": {name: len(value) for name, value in partitions.items()},
        "query_counts": query_counts,
        "partitions": partitions,
        "input_path": str(INPUTS),
        "input_sha256": sha256(INPUTS),
    }


def load_transitions() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(TRANSITIONS)):
        for line in Path(name).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("query_id"), row.get("state_id"), row.get("attempted_rule"))
            if key in seen or row.get("newly_enabled") is None:
                continue
            if row.get("termination_reason") == "roundtrip_failed":
                continue
            seen.add(key)
            rows.append(row)
    return rows


def aggregate(rows: list[dict], universe: set[str] | None = None) -> tuple[dict, set[str]]:
    if universe is None:
        universe = set()
        for row in rows:
            universe.update(row.get("enabled_before") or [])
            universe.update(row.get("enabled_after") or [])
            universe.update(row.get("newly_enabled") or [])
            universe.add(row["attempted_rule"])

    family_opportunity: dict[tuple, int] = defaultdict(int)
    family_hit: dict[tuple, int] = defaultdict(int)
    raw_hit: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        family = row["query_family"]
        source = row["attempted_rule"]
        before = set(row.get("enabled_before") or [])
        newly = set(row.get("newly_enabled") or [])
        for target in universe:
            if target == source or target in before:
                continue
            family_opportunity[(family, source, target)] += 1
            if target in newly:
                family_hit[(family, source, target)] += 1
                raw_hit[(source, target)] += 1

    stats: dict[tuple[str, str], dict] = {}
    pairs = {(source, target) for (_family, source, target) in family_opportunity}
    for source, target in pairs:
        families = [
            family
            for (family, a, b), count in family_opportunity.items()
            if a == source and b == target and count > 0
        ]
        rates = [
            family_hit.get((family, source, target), 0)
            / family_opportunity[(family, source, target)]
            for family in families
        ]
        hit_families = [
            family
            for family in families
            if family_hit.get((family, source, target), 0) > 0
        ]
        stats[(source, target)] = {
            "raw_query_support": raw_hit.get((source, target), 0),
            "family_support": len(hit_families),
            "opportunity_families": len(families),
            "family_balanced_hit": round(sum(rates), 8),
            "family_balanced_opportunity": len(rates),
            "structural_strength": round((sum(rates) + 1.0) / (len(rates) + 2.0), 8),
            "supporting_families": sorted(hit_families),
        }
    return stats, universe


def make_graph(stats: dict, universe: set[str], support: int, strength: float) -> dict:
    edges = []
    for (source, target), value in sorted(stats.items()):
        if value["family_support"] < support:
            continue
        if value["structural_strength"] < strength:
            continue
        edges.append({
            "from": source,
            "to": target,
            "reason": (
                "family-balanced enablement: "
                f"strength={value['structural_strength']:.3f}, "
                f"families={value['family_support']}/{value['opportunity_families']}"
            ),
            "observed_count": value["raw_query_support"],
            **value,
        })
    return {
        "nodes": sorted(universe),
        "edges": edges,
        "metadata": {
            "min_family_support": support,
            "min_structural_strength": strength,
            "estimator": "family_mean_with_beta_1_1",
            "split": "graph_train_only",
            "split_salt": SALT,
        },
    }


def graph_metrics(graph: dict, development_stats: dict) -> dict:
    edges = {(edge["from"], edge["to"]) for edge in graph["edges"]}
    out_degree: dict[str, int] = defaultdict(int)
    for source, _target in edges:
        out_degree[source] += 1
    degree_values = sorted(out_degree.values())

    development_hits = {
        pair for pair, value in development_stats.items() if value["family_support"] > 0
    }
    evaluable = {pair for pair in edges if pair in development_stats}
    reproduced = evaluable & development_hits
    recall = len(edges & development_hits) / len(development_hits) if development_hits else 0.0
    precision = len(reproduced) / len(evaluable) if evaluable else 0.0
    return {
        "n_edges": len(edges),
        "covered_nodes": len({node for edge in edges for node in edge}),
        "max_out_degree": max(degree_values, default=0),
        "median_out_degree": (
            degree_values[len(degree_values) // 2] if degree_values else 0
        ),
        "development_observed_edges": len(development_hits),
        "development_evaluable_graph_edges": len(evaluable),
        "development_reproduced_graph_edges": len(reproduced),
        "development_edge_recall": round(recall, 8),
        "development_edge_precision": round(precision, 8),
    }


def main() -> None:
    artifact_dir = HERE / "artifacts"
    graph_dir = artifact_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    query_rows = read_jsonl(INPUTS)
    split = freeze_split(query_rows)
    (artifact_dir / "family_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n"
    )

    transitions = load_transitions()
    train_families = set(split["partitions"]["graph_train"])
    development_families = set(split["partitions"]["development"])
    train_rows = [row for row in transitions if row["query_family"] in train_families]
    development_rows = [
        row for row in transitions if row["query_family"] in development_families
    ]
    if not train_rows or not development_rows:
        raise RuntimeError("family split did not match transition corpus")

    train_stats, universe = aggregate(train_rows)
    development_stats, _ = aggregate(development_rows, universe=universe)
    summary = []
    for support in SUPPORT_GRID:
        for strength in STRENGTH_GRID:
            graph = make_graph(train_stats, universe, support, strength)
            graph_name = f"rdg_family_s{support}_rho{int(round(strength * 100)):02d}.json"
            graph_path = graph_dir / graph_name
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
            summary.append({
                "min_family_support": support,
                "min_structural_strength": strength,
                "graph_path": str(graph_path),
                "graph_sha256": sha256(graph_path),
                **graph_metrics(graph, development_stats),
            })

    result = {
        "protocol": "latency-blind family-disjoint RDG sweep v1",
        "transition_glob": TRANSITIONS,
        "transition_rows_total": len(transitions),
        "transition_rows_graph_train": len(train_rows),
        "transition_rows_development": len(development_rows),
        "held_out_scored": False,
        "support_grid": SUPPORT_GRID,
        "strength_grid": STRENGTH_GRID,
        "graphs": summary,
    }
    result_path = artifact_dir / "offline_sweep.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    with (artifact_dir / "offline_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps({
        "split": split["family_counts"],
        "query_counts": split["query_counts"],
        "train_transition_rows": len(train_rows),
        "development_transition_rows": len(development_rows),
        "graphs": len(summary),
        "result": str(result_path),
    }, indent=2))


if __name__ == "__main__":
    main()
