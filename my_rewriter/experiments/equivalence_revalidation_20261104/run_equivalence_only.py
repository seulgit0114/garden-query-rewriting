#!/usr/bin/env python3
"""Resumable strict equivalence-only runner for the server-74 campaign."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import evaluate_arm_snapshot as protocol


PROTOCOL = "paper_strict_sqlsolver10_full300_sf1_1000_v1"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--sf1-database", required=True)
    p.add_argument("--schema-file", type=Path, required=True)
    p.add_argument("--solver-timeout", type=int, default=10)
    p.add_argument("--full-timeout", type=int, default=300)
    p.add_argument("--sf1-timeout", type=int, default=1000)
    p.add_argument("--workload", required=True)
    args = p.parse_args()

    pairs = read_jsonl(args.pairs)
    prior_rows = read_jsonl(args.output)
    prior = {
        row["pair_hash"]: row for row in prior_rows
        if row.get("protocol") == PROTOCOL
    }
    completed: list[dict] = []
    schema = args.schema_file.read_text()
    protocol.SYSTEM = "garden_paper_revalidation"
    protocol.MICRO_EVIDENCE = {}  # strict protocol: never accept micro evidence

    for position, pair in enumerate(pairs, 1):
        pair_hash = pair["pair_hash"]
        if pair_hash in prior:
            result = prior[pair_hash]
            print(f"[EQ {position}/{len(pairs)}] {pair_hash[:12]} resumed", flush=True)
        else:
            verdict = protocol.verify_pair(
                pair, args.database, args.sf1_database, schema,
                args.solver_timeout, args.full_timeout, args.sf1_timeout,
            )
            result = {
                "protocol": PROTOCOL,
                "workload": args.workload,
                "pair_hash": pair_hash,
                "query_id": pair.get("source_query_id"),
                "original_sha256": pair["original_sha256"],
                "rewrite_sha256": pair["rewrite_sha256"],
                "consumers": pair.get("consumers", []),
                **verdict,
            }
            print(
                f"[EQ {position}/{len(pairs)}] {pair_hash[:12]} "
                f"{result['sqlsolver_verdict']} {result['equivalence_status']}",
                flush=True,
            )
        completed.append(result)
        checkpoint(args.output, completed)
        counts = collections.Counter(row["equivalence_status"] for row in completed)
        atomic_json(args.output.with_name("progress.json"), {
            "protocol": PROTOCOL,
            "workload": args.workload,
            "completed": len(completed),
            "total": len(pairs),
            "status_counts": dict(counts),
            "current_pair_hash": pair_hash,
        })

    counts = collections.Counter(row["equivalence_status"] for row in completed)
    atomic_json(args.output.with_name("equivalence_summary.json"), {
        "protocol": PROTOCOL,
        "workload": args.workload,
        "total_pairs": len(completed),
        "status_counts": dict(counts),
        "sqlsolver_timeout_s": args.solver_timeout,
        "full_scale_timeout_s": args.full_timeout,
        "sf1_timeout_s": args.sf1_timeout,
        "full_scale_database": args.database,
        "sf1_database": args.sf1_database,
    })
    print(f"EQUIVALENCE_COMPLETE workload={args.workload} pairs={len(completed)}", flush=True)


if __name__ == "__main__":
    main()
