#!/usr/bin/env python3
"""Run a strict subset and merge every checkpoint into canonical workload order."""

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


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--canonical-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--sf1-database", required=True)
    parser.add_argument("--schema-file", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--solver-timeout", type=int, default=10)
    parser.add_argument("--full-timeout", type=int, default=300)
    parser.add_argument("--sf1-timeout", type=int, default=1000)
    args = parser.parse_args()

    batch = read_jsonl(args.pairs)
    canonical = read_jsonl(args.canonical_pairs)
    canonical_by_hash = {row["pair_hash"]: row for row in canonical}
    batch_hashes = [row["pair_hash"] for row in batch]
    if len(batch_hashes) != len(set(batch_hashes)):
        raise RuntimeError("duplicate pair hash in method batch")
    unknown = set(batch_hashes) - set(canonical_by_hash)
    if unknown:
        raise RuntimeError(f"batch contains hashes outside canonical target: {sorted(unknown)[:3]}")

    prior_rows = read_jsonl(args.output)
    prior = {
        row["pair_hash"]: row
        for row in prior_rows
        if row.get("protocol") == PROTOCOL
    }
    if set(prior) - set(canonical_by_hash):
        raise RuntimeError("existing output contains a non-canonical pair")

    schema = args.schema_file.read_text()
    protocol.SYSTEM = "garden_paper_revalidation"
    protocol.MICRO_EVIDENCE = {}

    def checkpoint(current_hash: str) -> None:
        ordered = [prior[row["pair_hash"]] for row in canonical if row["pair_hash"] in prior]
        atomic_jsonl(args.output, ordered)
        counts = collections.Counter(row["equivalence_status"] for row in ordered)
        atomic_json(args.output.with_name("progress.json"), {
            "protocol": PROTOCOL,
            "workload": args.workload,
            "completed": len(ordered),
            "total": len(canonical),
            "status_counts": dict(counts),
            "current_pair_hash": current_hash,
            "method_priority_phase": args.phase,
        })

    for position, pair in enumerate(batch, 1):
        pair_hash = pair["pair_hash"]
        if pair_hash in prior:
            print(f"[METHOD {args.phase} {position}/{len(batch)}] {pair_hash[:12]} resumed", flush=True)
            continue
        verdict = protocol.verify_pair(
            pair, args.database, args.sf1_database, schema,
            args.solver_timeout, args.full_timeout, args.sf1_timeout,
        )
        prior[pair_hash] = {
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
            f"[METHOD {args.phase} {position}/{len(batch)}] {pair_hash[:12]} "
            f"{prior[pair_hash]['sqlsolver_verdict']} {prior[pair_hash]['equivalence_status']}",
            flush=True,
        )
        checkpoint(pair_hash)

    if batch:
        checkpoint(batch[-1]["pair_hash"])
    print(f"METHOD_BATCH_COMPLETE phase={args.phase} workload={args.workload} pairs={len(batch)}", flush=True)


if __name__ == "__main__":
    main()
