#!/usr/bin/env python3
"""Extract the final GARDEN LINE2 decision into the latency-runner schema."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

from sqlglot import exp, parse
from sqlglot.errors import ParseError


LINE2 = re.compile(r"Rewrite Execution Results: (\{.*'_seq': 'LINE2'.*\})")


def norm(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def validate(sql: str) -> tuple[bool, str | None]:
    if not sql.strip():
        return False, "empty_output"
    try:
        statements = [item for item in parse(sql, read="postgres") if item]
    except ParseError as exc:
        return False, f"parse_error: {exc}"
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return False, "not_exactly_one_query"
    return True, None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()
    rows = read_jsonl(args.inputs)
    timing_path = args.arm / "01_rewrite/query_timing.jsonl"
    timings = {row["query_id"]: row for row in read_jsonl(timing_path)}
    token_path = args.arm / "01_rewrite/token_usage.jsonl"
    tokens = read_jsonl(token_path) if token_path.exists() else []
    records = []
    for source in rows:
        query_id = source["query_id"]
        log_path = args.arm / "01_rewrite/query_logs" / f"{query_id}.log"
        matches = []
        for line in log_path.read_text(errors="replace").splitlines():
            match = LINE2.search(line)
            if match:
                matches.append(ast.literal_eval(match.group(1)))
        if not matches:
            raise RuntimeError(f"missing LINE2 for {query_id}")
        result = matches[-1]
        rewritten = str(result.get("output_sql") or "")
        parseable, parse_error = validate(rewritten)
        same = norm(rewritten) == norm(source["original_query"])
        records.append({
            "query_id": query_id,
            "query_family": source["query_family"],
            "source_group": source["source_group"],
            "original_query": source["original_query"],
            "rewritten_query": rewritten,
            "used_rules": result.get("used_rules"),
            "estimated_output_cost": result.get("output_cost"),
            "rewrite_generated": bool(parseable and not same),
            "rewrite_parseable": parseable,
            "rewrite_parse_error": parse_error,
            "rewrite_empty": not bool(rewritten.strip()),
            "rewrite_same_as_original": same,
            "rewrite_invalid": not parseable,
            "total_rewrite_time_ms": timings[query_id]["elapsed_ms"],
        })
    if len(records) != 69 or len({row["query_id"] for row in records}) != 69:
        raise RuntimeError("rewrite denominator mismatch")
    output = args.arm / "01_rewrite/rewrites.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elapsed = [float(row["total_rewrite_time_ms"]) for row in records]
    summary = {
        "model": args.model,
        "query_count": len(records),
        "generated_count": sum(row["rewrite_generated"] for row in records),
        "same_or_empty_count": sum(row["rewrite_same_as_original"] or row["rewrite_empty"] for row in records),
        "invalid_count": sum(row["rewrite_invalid"] for row in records),
        "rewrite_time_ms": {
            "sum": sum(elapsed), "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "p75": percentile(elapsed, 0.75), "p90": percentile(elapsed, 0.90),
            "p95": percentile(elapsed, 0.95),
        },
        "token_event_count": len(tokens),
    }
    (args.arm / "01_rewrite/rewrite_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
