# Strict equivalence revalidation (2026-11-04)

This audit maps the latency table supplied on 2026-11-04 back to preserved
rewrite artifacts and revalidates exact SQL pairs on server 74.

- `source_inventory.csv`: all 53 method/workload cells, observed statistics,
  provenance, scope, and scheduling decision.
- `inventory_summary.json`: 33 confirmed cells and 20 unresolved cells.
- `campaign_74_staging/`: normalized arm inputs and workload-deduplicated pairs.
- `remote74_results/`: periodically synchronized verification checkpoints.
- `verified_arms/`: generated only after all four workload queues complete.
- `remeasurement_manifest.csv`: generated after completion as a record of the
  verified arm inputs. No latency measurement is started automatically.
- `targeted_latency_remeasurement/`: after strict verification completes,
  measures only rewrites that previously fell back because equivalence timed
  out and are newly verified by this campaign. Previously verified pairs are
  never placed in this latency input.

Targeted latency starts only after a full canonical-original preflight for the
same workload is within 10% of the reference mean (TPC-H 90.4 s, DSB 36.1 s,
Calcite 100.3 s, and Calcite(zipf) 96.5 s). Each database has one global lock;
arms sharing a database run sequentially, while the four distinct databases
run in parallel. Every arm uses a fresh cache path with persisted query caches
disabled and follows warm-up plus forward/reverse workload-level crossover.

Remote campaign:
`${GARDEN_ROOT}/results/equivalence_revalidation_20261104`

Protocol: SQLSolver (10 s), then full-scale result comparison (300 s per SQL),
then SF=1 result comparison (1,000 s per SQL) only when full-scale execution
times out. Ordered output is compared when the outer query has `ORDER BY`;
otherwise duplicate-preserving multiset comparison is used. No micro-ensemble
or previous equivalence label is accepted as a verdict.

From checkpoint 71 of the Calcite queue, server 74 uses the dedicated
equivalence-session performance profile recorded in
`server74_equivalence_performance_profile.json`.  It enables indexed access,
larger per-query memory, NVMe-aware planner costs, and up to eight workers per
gather.  The verification timeouts and result-comparison protocol are
unchanged, and the profile is not used for latency measurement.

The deployed targeted queue contains 161 arm-query instances represented by
123 unique SQL pairs (Calcite 35, Calcite(zipf) 38, TPC-H 31, DSB 19).  A CPU
policy boundary occurred on server 74 at approximately 2026-11-05 13:19:30
KST: Calcite positions 1--8 completed before it and position 9 ran across it.
After the original 123-pair queue is complete and idle, those exact nine pair
hashes are revalidated to a separate NVMe output under the post-change CPU
policy.  Only after exact hash/protocol/cardinality checks are their verdicts
merged by `pair_hash`; all original results, consumer mappings, CPU
provenance, and the standalone prefix result are retained.

`monitor_and_collect_74.sh` never restarts the original queue.  It releases
materialization and the server-208 targeted latency queue only after the
remote `final_policy_boundary_complete.json` marker is collected and the
final 123-pair/161-instance integrity report succeeds.  Server 208's CPU
policy is outside this change and is explicitly recorded as unchanged.
