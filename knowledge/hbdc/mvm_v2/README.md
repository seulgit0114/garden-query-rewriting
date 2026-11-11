# MVM dataset v2

Expansion successor to `mvm_v1`. The original v1 snapshot is preserved under
`baseline/`; `mvm_v1` itself is never modified.

Current pre-collection state:

- 41 candidates excluded by the old normalizer were restored because their
  source evaluations explicitly mark them `verified_on_smaller_scale` and
  `latency_eligible=true`.
- Their original `equivalent=null` values are preserved. Eligibility is
  represented separately by `evaluation_eligible` and
  `equivalence_scope=smaller_scale`.
- Expansion sources contain 8 TPC-H-schema scenario families (24 instances)
  and 24 novel JOB-Complex structural families (29 instances).
- All 53 expansion instances passed PostgreSQL `EXPLAIN (FORMAT JSON)`.

The TPC-H-schema scenarios are research workload extensions, not official
TPC-H benchmark queries. Provenance and the pinned JOB-Complex commit are in
`collection/manifests/source_queries.json`.
