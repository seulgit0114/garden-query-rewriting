# Frozen knowledge

- `rdg/global_rdg_s2_rho04.json`: final cross-workload Rule Dependency Graph.
- `rdg/support/`: supporting rule statistics and earlier graph artifacts.
- `hbdc/mvm_v2/`: base execution-history knowledge and family-disjoint split.
- `hbdc/calcite_log_mined_v2/`: measured Calcite history delta.
- `hbdc/hbdc_calcite_sequence_guard_v1.json`: exact-sequence override.
- `distribution/`: query-relevant PostgreSQL statistics pools.

The JSON manifests may retain `${GARDEN_ROOT}` placeholders where the original
collection pipeline recorded local provenance paths. Runtime code resolves
knowledge from repository-relative paths and does not require those provenance
paths to exist.
