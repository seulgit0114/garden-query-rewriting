# MVM Reliability Prior -- Offline Evaluation

Offline, read-only comparison of two candidate-selection policies over
already-logged historical data:

1. **Cost-only baseline** -- accept a rewrite iff its estimated cost is
   lower than the original's (see `candidate_selection/cost_policy.py`).
2. **Cost + MVM reliability prior** -- the same cost decision, corrected
   by a train-only historical reliability prior over rule / rule-sequence
   history (see `candidate_selection/mvm/`).

This package never executes SQL, never calls an LLM, and never modifies
`portfolio_rewrite.py`, `analyze.py`, or anything under `query_sets/`. It
only reads `query_sets/mvm_v1/{candidates.jsonl,split.json}` and writes
new files under `experiments/mvm_reliability/outputs/`.

The MVM policy code (`candidate_selection/mvm/{prior,policy}.py`) has no
dependency on this experiment package -- it is the single implementation
a future online adapter would import too (see "Future online adapter"
below). `experiments/mvm_reliability/evaluate.py` is the only place that
imports it for offline replay.

## Data flow

```
query_sets/mvm_v1/candidates.jsonl, split.json   (read-only, canonical)
        |
        v
dataset.py: load + validate_dataset() (raises DatasetIntegrityError on any
            problem -- duplicate ids, family leakage, non-finite usable
            cost/latency, unknown workload, empty query_family, a usable
            candidate whose family is in neither split)
        |
        v
evaluate.py: train_rows = filter_split(records, split, 'train')
             prior_table = PriorTable.build(train_rows)      <- train only
             test_groups = group_usable_by_query(test split) <- by (workload, query)
             for each group: per-candidate decide(), then group-level
             cost-only / MVM / oracle selection, then latency ground truth
             attached only for scoring
        |
        v
metrics.py: candidate-level TP/FP/FN/TN + accuracy, group-level latency stats
report.py:  writes outputs/<timestamp>_<run_name>/ (atomic writes only)
```

## Grouping (multi-candidate handling)

Real inspection of `candidates.jsonl` (not assumed) found that 159 of 227
distinct `(workload, query)` pairs have 2-4 logged rewrite candidates for
the same original query. Candidates are grouped by `(workload, query)` in
`dataset.group_usable_by_query()`, and each policy picks at most one
candidate per group (or "keep original"):

- **Cost-only group pick**: among candidates the cost policy accepts
  (`rewritten_cost < original_cost`), the one with the lowest
  `rewritten_cost` -- mirrors `_select_best_candidate()`'s
  `min(all_valid, key=cost)` rule in `portfolio_rewrite.py`.
- **MVM group pick**: among candidates MVM accepts, the one with the
  highest prior reliability, cost as tiebreaker.
- **Oracle**: the candidate with the lowest actual `rewritten_latency_ms`,
  if it beats the group's original latency; otherwise "keep original".

**Caveat, stated explicitly**: candidates grouped under one original query
may come from different ablation arms (`source_ablations`: `gate1_only`,
`gate2_only`, `both`). They were not necessarily all present as
simultaneous choices in one single pipeline run. Group-based evaluation
here is therefore an approximation of "if these were the candidate pool
available", not a literal replay of one run's multi-candidate ranking.
This is why the per-candidate confusion metrics (TP/FP/FN/TN, accuracy)
are reported independently of grouping -- those hold regardless of
whether candidates were ever co-present.

A `(workload, query)` group with only one usable candidate reduces
naturally to a binary original-vs-rewrite decision -- no special-casing
is needed in the code.

## MVM policy: exact rule

Not a learned/weighted model. Two fixed overrides over the cost-only
decision, both threshold-based and both listed in
`configs/mvm_v1.json::mvm_policy`:

```
cost_decision = 'rewrite' if rewritten_cost < original_cost else 'original'   (strict)
prior = PriorTable.lookup(rules, workload, min_support)   # see fallback below

if prior.source == 'none':
    mvm_decision = cost_decision          # no train data at all -> defer to cost-only

elif cost_decision == 'rewrite' and prior.reliability < fp_guard_threshold:
    mvm_decision = 'original'             # FP guard

elif cost_decision == 'original' and prior.reliability >= fn_recovery_threshold:
    mvm_decision = 'rewrite'              # FN recovery

else:
    mvm_decision = cost_decision          # keep cost-only decision
```

Default thresholds (`configs/mvm_v1.json`): `fp_guard_threshold=0.4`,
`fn_recovery_threshold=0.6`, `min_support=5`, Laplace/Beta(1,1) smoothing.
Rationale: symmetric around the 0.5 random-guess baseline, so MVM only
overrides cost when the matching prior is clearly on the other side of
even odds, and only when there is at least `min_support` train
observations backing that prior -- a low-support prior at exactly 0.5
(e.g. 0 observations, `laplace_success_rate = 1/2`) can never trigger
either override at these thresholds. `MVMConfig.__post_init__` enforces
`0 <= fp_guard_threshold <= fn_recovery_threshold <= 1`.

### Prior fallback order

1. exact rule-sequence prior (`' > '.join(rules)`, or `'(no_rule)'`)
2. individual-rule prior -- among the candidate's distinct rules, the one
   with the most train support (highest-evidence rule, not an average
   across heterogeneous rules)
3. workload-level global prior (all train rows for that workload)
4. dataset-wide global prior (all train rows)

A step is skipped when the key has no train entry, or `count <
min_support`. Every decision records `prior_source`, `prior_key`,
`support`, `reliability`, and a human-readable `fallback_reason` string
explaining exactly which steps were tried and why each was skipped.

## Metrics

Two independent views, see `metrics.py`:

- **Candidate-level** (one row per logged rewrite candidate, test split):
  TP/FP/FN/TN and accuracy for both policies, FP-prevented / FN-recovered
  counts and rates, correct-decisions-broken count, new-FP / new-FN
  counts. Also broken down by workload, by individual rule, by full rule
  sequence, and by prior-support bucket (`0`, `1-4`, `5-9`, `10-19`,
  `20+`).
- **Group-level** (one row per original query, test split): selected
  latency sum/mean/median/p75/p95 for cost-only, MVM, original-only, and
  oracle; `oracle_gap_available_ms` (total headroom oracle has over
  cost-only, summed only over groups where oracle beats cost-only) and
  `oracle_gap_recovered_rate` (total latency MVM actually saved over
  cost-only, divided by that headroom).

## Results storage

Every run writes to a **new** directory
`outputs/<UTC-timestamp>_<run-name>/` (`report.py::make_run_dir` raises
`FileExistsError` rather than reuse or overwrite one). Contents:
`config.json`, `dataset_snapshot.json` (input file sha256 + counts + git
commit/dirty state), `rule_priors.json` (this run's recomputed prior
table, for inspection -- independent from the pre-existing
`query_sets/mvm_v1/rule_priors.json`), `candidate_decisions.jsonl` (full
per-candidate trace), `workload_metrics.json`, `summary.json`,
`report.md`, `logs/`.

`query_sets/mvm_v1/results/mvm_evaluation.json` (the existing `not_run`
stub) is **never** touched by an ordinary run. It is only overwritten when
`run_mvm_evaluation.py` is invoked with `--adopt-official`, which records
the officially-adopted run's id, output path, and headline summary.

## Fast validation (optional, not yet enabled)

Not implemented in this pass. `configs/mvm_v1.json` has no
`fast_validation` section yet; the existing `train`/`test` split from
`split.json` is used as-is (`test` split is never re-partitioned). Adding
an opt-in, train-family-only diagnostic/random-guard split is future
work and must not alter how `test` is read.

## Future online adapter connection point

Not wired in this change. `candidate_selection/mvm/adapter.py` defines
the conversion (`all_valid_entry_to_features`) from a
`portfolio_rewrite.py` `all_valid` entry (`{'family', 'cost', 'result'}`,
`result['used_rules']`) to `CandidateFeatures`, and an illustrative
`select_group_with_mvm()`. The intended attachment point is
`_select_best_candidate()` in `portfolio_rewrite.py` (currently
`min(all_valid, key=lambda x: x['cost'])`), which is not modified here.

## Known data/evaluation limitations

- Only candidates whose SQL, cost, and latency were already logged can be
  replayed; anything discarded earlier in the live pipeline is invisible
  to this dataset.
- Group-based multi-candidate evaluation approximates a candidate pool
  from candidates that may come from different ablation-arm runs (see
  "Grouping" above).
- `rules` mixes real Calcite optimizer rule names with NL-family pseudo
  tags (`C_cte`, `C_intent`, `C_join`, `C_pred`); the prior treats both
  the same way, so a "rule" prior for an NL family is really a
  family-level reliability estimate.
- Some `usable` candidates carry `FILTER_SUB_QUERY_TO_CORRELATE`-family
  rules; the on-disk `query_sets/mvm_v1/README.md` states these are
  excluded, but `build_mvm_repository.py`'s actual code includes them and
  25 such candidates were found in `candidates.jsonl` -- this
  implementation includes them like any other rule; excluding them is a
  policy decision left to a future config change, not made here.
- Test-split group sizes are small (currently 89 usable test candidates
  across a handful of groups per workload); metric breakdowns by rule or
  rule-sequence can have very low support and should be read with that in
  mind.
