"""
Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m pytest experiments/mvm_reliability/tests/test_prior.py -v
"""

import sys
import unittest
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[3]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from candidate_selection.mvm.prior import (
    NO_RULE_KEY,
    PriorTable,
    compute_rule_priors,
    compute_sequence_priors,
    is_positive,
    sequence_key,
)
from experiments.mvm_reliability.schemas import CandidateRecord


def make_record(**overrides) -> CandidateRecord:
    base = dict(
        id='id-1',
        workload='calcite',
        query='q1',
        query_family='q1',
        original_sql='select 1',
        rewritten_sql='select 1',
        rules=('RULE_A',),
        source_ablations=('both',),
        original_cost=10.0,
        rewritten_cost=5.0,
        original_latency_ms=100.0,
        rewritten_latency_ms=50.0,
        equivalent=True,
        equivalence_status='equivalent_full_scale',
        label='TP',
        usable=True,
        exclude_reason=None,
    )
    base.update(overrides)
    return CandidateRecord(**base)


class TestIsPositive(unittest.TestCase):
    def test_positive_from_latency_only_ignores_label_field(self):
        # label deliberately says FN (not positive-by-label-convention);
        # is_positive() must only look at the two latency fields.
        row = make_record(label='FN', original_latency_ms=100.0, rewritten_latency_ms=50.0)
        self.assertTrue(is_positive(row))

        row2 = make_record(label='TP', original_latency_ms=50.0, rewritten_latency_ms=100.0)
        self.assertFalse(is_positive(row2))

    def test_tie_is_not_positive(self):
        row = make_record(original_latency_ms=100.0, rewritten_latency_ms=100.0)
        self.assertFalse(is_positive(row))


class TestComputePriors(unittest.TestCase):
    def test_rule_prior_counts_and_rates(self):
        rows = [
            make_record(id='1', rules=('RULE_A',), original_latency_ms=100, rewritten_latency_ms=50),  # positive
            make_record(id='2', rules=('RULE_A',), original_latency_ms=100, rewritten_latency_ms=150),  # negative
            make_record(id='3', rules=('RULE_A',), original_latency_ms=100, rewritten_latency_ms=80),  # positive
        ]
        priors = compute_rule_priors(rows, alpha=1.0, beta=1.0)
        entry = priors['RULE_A']
        self.assertEqual(entry.count, 3)
        self.assertEqual(entry.positive, 2)
        self.assertEqual(entry.negative, 1)
        self.assertAlmostEqual(entry.success_rate, 2 / 3)
        self.assertAlmostEqual(entry.laplace_success_rate, (2 + 1) / (3 + 2))
        expected_improvement = ((100 - 50) / 100 * 100 + (100 - 150) / 100 * 100 + (100 - 80) / 100 * 100) / 3
        self.assertAlmostEqual(entry.mean_latency_improvement_pct, expected_improvement)

    def test_row_with_multiple_rules_counted_once_per_distinct_rule(self):
        rows = [make_record(id='1', rules=('RULE_A', 'RULE_B', 'RULE_A'))]
        priors = compute_rule_priors(rows)
        self.assertEqual(priors['RULE_A'].count, 1)
        self.assertEqual(priors['RULE_B'].count, 1)

    def test_sequence_key_preserves_order_and_empty_rules(self):
        self.assertEqual(sequence_key(('A', 'B')), 'A > B')
        self.assertEqual(sequence_key(('B', 'A')), 'B > A')
        self.assertEqual(sequence_key(()), NO_RULE_KEY)

    def test_sequence_prior_distinct_from_rule_prior(self):
        rows = [
            make_record(id='1', rules=('A', 'B')),
            make_record(id='2', rules=('B', 'A')),
        ]
        sequences = compute_sequence_priors(rows)
        self.assertEqual(sequences['A > B'].count, 1)
        self.assertEqual(sequences['B > A'].count, 1)


class TestPriorTableFallback(unittest.TestCase):
    def setUp(self):
        # 6 train rows for sequence 'A > B' (all positive) -> support 6
        # 6 train rows for rule 'A' alone (all positive) via a different sequence
        # workload has other rows too; global covers everything.
        self.rows = (
            [make_record(id=f'seq-{i}', rules=('A', 'B'), workload='calcite',
                          original_latency_ms=100, rewritten_latency_ms=10) for i in range(6)]
            + [make_record(id=f'rule-{i}', rules=('A',), workload='calcite',
                            original_latency_ms=100, rewritten_latency_ms=10) for i in range(6)]
            + [make_record(id=f'wl-{i}', rules=('Z',), workload='calcite',
                            original_latency_ms=100, rewritten_latency_ms=150) for i in range(6)]
            + [make_record(id=f'other-wl-{i}', rules=('Y',), workload='dsb',
                            original_latency_ms=100, rewritten_latency_ms=10) for i in range(6)]
        )
        self.table = PriorTable.build(self.rows, alpha=1.0, beta=1.0)

    def test_exact_sequence_hit(self):
        result = self.table.lookup(('A', 'B'), 'calcite', min_support=5)
        self.assertEqual(result.source, 'rule_sequence')
        self.assertEqual(result.key, 'A > B')
        self.assertEqual(result.support, 6)

    def test_falls_back_to_rule_when_sequence_unsupported(self):
        # ('A', 'B', 'A') as a sequence has never been observed -> falls
        # back to individual rule 'A', which has support 6+6=12 (rule 'A'
        # appears in both the seq-* and rule-* rows).
        result = self.table.lookup(('A', 'B', 'A'), 'calcite', min_support=5)
        self.assertEqual(result.source, 'rule')
        self.assertEqual(result.key, 'A')
        self.assertIn('not observed in train priors', result.fallback_reason)

    def test_falls_back_to_workload_when_rule_unsupported(self):
        # Rule 'Q' was never observed at all -> sequence miss, rule miss,
        # falls back to workload-level prior for 'calcite'.
        result = self.table.lookup(('Q',), 'calcite', min_support=5)
        self.assertEqual(result.source, 'workload_global')
        self.assertEqual(result.key, 'calcite')

    def test_falls_back_to_global_when_workload_unsupported(self):
        # Workload 'imdb' has zero train rows in this fixture.
        result = self.table.lookup(('Q',), 'imdb', min_support=5)
        self.assertEqual(result.source, 'global')
        self.assertIsNone(result.key)

    def test_min_support_forces_fallback_even_when_key_exists(self):
        # 'A > B' exists with support 6, but min_support=10 disqualifies it.
        result = self.table.lookup(('A', 'B'), 'calcite', min_support=10)
        self.assertNotEqual(result.source, 'rule_sequence')

    def test_none_source_when_absolutely_no_train_data(self):
        empty_table = PriorTable.build([], alpha=1.0, beta=1.0)
        result = empty_table.lookup(('A',), 'calcite', min_support=5)
        self.assertEqual(result.source, 'none')
        self.assertEqual(result.support, 0)


if __name__ == '__main__':
    unittest.main()
