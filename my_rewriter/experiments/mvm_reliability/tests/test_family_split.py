"""
Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m pytest experiments/mvm_reliability/tests/test_family_split.py -v
"""

import sys
import unittest
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[3]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.dataset import (
    DatasetIntegrityError,
    load_candidates_jsonl,
    load_split,
    validate_dataset,
)
from experiments.mvm_reliability.schemas import CandidateRecord, SplitInfo

MVM_V1_DIR = MY_REWRITER_DIR / 'query_sets' / 'mvm_v1'


def make_record(**overrides) -> CandidateRecord:
    base = dict(
        id='id-1',
        workload='calcite',
        query='q1',
        query_family='q1',
        original_sql='select 1',
        rewritten_sql='select 1',
        rules=('SOME_RULE',),
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


class TestRealDatasetFamilySplit(unittest.TestCase):
    """Test 1/2 against the real mvm_v1 dataset: no family is in both
    train and test, and every usable candidate belongs to exactly one
    split."""

    @classmethod
    def setUpClass(cls):
        cls.records = load_candidates_jsonl(MVM_V1_DIR / 'candidates.jsonl')
        cls.split = load_split(MVM_V1_DIR / 'split.json')

    def test_no_family_overlap_per_workload(self):
        for workload, train_fams in self.split.train_families.items():
            test_fams = self.split.test_families.get(workload, frozenset())
            self.assertEqual(train_fams & test_fams, frozenset(), msg=f'overlap in {workload}')

    def test_stated_family_counts_match(self):
        expected = {
            'calcite': (23, 5),
            'dsb': (26, 6),
            'imdb': (26, 6),
            'tpch': (15, 3),
        }
        for workload, (train_n, test_n) in expected.items():
            self.assertEqual(len(self.split.train_families[workload]), train_n)
            self.assertEqual(len(self.split.test_families[workload]), test_n)

    def test_real_dataset_passes_validation(self):
        # Should not raise.
        validate_dataset(self.records, self.split)

    def test_every_usable_candidate_in_exactly_one_split(self):
        for record in self.records:
            if not record.usable:
                continue
            split_name = self.split.split_of(record.workload, record.query_family)
            self.assertIn(split_name, ('train', 'test'), msg=f'candidate {record.id} has no split')


class TestSyntheticIntegrityFailures(unittest.TestCase):
    """Test 1/2 with deliberately broken synthetic inputs: validate_dataset
    must raise, not silently continue."""

    def test_overlapping_family_raises(self):
        split = SplitInfo(
            method='test',
            hash_spec='test',
            train_families={'calcite': frozenset({'fam_a', 'fam_shared'})},
            test_families={'calcite': frozenset({'fam_b', 'fam_shared'})},
        )
        records = [make_record(id='r1', query_family='fam_a')]
        with self.assertRaises(DatasetIntegrityError):
            validate_dataset(records, split)

    def test_usable_candidate_family_missing_from_both_splits_raises(self):
        split = SplitInfo(
            method='test',
            hash_spec='test',
            train_families={'calcite': frozenset({'fam_a'})},
            test_families={'calcite': frozenset({'fam_b'})},
        )
        records = [make_record(id='r1', query_family='fam_orphan', usable=True)]
        with self.assertRaises(DatasetIntegrityError):
            validate_dataset(records, split)

    def test_non_usable_candidate_missing_from_split_does_not_raise(self):
        split = SplitInfo(
            method='test',
            hash_spec='test',
            train_families={'calcite': frozenset({'fam_a'})},
            test_families={'calcite': frozenset({'fam_b'})},
        )
        records = [make_record(id='r1', query_family='fam_orphan', usable=False)]
        validate_dataset(records, split)  # should not raise


if __name__ == '__main__':
    unittest.main()
