"""
Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m pytest experiments/mvm_reliability/tests/test_dataset.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[3]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from experiments.mvm_reliability.dataset import (
    DatasetIntegrityError,
    build_snapshot,
    filter_split,
    group_usable_by_query,
    load_candidates_jsonl,
    load_split,
    sha256_file,
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


SIMPLE_SPLIT = SplitInfo(
    method='test',
    hash_spec='test',
    train_families={'calcite': frozenset({'q1'})},
    test_families={'calcite': frozenset({'q2'})},
)


class TestValidateDataset(unittest.TestCase):
    def test_duplicate_id_raises(self):
        records = [make_record(id='dup'), make_record(id='dup')]
        with self.assertRaises(DatasetIntegrityError) as ctx:
            validate_dataset(records, SIMPLE_SPLIT)
        self.assertIn('duplicate candidate id', str(ctx.exception))

    def test_unknown_workload_raises(self):
        records = [make_record(workload='not_a_real_workload')]
        with self.assertRaises(DatasetIntegrityError) as ctx:
            validate_dataset(records, SIMPLE_SPLIT)
        self.assertIn('not in allowed set', str(ctx.exception))

    def test_empty_query_family_raises(self):
        records = [make_record(query_family='')]
        with self.assertRaises(DatasetIntegrityError) as ctx:
            validate_dataset(records, SIMPLE_SPLIT)
        self.assertIn('empty query_family', str(ctx.exception))

    def test_non_finite_cost_on_usable_candidate_raises(self):
        records = [make_record(usable=True, rewritten_cost=float('nan'))]
        with self.assertRaises(DatasetIntegrityError) as ctx:
            validate_dataset(records, SIMPLE_SPLIT)
        self.assertIn('rewritten_cost', str(ctx.exception))

    def test_non_finite_cost_on_non_usable_candidate_does_not_raise(self):
        records = [make_record(usable=False, rewritten_cost=float('nan'), query_family='q_orphan')]
        # non-usable candidates are allowed to have missing/invalid values,
        # and are not required to belong to any split
        validate_dataset(records, SIMPLE_SPLIT)

    def test_valid_dataset_does_not_raise(self):
        records = [make_record(id='ok', query_family='q1')]
        validate_dataset(records, SIMPLE_SPLIT)

    def test_all_problems_reported_together(self):
        records = [make_record(id='dup'), make_record(id='dup', workload='nope')]
        with self.assertRaises(DatasetIntegrityError) as ctx:
            validate_dataset(records, SIMPLE_SPLIT)
        message = str(ctx.exception)
        self.assertIn('duplicate candidate id', message)
        self.assertIn('not in allowed set', message)


class TestChecksum(unittest.TestCase):
    def test_same_content_same_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.txt'
            path.write_text('hello world')
            h1 = sha256_file(path)
            h2 = sha256_file(path)
            self.assertEqual(h1, h2)

    def test_changed_content_changes_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.txt'
            path.write_text('hello world')
            h1 = sha256_file(path)
            path.write_text('hello world!')
            h2 = sha256_file(path)
            self.assertNotEqual(h1, h2)

    def test_real_dataset_snapshot_checksums_are_stable(self):
        records = load_candidates_jsonl(MVM_V1_DIR / 'candidates.jsonl')
        split = load_split(MVM_V1_DIR / 'split.json')
        snap1 = build_snapshot(MVM_V1_DIR / 'candidates.jsonl', MVM_V1_DIR / 'split.json', records, split)
        snap2 = build_snapshot(MVM_V1_DIR / 'candidates.jsonl', MVM_V1_DIR / 'split.json', records, split)
        self.assertEqual(snap1.candidates_sha256, snap2.candidates_sha256)
        self.assertEqual(snap1.split_sha256, snap2.split_sha256)


class TestGroupingAndSplitFilter(unittest.TestCase):
    def test_group_usable_by_query_excludes_non_usable(self):
        records = [
            make_record(id='a', usable=True, query='q1'),
            make_record(id='b', usable=False, query='q1'),
            make_record(id='c', usable=True, query='q2'),
        ]
        groups = group_usable_by_query(records)
        self.assertEqual(set(groups.keys()), {('calcite', 'q1'), ('calcite', 'q2')})
        self.assertEqual(len(groups[('calcite', 'q1')]), 1)
        self.assertEqual(groups[('calcite', 'q1')][0].id, 'a')

    def test_filter_split_train_excludes_test_rows(self):
        records = [
            make_record(id='train-row', query_family='q1', usable=True),
            make_record(id='test-row', query_family='q2', usable=True),
        ]
        train = filter_split(records, SIMPLE_SPLIT, 'train')
        test = filter_split(records, SIMPLE_SPLIT, 'test')
        self.assertEqual([r.id for r in train], ['train-row'])
        self.assertEqual([r.id for r in test], ['test-row'])

    def test_real_dataset_usable_and_split_counts(self):
        records = load_candidates_jsonl(MVM_V1_DIR / 'candidates.jsonl')
        split = load_split(MVM_V1_DIR / 'split.json')
        usable = [r for r in records if r.usable]
        train = filter_split(records, split, 'train')
        test = filter_split(records, split, 'test')
        self.assertEqual(len(records), 556)
        self.assertEqual(len(usable), 416)
        self.assertEqual(len(train) + len(test), len(usable))


if __name__ == '__main__':
    unittest.main()
