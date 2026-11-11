"""
Run from ${GARDEN_ROOT}/my_rewriter:
    conda run -n rbot310 python3 -m pytest experiments/mvm_reliability/tests/test_policy.py -v
"""

import dataclasses
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

MY_REWRITER_DIR = Path(__file__).resolve().parents[3]
if str(MY_REWRITER_DIR) not in sys.path:
    sys.path.insert(0, str(MY_REWRITER_DIR))

from candidate_selection.cost_policy import CostOnlyPolicy
from candidate_selection.interfaces import CandidateFeatures
from candidate_selection.mvm.policy import MVMConfig, MVMPolicy
from candidate_selection.mvm.prior import PriorTable
from experiments.mvm_reliability.evaluate import (
    ExperimentConfig,
    build_group_result,
    decide_candidate,
    run_evaluation,
)
from experiments.mvm_reliability.dataset import load_candidates_jsonl, load_split
from experiments.mvm_reliability.metrics import summarize_candidates
from experiments.mvm_reliability.report import make_run_dir
from experiments.mvm_reliability.schemas import CandidateRecord

MVM_V1_DIR = MY_REWRITER_DIR / 'query_sets' / 'mvm_v1'


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


class TestCandidateFeaturesHasNoGroundTruth(unittest.TestCase):
    def test_no_latency_or_label_fields(self):
        field_names = {f.name for f in dataclasses.fields(CandidateFeatures)}
        for name in field_names:
            self.assertNotIn('latency', name)
            self.assertNotIn('label', name)

    def test_rejects_non_finite_cost(self):
        with self.assertRaises(ValueError):
            CandidateFeatures(
                candidate_id='x', workload='calcite', query='q', query_family='q',
                rules=(), original_cost=float('nan'), rewritten_cost=1.0,
            )


class TestCostOnlyPolicy(unittest.TestCase):
    def _features(self, original_cost, rewritten_cost):
        return CandidateFeatures(
            candidate_id='x', workload='calcite', query='q', query_family='q',
            rules=(), original_cost=original_cost, rewritten_cost=rewritten_cost,
        )

    def test_strict_tie_is_original(self):
        policy = CostOnlyPolicy(strict=True)
        self.assertEqual(policy.decide(self._features(10.0, 10.0)), 'original')

    def test_non_strict_tie_is_rewrite(self):
        policy = CostOnlyPolicy(strict=False)
        self.assertEqual(policy.decide(self._features(10.0, 10.0)), 'rewrite')

    def test_lower_cost_is_rewrite(self):
        policy = CostOnlyPolicy(strict=True)
        self.assertEqual(policy.decide(self._features(10.0, 9.99)), 'rewrite')

    def test_higher_cost_is_original(self):
        policy = CostOnlyPolicy(strict=True)
        self.assertEqual(policy.decide(self._features(10.0, 10.01)), 'original')


class TestMVMPolicyOverrides(unittest.TestCase):
    def setUp(self):
        # 10 train rows, all rule 'UNRELIABLE', mostly negative -> low reliability
        unreliable_rows = (
            [make_record(id=f'u-pos-{i}', rules=('UNRELIABLE',), original_latency_ms=100, rewritten_latency_ms=50)
             for i in range(1)]
            + [make_record(id=f'u-neg-{i}', rules=('UNRELIABLE',), original_latency_ms=100, rewritten_latency_ms=150)
               for i in range(9)]
        )
        # 10 train rows, all rule 'RELIABLE', mostly positive -> high reliability
        reliable_rows = (
            [make_record(id=f'r-pos-{i}', rules=('RELIABLE',), original_latency_ms=100, rewritten_latency_ms=50)
             for i in range(9)]
            + [make_record(id=f'r-neg-{i}', rules=('RELIABLE',), original_latency_ms=100, rewritten_latency_ms=150)
               for i in range(1)]
        )
        self.prior_table = PriorTable.build(unreliable_rows + reliable_rows, alpha=1.0, beta=1.0)
        self.config = MVMConfig(min_support=5, fp_guard_threshold=0.4, fn_recovery_threshold=0.6)
        self.policy = MVMPolicy(self.prior_table, self.config)

    def _features(self, rules, original_cost, rewritten_cost):
        return CandidateFeatures(
            candidate_id='x', workload='calcite', query='q', query_family='q',
            rules=rules, original_cost=original_cost, rewritten_cost=rewritten_cost,
        )

    def test_fp_guard_overrides_unreliable_rewrite_to_original(self):
        # cost says rewrite (cost lower), but rule is historically unreliable
        features = self._features(('UNRELIABLE',), original_cost=10.0, rewritten_cost=5.0)
        decision = self.policy.decide_detailed(features)
        self.assertEqual(decision.decision, 'original')
        self.assertIn('fp_guard', decision.reason)
        self.assertTrue(decision.reliability < self.config.fp_guard_threshold)

    def test_fn_recovery_overrides_reliable_original_to_rewrite(self):
        # cost says original (cost not lower), but rule is historically reliable
        features = self._features(('RELIABLE',), original_cost=10.0, rewritten_cost=10.0)
        decision = self.policy.decide_detailed(features)
        self.assertEqual(decision.decision, 'rewrite')
        self.assertIn('fn_recovery', decision.reason)
        self.assertTrue(decision.reliability >= self.config.fn_recovery_threshold)

    def test_no_prior_evidence_defers_to_cost_decision(self):
        features = self._features(('NEVER_SEEN_RULE_XYZ',), original_cost=10.0, rewritten_cost=5.0)
        # workload/global prior exists from setUp's train rows, so this
        # will actually fall back rather than hit 'none' -- use a
        # policy built from an empty prior table to force 'none'.
        empty_policy = MVMPolicy(PriorTable.build([]), self.config)
        decision = empty_policy.decide_detailed(features)
        self.assertEqual(decision.prior_source, 'none')
        self.assertEqual(decision.decision, 'rewrite')  # same as cost-only decision

    def test_reliable_rewrite_stays_rewrite_no_override_needed(self):
        features = self._features(('RELIABLE',), original_cost=10.0, rewritten_cost=5.0)
        decision = self.policy.decide_detailed(features)
        self.assertEqual(decision.decision, 'rewrite')
        self.assertEqual(self.policy.cost_policy.decide(features), 'rewrite')

    def test_mvm_config_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            MVMConfig(fp_guard_threshold=0.7, fn_recovery_threshold=0.3)


class TestGroupLatencySelection(unittest.TestCase):
    def test_group_selects_correct_latency_for_original_and_rewrite(self):
        # Cost-only: candidate A has lower cost than original -> rewrite;
        # its rewritten_latency_ms should be the cost-selected latency.
        records = [
            make_record(id='A', query='q', rules=(), original_cost=10.0, rewritten_cost=5.0,
                        original_latency_ms=100.0, rewritten_latency_ms=42.0, label='TP'),
        ]
        cost_policy = CostOnlyPolicy(strict=True)
        mvm_policy = MVMPolicy(PriorTable.build([]), MVMConfig())
        result = build_group_result('calcite', 'q', 'q', records, cost_policy, mvm_policy)
        self.assertEqual(result.cost_selected_id, 'A')
        self.assertEqual(result.cost_selected_latency_ms, 42.0)
        # MVM has no prior evidence -> defers to cost decision -> same pick
        self.assertEqual(result.mvm_selected_id, 'A')
        self.assertEqual(result.mvm_selected_latency_ms, 42.0)

    def test_group_selects_original_latency_when_no_candidate_accepted(self):
        records = [
            make_record(id='A', query='q', rules=(), original_cost=10.0, rewritten_cost=20.0,
                        original_latency_ms=100.0, rewritten_latency_ms=200.0, label='TN'),
        ]
        cost_policy = CostOnlyPolicy(strict=True)
        mvm_policy = MVMPolicy(PriorTable.build([]), MVMConfig())
        result = build_group_result('calcite', 'q', 'q', records, cost_policy, mvm_policy)
        self.assertIsNone(result.cost_selected_id)
        self.assertEqual(result.cost_selected_latency_ms, result.original_latency_ms)

    def test_oracle_picks_lowest_actual_latency_candidate(self):
        records = [
            make_record(id='slow', query='q', rules=(), original_cost=10.0, rewritten_cost=9.0,
                        original_latency_ms=100.0, rewritten_latency_ms=90.0, label='TP'),
            make_record(id='fast', query='q', rules=(), original_cost=10.0, rewritten_cost=20.0,
                        original_latency_ms=100.0, rewritten_latency_ms=10.0, label='FN'),
        ]
        cost_policy = CostOnlyPolicy(strict=True)
        mvm_policy = MVMPolicy(PriorTable.build([]), MVMConfig())
        result = build_group_result('calcite', 'q', 'q', records, cost_policy, mvm_policy)
        self.assertEqual(result.oracle_selected_id, 'fast')
        self.assertEqual(result.oracle_latency_ms, 10.0)
        # cost-only only sees 'slow' as acceptable (lower cost); oracle beats it
        self.assertEqual(result.cost_selected_id, 'slow')


class TestConfusionMetrics(unittest.TestCase):
    def _decision(self, cost_decision, mvm_decision, actual_positive):
        from experiments.mvm_reliability.evaluate import CandidateDecision
        changed = cost_decision != mvm_decision
        return CandidateDecision(
            candidate_id='x', workload='calcite', query='q', query_family='q', rules=(),
            original_cost=10.0, rewritten_cost=5.0,
            decision=mvm_decision, cost_decision=cost_decision, mvm_decision=mvm_decision,
            reliability=0.5, support=10, prior_source='rule', prior_key='R', reason='test',
            changed_from_cost=changed, actual_positive=actual_positive,
            original_latency_ms=100.0, rewritten_latency_ms=50.0 if actual_positive else 150.0,
        )

    def test_fp_prevented(self):
        # cost=rewrite, actual negative (FP); mvm=original (prevented)
        decisions = [self._decision('rewrite', 'original', actual_positive=False)]
        summary = summarize_candidates(decisions)
        self.assertEqual(summary['fp_prevented']['count'], 1)
        self.assertEqual(summary['fp_prevented']['of_cost_fp'], 1)
        self.assertEqual(summary['new_fp'], 0)
        self.assertEqual(summary['new_fn'], 0)

    def test_fn_recovered(self):
        # cost=original, actual positive (FN); mvm=rewrite (recovered)
        decisions = [self._decision('original', 'rewrite', actual_positive=True)]
        summary = summarize_candidates(decisions)
        self.assertEqual(summary['fn_recovered']['count'], 1)
        self.assertEqual(summary['fn_recovered']['of_cost_fn'], 1)

    def test_new_fp_introduced(self):
        # cost=original, actual negative (TN, correct); mvm=rewrite -> new FP
        decisions = [self._decision('original', 'rewrite', actual_positive=False)]
        summary = summarize_candidates(decisions)
        self.assertEqual(summary['new_fp'], 1)
        self.assertEqual(summary['broken_correct_decisions'], 1)

    def test_new_fn_introduced(self):
        # cost=rewrite, actual positive (TP, correct); mvm=original -> new FN
        decisions = [self._decision('rewrite', 'original', actual_positive=True)]
        summary = summarize_candidates(decisions)
        self.assertEqual(summary['new_fn'], 1)
        self.assertEqual(summary['broken_correct_decisions'], 1)

    def test_unchanged_correct_decision_not_counted_as_broken(self):
        decisions = [self._decision('rewrite', 'rewrite', actual_positive=True)]
        summary = summarize_candidates(decisions)
        self.assertEqual(summary['broken_correct_decisions'], 0)
        self.assertEqual(summary['new_fp'], 0)
        self.assertEqual(summary['new_fn'], 0)


class TestReproducibility(unittest.TestCase):
    def test_same_input_same_config_same_decisions(self):
        records = load_candidates_jsonl(MVM_V1_DIR / 'candidates.jsonl')
        split = load_split(MVM_V1_DIR / 'split.json')
        config = ExperimentConfig(cost_strict=True, mvm=MVMConfig())

        result1 = run_evaluation(records, split, config)
        result2 = run_evaluation(records, split, config)

        trace1 = [d.to_dict() for d in result1.all_candidate_decisions]
        trace2 = [d.to_dict() for d in result2.all_candidate_decisions]
        self.assertEqual(trace1, trace2)


class TestRunDirNoOverwrite(unittest.TestCase):
    def test_second_call_with_same_timestamp_and_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp)
            fixed_now = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
            first = make_run_dir(outputs_dir, 'pilot', now=fixed_now)
            self.assertTrue(first.exists())
            with self.assertRaises(FileExistsError):
                make_run_dir(outputs_dir, 'pilot', now=fixed_now)

    def test_different_names_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp)
            fixed_now = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
            first = make_run_dir(outputs_dir, 'pilot-a', now=fixed_now)
            second = make_run_dir(outputs_dir, 'pilot-b', now=fixed_now)
            self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
