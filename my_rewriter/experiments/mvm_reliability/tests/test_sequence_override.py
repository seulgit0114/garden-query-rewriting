import json
import tempfile
import unittest
from pathlib import Path

from my_rewriter.candidate_selection.interfaces import CandidateFeatures, MVMDecision
from my_rewriter.candidate_selection.sequence_override import SequenceOverridePolicy


class DelegatePolicy:
    accept_train_count = 7
    reject_train_count = 3

    def decide_detailed(self, candidate):
        return MVMDecision(
            decision="rewrite", reliability=0.9, support=10,
            prior_source="delegate", prior_key="delegate", reason="delegate",
        )


class TestSequenceOverridePolicy(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "overrides.json"
        self.path.write_text(json.dumps({
            "overrides": [{
                "id": "calcite_apm_singleton",
                "workloads": ["calcite"],
                "exact_rules": ["AGGREGATE_PROJECT_MERGE"],
                "decision": "original",
                "reason": "test guard",
            }]
        }))
        self.policy = SequenceOverridePolicy.from_json(DelegatePolicy(), self.path)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def features(workload, rules):
        return CandidateFeatures(
            candidate_id="x", workload=workload, query="select 1",
            query_family="q", rules=rules, original_cost=10.0,
            rewritten_cost=5.0,
        )

    def test_suppresses_exact_calcite_singleton(self):
        decision = self.policy.decide_detailed(
            self.features("calcite", ("AGGREGATE_PROJECT_MERGE",))
        )
        self.assertEqual(decision.decision, "original")
        self.assertEqual(decision.prior_source, "sequence_override")

    def test_preserves_calcite_multi_rule_sequence(self):
        decision = self.policy.decide_detailed(self.features(
            "calcite", ("AGGREGATE_PROJECT_MERGE", "FILTER_INTO_JOIN")
        ))
        self.assertEqual(decision.decision, "rewrite")
        self.assertEqual(decision.prior_source, "delegate")

    def test_does_not_affect_other_workloads(self):
        decision = self.policy.decide_detailed(
            self.features("dsb", ("AGGREGATE_PROJECT_MERGE",))
        )
        self.assertEqual(decision.decision, "rewrite")
        self.assertEqual(decision.prior_source, "delegate")


if __name__ == "__main__":
    unittest.main()
