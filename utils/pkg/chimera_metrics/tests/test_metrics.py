"""Unit tests and parity verification for chimera_metrics."""

import unittest
from chimera_metrics import (
    compute_biopsy_decision_gate,
    compute_treatment_decision_gate,
    compute_confidence_score,
    compute_variable_weights_score,
    compute_important_decisive_f1,
    compute_tool_efficiency,
    compute_section_grounding,
    compute_recurrence_event_score,
    compute_recurrence_time_score,
    compute_concordance_index,
    compute_time_dependent_auc,
    evaluate_task1_case,
    evaluate_task2_case,
    evaluate_task3_cohort,
    ExperimentReporter,
)


class TestChimeraMetrics(unittest.TestCase):

    def test_task1_decision_gate(self):
        # Exact match
        score, gt_n, pr_n, _ = compute_biopsy_decision_gate("yes", "yes")
        self.assertEqual(score, 1.0)
        self.assertEqual(gt_n, "yes")
        self.assertEqual(pr_n, "yes")

        # Mismatch
        score, _, _, _ = compute_biopsy_decision_gate("yes", "no")
        self.assertEqual(score, 0.0)

        # Dict formats
        score, _, _, _ = compute_biopsy_decision_gate({"biopsy_decision": True}, {"biopsy_decision": "yes"})
        self.assertEqual(score, 1.0)

    def test_task2_decision_gate(self):
        score, gt_n, pr_n, _ = compute_treatment_decision_gate("active_surveillance", "active_surveillance")
        self.assertEqual(score, 1.0)
        self.assertEqual(gt_n, "active_surveillance")

        score, _, _, _ = compute_treatment_decision_gate("active_surveillance", "watchful_waiting")
        self.assertEqual(score, 0.0)

    def test_confidence_score(self):
        # Same confidence -> 1.0
        self.assertEqual(compute_confidence_score({"confidence": "clear"}, {"confidence": "clear"}), 1.0)
        # clear (2) vs borderline (1) -> 1 - 1/2 = 0.5
        self.assertEqual(compute_confidence_score({"confidence": "clear"}, {"confidence": "borderline"}), 0.5)
        # clear (2) vs uncertain (0) -> 1 - 2/2 = 0.0
        self.assertEqual(compute_confidence_score({"confidence": "clear"}, {"confidence": "uncertain"}), 0.0)

    def test_variable_weights_score(self):
        gt = {"variable_weights": {"pirads": "decisive", "psa": "important"}}
        # Perfect match
        pred_perfect = {"variable_weights": {"pirads": "decisive", "psa": "important"}}
        self.assertEqual(compute_variable_weights_score(gt, pred_perfect), 1.0)

        # One step off in pirads (important vs decisive = 1/3 err), psa exact (0 err) -> mean err = 1/6 -> score = 5/6
        pred_off = {"variable_weights": {"pirads": "important", "psa": "important"}}
        self.assertAlmostEqual(compute_variable_weights_score(gt, pred_off), 5.0 / 6.0)

    def test_important_decisive_f1(self):
        gt = {"variable_weights": {"pirads": "decisive", "psa": "important", "age": "noted"}}
        pred = {"variable_weights": {"pirads": "decisive", "psa": "noted", "dre": "important"}}
        # gt decisive/important: {pirads, psa} (len 2)
        # pred decisive/important: {pirads, dre} (len 2)
        # intersection: {pirads} (len 1)
        # F1 = 2*1 / (2+2) = 0.5
        self.assertEqual(compute_important_decisive_f1(gt, pred), 0.5)

    def test_tool_efficiency(self):
        gt = {"reveal_sequence": ["radiology_report", "psa_trend"]}
        pred_exact = {"reveal_sequence": ["radiology_report", "psa_trend"]}
        self.assertEqual(compute_tool_efficiency(gt, pred_exact), 1.0)

        # Agent reveals 1 extra tool (pathology) -> approved: 2, total actual: 3 -> score 2/3
        pred_extra = {"reveal_sequence": ["radiology_report", "psa_trend", "pathology_report"]}
        self.assertAlmostEqual(compute_tool_efficiency(gt, pred_extra), 2.0 / 3.0)

        # Agent reveals 0 tools -> 1.0 (no unnecessary cost incurred)
        self.assertEqual(compute_tool_efficiency(gt, {"reveal_sequence": []}), 1.0)

    def test_task3_survival_scores(self):
        # Event match
        self.assertEqual(compute_recurrence_event_score({"event": 1}, {"event": 1}), 1.0)
        self.assertEqual(compute_recurrence_event_score({"event": 1}, {"event": 0}), 0.0)

        # Time score observed event (event = 1, gt_t = 12, pred_t = 12 -> 1.0)
        gt_obs = {"event": 1, "months_to_recurrence": 12.0}
        pred_exact = {"event": 1, "months_to_recurrence": 12.0}
        self.assertEqual(compute_recurrence_time_score(gt_obs, pred_exact), 1.0)

        # Time score observed event (event = 1, gt_t = 12, pred_t = 6 -> 1 - |6-12|/12 = 0.5)
        pred_early = {"event": 1, "months_to_recurrence": 6.0}
        self.assertEqual(compute_recurrence_time_score(gt_obs, pred_early), 0.5)

        # Time score censored (event = 0, gt_t = 12, pred_t = 24 >= 12 -> 1.0)
        gt_cens = {"event": 0, "months_to_recurrence": 12.0}
        pred_late = {"event": 0, "months_to_recurrence": 24.0}
        self.assertEqual(compute_recurrence_time_score(gt_cens, pred_late), 1.0)

    def test_concordance_and_td_auc(self):
        times = [10.0, 20.0, 30.0, 40.0]
        events = [1, 1, 0, 1]
        # Perfect predictions: shorter predicted time for earlier failures
        preds_perfect = [10.0, 20.0, 30.0, 40.0]
        c_idx = compute_concordance_index(times, preds_perfect, events)
        self.assertIsNotNone(c_idx)
        self.assertEqual(c_idx, 1.0)

        auc_dict = compute_time_dependent_auc(times, preds_perfect, events, horizons=(15.0, 25.0))
        self.assertIn(15.0, auc_dict)
        self.assertIn(25.0, auc_dict)

    def test_reporters(self):
        gt = {
            "biopsy_decision": "yes",
            "confidence": "clear",
            "variable_weights": {"pirads": "decisive", "psa": "important"},
            "reveal_sequence": ["radiology_report"]
        }
        pred = {
            "biopsy_decision": "yes",
            "confidence": "clear",
            "variable_weights": {"pirads": "decisive", "psa": "important"},
            "reveal_sequence": ["radiology_report"]
        }
        res1 = evaluate_task1_case("BX_01", gt, pred)
        summary1 = ExperimentReporter.format_task1_summary([res1])
        self.assertIn("BX_01", summary1)
        self.assertIn("Mean Decision Gate: 1.0000", summary1)


if __name__ == "__main__":
    unittest.main()
