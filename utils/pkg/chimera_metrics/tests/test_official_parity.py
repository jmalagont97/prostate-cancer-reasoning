"""Small differential smoke tests for parity with the authors' evaluator."""

from chimera_metrics.core.base import norm_decision, norm_treatment_decision
from chimera_metrics.task2.decision import compute_treatment_decision_gate


def test_task2_official_nested_primary_decision_matches():
    gt = {"treatment_recommendation": {"primary": "active_surveillance"}}
    pred = {"treatment_recommendation": {"primary": "active_surveillance"}}

    score, gt_value, pred_value, _ = compute_treatment_decision_gate(gt, pred)

    assert score == 1.0
    assert gt_value == "active_surveillance"
    assert pred_value == "active_surveillance"


def test_decision_normalization_matches_official_string_contract():
    assert norm_decision("yes") == "yes"
    assert norm_decision("no") == "no"
    assert norm_decision(True) is None
    assert norm_decision(1) is None


def test_treatment_normalization_reads_only_official_primary_field():
    record = {"treatment_recommendation": {"primary": "active_surveillance"}}
    assert norm_treatment_decision(record) == "active_surveillance"
    assert norm_treatment_decision({"action": "active_surveillance"}) is None
