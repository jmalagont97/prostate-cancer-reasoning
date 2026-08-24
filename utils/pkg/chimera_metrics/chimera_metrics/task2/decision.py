"""Task 2: Treatment Recommendation Decision Gate Evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from chimera_metrics.core.base import BaseMetric, norm_treatment_decision
from chimera_metrics.core.types import MetricResult


def compute_treatment_decision_gate(
    gt: Any,
    pred: Any,
) -> Tuple[float, Optional[str], Optional[str], str]:
    """
    Computes exact match decision gate for Task 2 (treatment recommendation).

    Valid categories: watchful_waiting, active_surveillance, continued_surveillance, active_treatment.

    Equation:
        S_gate = 1.0 if y_pred == y_true else 0.0

    Args:
        gt: Ground truth dictionary or string recommendation.
        pred: Predicted dictionary or string recommendation.

    Returns:
        Tuple of (score, gt_normalized, pred_normalized, explanation).
    """
    gt_norm = norm_treatment_decision(gt)
    pred_norm = norm_treatment_decision(pred)

    if gt_norm is None:
        return 0.0, None, pred_norm, "Ground truth treatment recommendation missing or invalid"
    if pred_norm is None:
        return 0.0, gt_norm, None, "Predicted treatment recommendation missing or invalid"

    if gt_norm == pred_norm:
        return 1.0, gt_norm, pred_norm, "Treatment recommendation matched"
    return 0.0, gt_norm, pred_norm, f"Treatment decision mismatch: gt={gt_norm!r} vs pred={pred_norm!r}"


class TreatmentDecisionGateMetric(BaseMetric):
    """Metric wrapper for Task 2 decision gate."""

    def __init__(self):
        super().__init__(
            name="task2_treatment_decision_gate",
            description="Exact match gate for multiclass prostate cancer treatment recommendation.",
        )

    def evaluate(self, gt: Any, pred: Any) -> MetricResult:
        score, gt_n, pred_n, msg = compute_treatment_decision_gate(gt, pred)
        return MetricResult(
            name=self.name,
            value=score,
            passed=(score == 1.0),
            details={"gt_decision": gt_n, "pred_decision": pred_n, "message": msg},
            description=self.description,
        )
