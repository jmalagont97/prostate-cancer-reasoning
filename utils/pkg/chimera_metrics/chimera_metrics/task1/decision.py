"""Task 1: Biopsy Decision Gate Evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from chimera_metrics.core.base import BaseMetric, norm_decision
from chimera_metrics.core.types import MetricResult


def compute_biopsy_decision_gate(
    gt: Any,
    pred: Any,
) -> Tuple[float, Optional[str], Optional[str], str]:
    """
    Computes exact match decision gate for Task 1 (biopsy decision).

    Equation:
        S_gate = 1.0 if y_pred == y_true else 0.0

    Args:
        gt: Ground truth dictionary or string/bool value.
        pred: Predicted dictionary or string/bool value.

    Returns:
        Tuple of (score, gt_normalized, pred_normalized, explanation).
    """
    gt_val = gt.get("biopsy_decision") if isinstance(gt, dict) else gt
    pred_val = pred.get("biopsy_decision") if isinstance(pred, dict) else pred

    gt_norm = norm_decision(gt_val)
    pred_norm = norm_decision(pred_val)

    if gt_norm is None:
        return 0.0, None, pred_norm, "Ground truth biopsy decision missing or invalid"
    if pred_norm is None:
        return 0.0, gt_norm, None, "Predicted biopsy decision missing or invalid"

    if gt_norm == pred_norm:
        return 1.0, gt_norm, pred_norm, "Biopsy decision matched"
    return 0.0, gt_norm, pred_norm, f"Biopsy decision mismatch: gt={gt_norm!r} vs pred={pred_norm!r}"


class BiopsyDecisionGateMetric(BaseMetric):
    """Metric wrapper for Task 1 decision gate."""

    def __init__(self):
        super().__init__(
            name="task1_biopsy_decision_gate",
            description="Exact match gate for binary biopsy indication.",
        )

    def evaluate(self, gt: Any, pred: Any) -> MetricResult:
        score, gt_n, pred_n, msg = compute_biopsy_decision_gate(gt, pred)
        return MetricResult(
            name=self.name,
            value=score,
            passed=(score == 1.0),
            details={"gt_decision": gt_n, "pred_decision": pred_n, "message": msg},
            description=self.description,
        )
