"""Task 3: Per-case Biochemical Recurrence (BCR) Prognostic Scores."""

from __future__ import annotations

from typing import Any, Dict, Optional

from chimera_metrics.core.base import (
    BaseMetric,
    norm_event,
    norm_months,
)
from chimera_metrics.core.types import MetricResult


def compute_recurrence_event_score(gt: Dict[str, Any], pred: Dict[str, Any]) -> Optional[float]:
    """
    Computes agreement on binary recurrence event occurrence.

    Equation:
        S_event = 1.0 if e_pred == e_gt else 0.0
    """
    g = norm_event(gt.get("event"))
    p = norm_event(pred.get("event"))
    if g is None or p is None:
        return None
    return 1.0 if g == p else 0.0


def compute_recurrence_time_score(gt: Dict[str, Any], pred: Dict[str, Any]) -> Optional[float]:
    """
    Censoring-aware closeness of predicted time-to-recurrence (months).

    Equation:
        Scale = max(T_gt, 1.0)
        If e_gt == 1 (observed BCR):
            S_time = max(0.0, 1.0 - |T_pred - T_gt| / Scale)
        If e_gt == 0 (censored at T_gt):
            If T_pred >= T_gt:
                S_time = 1.0
            Else:
                S_time = max(0.0, 1.0 - (T_gt - T_pred) / Scale)
    """
    g_event = norm_event(gt.get("event"))
    g_t = norm_months(gt.get("months_to_recurrence"))
    p_t = norm_months(pred.get("months_to_recurrence"))

    if g_event not in (0, 1) or g_t is None or p_t is None:
        return None

    scale = max(g_t, 1.0)
    if g_event == 1:
        return max(0.0, 1.0 - abs(p_t - g_t) / scale)

    # Censored: predicting recurrence at or after the follow-up time is fully consistent.
    if p_t >= g_t:
        return 1.0
    return max(0.0, 1.0 - (g_t - p_t) / scale)


class RecurrenceCaseScoreMetric(BaseMetric):
    """Metric wrapper for per-case BCR prognosis evaluation."""

    def __init__(self):
        super().__init__(
            name="task3_case_prognosis",
            description="Per-case evaluation for event match and censoring-aware recurrence time.",
        )

    def evaluate(self, gt: Dict[str, Any], pred: Dict[str, Any]) -> MetricResult:
        ev_score = compute_recurrence_event_score(gt, pred)
        tm_score = compute_recurrence_time_score(gt, pred)

        details = {
            "event_score": ev_score,
            "time_score": tm_score,
            "gt_event": norm_event(gt.get("event")),
            "pred_event": norm_event(pred.get("event")),
            "gt_months": norm_months(gt.get("months_to_recurrence")),
            "pred_months": norm_months(pred.get("months_to_recurrence")),
        }
        return MetricResult(
            name=self.name,
            value=tm_score,
            details=details,
            description=self.description,
        )
