"""Task 3: Time-Dependent Cumulative/Dynamic AUC with Inverse Probability of Censoring Weighting (IPCW)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from chimera_metrics.core.base import BaseMetric
from chimera_metrics.core.types import MetricResult

DEFAULT_HORIZONS: Tuple[float, ...] = (12.0, 24.0, 36.0, 60.0)


def compute_censoring_km(
    times: List[float],
    events: List[int],
) -> List[Tuple[float, float]]:
    """
    Computes Kaplan-Meier estimate of the censoring survival distribution G(t) = P(C > t).

    Returned as ascending (time, G) steps. Censoring is the event of interest (event == 0).
    """
    order = sorted(range(len(times)), key=lambda i: times[i])
    steps: List[Tuple[float, float]] = [(float("-inf"), 1.0)]
    at_risk = len(order)
    g = 1.0
    i = 0

    while i < len(order):
        t = times[order[i]]
        tied = 0
        censored = 0
        while i + tied < len(order) and times[order[i + tied]] == t:
            if events[order[i + tied]] == 0:
                censored += 1
            tied += 1
        if at_risk > 0 and censored > 0:
            g *= 1.0 - censored / at_risk
        steps.append((t, g))
        at_risk -= tied
        i += tied

    return steps


def km_at_time(steps: List[Tuple[float, float]], target_time: float) -> float:
    """Evaluates Kaplan-Meier step function at given time."""
    g = 1.0
    for t, val in steps:
        if t <= target_time:
            g = val
        else:
            break
    return g


def compute_single_horizon_auc(
    times: List[float],
    preds: List[float],
    events: List[int],
    horizon: float,
    km_steps: Optional[List[Tuple[float, float]]] = None,
) -> Optional[float]:
    """
    Computes cumulative/dynamic time-dependent AUC at a single horizon t* via IPCW (Uno et al.).

    Args:
        times: Observed times.
        preds: Predicted recurrence times (risk = -pred).
        events: Event indicators (1=event, 0=censored).
        horizon: Evaluation horizon t* in months.
        km_steps: Precomputed censoring Kaplan-Meier steps.
    """
    if km_steps is None:
        km_steps = compute_censoring_km(times, events)

    cases: List[Tuple[float, float]] = []
    controls: List[Tuple[float, float]] = []

    for t, pred_t, ev in zip(times, preds, events):
        risk = -pred_t  # Shorter predicted time = higher risk
        if t <= horizon and ev == 1:
            g = km_at_time(km_steps, t)
            if g > 0.0:
                cases.append((risk, 1.0 / g))
        elif t > horizon:
            g = km_at_time(km_steps, horizon)
            if g > 0.0:
                controls.append((risk, 1.0 / g))

    if not cases or not controls:
        return None

    num = 0.0
    for case_risk, case_w in cases:
        for ctrl_risk, ctrl_w in controls:
            if case_risk > ctrl_risk:
                num += case_w * ctrl_w
            elif case_risk == ctrl_risk:
                num += 0.5 * case_w * ctrl_w

    den = sum(w for _, w in cases) * sum(w for _, w in controls)
    return (num / den) if den > 0 else None


def compute_time_dependent_auc(
    times: List[float],
    preds: List[float],
    events: List[int],
    horizons: Tuple[float, ...] = DEFAULT_HORIZONS,
) -> Dict[float, Optional[float]]:
    """
    Computes time-dependent AUC across multiple time horizons.

    Returns:
        Dictionary mapping each horizon (months) to its IPCW AUC score.
    """
    km_steps = compute_censoring_km(times, events)
    results: Dict[float, Optional[float]] = {}
    for h in horizons:
        results[h] = compute_single_horizon_auc(times, preds, events, h, km_steps=km_steps)
    return results


class TimeDependentAUCMetric(BaseMetric):
    """Metric wrapper for IPCW Time-Dependent AUC."""

    def __init__(self, horizons: Tuple[float, ...] = DEFAULT_HORIZONS):
        super().__init__(
            name="task3_time_dependent_auc",
            description="IPCW Cumulative/Dynamic Time-Dependent AUC (Uno et al.) at fixed horizons.",
        )
        self.horizons = horizons

    def evaluate(
        self,
        times: List[float],
        preds: List[float],
        events: List[int],
    ) -> MetricResult:
        auc_dict = compute_time_dependent_auc(times, preds, events, self.horizons)
        valid_scores = [v for v in auc_dict.values() if v is not None]
        mean_auc = sum(valid_scores) / len(valid_scores) if valid_scores else None

        return MetricResult(
            name=self.name,
            value=mean_auc,
            details={"horizons_auc": auc_dict, "horizons_evaluated": list(self.horizons)},
            description=self.description,
        )
