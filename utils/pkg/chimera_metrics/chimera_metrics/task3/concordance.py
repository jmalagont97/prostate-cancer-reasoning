"""Task 3: Harrell's Concordance Index for Survival Ranking."""

from __future__ import annotations

from typing import List, Optional

from chimera_metrics.core.base import BaseMetric
from chimera_metrics.core.types import MetricResult


def compute_concordance_index(
    times: List[float],
    preds: List[float],
    events: List[int],
) -> Optional[float]:
    """
    Computes Harrell's concordance index (C-index) for right-censored survival outcomes.

    Note:
        `preds` are predicted months-to-recurrence (shorter time => higher risk).
        A pair (i, j) is admissible if T_i < T_j and subject i experienced the event (events[i] == 1).
        Concordance occurs if preds[i] < preds[j] (patient i is predicted to fail earlier).

    Equation:
        C = sum_{i, j: T_i < T_j, E_i = 1} [ I(pred_i < pred_j) + 0.5 * I(pred_i == pred_j) ] / sum_{i, j: T_i < T_j, E_i = 1} 1
    """
    if not (len(times) == len(preds) == len(events)):
        raise ValueError("Lengths of times, preds, and events must match.")

    n = len(times)
    num = 0.0
    den = 0.0

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j or not (times[i] < times[j]):
                continue
            den += 1.0
            if preds[i] < preds[j]:
                num += 1.0
            elif preds[i] == preds[j]:
                num += 0.5

    return (num / den) if den > 0 else None


class HarrellCIndexMetric(BaseMetric):
    """Metric wrapper for Harrell's C-Index."""

    def __init__(self):
        super().__init__(
            name="task3_harrell_c_index",
            description="Harrell's Concordance Index on right-censored BCR time predictions.",
        )

    def evaluate(
        self,
        times: List[float],
        preds: List[float],
        events: List[int],
    ) -> MetricResult:
        c_idx = compute_concordance_index(times, preds, events)
        return MetricResult(
            name=self.name,
            value=c_idx,
            details={"num_cases": len(times), "admissible_pairs_found": (c_idx is not None)},
            description=self.description,
        )
