"""Task 2: Deterministic Reasoning Scores for Treatment Recommendation."""

from chimera_metrics.core.base import BaseMetric
from chimera_metrics.core.types import MetricResult
from chimera_metrics.task1.reasoning import (
    compute_confidence_score,
    compute_important_decisive_f1,
    compute_set_f1,
    compute_variable_weights_score,
    extract_important_decisive_set,
)


class Task2ReasoningMetric(BaseMetric):
    """Aggregated reasoning metric for Task 2."""

    def __init__(self):
        super().__init__(
            name="task2_reasoning_composite",
            description="Composite metric covering confidence, variable weights (17 variables) and decisive factors.",
        )

    def evaluate(self, gt: dict, pred: dict) -> MetricResult:
        from statistics import mean
        conf = compute_confidence_score(gt, pred)
        weights = compute_variable_weights_score(gt, pred)
        f1 = compute_important_decisive_f1(gt, pred)

        components = [c for c in (conf, weights, f1) if c is not None]
        composite_score = mean(components) if components else None

        return MetricResult(
            name=self.name,
            value=composite_score,
            details={
                "confidence_score": conf,
                "variable_weights_score": weights,
                "important_decisive_f1": f1,
            },
            description=self.description,
        )


__all__ = [
    "compute_confidence_score",
    "compute_variable_weights_score",
    "extract_important_decisive_set",
    "compute_set_f1",
    "compute_important_decisive_f1",
    "Task2ReasoningMetric",
]
