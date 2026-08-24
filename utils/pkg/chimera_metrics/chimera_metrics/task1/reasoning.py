"""Task 1: Deterministic Reasoning Scores (Confidence, Variable Weights, Important/Decisive Factors)."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from chimera_metrics.core.base import (
    BaseMetric,
    norm_confidence,
    norm_weight,
)
from chimera_metrics.core.types import (
    CONF_MAP,
    IMPORTANT_OR_DECISIVE,
    WEIGHT_MAP,
    MetricResult,
)


def compute_confidence_score(gt: Dict[str, Any], pred: Dict[str, Any]) -> Optional[float]:
    """
    Computes normalized ordinal distance for confidence score.

    Equation:
        S_conf = 1.0 - |rank(c_pred) - rank(c_gt)| / max_distance
        where rank in {uncertain: 0, borderline: 1, clear: 2}, max_distance = 2.
    """
    gt_conf = norm_confidence(gt.get("confidence"))
    pred_conf = norm_confidence(pred.get("confidence"))

    if gt_conf is None or pred_conf is None:
        return None

    dist = abs(CONF_MAP[gt_conf] - CONF_MAP[pred_conf])
    max_dist = max(CONF_MAP.values()) - min(CONF_MAP.values())  # 2
    return 1.0 - (dist / max_dist)


def compute_variable_weights_score(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    default_missing: str = "not_used",
) -> Optional[float]:
    """
    Computes mean inverse normalized MAE across clinical variable weights.

    Equation:
        S_weights = 1.0 - (1 / |V|) * sum_{v in V} [ |rank(w_pred(v)) - rank(w_gt(v))| / max_weight ]
        where rank in {not_used: 0, noted: 1, important: 2, decisive: 3}, max_weight = 3.
    """
    gt_w = gt.get("variable_weights") or {}
    pr_w = pred.get("variable_weights") or {}

    if not isinstance(gt_w, dict) or not gt_w:
        return None

    max_w = max(WEIGHT_MAP.values()) - min(WEIGHT_MAP.values())  # 3
    errors: List[float] = []

    for var, gv in gt_w.items():
        g = norm_weight(gv)
        if g is None:
            continue
        p = norm_weight(pr_w.get(var, default_missing)) or default_missing
        errors.append(abs(WEIGHT_MAP[g] - WEIGHT_MAP[p]) / max_w)

    if not errors:
        return None
    return 1.0 - mean(errors)


def extract_important_decisive_set(weights: Dict[str, Any]) -> Set[str]:
    """Extracts variables marked as 'important' or 'decisive'."""
    if not isinstance(weights, dict):
        return set()
    out = set()
    for var, val in weights.items():
        v = norm_weight(val)
        if v in IMPORTANT_OR_DECISIVE:
            out.add(var)
    return out


def compute_set_f1(gt_set: Set[str], pred_set: Set[str]) -> float:
    """
    Computes set-based F1-score between ground truth and predicted variable sets.

    Equation:
        F1 = 2 * |gt_set inter pred_set| / (|gt_set| + |pred_set|)
    """
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0

    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def compute_important_decisive_f1(gt: Dict[str, Any], pred: Dict[str, Any]) -> Optional[float]:
    """Computes F1-score on important and decisive variables."""
    gt_set = extract_important_decisive_set(gt.get("variable_weights") or {})
    pred_set = extract_important_decisive_set(pred.get("variable_weights") or {})
    return compute_set_f1(gt_set, pred_set)


class Task1ReasoningMetric(BaseMetric):
    """Aggregated reasoning metric for Task 1."""

    def __init__(self):
        super().__init__(
            name="task1_reasoning_composite",
            description="Composite metric covering confidence, variable weights and decisive factors.",
        )

    def evaluate(self, gt: Dict[str, Any], pred: Dict[str, Any]) -> MetricResult:
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
