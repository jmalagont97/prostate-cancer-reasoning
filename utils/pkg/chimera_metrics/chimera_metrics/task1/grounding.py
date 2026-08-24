"""Task 1 & Task 2: Grounding and Cost-Aware Tool Efficiency Metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from chimera_metrics.core.base import (
    BaseMetric,
    extract_reveal_keys,
    norm_weight,
)
from chimera_metrics.core.types import (
    SECTION_KEY_TO_REVEAL_NAME,
    MetricResult,
)


def compute_cost_aware_tool_efficiency(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Uniform-cost asymmetric tool efficiency score (precision of agent tool usage).

    Rule:
      - Penalize agent tools that the pathologist did NOT use (extra tools).
      - Do NOT penalize missing pathologist tools (under-use is permitted).
      - Ignore reveal order.
      - Each extra tool incurs a uniform cost.

    Equation:
      Score = |agent_tools inter pathologist_tools| / |agent_tools|
      If |agent_tools| == 0: Score = 1.0 (no unnecessary cost incurred).

    Returns:
        Tuple of (score in [0, 1], details_dict).
    """
    expected = set(extract_reveal_keys(gt))
    actual = set(extract_reveal_keys(pred))

    if not actual:
        return 1.0, {
            "expected_tools": sorted(expected),
            "actual_tools": [],
            "approved_tools": [],
            "extra_tools": [],
            "missing_tools_not_penalized": sorted(expected),
            "n_extra": 0,
            "n_actual": 0,
            "policy": "no_actual_tools_no_extra_cost",
        }

    extra = actual - expected
    approved = actual & expected
    missing = expected - actual
    score = len(approved) / len(actual)

    return score, {
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual),
        "approved_tools": sorted(approved),
        "extra_tools": sorted(extra),
        "missing_tools_not_penalized": sorted(missing),
        "n_extra": len(extra),
        "n_actual": len(actual),
        "policy": "penalize_extra_only_ignore_order_uniform_cost",
    }


def compute_tool_efficiency(gt: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """Convenience shorthand returning float score."""
    score, _ = compute_cost_aware_tool_efficiency(gt, pred)
    return score


def compute_section_grounding(
    pred: Dict[str, Any],
    section_mapping: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Penalizes variables weighted above 'not_used' whose source section was never revealed.

    Equation:
        Score = n_grounded / (n_grounded + n_ungrounded)
        Returns None if total weighted active variables is 0.
    """
    if not section_mapping:
        return None, {
            "grounded_variables": [],
            "ungrounded_variables": [],
            "ungradable_variables": [],
            "total_weighted": 0,
            "revealed_sections": [],
        }

    var_to_sections = section_mapping.get("variable_to_sections", {})
    always_available = set(
        section_mapping.get("always_available_variables", {}).get("variables", [])
    )

    revealed = set(extract_reveal_keys(pred))
    weights = pred.get("variable_weights") or {}

    grounded: List[str] = []
    ungrounded: List[str] = []
    ungradable: List[str] = []

    for var, weight_val in weights.items():
        w = norm_weight(weight_val)
        if w is None or w == "not_used":
            continue

        if var in always_available:
            grounded.append(var)
            continue

        var_info = var_to_sections.get(var, {})
        primary_sections = var_info.get("primary_sections", [])
        always_avail_flag = var_info.get("always_available_baseline", False)

        if always_avail_flag or not primary_sections:
            grounded.append(var)
            continue

        primary_names = [
            SECTION_KEY_TO_REVEAL_NAME[s]
            for s in primary_sections
            if s in SECTION_KEY_TO_REVEAL_NAME
        ]
        if not primary_names:
            ungradable.append(var)
            continue

        if any(s in revealed for s in primary_names):
            grounded.append(var)
        else:
            ungrounded.append(var)

    total = len(grounded) + len(ungrounded)
    if total == 0:
        return None, {
            "grounded_variables": [],
            "ungrounded_variables": [],
            "ungradable_variables": sorted(ungradable),
            "total_weighted": 0,
            "revealed_sections": sorted(revealed),
        }

    score = len(grounded) / total
    return score, {
        "grounded_variables": sorted(grounded),
        "ungrounded_variables": sorted(ungrounded),
        "ungradable_variables": sorted(ungradable),
        "total_weighted": total,
        "revealed_sections": sorted(revealed),
    }


class SectionGroundingMetric(BaseMetric):
    """Metric wrapper for Section Grounding."""

    def __init__(self, section_mapping: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="section_grounding",
            description="Evaluates whether actively weighted variables are backed by revealed EHR sections.",
        )
        self.section_mapping = section_mapping

    def evaluate(self, pred: Dict[str, Any]) -> MetricResult:
        score, details = compute_section_grounding(pred, self.section_mapping)
        return MetricResult(
            name=self.name,
            value=score,
            details=details,
            description=self.description,
        )


class ToolEfficiencyMetric(BaseMetric):
    """Metric wrapper for Tool Efficiency."""

    def __init__(self):
        super().__init__(
            name="tool_efficiency_precision",
            description="Tool efficiency precision: penalizes unnecessary extra reveals against ground truth.",
        )

    def evaluate(self, gt: Dict[str, Any], pred: Dict[str, Any]) -> MetricResult:
        score, details = compute_cost_aware_tool_efficiency(gt, pred)
        return MetricResult(
            name=self.name,
            value=score,
            details=details,
            description=self.description,
        )
