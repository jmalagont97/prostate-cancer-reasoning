"""Base abstract classes and normalization utilities for CHIMERA-Agent metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from chimera_metrics.core.types import (
    CONF_MAP,
    WEIGHT_ALIAS,
    WEIGHT_MAP,
    MetricResult,
)


def norm_decision(val: Any) -> Optional[str]:
    """Normalize binary biopsy decision using the official string contract."""
    if not isinstance(val, str):
        return None
    value = val.strip().lower()
    return value if value in {"yes", "no"} else None


def norm_treatment_decision(val: Any) -> Optional[str]:
    """Normalize treatment recommendation using the official primary field."""
    if val is None:
        return None
    if isinstance(val, dict):
        recommendation = val.get("treatment_recommendation")
        if not isinstance(recommendation, dict):
            return None
        val = recommendation.get("primary")
    if not isinstance(val, str):
        return None
    s = val.strip().lower().replace("-", "_")
    s = "_".join(s.split())
    valid_decisions = {
        "watchful_waiting",
        "active_surveillance",
        "continued_surveillance",
        "active_treatment",
    }
    return s if s in valid_decisions else None


def norm_confidence(val: Any) -> Optional[str]:
    """Normalize confidence level string."""
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in CONF_MAP else None


def norm_weight(val: Any) -> Optional[str]:
    """Normalize variable importance weight."""
    if val is None:
        return None
    s = str(val).strip().lower().replace("-", "_").replace(" ", "_")
    s = WEIGHT_ALIAS.get(s, s)
    return s if s in WEIGHT_MAP else None


def norm_event(val: Any) -> Optional[int]:
    """Normalize binary event indicator (0/1)."""
    if val is None:
        return None
    try:
        iv = int(val)
        return iv if iv in (0, 1) else None
    except (TypeError, ValueError):
        return None


def norm_months(val: Any) -> Optional[float]:
    """Normalize survival/recurrence time in months."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f >= 0.0 else None
    except (TypeError, ValueError):
        return None


def extract_reveal_keys(pred: Dict[str, Any]) -> List[str]:
    """Extract flat reveal sequence section names from prediction payload."""
    seq = pred.get("reveal_sequence") or pred.get("revealed_sections") or []
    if isinstance(seq, list):
        out: List[str] = []
        for item in seq:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict) and "section_name" in item:
                out.append(str(item["section_name"]).strip())
            elif isinstance(item, dict) and "name" in item:
                out.append(str(item["name"]).strip())
        return out
    return []


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> MetricResult:
        """Compute the metric and return a MetricResult instance."""
        pass
