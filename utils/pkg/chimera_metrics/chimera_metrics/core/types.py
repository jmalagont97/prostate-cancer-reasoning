"""Standard types, enums, mappings and data containers for CHIMERA-Agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ConfidenceLevel(str, Enum):
    UNCERTAIN = "uncertain"
    BORDERLINE = "borderline"
    CLEAR = "clear"


class ImportanceWeight(str, Enum):
    NOT_USED = "not_used"
    NOTED = "noted"
    IMPORTANT = "important"
    DECISIVE = "decisive"


class BiopsyDecision(str, Enum):
    YES = "yes"
    NO = "no"


class TreatmentDecision(str, Enum):
    WATCHFUL_WAITING = "watchful_waiting"
    ACTIVE_SURVEILLANCE = "active_surveillance"
    CONTINUED_SURVEILLANCE = "continued_surveillance"
    ACTIVE_TREATMENT = "active_treatment"


CONF_MAP: Dict[str, int] = {
    "uncertain": 0,
    "borderline": 1,
    "clear": 2,
}

WEIGHT_MAP: Dict[str, int] = {
    "not_used": 0,
    "noted": 1,
    "important": 2,
    "decisive": 3,
}

WEIGHT_ALIAS: Dict[str, str] = {
    "not_revealed": "not_used",
}

IMPORTANT_OR_DECISIVE: Set[str] = {"important", "decisive"}

SECTION_KEY_TO_REVEAL_NAME: Dict[str, str] = {
    "section_s3-prev": "previous_notes",
    "section_s3-labs": "laboratory_results",
    "section_s3-psa": "psa_trend",
    "section_s3-mri": "radiology_report",
    "section_s3-path": "pathology_report",
    "section_s3-fh": "family_history",
}


@dataclass
class MetricResult:
    """Standardized return structure for any evaluated metric."""
    name: str
    value: Optional[float]
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Task1CaseResult:
    """Evaluation result for a single Task 1 case."""
    case_id: str
    decision_gate: float
    decision_gt: Optional[str]
    decision_pred: Optional[str]
    confidence_score: Optional[float]
    variable_weights_score: Optional[float]
    important_decisive_f1: Optional[float]
    tool_efficiency_score: Optional[float]
    section_grounding_score: Optional[float]
    overall_case_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task2CaseResult:
    """Evaluation result for a single Task 2 case."""
    case_id: str
    decision_gate: float
    decision_gt: Optional[str]
    decision_pred: Optional[str]
    confidence_score: Optional[float]
    variable_weights_score: Optional[float]
    important_decisive_f1: Optional[float]
    tool_efficiency_score: Optional[float]
    section_grounding_score: Optional[float]
    overall_case_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task3CaseResult:
    """Evaluation result for a single Task 3 case."""
    case_id: str
    event_score: Optional[float]
    time_score: Optional[float]
    event_gt: Optional[int]
    event_pred: Optional[int]
    time_gt: Optional[float]
    time_pred: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task3CohortResult:
    """Evaluation results for the full Task 3 cohort."""
    c_index: Optional[float]
    td_auc: Dict[float, Optional[float]]
    mean_event_score: Optional[float]
    mean_time_score: Optional[float]
    num_cases: int
    details: Dict[str, Any] = field(default_factory=dict)
