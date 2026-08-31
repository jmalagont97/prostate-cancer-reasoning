"""Structured output contract — Task 2 only (fixed, mirrors GC Interface 1 submission)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Weight(StrEnum):
    NOT_USED = "not_used"
    NOTED = "noted"
    IMPORTANT = "important"
    DECISIVE = "decisive"


class Confidence(StrEnum):
    CLEAR = "clear"
    BORDERLINE = "borderline"
    UNCERTAIN = "uncertain"


class TreatmentDecision(StrEnum):
    ACTIVE_SURVEILLANCE = "Active Surveillance"
    RADICAL_PROSTATECTOMY = "Radical Prostatectomy"
    RADIATION_THERAPY = "Radiation Therapy"
    ANDROGEN_DEPRIVATION_THERAPY = "Androgen Deprivation Therapy"


# ---------------------------------------------------------------------------
# Reveal-sequence correspondence for Task 2.
#
# Deterministic evidence-source projection derived from the submodel-assigned
# variable weights.
# ---------------------------------------------------------------------------


class RevealCategory(StrEnum):
    FAMILY_HISTORY = "family_history"
    PREVIOUS_NOTES = "previous_notes"
    LABORATORY_RESULTS = "laboratory_results"
    PSA_TREND = "psa_trend"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"


# Variables available in the basal patient context do not require grounding.
ALWAYS_AVAILABLE_VARIABLES = {"age", "psa"}

# Task-2 variable -> one or more normative evidence-source categories.
VARIABLE_REVEAL_MAP: dict[str, tuple[str, ...]] = {
    "age": (),
    "psa": (),
    "psad": (RevealCategory.RADIOLOGY_REPORT.value,),
    "pirads": (RevealCategory.RADIOLOGY_REPORT.value,),
    "ct": (RevealCategory.RADIOLOGY_REPORT.value,),
    "cspca": (RevealCategory.RADIOLOGY_REPORT.value,),
    "fh": (RevealCategory.FAMILY_HISTORY.value,),
    "bx_isup": (RevealCategory.PATHOLOGY_REPORT.value,),
    "bx_gl_prim": (RevealCategory.PATHOLOGY_REPORT.value,),
    "bx_gl_sec": (RevealCategory.PATHOLOGY_REPORT.value,),
    "comorbidity": (),
}

# Canonical order is deterministic only.
_CANONICAL_REVEAL_ORDER = (
    RevealCategory.FAMILY_HISTORY.value,
    RevealCategory.PREVIOUS_NOTES.value,
    RevealCategory.LABORATORY_RESULTS.value,
    RevealCategory.PSA_TREND.value,
    RevealCategory.RADIOLOGY_REPORT.value,
    RevealCategory.PATHOLOGY_REPORT.value,
)
_WEIGHT_RANK = {"decisive": 3, "important": 2, "noted": 1, "not_used": 0}


def build_reveal_sequence(variable_weights: dict[str, str]) -> list[str]:
    """Project active weighted variables onto normative reveal categories."""
    active_categories: set[str] = set()
    for variable, weight in variable_weights.items():
        if variable in ALWAYS_AVAILABLE_VARIABLES:
            continue
        if _WEIGHT_RANK.get(str(weight), 0) <= 0:
            continue
        active_categories.update(VARIABLE_REVEAL_MAP.get(variable, ()))

    return [
        category
        for category in _CANONICAL_REVEAL_ORDER
        if category in active_categories
    ]


class Task2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: Literal[2] = 2
    treatment_decision: TreatmentDecision
    confidence: Confidence
    variable_weights: dict[str, Weight] = Field(default_factory=dict)
    reveal_sequence: list[
        Literal[
            "family_history",
            "previous_notes",
            "laboratory_results",
            "psa_trend",
            "radiology_report",
            "pathology_report",
        ]
    ] = Field(default_factory=list)
    reasoning: str = Field(min_length=40)
