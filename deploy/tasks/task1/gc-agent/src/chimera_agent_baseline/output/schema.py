"""Structured output contract — Task 1 only (fixed, mirrors GC submission)."""

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


# ---------------------------------------------------------------------------
# Reveal-sequence correspondence.
#
# With fixed orchestration there is no incremental tool-gated revelation;
# ``reveal_sequence`` is instead derived from the submodel-assigned variable
# weights: every variable rated above ``not_used`` contributed evidence, and
# its evidence SOURCE (category) enters the sequence. Categories mirror the
# urologist forms' export vocabulary where possible.
# ---------------------------------------------------------------------------


class RevealCategory(StrEnum):
    FAMILY_HISTORY = "family_history"
    PREVIOUS_NOTES = "previous_notes"
    LABORATORY_RESULTS = "laboratory_results"
    PSA_TREND = "psa_trend"
    RADIOLOGY_REPORT = "radiology_report"


# task1 weight-variable -> evidence source it was extracted from.
VARIABLE_REVEAL_MAP: dict[str, str] = {
    "age": RevealCategory.PREVIOUS_NOTES.value,
    "dre": RevealCategory.PREVIOUS_NOTES.value,
    "bx": RevealCategory.PREVIOUS_NOTES.value,
    "comorbidity": RevealCategory.PREVIOUS_NOTES.value,
    "fh": RevealCategory.FAMILY_HISTORY.value,
    "psa": RevealCategory.LABORATORY_RESULTS.value,
    "psad": RevealCategory.LABORATORY_RESULTS.value,
    "vol": RevealCategory.RADIOLOGY_REPORT.value,
    "cspca": RevealCategory.RADIOLOGY_REPORT.value,
    "pirads": RevealCategory.RADIOLOGY_REPORT.value,
}

# Ordinal ranking used to order the sequence: decisive sources first.
_WEIGHT_RANK = {"decisive": 3, "important": 2, "noted": 1, "not_used": 0}


def build_reveal_sequence(variable_weights: dict[str, str]) -> list[str]:
    """Evidence-source categories for all variables weighted above not_used.

    Ordered by descending assigned importance; ties keep the canonical
    VARIABLE_REVEAL_MAP order. Duplicated categories collapse to one.
    """
    ranked = sorted(
        (
            (var, w)
            for var, w in variable_weights.items()
            if _WEIGHT_RANK.get(w, 0) > 0 and var in VARIABLE_REVEAL_MAP
        ),
        key=lambda vw: -_WEIGHT_RANK.get(vw[1], 0),
    )
    seen: list[str] = []
    for var, _w in ranked:
        cat = VARIABLE_REVEAL_MAP[var]
        if cat not in seen:
            seen.append(cat)
    return seen


class Task1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: Literal[1] = 1
    biopsy_decision: bool = Field(description="True = recommend biopsy.")
    confidence: Confidence
    variable_weights: dict[str, Weight] = Field(default_factory=dict)
    reveal_sequence: list[
        Literal[
            "family_history",
            "previous_notes",
            "laboratory_results",
            "psa_trend",
            "radiology_report",
        ]
    ] = Field(default_factory=list)
    reasoning: str = Field(min_length=40)
