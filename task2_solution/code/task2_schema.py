"""Task2 output contract. Verified from `external/CHIMERA-agent/evaluation/evaluate.py`
(`process_interf1`, `VALID_TREATMENT_DECISIONS`, `GT_FILENAMES["task2"]`), not the webpage alone --
mirrors task1's schema.py exactly: two output files, same reasoning-object shape.

Two independent sources confirm the shape: process_interf1() reads
`prostate-treatment-decision.json` (a bare string) + `prostate-treatment-decision-reasoning.json`
(confidence/variable_weights/reveal_sequence/free_text), flattening the decision into
`treatment_recommendation.primary` only for scoring -- the ON-DISK file is the bare string, same
convention as task1's `biopsy_decision`.
"""

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
    WATCHFUL_WAITING = "watchful_waiting"
    ACTIVE_SURVEILLANCE = "active_surveillance"
    CONTINUED_SURVEILLANCE = "continued_surveillance"
    ACTIVE_TREATMENT = "active_treatment"


# variable -> tool required for grounding eligibility, confirmed from
# external/CHIMERA-agent/evaluation/ground_truth/section_variable_mapping.json's
# variable_to_sections (primary_sections), NOT assumed from the webpage's prose description.
# None = always-available (readable from the patient card, no reveal needed) or structurally
# ungroundable (comorbidity's primary section, section_s3-comorb, has no reveal-vocabulary entry
# at all -- same category as task1's "bx").
TASK2_VARIABLES: dict[str, str | None] = {
    "age": None,
    "psa": None,
    "psad": "get_mri_report",
    "pirads": "get_mri_report",
    "cspca": "get_mri_report",
    "ct": "get_mri_report",
    "bx_isup": "get_pathology_report",
    "bx_gl_prim": "get_pathology_report",
    "bx_gl_sec": "get_pathology_report",
    "comorbidity": None,  # structurally ungroundable, see STRUCTURALLY_UNGROUNDABLE_TASK2 below
    "fh": "get_family_history",
}

# comorbidity's primary_sections (section_s3-comorb) is not in SECTION_KEY_TO_REVEAL_NAME at all
# (only prev/labs/psa/mri/path/fh are) -- no reveal_sequence entry can ever ground it, confirmed
# directly from evaluate.py's SECTION_KEY_TO_REVEAL_NAME and section_variable_mapping.json.
# Mirrors task1's STRUCTURALLY_UNGROUNDABLE_TASK1 = {"bx"} exactly.
STRUCTURALLY_UNGROUNDABLE_TASK2 = {"comorbidity"}


def eligible_variables(called_tools: set[str]) -> list[str]:
    return [v for v, req in TASK2_VARIABLES.items() if req is None or req in called_tools]


class Task2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    treatment_recommendation: TreatmentDecision
    confidence: Confidence
    variable_weights: dict[str, Weight] = Field(default_factory=dict)
    reveal_sequence: list[Literal[
        "family_history", "previous_notes", "laboratory_results", "psa_trend", "radiology_report",
        "pathology_report",
    ]] = Field(default_factory=list)
    reasoning: str = Field(min_length=40)


def normalise_to_full_shape(raw_weights: dict) -> dict:
    weights = dict(raw_weights)
    for var in TASK2_VARIABLES:
        weights.setdefault(var, Weight.NOT_USED.value)
    return weights


def enforce_eligibility(raw_weights: dict, called_tools: set[str], clamp_ungroundable: bool = False) -> dict:
    """Zero out any weight for a variable that isn't eligible given the tools actually called.
    clamp_ungroundable defaults False, mirroring task1's measured finding that clamping the
    structurally-ungroundable variable costs more on variable_weight_score/factor_f1 than it gains
    on section_grounding_score -- not yet re-measured for task2's `comorbidity`, kept as an
    available, honest choice rather than assumed to transfer."""
    eligible = set(eligible_variables(called_tools))
    out = dict(raw_weights)
    for var in TASK2_VARIABLES:
        if var not in eligible:
            out[var] = Weight.NOT_USED.value
    if clamp_ungroundable:
        for var in STRUCTURALLY_UNGROUNDABLE_TASK2:
            out[var] = Weight.NOT_USED.value
    return out
