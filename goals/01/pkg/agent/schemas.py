#!/usr/bin/env python3
"""
agent/schemas.py

Pydantic schemas defining the ground-truth Pathology Reasoning output structure.
Identical to raw dataset 'prostate-biopsy-decision-reasoning.json'.
"""

from typing import Literal, List
from pydantic import BaseModel, Field


class VariableWeights(BaseModel):
    age: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for patient age")
    fh: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for family history")
    cspca: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for csPCa history")
    pirads: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for PI-RADS score")
    vol: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for prostate volume")
    psa: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for PSA level")
    comorbidity: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for comorbidities")
    psad: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for PSA density")
    dre: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for DRE findings")
    bx: Literal["not_used", "noted", "important", "decisive"] = Field(..., description="Weight for prior biopsy")


class PathologyReasoningOutput(BaseModel):
    biopsy_decision: Literal["yes", "no"] = Field(
        ..., description="Final clinical recommendation on prostate biopsy"
    )
    confidence: Literal["uncertain", "borderline", "clear"] = Field(
        ..., description="Clinical confidence grade"
    )
    variable_weights: VariableWeights = Field(
        ..., description="Importance weights assigned to the 10 target clinical variables"
    )
    reveal_sequence: List[str] = Field(
        ..., description="Ordered list of evidence sections to reveal"
    )
    free_text: str = Field(
        ..., description="Clinical rationale summarizing the reasoning for the decision"
    )
