"""Task 2 Contract and Output Schema Tests."""

import pytest

from chimera_agent.tasks.task2.baseline.output.schema import (
    Confidence,
    Task2Output,
    TreatmentDecision,
    Weight,
    build_reveal_sequence,
)


def test_task2_output_validation_valid():
    weights = {
        "age": Weight.IMPORTANT,
        "bx_gl_prim": Weight.IMPORTANT,
        "bx_gl_sec": Weight.DECISIVE,
        "bx_isup": Weight.DECISIVE,
        "comorbidity": Weight.NOTED,
        "cspca": Weight.NOT_USED,
        "ct": Weight.NOTED,
        "fh": Weight.NOTED,
        "pirads": Weight.IMPORTANT,
        "psa": Weight.NOTED,
        "psad": Weight.NOTED,
    }
    rev_seq = build_reveal_sequence({k: v.value for k, v in weights.items()})

    out = Task2Output(
        case_id="T2-004",
        task=2,
        treatment_decision=TreatmentDecision.ACTIVE_SURVEILLANCE,
        confidence=Confidence.CLEAR,
        variable_weights=weights,
        reveal_sequence=rev_seq,  # type: ignore[arg-type]
        reasoning="Patient with high-risk localized prostate cancer with Gleason 4+3 and PSA 12 ng/mL.",
    )

    data = out.model_dump(mode="json")
    assert data["treatment_decision"] == "Active Surveillance"
    assert data["confidence"] == "clear"
    assert "pathology_report" in data["reveal_sequence"]
    assert "radiology_report" in data["reveal_sequence"]
    assert "family_history" in data["reveal_sequence"]


def test_task2_reveal_sequence_empty_when_all_not_used():
    weights = {k: "not_used" for k in ["age", "psa", "bx_isup", "pirads", "fh"]}
    assert build_reveal_sequence(weights) == []


@pytest.mark.parametrize(
    "decision",
    [
        "Active Surveillance",
        "Radical Prostatectomy",
        "Radiation Therapy",
        "Androgen Deprivation Therapy",
    ],
)
def test_task2_treatment_contract_uses_model_classes(decision):
    assert TreatmentDecision(decision).value == decision


@pytest.mark.parametrize("decision", ["no", "yes", "active_treatment", "watchful_waiting"])
def test_task2_treatment_contract_rejects_non_model_classes(decision):
    with pytest.raises(ValueError):
        TreatmentDecision(decision)
