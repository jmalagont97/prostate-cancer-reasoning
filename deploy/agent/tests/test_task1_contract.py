"""Task 1 contract tests — preserve baseline output schema."""

import json

import pytest
from pydantic import ValidationError

from chimera_agent.tasks.task1.baseline.output.schema import Task1Output


def test_task1_output_valid_minimal():
    # Minimal valid payload per baseline schema
    payload = {
        "case_id": "case_001",
        "task": 1,
        "biopsy_decision": True,
        "confidence": "borderline",
        "variable_weights": {"psa": "important", "pirads": "decisive"},
        "reveal_sequence": ["laboratory_results", "radiology_report"],
        "reasoning": "This is a sufficiently long reasoning free_text with more than forty characters anchored on neighbor exemplars.",
    }
    obj = Task1Output.model_validate(payload)
    assert obj.case_id == "case_001"
    assert obj.biopsy_decision is True


def test_task1_output_rejects_extra_keys():
    payload = {
        "case_id": "case_001",
        "task": 1,
        "biopsy_decision": False,
        "confidence": "clear",
        "variable_weights": {},
        "reveal_sequence": [],
        "reasoning": "x" * 50,
        "extra_key": "should_fail",
    }
    with pytest.raises(ValidationError):
        Task1Output.model_validate(payload)


def test_task1_output_rejects_short_reasoning():
    payload = {
        "case_id": "case_001",
        "task": 1,
        "biopsy_decision": False,
        "confidence": "uncertain",
        "variable_weights": {},
        "reveal_sequence": [],
        "reasoning": "too short",
    }
    with pytest.raises(ValidationError):
        Task1Output.model_validate(payload)


def test_reveal_sequence_helper():
    from chimera_agent.tasks.task1.baseline.output.schema import build_reveal_sequence

    weights = {"psa": "decisive", "pirads": "important", "age": "noted"}
    seq = build_reveal_sequence(weights)
    assert isinstance(seq, list)
    # age and baseline PSA are always available; only PIRADS reveals MRI.
    assert seq == ["radiology_report"]


def test_reveal_sequence_maps_pathology_and_psa_trend():
    from chimera_agent.tasks.task1.baseline.output.schema import build_reveal_sequence

    weights = {"bx": "decisive", "psav": "important"}
    assert build_reveal_sequence(weights) == [
        "previous_notes",
        "psa_trend",
        "pathology_report",
    ]


def test_reveal_sequence_excludes_not_used_variables():
    from chimera_agent.tasks.task1.baseline.output.schema import build_reveal_sequence

    weights = {"age": "decisive", "psa": "important", "pirads": "not_used"}
    assert build_reveal_sequence(weights) == []
