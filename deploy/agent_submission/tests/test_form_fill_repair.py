"""Tests for the LLM JSON repair path (single quotes, fences, trailing commas)."""

import json

from chimera_agent.tasks.task1.baseline.form_fill import (
    _extract_json_object,
    _repair_json,
)


def test_repair_single_quoted_keys_and_values():
    raw = """{
  'case_id': 'case_001',
  'task': 1,
  'biopsy_decision': true,
  'confidence': 'clear',
  'variable_weights': {'psa': 'important'},
  'reasoning': 'PSA 3.7 ng/mL with PI-RADS 2 supports deferring biopsy.',
}"""
    repaired = _repair_json(raw)
    obj = json.loads(repaired)
    assert obj["case_id"] == "case_001"
    assert obj["task"] == 1
    assert obj["biopsy_decision"] is True
    assert obj["confidence"] == "clear"
    assert obj["variable_weights"]["psa"] == "important"


def test_repair_drops_fences_and_trailing_commas():
    raw = '```json\n{"case_id": "c1", "task": 1, "reasoning": "x",}\n```'
    extracted = _extract_json_object(raw)
    repaired = _repair_json(extracted)
    obj = json.loads(repaired)
    assert obj["case_id"] == "c1"
