import json
from pathlib import Path

from chimera_agent.tasks.task3.baseline.output.schema import Task3Outcome, Task3Reasoning


def test_task3_outcome_matches_official_fields():
    outcome = Task3Outcome.model_validate({"months_to_recurrence": 14.9, "event": 0})
    assert outcome.model_dump() == {"months_to_recurrence": 14.9, "event": 0}


def test_task3_outcome_rejects_extra_fields():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Task3Outcome.model_validate({"months_to_recurrence": 14.9, "event": 0, "reasoning": "x"})


def test_task3_reasoning_is_json_string():
    reasoning = Task3Reasoning.model_validate("Clinical justification.")
    assert reasoning.root == "Clinical justification."


def test_task3_prompt_template_exists():
    path = Path("deploy/agent/templates/prompts/task3/agent_prompt.j2")
    assert path.is_file()
    text = path.read_text()
    assert "months_to_recurrence" in text
    assert "event" in text
    assert "Gemma" in text
