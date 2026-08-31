"""No fake predictions validation."""

import pytest

from chimera_agent.dispatch import get_task_runner


def test_invalid_task_does_not_load_models():
    # Validation happens before any model load; we test that no file is accessed
    # Indirectly: get_task_runner raises before touching SUBMODELS_DIR
    with pytest.raises(ValueError):
        get_task_runner(99)
    with pytest.raises(ValueError):
        get_task_runner(0)
