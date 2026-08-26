"""No fake predictions for Tasks 2/3 — they must fail controllably."""

import pytest

from chimera_agent.dispatch import get_task_runner


@pytest.mark.parametrize("tid", [2, 3])
def test_task2_3_not_fake_predictions(tid):
    runner = get_task_runner(tid)
    # Must not return dummy JSON; must raise NotImplementedError when called
    import asyncio

    with pytest.raises(NotImplementedError):
        asyncio.run(runner(cfg=None))  # type: ignore[arg-type]


def test_invalid_task_does_not_load_models():
    # Validation happens before any model load; we test that no file is accessed
    # Indirectly: get_task_runner raises before touching SUBMODELS_DIR
    with pytest.raises(ValueError):
        get_task_runner(99)
    with pytest.raises(ValueError):
        get_task_runner(0)
