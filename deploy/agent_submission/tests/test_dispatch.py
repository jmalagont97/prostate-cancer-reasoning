"""Tests for the generalist dispatcher."""

import pytest

from chimera_agent.dispatch import validate_task_id, get_task_runner


def test_validate_accepts_1_2_3():
    for tid in (1, 2, 3):
        assert validate_task_id(tid) == tid


@pytest.mark.parametrize("bad", [0, 4, -1, 99, "1", 1.0, None])
def test_validate_rejects(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_task_id(bad)


def test_get_task1_runner_is_callable():
    r = get_task_runner(1)
    assert callable(r)
    assert r.__name__ == "run_task1"


def test_task2_and_3_runners_are_callable():
    import asyncio

    async def _check(tid):
        runner = get_task_runner(tid)
        assert callable(runner)

    asyncio.run(_check(2))
    asyncio.run(_check(3))


def test_invalid_task_raises_before_model_load():
    with pytest.raises(ValueError, match="Unsupported task_id"):
        get_task_runner(99)
