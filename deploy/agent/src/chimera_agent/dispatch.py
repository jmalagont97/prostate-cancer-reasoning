"""Task dispatcher for the generalist agent.

Validates task_id explicitly and dispatches to the task-specific runner.
No model loading occurs before validation, and Tasks 2/3 intentionally
fail with a controlled error until their implementations are supplied.
"""

from __future__ import annotations

from typing import Any

VALID_TASK_IDS = {1, 2, 3}


def validate_task_id(task_id: Any) -> int:
    """Validate task_id is an int in {1,2,3}, else raise ValueError."""
    if not isinstance(task_id, int):
        raise ValueError(f"task_id must be int, got {type(task_id).__name__}: {task_id!r}")
    if task_id not in VALID_TASK_IDS:
        raise ValueError(f"Unsupported task_id={task_id}; valid ids: {sorted(VALID_TASK_IDS)}")
    return task_id


def get_task_runner(task_id: int):
    """Return the async run_agent callable for a given task_id.

    Task 1 is fully implemented. Tasks 2/3 are not yet implemented and
    raise NotImplementedError when invoked.
    """
    validate_task_id(task_id)
    if task_id == 1:
        from chimera_agent.tasks.task1 import run_task1

        return run_task1
    if task_id == 2:
        from chimera_agent.tasks.task2 import run_task2

        return run_task2
    # task_id == 3
    from chimera_agent.tasks.task3 import run_task3

    return run_task3


def dispatch_task(task_id: int, cfg) -> Any:
    """Dispatch to the validated task runner (sync wrapper)."""
    runner = get_task_runner(validate_task_id(task_id))
    return runner
