"""Task 1 — Biopsy decision (fully implemented via baseline)."""

from __future__ import annotations

from chimera_agent.tasks.task1.baseline.run import run_agent as _baseline_run_agent


async def run_task1(cfg):
    """Entry point for Task 1: delegates to the conserved baseline orchestration."""
    await _baseline_run_agent(cfg)
