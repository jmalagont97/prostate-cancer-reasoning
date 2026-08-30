#!/usr/bin/env python3
"""Generalist inference entrypoint (GC-compatible).

- Loads base config explicitly and merges the task-specific override via
  OmegaConf.merge (never relying on Hydra defaults handling).
- Detects the GC input at /input/structured-prompt.json or /input/task1/...
  and stages it so the conserved Task-1 baseline loader (task1/agent_input)
  can find it without any baseline modification.
- Validates task_id via the dispatcher BEFORE loading any pickles.
- Dispatches to the task-specific runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

from chimera_agent.dispatch import validate_task_id, get_task_runner


def load_merged_config(base_path: str | Path) -> OmegaConf:
    base_path = Path(base_path)
    base = OmegaConf.load(str(base_path))
    # Determine intended task_id: from base or from CLI-merged? For now base's task_id.
    # If base has task_id, try to merge its task override explicitly.
    task_id = None
    try:
        task_id = int(base.agent.task_id)  # type: ignore[attr-defined]
    except Exception:
        task_id = None
    if task_id is not None:
        override_path = base_path.parent / "tasks" / f"task{task_id}.yaml"
        if override_path.is_file():
            override = OmegaConf.load(str(override_path))
            # OmegaConf.merge handles deep merge; override wins.
            base = OmegaConf.merge(base, override)
    return base


def stage_gc_input_if_needed(cfg) -> None:
    """If running on GC (/input exists), stage flat structured-prompt.json into baseline layout.

    Baseline expects <data_root>/task1/agent_input/<case_id>/structured-prompt.json.
    GC may provide /input/structured-prompt.json (single case) or /input/task1/...
    This helper creates /tmp/gc_cases so baseline loader works unmodified.
    """
    gc_flat = Path("/input/structured-prompt.json")
    task_id = int(getattr(cfg.agent, "task_id", 1) or 1)
    if gc_flat.is_file():
        try:
            payload_task = json.loads(gc_flat.read_text()).get("task")
            if payload_task is not None:
                task_id = validate_task_id(int(payload_task))
                cfg.agent.task_id = task_id
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    task_slug = f"task{task_id}"
    gc_task_dir = Path("/input") / task_slug / "agent_input"
    data_root = Path(str(cfg.paths.data_root))
    output_dir = Path(str(cfg.paths.output_dir))

    # Ensure output_dir exists (GC expects /output)
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if gc_flat.is_file():
        # Single-case GC contract: stage it under /tmp/gc_cases/task1/agent_input/<case_id>/
        try:
            payload = json.loads(gc_flat.read_text())
            case_id = str(payload.get("case_id", "case"))
        except Exception:
            case_id = "case"
        tmp_root = Path("/tmp/gc_cases")
        target_dir = tmp_root / task_slug / "agent_input" / case_id
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(gc_flat), str(target_dir / "structured-prompt.json"))
        # Preserve companion files required by the trained predict_raw() models
        # (clinical text/history and MRI neural representation) alongside the
        # structured prompt. The model tools read these files from case_path.
        for companion in Path("/input").iterdir():
            if companion.is_file() and companion.name != "structured-prompt.json":
                shutil.copy2(str(companion), str(target_dir / companion.name))
        cfg.paths.data_root = str(tmp_root)
        cfg.paths.input_dir = "/input"
        cfg.paths.output_dir = str(output_dir)
        print(f"[inference] staged GC flat input {gc_flat} -> {target_dir}/structured-prompt.json", file=sys.stderr)
        return

    if gc_task_dir.is_dir():
        # GC already provides the baseline layout under /input
        cfg.paths.data_root = "/input"
        print(f"[inference] GC task layout detected at {gc_task_dir}", file=sys.stderr)
        return

    if Path("/input").is_dir() and any(Path("/input").iterdir()):
        # Generic GC: /input contains case subdirs directly
        # Check if any subdir has structured-prompt.json
        has_cases = any((p / "structured-prompt.json").exists() for p in Path("/input").iterdir() if p.is_dir())
        if has_cases:
            tmp_root = Path("/tmp/gc_cases")
            target_base = tmp_root / task_slug / "agent_input"
            target_base.mkdir(parents=True, exist_ok=True)
            for p in Path("/input").iterdir():
                if p.is_dir() and (p / "structured-prompt.json").exists():
                    dest = target_base / p.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(str(p), str(dest))
            cfg.paths.data_root = str(tmp_root)
            print(f"[inference] staged GC generic /input -> {target_base}", file=sys.stderr)
            return

    # Local dev: keep data_root as-is
    return


def detect_task_id(cfg) -> int:
    """Detect task_id from cfg or from GC input payload."""
    # Prefer explicit cfg
    try:
        tid = cfg.agent.task_id  # type: ignore[attr-defined]
        if tid is not None:
            return validate_task_id(int(tid))
    except Exception as e:
        # if cfg has no task_id, fall through to input inspection
        pass

    # Try GC flat payload
    gc_flat = Path("/input/structured-prompt.json")
    if gc_flat.is_file():
        try:
            payload = json.loads(gc_flat.read_text())
            if "task" in payload:
                return validate_task_id(int(payload["task"]))
        except Exception:
            pass

    # Fallback to base config default
    try:
        return validate_task_id(int(cfg.agent.task_id))  # type: ignore[attr-defined]
    except Exception:
        pass
    # Ultimate fallback: Task 1 (conserved)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="CHIMERA Generalist Agent inference")
    parser.add_argument("--config", default="configs/config.yaml", help="Base config path")
    args = parser.parse_args()

    cfg = load_merged_config(args.config)

    # Stage GC input before anything else, then re-merge task override if task_id changed from input
    stage_gc_input_if_needed(cfg)

    # Detect final task_id (from cfg or from GC payload) and validate early
    task_id = detect_task_id(cfg)
    validate_task_id(task_id)

    # If detected task_id differs from cfg, re-merge that task's override
    try:
        cfg_tid = int(cfg.agent.task_id)  # type: ignore[attr-defined]
    except Exception:
        cfg_tid = None
    override_path = Path(args.config).parent / "tasks" / f"task{task_id}.yaml"
    if override_path.is_file():
        override = OmegaConf.load(str(override_path))
        cfg = OmegaConf.merge(cfg, override)
    cfg.agent.task_id = task_id  # type: ignore[attr-defined]
    cfg.agent.tasks = [task_id]  # type: ignore[attr-defined]

    print(f"[inference] dispatching Task {task_id}", file=sys.stderr)
    runner = get_task_runner(task_id)
    import asyncio

    asyncio.run(runner(cfg))


if __name__ == "__main__":
    main()
