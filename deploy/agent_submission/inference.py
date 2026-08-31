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


def _detect_generic_input_task(input_dir: Path) -> int | None:
    """Return the task declared by direct case directories, rejecting mixed batches."""
    task_ids: set[int] = set()
    for case_dir in sorted(input_dir.iterdir()):
        prompt = case_dir / "structured-prompt.json"
        if not case_dir.is_dir() or not prompt.is_file():
            continue
        payload = json.loads(prompt.read_text())
        if "task" in payload:
            task_ids.add(validate_task_id(int(payload["task"])))
    if len(task_ids) > 1:
        raise ValueError(f"Generic /input batch contains mixed tasks: {sorted(task_ids)}")
    return next(iter(task_ids), None)


def _detect_input_task() -> int | None:
    """Detect task from every supported GC layout; never infer Task 1 by default."""
    input_dir = Path("/input")
    if not input_dir.is_dir():
        return None

    task_ids: set[int] = set()
    flat_prompt = input_dir / "structured-prompt.json"
    if flat_prompt.is_file():
        payload = json.loads(flat_prompt.read_text())
        if "task" in payload:
            task_ids.add(validate_task_id(int(payload["task"])))

    # Canonical nested GC layout: /input/taskN/agent_input/<case>/structured-prompt.json
    for task_dir in sorted(input_dir.glob("task[123]")):
        if not task_dir.is_dir():
            continue
        task_id = validate_task_id(int(task_dir.name.removeprefix("task")))
        agent_input = task_dir / "agent_input"
        if not agent_input.is_dir():
            continue
        declared_ids: set[int] = set()
        for prompt in agent_input.glob("*/structured-prompt.json"):
            payload = json.loads(prompt.read_text())
            if "task" in payload:
                declared_ids.add(validate_task_id(int(payload["task"])))
        if len(declared_ids) > 1 or (declared_ids and task_id not in declared_ids):
            raise ValueError(f"Task id conflicts with nested input layout at {task_dir}")
        task_ids.add(task_id)

    direct_task = _detect_generic_input_task(input_dir)
    if direct_task is not None:
        task_ids.add(direct_task)
    if len(task_ids) > 1:
        raise ValueError(f"Input contains mixed or conflicting tasks: {sorted(task_ids)}")
    return next(iter(task_ids), None)


def stage_gc_input_if_needed(cfg) -> None:
    """If running on GC (/input exists), stage flat structured-prompt.json into baseline layout.

    Baseline expects <data_root>/task1/agent_input/<case_id>/structured-prompt.json.
    GC may provide /input/structured-prompt.json (single case) or /input/task1/...
    This helper creates /tmp/gc_cases so baseline loader works unmodified.
    """
    gc_flat = Path("/input/structured-prompt.json")
    detected_task = _detect_input_task()
    if detected_task is None and Path("/input").is_dir() and any(Path("/input").iterdir()):
        raise ValueError("Cannot determine task_id from /input; refusing to run")
    task_id = detected_task
    if task_id is None:
        task_id = validate_task_id(int(getattr(cfg.agent, "task_id", 1)))
    cfg.agent.task_id = task_id
    if gc_flat.is_file():
        try:
            payload_task = json.loads(gc_flat.read_text()).get("task")
            if payload_task is not None:
                task_id = validate_task_id(int(payload_task))
                cfg.agent.task_id = task_id
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # Generic GC input: infer the task from the first case before selecting
    # the staging namespace. The base config defaults to Task 1, but a batch
    # may contain structured prompts for another task.
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
    detected_task = _detect_input_task()
    if detected_task is not None:
        return detected_task

    # Prefer explicit cfg only when no task can be read from the input.
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
    raise ValueError("Cannot determine task_id; refusing to run")


def _task_from_config_path(config_path: str | Path) -> int | None:
    """Return a task only for an explicitly task-scoped config file."""
    name = Path(config_path).name
    if not (name.startswith("task") and name.endswith(".yaml")):
        return None
    try:
        return validate_task_id(int(name[4:-5]))
    except (ValueError, TypeError):
        return None


_GC_OUTPUTS = {
    1: (
        "prostate-biopsy-decision.json",
        "prostate-biopsy-decision-reasoning.json",
    ),
    2: (
        "prostate-treatment-decision.json",
        "prostate-treatment-decision-reasoning.json",
    ),
    3: (
        "prostate-time-to-recurrence-or-last-follow-up.json",
        "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
    ),
}


def publish_gc_outputs(task_id: int, output_dir: str | Path = "/output") -> None:
    """Expose exactly the official output pair directly under Grand Challenge's output root."""
    task_id = validate_task_id(task_id)
    root = Path(output_dir)
    names = _GC_OUTPUTS[task_id]
    selected: dict[str, Path] = {}

    for name in names:
        direct = root / name
        nested = [path for path in root.glob(f"task{task_id}/**/{name}") if path.is_file()]
        if direct.is_file():
            selected[name] = direct
        elif nested:
            selected[name] = nested[0]
        else:
            raise FileNotFoundError(
                f"Required Grand Challenge output is missing for Task {task_id}: {name}"
            )

    root.mkdir(parents=True, exist_ok=True)
    for name, source in selected.items():
        try:
            content = source.read_text(encoding="utf-8")
            json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON in Grand Challenge output: {source}") from exc
        destination = root / name
        if source.resolve() != destination.resolve():
            destination.write_text(content, encoding="utf-8")


def run_inference(config_path: str | Path = "configs/config.yaml") -> None:
    parser = argparse.ArgumentParser(description="CHIMERA Generalist Agent inference")
    config_path = Path(config_path)
    cfg = load_merged_config(config_path)

    input_task = _detect_input_task()
    config_task = _task_from_config_path(config_path)
    if input_task is not None and config_task is not None and input_task != config_task:
        raise ValueError(
            f"Task id conflict: input declares Task {input_task}, "
            f"but config declares Task {config_task}; refusing to run"
        )

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
    override_path = config_path.parent / "tasks" / f"task{task_id}.yaml"
    if override_path.is_file():
        override = OmegaConf.load(str(override_path))
        cfg = OmegaConf.merge(cfg, override)
    cfg.agent.task_id = task_id  # type: ignore[attr-defined]
    cfg.agent.tasks = [task_id]  # type: ignore[attr-defined]

    print(f"[inference] dispatching Task {task_id}", file=sys.stderr)
    runner = get_task_runner(task_id)
    import asyncio

    asyncio.run(runner(cfg))
    output_dir = str(getattr(getattr(cfg, "paths", None), "output_dir", "/output"))
    publish_gc_outputs(task_id, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="CHIMERA Generalist Agent inference")
    parser.add_argument("--config", default="configs/config.yaml", help="Base config path")
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
