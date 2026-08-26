#!/usr/bin/env python3
"""Run the Task-1 agent on every locally available case for exp_10."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGENT = ROOT / "deploy" / "agent"
SOURCE = ROOT / "data" / "raw" / "train_release" / "task1"
RUNTIME = ROOT / "experiments" / "exp_10" / "runtime"
RESULTS = ROOT / "experiments" / "exp_10" / "results" / "deploy_agent"
MODELS = ROOT / "deploy" / "tasks" / "task1" / "training" / "models"


def discover() -> list[Path]:
    return sorted(
        p for p in SOURCE.iterdir()
        if p.is_dir()
        and (p / "structured-prompt.json").is_file()
    )


def stage(cases: list[Path]) -> list[str]:
    base = RUNTIME / "task1" / "agent_input"
    base.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases:
        target = base / case.name
        if target.exists() or target.is_symlink():
            target.unlink() if target.is_symlink() else shutil.rmtree(target)
        target.symlink_to(case, target_is_directory=True)
        manifest.append(case.name)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cases = discover()
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No Task-1 cases with structured-prompt.json found")
    ids = stage(cases)

    sys.path.insert(0, str(AGENT / "src"))
    from omegaconf import OmegaConf
    from chimera_agent.dispatch import get_task_runner

    cfg = OmegaConf.load(str(AGENT / "configs" / "config.yaml"))
    cfg.paths.data_root = str(RUNTIME)
    cfg.paths.output_dir = str(RESULTS / "outputs")
    cfg.paths.submodels_dir = str(MODELS)
    cfg.agent.pids = ids
    cfg.agent.limit = len(ids)
    cfg.logging.level = "WARNING"

    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    print(f"[exp10] starting 0/{len(ids)} cases", flush=True)
    start = time.monotonic()
    asyncio.run(get_task_runner(1)(cfg))
    elapsed = time.monotonic() - start
    print(f"[exp10] runner finished {len(ids)}/{len(ids)} cases elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
