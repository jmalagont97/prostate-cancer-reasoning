"""Fixed-orchestration Task 3 runner: Cox tool first, Gemma reasoning second."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from chimera_agent.tasks.task1.baseline.mcp_client import SubtaskMCPClient
from chimera_agent.tasks.task1.baseline.models.ollama_compat import build_model
from chimera_agent.tasks.task3.baseline.form_fill import make_reasoning_node
from chimera_agent.tasks.task3.baseline.output.schema import Task3Outcome, Task3Reasoning

log = logging.getLogger(__name__)
_TOOL = "predict_bcr_task3"
_OUTCOME = "prostate-time-to-recurrence-or-last-follow-up.json"
_REASONING = "prostate-time-to-recurrence-or-last-follow-up-reasoning.json"


def _load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for prompt_path in sorted(cases_dir.glob("*/structured-prompt.json")):
        payload = json.loads(prompt_path.read_text())
        case_id = str(payload.get("case_id", prompt_path.parent.name))
        cases.append({"case_id": case_id, "case_path": str(prompt_path.parent), "context": json.dumps(payload, indent=2)})
    return cases


async def _run_case(client, model, case: dict) -> tuple[dict, str]:
    raw = await client.call(_TOOL, {"case_path": case["case_path"]})
    prediction = json.loads(raw)
    outcome = Task3Outcome.model_validate(prediction)
    reasoning = make_reasoning_node(model)({"case_id": case["case_id"], "context": case["context"], "prediction": outcome.model_dump()})
    Task3Reasoning.model_validate(reasoning)
    return outcome.model_dump(), reasoning


async def run_task3(cfg) -> None:
    cases_dir = Path(str(cfg.paths.data_root)) / "task3" / "agent_input"
    output_dir = Path(str(cfg.paths.output_dir)) / "task3"
    cases = _load_cases(cases_dir)
    if cfg.agent.pids:
        allowed = set(cfg.agent.pids)
        cases = [case for case in cases if case["case_id"] in allowed]
    if cfg.agent.limit:
        cases = cases[: int(cfg.agent.limit)]
    model = build_model(cfg)
    async with SubtaskMCPClient(cfg) as client:
        for case in cases:
            outcome, reasoning = await _run_case(client, model, case)
            out_dir = output_dir / case["case_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / _OUTCOME).write_text(json.dumps(outcome, indent=4))
            (out_dir / _REASONING).write_text(json.dumps(reasoning, indent=4))
            log.info("Task 3 case %s done -> %s", case["case_id"], out_dir)


def main() -> None:
    import asyncio
    from omegaconf import OmegaConf
    import sys
    asyncio.run(run_task3(OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")))


if __name__ == "__main__":
    main()
