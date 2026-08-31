"""Fixed-orchestration runner (Task 1).

Pipeline per case — no ReAct, the LLM never issues tool calls:

  1. Tool A: get_biopsy_decision_support  (ensemble -> decision + prob + neighbors)
  2. Tool B: get_confidence_estimate      (KDM      -> confidence)
  3. Tool C: get_variable_weights         (ExtraTrees + SHAP -> weights)
  4. form_fill: ONE Gemma 4 E2B call that redacts the reasoning free_text
     anchored on the neighbor exemplars and transcribes the hard fields,
     validated against Task1Output (Pydantic).
  5. Hard-field guard: decision / confidence / variable_weights are
     overwritten with the submodel values before writing to disk.

Writes ``<output_dir>/task1/<case>/prediction.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from chimera_agent.tasks.task1.baseline.case_loader import load_cases
from chimera_agent.tasks.task1.baseline.form_fill import make_form_fill_node
from chimera_agent.tasks.task1.baseline.models.ollama_compat import build_model
from chimera_agent.tasks.task1.baseline.output.schema import build_reveal_sequence
from chimera_agent.tasks.task1.baseline.utils import setup_logging
from chimera_agent.tasks.task1.baseline.mcp_client import SubtaskMCPClient

log = logging.getLogger(__name__)


async def _run_case(client: SubtaskMCPClient, case: dict, model, cfg) -> dict:
    case_id = case["case_id"]
    case_path = case["case_path"]

    # --- Fixed orchestration: A -> B -> C -------------------------------
    a_raw = await client.call("get_biopsy_decision_support", {"case_path": case_path})
    b_raw = await client.call("get_confidence_estimate", {"case_path": case_path})
    c_raw = await client.call("get_variable_weights", {"case_path": case_path})

    tool_outputs = {
        "biopsy_support": json.loads(a_raw),
        "confidence_estimate": json.loads(b_raw),
        "variable_weights_estimate": json.loads(c_raw),
    }
    log.info(
        "case %s | A.decision=%s p=%.4f | B.confidence=%s | C.decisive=%s",
        case_id,
        tool_outputs["biopsy_support"]["decision"],
        tool_outputs["biopsy_support"]["biopsy_probability"],
        tool_outputs["confidence_estimate"]["confidence"],
        ",".join(tool_outputs["variable_weights_estimate"]["decisive_factors"]) or "-",
    )

    # --- Single LLM synthesis call ---------------------------------------
    form_fill = make_form_fill_node(model, max_retries=cfg.agent.form_fill.max_retries)
    state = {
        "case_id": case_id,
        "task": 1,
        "context": case["context"],
        "tool_outputs": tool_outputs,
    }
    result = form_fill(state)
    payload = result["structured_response"]

    # --- Hard-field guard: submodels win, always -------------------------
    payload["case_id"] = case_id
    payload["task"] = 1
    payload["biopsy_decision"] = tool_outputs["biopsy_support"]["decision"] == "yes"
    payload["confidence"] = tool_outputs["confidence_estimate"]["confidence"]
    for var, w in tool_outputs["variable_weights_estimate"]["variable_weights"].items():
        payload.setdefault("variable_weights", {})[var] = w
    # Reveal sequence derived deterministically from the assigned weights.
    payload["reveal_sequence"] = build_reveal_sequence(payload["variable_weights"])

    return payload


async def run_agent(cfg) -> None:
    setup_logging(cfg.logging.level)

    data_root = Path(cfg.paths.data_root)
    output_dir = Path(cfg.paths.output_dir)
    cases_dir = data_root / "task1" / "agent_input"

    # Resolve templates relative to agent package or GC mount; fallback to relative.
    _tmpl = Path(__file__).resolve().parents[5] / "templates" / "prompts" / "task1"
    _templates_dir = str(_tmpl) if _tmpl.is_dir() else "templates/prompts/task1"
    cases = load_cases(cases_dir, templates_dir=_templates_dir)
    if cfg.agent.pids:
        cases = [c for c in cases if c["case_id"] in set(cfg.agent.pids)]
    if cfg.agent.limit:
        cases = cases[: int(cfg.agent.limit)]
    log.info("Loaded %d task1 cases from %s", len(cases), cases_dir)

    model = build_model(cfg)
    import os as _os
    _os.environ["TASK_ID"] = str(getattr(getattr(cfg, "agent", {}), "task_id", 1) or 1)
    async with SubtaskMCPClient(cfg) as client:
        for case in cases:
            out_dir = output_dir / "task1" / case["case_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                payload = await _run_case(client, case, model, cfg)
            except Exception:
                log.exception("case %s failed — aborting loudly", case["case_id"])
                raise

            # GC contract: two flat files.
            decision = "yes" if payload["biopsy_decision"] else "no"
            reasoning = {
                "confidence": payload["confidence"],
                "variable_weights": payload["variable_weights"],
                "reveal_sequence": payload["reveal_sequence"],
                "free_text": payload["reasoning"],
            }
            (out_dir / "prostate-biopsy-decision.json").write_text(json.dumps(decision, indent=4))
            (out_dir / "prostate-biopsy-decision-reasoning.json").write_text(json.dumps(reasoning, indent=4))
            (out_dir / "prediction.json").write_text(json.dumps(payload, indent=4))
            log.info("case %s done -> %s", case["case_id"], out_dir)


def main() -> None:
    cfg = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
    asyncio.run(run_agent(cfg))


if __name__ == "__main__":
    main()
