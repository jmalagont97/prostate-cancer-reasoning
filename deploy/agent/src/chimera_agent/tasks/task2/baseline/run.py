"""Task 2 orchestrator: connects to the FastMCP server, invokes the 3 tools,
passes context to the form-fill node, and serializes the submission files.

Guarantees:
  1. Fixed tool invocation (subtask1_ensemble -> subtask2_knn -> subtask3_extratrees).
  2. Hard-field guard: submodel outputs overwrite the LLM-emitted values.
  3. Deterministic reveal_sequence via build_reveal_sequence.
  4. Exports:
     - prostate-treatment-decision.json
     - prostate-treatment-decision-reasoning.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from chimera_agent.tasks.task1.baseline.mcp_client import SubtaskMCPClient
from chimera_agent.tasks.task1.baseline.models.ollama_compat import build_model
from chimera_agent.tasks.task2.baseline.case_loader import CaseLoader
from chimera_agent.tasks.task2.baseline.form_fill import make_form_fill_node
from chimera_agent.tasks.task2.baseline.output.schema import (
    Confidence,
    Task2Output,
    TreatmentDecision,
    Weight,
    build_reveal_sequence,
)

log = logging.getLogger(__name__)


def _parse_tool_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse tool JSON: {raw!r}") from exc


async def run_task2(cfg: Any) -> None:
    """Run Task 2 across all cases discovered in data/task2/agent_input/."""
    data_root = Path(cfg.paths.data_root)
    task_dir = data_root / "task2" / "agent_input"
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _tmpl = Path(__file__).resolve().parents[5] / "templates" / "prompts" / "task2"
    _templates_dir = str(_tmpl) if _tmpl.is_dir() else "templates/prompts/task2"
    templates_dir = Path(getattr(getattr(cfg, "paths", None), "templates_dir", _templates_dir) or _templates_dir)
    loader = CaseLoader(templates_dir=templates_dir)
    cases = loader.discover_cases(task_dir)

    if not cases:
        log.warning("No cases found in %s; exiting cleanly", task_dir)
        return

    log.info("Task 2: processing %d cases from %s", len(cases), task_dir)

    async with SubtaskMCPClient(cfg) as mcp:
        llm = build_model(cfg)
        form_fill_node = make_form_fill_node(llm)

        for case_dir in cases:
            case_id = case_dir.name
            log.info("Processing case %s", case_id)

            payload = loader.load_case_payload(case_dir)
            context = loader.render_context(payload)

            raw_1 = await mcp.call("get_treatment_decision_support", {"case_path": str(case_dir)})
            raw_2 = await mcp.call("get_confidence_estimate", {"case_path": str(case_dir)})
            raw_3 = await mcp.call("get_variable_weights", {"case_path": str(case_dir)})

            tool_outputs = {
                "treatment_decision_support": _parse_tool_json(raw_1),
                "confidence_estimate": _parse_tool_json(raw_2),
                "variable_weights_estimate": _parse_tool_json(raw_3),
            }

            state = {
                "case_id": case_id,
                "context": context,
                "tool_outputs": tool_outputs,
            }

            result = form_fill_node(state)
            raw_response = result["structured_response"]

            # --- Hard-Field Guard ---
            # Overwrite decisions with deterministic submodel outputs.
            treatment_support = tool_outputs["treatment_decision_support"]
            conf_est = tool_outputs["confidence_estimate"]
            weights_est = tool_outputs["variable_weights_estimate"]

            submodel_decision = treatment_support.get("decision")
            submodel_conf = conf_est.get("confidence")
            if not submodel_decision:
                raise ValueError(f"Task 2.1 output for {case_id} is missing 'decision'")
            if not submodel_conf:
                raise ValueError(f"Task 2.2 output for {case_id} is missing 'confidence'")
            submodel_weights = weights_est.get("variable_weights", {})

            clean_weights = {
                k: Weight(str(v))
                for k, v in submodel_weights.items()
            }
            computed_reveal_seq = build_reveal_sequence(
                {k: v.value for k, v in clean_weights.items()}
            )

            guarded_output = Task2Output(
                case_id=case_id,
                task=2,
                treatment_decision=TreatmentDecision(submodel_decision),
                confidence=Confidence(submodel_conf),
                variable_weights=clean_weights,
                reveal_sequence=computed_reveal_seq,  # type: ignore[arg-type]
                reasoning=raw_response.get("reasoning", "").strip() or "Standard treatment recommendation based on clinical risk profile.",
            )

            _save_case_output(output_dir, guarded_output)
            log.info("Successfully exported Task 2 outputs for %s", case_id)


def _save_case_output(output_dir: Path, out: Task2Output) -> None:
    """Write Grand Challenge submission files."""
    decision_file = output_dir / "prostate-treatment-decision.json"
    with open(decision_file, "w", encoding="utf-8") as f:
        json.dump(out.treatment_decision.value, f, indent=2)

    reasoning_payload = {
        "confidence": out.confidence.value,
        "variable_weights": {k: v.value for k, v in out.variable_weights.items()},
        "reveal_sequence": out.reveal_sequence,
        "free_text": out.reasoning,
    }
    reasoning_file = output_dir / "prostate-treatment-decision-reasoning.json"
    with open(reasoning_file, "w", encoding="utf-8") as f:
        json.dump(reasoning_payload, f, indent=2)
