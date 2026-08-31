"""Terminal form-fill node for Task 2 — single prompt-and-parse LLM call.

The user prompt contains:
  1. The rendered case context.
  2. The three fixed Task 2 subtask tool outputs (decision, confidence, weights).
  3. Neighbor reasoning free-text exemplars from retrieved training cases.

The LLM generates a focused clinical justification (free_text reasoning).
Hard fields are subsequently overwritten by the orchestrator with deterministic
submodel predictions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from chimera_agent.tasks.task2.baseline.output.schema import Task2Output

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are completing the structured decision form for a prostate-cancer "
    "treatment case. Expert decision-support systems have ALREADY determined "
    "the treatment decision, the confidence level, and the variable weights — "
    "these values are GIVEN to you and must be transcribed EXACTLY. Your "
    "only task is to write the free-text reasoning that justifies them, "
    "modelled on the reference reasonings provided. Output a SINGLE JSON "
    "object matching the supplied shape EXACTLY — no extra keys, no "
    "markdown fences, no commentary."
)


def make_form_fill_node(model: BaseChatModel, max_retries: int = 3):
    """Return a node callable(state) -> {structured_response}."""

    def form_fill(state: dict[str, Any]) -> dict[str, Any]:
        case_id = state["case_id"]
        tools_out = state["tool_outputs"]
        treatment_support = tools_out["treatment_decision_support"]
        conf = tools_out["confidence_estimate"]
        weights_est = tools_out["variable_weights_estimate"]

        decision_str = treatment_support.get("decision")
        if not decision_str:
            raise ValueError("Task 2.1 model output is missing 'decision'")
        neighbors = treatment_support.get("neighbors_union", [])
        if isinstance(neighbors, dict):
            pids = neighbors.get("neighbor_pids", [])
            texts = neighbors.get("reasoning_free_texts", [])
            exemplars = [
                (pids[i] if i < len(pids) else "unknown", [], texts[i].strip() if i < len(texts) else "")
                for i in range(max(len(pids), len(texts)))
            ]
        else:
            exemplars = [
                (
                    d.get("pid", "unknown") if isinstance(d, dict) else str(d),
                    d.get("source_modalities", []) if isinstance(d, dict) else [],
                    d.get("reasoning", "").strip() if isinstance(d, dict) else "",
                )
                for d in neighbors
            ]
        exemplars = [(pid, src, t) for pid, src, t in exemplars if t and t != "Reasoning not found in file."]

        exemplar_block = "\n".join(
            f"  [{i + 1}] (sources: {', '.join(srcs) if srcs else 'unknown'} | "
            f'similar case "{pid}"): "{t}"'
            for i, (pid, srcs, t) in enumerate(exemplars[:6])
        ) or "  (none retrieved)"

        weights_block = "\n".join(
            f"  {k}: {v}"
            for k, v in weights_est.get("variable_weights", {}).items()
        )

        user_prompt = f"""Case ID: {case_id}
Task: 2 (treatment decision)

=== CASE CONTEXT (visible clinical panel) ===
{state['context']}

=== EXPERT DECISION-SUPPORT SYSTEM OUTPUTS (GIVEN VALUES — transcribe exactly) ===
1. Treatment decision: {decision_str}
   Modality probabilities: {json.dumps(treatment_support.get('modality_probabilities', {}))}
   Fused probabilities: {json.dumps(treatment_support.get('treatment_probabilities', {}))}
2. Confidence: {conf.get('confidence')}
3. Variable weights:
{weights_block}

=== REFERENCE REASONINGS FROM SIMILAR NEIGHBOR CASES (SECONDARY REFERENCE ONLY) ===
The texts below were written by expert urologists for OTHER patients with similar multimodal profiles.
They are provided ONLY as a reference for how experts structure and phrase their
clinical analysis — the depth, tone and argumentative pattern expected in this form.
They are NOT sources of facts: every clinical fact in your reasoning MUST
come exclusively from THIS patient's context above.
{exemplar_block}

Now fill out the form. Write the free_text reasoning under these STRICT requirements:
1. PRIORITIZE THIS PATIENT'S information: cite at least two EXACT findings
   from the case context above (e.g. "PSA 10.5 ng/mL", "ISUP Grade Group 3", "Gleason 4+3", "cT2a", "PI-RADS 4") — values must appear verbatim in the context.
2. Explicitly link each cited factor to the GIVEN decision ({decision_str}):
   explain why it supports this treatment path over alternatives.
3. End with a one-sentence conclusion stating the recommendation.
4. Neighbor texts may inform only the STYLE of the argument — never its facts.
5. Length: 2-4 sentences, focused and specific.

Output a SINGLE JSON object matching exactly this shape:
{{
  "case_id": "{case_id}",
  "task": 2,
  "treatment_decision": "{decision_str}",
  "confidence": "{conf.get('confidence', 'clear')}",
  "variable_weights": {{
{chr(10).join(f'    "{k}": "{v}",' for k, v in weights_est.get("variable_weights", {}).items())}
  }},
  "reasoning": "<your focused clinical justification>"
}}

Do NOT emit any other top-level keys. Do NOT wrap in markdown fences."""

        parser_retry_hint: str | None = None
        warnings: list[str] = []
        structured_response: dict[str, Any] | None = None

        for attempt in range(1, max_retries + 1):
            content = user_prompt if not parser_retry_hint else f"{user_prompt}\n\n{parser_retry_hint}"
            try:
                response = model.invoke(
                    [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
                )
                raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
                extracted = _extract_json_object(raw)
                repaired = _repair_json(extracted)
                obj_dict = json.loads(repaired)
                # Coerce treatment_decision / confidence / variable_weights to deterministic groundings if malformed
                obj_dict.pop("biopsy_decision", None)
                obj_dict["treatment_decision"] = decision_str
                obj_dict["confidence"] = conf.get("confidence", "clear")
                obj_dict["variable_weights"] = weights_est.get("variable_weights", {})
                obj_dict["case_id"] = case_id
                obj_dict["task"] = 2
                structured_response = Task2Output.model_validate(obj_dict).model_dump(mode="json")
                break
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                log.warning("form_fill parse failed (attempt %d) for %s: %s", attempt, case_id, exc)
                if attempt == max_retries:
                    break
                parser_retry_hint = (
                    "Your previous attempt did not validate. Error:\n"
                    f"{exc}\n\nEmit ONLY the JSON object matching the exact shape above."
                )

        if structured_response is None:
            raise RuntimeError(
                f"form_fill: case {case_id}: all {max_retries} attempts failed. warnings={warnings}"
            )

        return {"structured_response": structured_response, "form_fill_warnings": warnings}

    return form_fill


def _extract_json_object(text: str) -> str:
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _repair_json(text: str) -> str:
    if not text:
        return text
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        return text
    end = text.rfind("}")
    if end < 0:
        end = len(text)
    text = text[start : end + 1]

    text = re.sub(r"(?<![A-Za-z0-9_])'([A-Za-z_][A-Za-z0-9_]*)'(\s*:)", r'"\1"\2', text)
    text = re.sub(r":\s*'([^']*)'", lambda m: f': "{m.group(1)}"', text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text
