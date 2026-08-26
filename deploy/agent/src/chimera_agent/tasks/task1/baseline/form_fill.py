"""Terminal form-fill node — single prompt-and-parse LLM call.

Adapted from the baseline: there is no ReAct transcript. The user prompt
contains (1) the rendered case context, (2) the three fixed subtask tool
outputs, and (3) the neighbor reasoning exemplars. The model redacts a
coherent free_text reasoning anchored on the exemplars and transcribes
the hard fields as given. Output is validated against Task1Output.

Hard fields are later overwritten by the orchestrator with the submodel
values regardless of what the LLM emits — the LLM never decides.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from chimera_agent.tasks.task1.baseline.output.schema import Task1Output

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are completing the structured decision form for a prostate-cancer "
    "biopsy case. Expert decision-support systems have ALREADY determined "
    "the biopsy decision, the confidence level, and the variable weights — "
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
        biopsy = tools_out["biopsy_support"]
        conf = tools_out["confidence_estimate"]
        weights_est = tools_out["variable_weights_estimate"]

        decision_str = biopsy.get("decision", "no")
        neighbors = biopsy.get("neighbors_union", {})
        exemplars = [
            (
                d.get("neighbor_pid", "unknown"),
                d.get("source_modalities", []),
                d.get("free_text", "").strip(),
            )
            for d in neighbors.get("neighbors_detail", [])
        ]
        exemplars = [(pid, src, t) for pid, src, t in exemplars if t]

        exemplar_block = "\n".join(
            f"  [{i + 1}] (sources: {', '.join(srcs) if srcs else 'unknown'} | "
            f'similar case "{pid}"): "{t}"'
            for i, (pid, srcs, t) in enumerate(exemplars)
        ) or "  (none retrieved)"

        weights_block = "\n".join(f"  {k}: {v}" for k, v in weights_est.get("variable_weights", {}).items())

        user_prompt = f"""Case ID: {case_id}
Task: 1 (biopsy decision)

=== CASE CONTEXT (visible clinical panel) ===
{state['context']}

=== EXPERT DECISION-SUPPORT SYSTEM OUTPUTS (GIVEN VALUES — transcribe exactly) ===
1. Biopsy decision: {decision_str}   (fused probability: {biopsy.get('biopsy_probability')})
   Modality probabilities: tabular={biopsy['modality_probabilities']['tabular']}, mri={biopsy['modality_probabilities']['mri']}, text={biopsy['modality_probabilities']['text']}
2. Confidence: {conf.get('confidence')}
3. Variable weights:
{weights_block}

=== REFERENCE REASONINGS FROM SIMILAR NEIGHBOR CASES (SECONDARY REFERENCE ONLY) ===
The fused probability above is a weighted vote of three modality experts:
  tabular expert (weight 0.55) -> probability {biopsy['modality_probabilities']['tabular']}
  MRI expert      (weight 0.20) -> probability {biopsy['modality_probabilities']['mri']}
  text expert     (weight 0.25) -> probability {biopsy['modality_probabilities']['text']}

The texts below were written by expert urologists for OTHER patients. Each
neighbor lists which modality experts retrieved it ("sources"): a neighbor
corroborated by SEVERAL modalities reflects a more globally similar patient
and deserves more weight as a stylistic anchor; a neighbor retrieved by a
single low-weight modality is weaker evidence of true similarity.
They are provided ONLY as a reference for how experts structure and phrase their
analysis — the depth, tone and argumentative pattern expected in this form.
They are NOT sources of facts: every clinical fact in your reasoning MUST
come exclusively from THIS patient's context above. If a neighbor text
mentions a finding that THIS patient does not have, it must be ignored.
{exemplar_block}

Now fill out the form. Write the free_text reasoning under these STRICT
requirements:
1. PRIORITIZE THIS PATIENT'S information: cite at least two EXACT findings
   from the case context above (e.g. "PSA 16.5 ng/mL", "PI-RADS 5 lesion",
   "PSAD 0.567") — values must appear verbatim in the context.
2. Explicitly link each cited factor to the GIVEN decision ({decision_str}):
   explain why it supports recommending (or deferring) biopsy.
3. End with a one-sentence conclusion stating the recommendation.
4. Neighbor texts may inform only the STYLE of the argument — never its facts.
5. Length: 2-4 sentences, focused and specific.

Output a SINGLE JSON object matching exactly this shape:
{{
  "case_id": "{case_id}",
  "task": 1,
  "biopsy_decision": <true | false>,
  "confidence": <"clear" | "borderline" | "uncertain">,
  "variable_weights": {{
{chr(10).join(f'    "{k}": <"not_used" | "noted" | "important" | "decisive">,' for k in weights_est.get("variable_weights", {}))}
  }},
  "reasoning": "<your focused justification>"
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
                structured_response = Task1Output.model_validate(obj_dict).model_dump(mode="json")
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


def _repair_json(text: str) -> str:
    """Normalize LLM-emitted pseudo-JSON into strict JSON.

    Gemma often returns single-quoted keys/values or trailing commas.
    We do a conservative pass: strip fences, unwrap code blocks, fix
    single quotes, and drop trailing commas before the closing brace.
    """
    if not text:
        return text
    # Remove markdown fences.
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

    # Replace single-quoted keys/values with double quotes. This is a
    # heuristic that works for the flat Task1Output shape we control.
    # 1) keys like 'case_id': -> "case_id":
    text = re.sub(r"(?<![A-Za-z0-9_])'([A-Za-z_][A-Za-z0-9_]*)'(\s*:)", r'"\1"\2', text)
    # 2) string values: : 'value' -> : "value"
    text = re.sub(r":\s*'([^']*)'", lambda m: f': "{m.group(1)}"', text)
    # 3) drop trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        return text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]
