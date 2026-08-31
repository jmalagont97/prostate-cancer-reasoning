"""Single Ollama/Gemma synthesis call for Task 3 clinical reasoning."""
from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM = (
    "You are a specialist urologist writing a concise clinical justification. "
    "The Cox model has already produced the official time and event. Do not "
    "recalculate, contradict, or alter either value. Use only facts in the "
    "patient context. Do not invent findings. Return plain text only, 2-4 sentences."
)


def _clean(text: Any) -> str:
    text = text if isinstance(text, str) else str(text)
    text = re.sub(r"```(?:text|markdown)?\s*|```", "", text).strip()
    return text.strip('"')


def make_reasoning_node(model: BaseChatModel):
    def generate(state: dict[str, Any]) -> str:
        result = state["prediction"]
        prompt = f"""Case ID: {state['case_id']}

=== PRIMARY STRUCTURED PATIENT INFORMATION ===
{state['context']}

=== LOCKED COX MODEL OUTPUT ===
months_to_recurrence: {result['months_to_recurrence']}
event: {result['event']}

Justify this result as a urologist. Cite at least two available patient findings,
link them to recurrence risk and the locked result, and mention uncertainty or
follow-up limitations when appropriate. Never introduce facts absent from the
primary information. Do not output JSON, headings, markdown, or the numeric
fields as a separate prediction; output only the justification text."""
        response = model.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
        reasoning = _clean(response.content)
        if not reasoning:
            raise ValueError("Gemma returned empty Task-3 reasoning")
        return reasoning

    return generate
