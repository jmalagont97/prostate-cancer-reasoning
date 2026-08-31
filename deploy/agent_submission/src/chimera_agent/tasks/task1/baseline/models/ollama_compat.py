"""Ollama-backed chat model builder (Gemma 4 E2B — invariant).

Wraps ``langchain_ollama.ChatOllama`` pointing at an Ollama daemon served
INSIDE the same container on localhost. No tool-calling is required from
the model: form_fill uses prompt-and-parse exclusively.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def build_model(cfg) -> BaseChatModel:
    """Build the single LLM used for the form-fill synthesis step."""
    from langchain_ollama import ChatOllama

    assert cfg.model.provider == "ollama", f"unsupported provider {cfg.model.provider}"
    assert "e2b" in str(cfg.model.model_id).lower(), (
        "Gemma 4 E2B is the invariant model — refusing to start with "
        f"model_id={cfg.model.model_id}"
    )

    return ChatOllama(
        base_url=cfg.model.base_url,
        model=cfg.model.model_id,
        temperature=float(cfg.generation.temperature),
        num_ctx=int(cfg.generation.num_ctx),
        num_predict=int(cfg.generation.num_predict),
        timeout=int(cfg.model.request_timeout),
    )
