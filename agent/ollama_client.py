#!/usr/bin/env python3
"""
agent/ollama_client.py

Ollama Local LLM Client for Pathology Reasoning Agent.
Communicates with local Ollama server running Gemma model (gemma:2b / gemma2:2b)
and uses Pydantic schema validation for structured JSON generation.
"""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from schemas import PathologyReasoningOutput

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"


def check_ollama_status(host="http://localhost:11434"):
    """Checks if Ollama server is running locally."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


def query_ollama_gemma(prompt, system_prompt=None, model=DEFAULT_MODEL, temperature=0.1):
    """
    Queries local Ollama Gemma model with JSON mode formatting.

    Parameters:
      prompt (str): Main user prompt detailing patient and tool outputs.
      system_prompt (str, optional): System prompt specifying agent identity and JSON format.
      model (str): Ollama model tag (default: gemma:2b).
      temperature (float): Sampling temperature.

    Returns:
      dict: Parsed JSON response from LLM.
    """
    is_running, available_models = check_ollama_status()
    if not is_running:
        raise ConnectionError(
            "Ollama server is not running on http://localhost:11434. "
            "Please start Ollama using 'ollama serve' in your terminal."
        )

    # Prepare payload with JSON format constraint
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "options": {
            "temperature": temperature,
            "seed": 42
        },
        "stream": False
    }
    if system_prompt:
        payload["system"] = system_prompt

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))
            raw_response = res_body.get("response", "")
            parsed_json = json.loads(raw_response)
            return parsed_json
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"Ollama API Error ({e.code}): {err_msg}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nRaw output: {raw_response}")
