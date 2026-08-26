"""Runtime case loader (Task 1 only).

Reads the per-patient ``<case>/structured-prompt.json`` and renders the
contextualization narrative through the Jinja template. Returns one query
dict per case: ``{case_id, task, context}``.

The companion clinical-data and neural-representations files are read by
the MCP server tools, never inlined here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_baseline_prompt(
    payload: dict[str, Any],
    templates_dir: Path | str = "templates/prompts",
    template_name: str = "agent_prompt.j2",
) -> str:
    """Render the agent prompt from a ``structured-prompt.json`` payload."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(templates_dir)), keep_trailing_newline=True)
    return env.get_template(template_name).render(**payload)


def load_cases(
    cases_dir: Path | str,
    *,
    templates_dir: Path | str = "templates/prompts",
    template_name: str = "agent_prompt.j2",
) -> list[dict[str, Any]]:
    """Load all Task-1 cases under ``cases_dir`` (subdirs with structured-prompt.json)."""
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"cases_dir {cases_dir} is not a directory")

    out: list[dict[str, Any]] = []
    for sub in sorted(p for p in cases_dir.iterdir() if p.is_dir() and (p / "structured-prompt.json").exists()):
        prompt_path = sub / "structured-prompt.json"
        try:
            payload = json.loads(prompt_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{prompt_path}: invalid JSON: {e}") from e
        out.append(
            {
                "case_id": payload["case_id"],
                "task": 1,
                "case_path": str(sub),
                "context": render_baseline_prompt(payload, templates_dir, template_name),
            }
        )
    if not out:
        raise FileNotFoundError(f"No cases found in {cases_dir}")
    return out
