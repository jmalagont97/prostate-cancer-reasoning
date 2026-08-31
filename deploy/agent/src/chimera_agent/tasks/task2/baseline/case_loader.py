"""Case loader: discovers and contextualizes Task-2 cases via Jinja."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class CaseLoader:
    def __init__(self, templates_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def discover_cases(self, task_dir: Path) -> list[Path]:
        """Return patient directories inside `task_dir`."""
        if not task_dir.is_dir():
            return []
        cases = [
            p
            for p in task_dir.iterdir()
            if p.is_dir() and (p / "structured-prompt.json").is_file()
        ]
        return sorted(cases, key=lambda p: p.name)

    def load_case_payload(self, case_dir: Path) -> dict[str, Any]:
        """Read structured-prompt.json from a case directory."""
        prompt_file = case_dir / "structured-prompt.json"
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Missing structured-prompt.json in {case_dir}")
        with open(prompt_file, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("case_id", case_dir.name)
        data.setdefault("pid", case_dir.name)
        return data

    def render_context(self, case_payload: dict[str, Any]) -> str:
        """Render agent_prompt.j2."""
        template = self.env.get_template("agent_prompt.j2")
        return template.render(**case_payload)
