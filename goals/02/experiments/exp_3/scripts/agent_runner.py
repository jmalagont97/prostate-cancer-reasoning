#!/usr/bin/env python3
"""agent_runner.py

Execution runner for agents in exp_3.
Manages tool binding, Ollama LLM communication, native function-calling execution loops,
and robust JSON schema response parsing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI


def resolve_paths() -> Tuple[Path, Path, Path, Path]:
    """Resolve paths for exp_3."""
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent
    agents_dir = scripts_dir / "declarations" / "agents"
    tools_decl_dir = scripts_dir / "declarations" / "tools"
    code_tools_dir = scripts_dir / "tools"
    return scripts_dir, agents_dir, tools_decl_dir, code_tools_dir


def load_agent_declaration(agent_identifier: str, agents_dir: Path) -> Dict[str, Any]:
    """Load agent declaration JSON."""
    clean_name = agent_identifier.replace(".json", "")
    target_path = agents_dir / f"{clean_name}.json"
    if not target_path.exists():
        raise FileNotFoundError(f"Agent declaration '{clean_name}.json' not found in {agents_dir}")

    with open(target_path, "r", encoding="utf-8") as f:
        declaration = json.load(f)
    declaration["_file_path"] = str(target_path)
    return declaration


def bind_tools(
    tool_configs: List[Dict[str, Any]], tools_decl_dir: Path, code_tools_dir: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    """Bind declared tools to OpenAI function calling schemas and executable functions."""
    openai_tools: List[Dict[str, Any]] = []
    executors: Dict[str, Callable[..., Any]] = {}

    for t_conf in tool_configs:
        tool_id = t_conf.get("tool_id")
        if not tool_id or not t_conf.get("enabled", True):
            continue

        decl_file = tools_decl_dir / f"{tool_id}.json"
        if not decl_file.exists():
            continue

        with open(decl_file, "r", encoding="utf-8") as f:
            t_decl = json.load(f)

        schema = {
            "type": "function",
            "function": {
                "name": tool_id,
                "description": t_decl.get("description", ""),
                "parameters": t_decl.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        openai_tools.append(schema)

        tool_script = code_tools_dir / f"{tool_id}.py"
        if tool_script.exists():
            spec = importlib.util.spec_from_file_location(tool_id, tool_script)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                func = getattr(mod, tool_id, None)
                if callable(func):
                    executors[tool_id] = func

    return openai_tools, executors


def render_system_prompt(declaration: Dict[str, Any], simulated_inputs: Optional[str] = None) -> str:
    """Render mustache system prompt template from agent declaration."""
    p = declaration.get("prompt", {})
    template = p.get("template", "{{role}}\n\n{{task}}\n\n{{rules}}")

    role_str = p.get("role", "")
    task_str = p.get("task", "")

    rules = p.get("rules", [])
    rules_str = "\n".join(f"- {r}" for r in rules) if isinstance(rules, list) else str(rules)

    tools_cfg = p.get("tools", {})
    tools_desc = tools_cfg.get("description", "")
    usage_policy = tools_cfg.get("usage_policy", [])
    policy_str = "\n".join(f"- {u}" for u in usage_policy) if isinstance(usage_policy, list) else str(usage_policy)
    tools_rendered = f"{tools_desc}\n{policy_str}".strip()

    output_fmt = p.get("output_format", {})
    output_fmt_str = json.dumps(output_fmt, indent=2) if isinstance(output_fmt, dict) else str(output_fmt)

    inputs_str = simulated_inputs or "Standalone Session"

    rendered = template
    rendered = rendered.replace("{{role}}", role_str)
    rendered = rendered.replace("{{task}}", task_str)
    rendered = rendered.replace("{{rules}}", rules_str)
    rendered = rendered.replace("{{tools}}", tools_rendered)
    rendered = rendered.replace("{{output_format}}", output_fmt_str)
    rendered = rendered.replace("{{inputs}}", inputs_str)

    return rendered


class SingleAgentRunner:
    """Runner for Ollama LLM execution with tool calling and structured output parsing."""

    def __init__(
        self,
        agent_declaration: Dict[str, Any],
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        verbose: bool = True,
    ):
        self.declaration = agent_declaration
        self.verbose = verbose
        scripts_dir, _, tools_decl_dir, code_tools_dir = resolve_paths()

        model_cfg = self.declaration.get("model", {})
        conn_cfg = model_cfg.get("connection", {})

        self.base_url = base_url_override or conn_cfg.get("base_url") or "http://localhost:11434/v1"
        self.api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.model_name = model_override or model_cfg.get("model_name") or "gemma4:e2b-64k"

        params = model_cfg.get("parameters", {})
        self.temperature = float(params.get("temperature", 0.0))
        self.max_tokens = int(params.get("max_tokens", 4096))

        tool_configs = self.declaration.get("tools", [])
        self.openai_tools, self.tool_executors = bind_tools(tool_configs, tools_decl_dir, code_tools_dir)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def run(self, user_prompt: str, simulated_inputs: Optional[str] = None) -> Dict[str, Any]:
        """Execute agent invocation with multi-turn tool handling."""
        system_prompt = render_system_prompt(self.declaration, simulated_inputs=simulated_inputs)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_iterations = 6
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            call_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.openai_tools:
                call_kwargs["tools"] = self.openai_tools

            response = self.client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]
            msg = choice.message

            # Check if LLM requested tool calls
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    if self.verbose:
                        print(f"  [Tool Call] {fn_name}({args})")

                    executor = self.tool_executors.get(fn_name)
                    if executor:
                        try:
                            result = executor(**args)
                        except Exception as exc:
                            result = {"ok": False, "error": str(exc)}
                    else:
                        result = {"ok": False, "error": f"Tool '{fn_name}' not implemented."}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(result),
                    })
                continue

            # Model finished turn and produced text output
            content = msg.content or ""
            parsed_json = self._extract_json(content)
            return {
                "raw_text": content,
                "parsed": parsed_json,
                "agent_id": self.declaration.get("id"),
                "turns": iteration,
            }

        return {"raw_text": "", "parsed": None, "error": "Exceeded max tool iterations."}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse structured JSON from model response text."""
        # 1. Clean markdown fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except Exception:
                pass

        # 2. Match outermost curly braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return None
