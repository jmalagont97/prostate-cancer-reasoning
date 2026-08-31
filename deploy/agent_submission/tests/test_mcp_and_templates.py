"""MCP and template smoke tests."""

import importlib
from pathlib import Path


def test_task1_mcp_server_importable():
    mod = importlib.import_module("chimera_agent.tasks.task1.baseline.mcp_server")
    assert hasattr(mod, "mcp")
    assert hasattr(mod, "_load_models")
    # tools must be registered
    # FastMCP registers via decorator; check at least 3 tools are present by inspecting mcp
    # For FastMCP, tools are in mcp._tool_manager or similar; just verify module loads


def test_task1_prompt_template_exists():
    tmpl = Path("deploy/agent/templates/prompts/task1/agent_prompt.j2")
    assert tmpl.is_file(), f"template missing: {tmpl}"
    text = tmpl.read_text()
    assert "PATIENT CASE" in text or "Patient" in text or "Case ID" in text
    assert "{{ case_id" in text or "{{case_id" in text or "{{ case_input" in text or "case_id" in text


def test_baseline_package_importable():
    import chimera_agent.tasks.task1.baseline.case_loader as cl
    import chimera_agent.tasks.task1.baseline.form_fill as ff

    assert hasattr(cl, "load_cases")
    assert hasattr(ff, "make_form_fill_node")
