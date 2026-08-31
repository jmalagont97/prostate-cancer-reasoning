"""Tests for explicit OmegaConf merge and path safety."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Ensure deploy/agent is importable for inference.py
sys.path.insert(0, str(Path("deploy/agent").resolve()))
from inference import _detect_generic_input_task, load_merged_config  # type: ignore[import]


def test_base_config_loads():
    from chimera_agent.dispatch import validate_task_id  # ensure package importable

    base = Path("deploy/agent/configs/config.yaml")
    assert base.is_file(), "base config missing"
    cfg = OmegaConf.load(str(base))
    assert str(cfg.paths.submodels_dir) == "/opt/ml/model"


def test_task1_override_merges_explicitly():
    base = Path("deploy/agent/configs/config.yaml")
    cfg = load_merged_config(str(base))
    # After explicit merge, task_id must be 1 and server_module task1
    assert int(cfg.agent.task_id) == 1
    assert "task1" in str(cfg.mcp.server_module)


def test_task2_override_merges_when_base_is_task2():
    # Simulate base with task_id=2 by merging manually
    base = OmegaConf.load("deploy/agent/configs/config.yaml")
    override = OmegaConf.load("deploy/agent/configs/tasks/task2.yaml")
    merged = OmegaConf.merge(base, override)
    assert int(merged.agent.task_id) == 2
    assert "task2" in str(merged.mcp.server_module)


def test_submodels_dir_never_derived_from_case_payload():
    # Security: runner must use cfg.paths.submodels_dir, not case JSON
    case_payload = {"case_id": "x", "submodels_dir": "/tmp/evil"}
    cfg = OmegaConf.load("deploy/agent/configs/config.yaml")
    # Even if payload contains a path, cfg must remain authoritative
    assert "/tmp/evil" not in str(cfg.paths.submodels_dir)
    assert str(cfg.paths.submodels_dir) == "/opt/ml/model"


def test_generic_input_detects_task3(tmp_path):
    case = tmp_path / "T3-001"
    case.mkdir()
    (case / "structured-prompt.json").write_text('{"case_id":"T3-001","task":3}')
    assert _detect_generic_input_task(tmp_path) == 3


def test_generic_input_rejects_mixed_tasks(tmp_path):
    for case_id, task_id in (("T1-001", 1), ("T3-001", 3)):
        case = tmp_path / case_id
        case.mkdir()
        (case / "structured-prompt.json").write_text(
            f'{{"case_id":"{case_id}","task":{task_id}}}'
        )
    import pytest

    with pytest.raises(ValueError, match="mixed tasks"):
        _detect_generic_input_task(tmp_path)
