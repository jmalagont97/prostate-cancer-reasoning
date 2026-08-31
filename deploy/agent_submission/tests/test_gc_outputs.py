import importlib.util
import json
from pathlib import Path

import pytest


_INFERENCE_PATH = Path(__file__).parents[1] / "inference.py"
_SPEC = importlib.util.spec_from_file_location("submission_inference", _INFERENCE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_INFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INFERENCE)
publish_gc_outputs = _INFERENCE.publish_gc_outputs


def test_publish_gc_outputs_flattens_nested_task_output(tmp_path):
    case_dir = tmp_path / "task1" / "PT-001"
    case_dir.mkdir(parents=True)
    (case_dir / "prostate-biopsy-decision.json").write_text(json.dumps("yes"))
    (case_dir / "prostate-biopsy-decision-reasoning.json").write_text(
        json.dumps({"confidence": "clear"})
    )
    (case_dir / "prediction.json").write_text(json.dumps({"internal": True}))

    publish_gc_outputs(1, tmp_path)

    assert json.loads((tmp_path / "prostate-biopsy-decision.json").read_text()) == "yes"
    assert json.loads(
        (tmp_path / "prostate-biopsy-decision-reasoning.json").read_text()
    ) == {"confidence": "clear"}


def test_publish_gc_outputs_requires_both_official_files(tmp_path):
    case_dir = tmp_path / "task3" / "T3-001"
    case_dir.mkdir(parents=True)
    (case_dir / "prostate-time-to-recurrence-or-last-follow-up.json").write_text(
        json.dumps({"months_to_recurrence": 1.6, "event": 1})
    )

    with pytest.raises(FileNotFoundError, match="reasoning"):
        publish_gc_outputs(3, tmp_path)
