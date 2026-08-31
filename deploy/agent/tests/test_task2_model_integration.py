"""Integration checks for the canonical Task-2 model artifacts."""

import json
import os
from pathlib import Path
from typing import Any


def test_canonical_task2_artifacts_load_and_expose_predict_raw():
    model_dir = Path("deploy/tasks/task2/training/models").resolve()
    assert model_dir.is_dir()

    os.environ["SUBMODELS_DIR"] = str(model_dir)
    from chimera_agent.tasks.task2.baseline import mcp_server

    mcp_server._MODELS.clear()
    loaded: dict[str, Any] = mcp_server._load_models()
    models = loaded

    assert set(models) == {"ensemble", "knn", "extratrees"}
    assert all(hasattr(model, "predict_raw") for model in models.values())
    assert models["ensemble"].weight_tab == 0.50
    assert models["ensemble"].weight_mri == 0.05
    assert models["ensemble"].weight_txt == 0.45

    # Direct predict_raw verification on raw case
    raw_case_dir = Path("data/raw/train_release/task2/T2-004")
    if raw_case_dir.is_dir():
        res1_raw = mcp_server.get_treatment_decision_support(str(raw_case_dir))
        res1 = json.loads(res1_raw)
        assert "decision" in res1
        assert "treatment_probabilities" in res1
        assert "neighbors_union" in res1

        res2_raw = mcp_server.get_confidence_estimate(str(raw_case_dir))
        res2 = json.loads(res2_raw)
        assert res2["confidence"] in ["clear", "borderline", "uncertain"]

        res3_raw = mcp_server.get_variable_weights(str(raw_case_dir))
        res3 = json.loads(res3_raw)
        assert "variable_weights" in res3
        assert len(res3["variable_weights"]) == 11
