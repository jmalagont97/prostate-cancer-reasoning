"""Integration checks for the canonical Task-1 model artifacts."""

import os
from pathlib import Path
from typing import Any


def test_canonical_task1_artifacts_load_and_expose_predict_raw():
    model_dir = Path("deploy/tasks/task1/training/models").resolve()
    assert model_dir.is_dir()

    os.environ["SUBMODELS_DIR"] = str(model_dir)
    from chimera_agent.tasks.task1.baseline import mcp_server

    mcp_server._MODELS.clear()
    loaded: dict[str, Any] = mcp_server._load_models()
    models = loaded

    assert set(models) == {"ensemble", "kdm", "extratrees"}
    assert all(hasattr(model, "predict_raw") for model in models.values())
    assert models["ensemble"].w_tab == 0.15
    assert models["ensemble"].w_mri == 0.05
    assert models["ensemble"].w_txt == 0.80
    assert models["ensemble"].threshold == 0.56
    assert models["kdm"].n_mri_pca == 2
    estimator = models["extratrees"].classifiers["age"]
    assert estimator.n_estimators == 50
    assert estimator.max_depth == 3
    assert estimator.min_samples_leaf == 2
