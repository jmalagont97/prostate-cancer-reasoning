"""MCP server (Task 1): three fixed subtask tools over the trained pickles.

Each tool wraps one self-contained ``predict_raw`` artifact. The server
runs as a stdio subprocess; tools are discovered via ``tools/list`` and
invoked by the orchestrator in fixed order — never by the LLM.

  - get_biopsy_decision_support : Subtask1LateFusionEnsemble
        -> preliminary decision, fused probability, KNN neighbors with
           reference reasoning free_texts.
  - get_confidence_estimate     : Subtask2DirectKDMClassifier
        -> preliminary confidence (uncertain/borderline/clear).
  - get_variable_weights        : Subtask3JointExtraTrees
        -> preliminary per-variable weights + decisive/important factors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# The pickles were serialized from deploy.tasks.task1.training.model_definitions;
# joblib.load requires that exact import path to resolve at unpickling time.
_REPO_ROOT = Path(__file__).resolve().parents[7]  # .../<repo root>
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib  # noqa: E402

log = logging.getLogger(__name__)

mcp = FastMCP("chimera-task1-submodels")

_MODELS: dict[str, object] = {}


def _load_models() -> dict[str, object]:
    if _MODELS:
        return _MODELS
    models_dir = Path(os.environ.get("SUBMODELS_DIR", "models"))
    mapping = {
        "ensemble": "subtask1_ensemble.pkl",
        "kdm": "subtask2_kdm.pkl",
        "extratrees": "subtask3_extratrees.pkl",
    }
    for key, fname in mapping.items():
        path = models_dir / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Task-1 model is missing: {path}. "
                "Mount the canonical deploy/tasks/task1/training/models artifacts "
                "at SUBMODELS_DIR."
            )
        log.info("Loading %s from %s", key, path)
        _MODELS[key] = joblib.load(path)
    return _MODELS


@mcp.tool()
def get_biopsy_decision_support(case_path: str) -> str:
    """Subtask-1 late-fusion ensemble: returns the preliminary biopsy decision,
    fused csPCa probability, modality probabilities, and the union of KNN
    neighbor cases with their reference reasoning free_texts."""
    res = _load_models()["ensemble"].predict_raw(case_path)
    return json.dumps(res)


@mcp.tool()
def get_confidence_estimate(case_path: str) -> str:
    """Subtask-2 Direct-KDM classifier: returns the preliminary confidence
    label (uncertain / borderline / clear) with class probabilities."""
    res = _load_models()["kdm"].predict_raw(case_path)
    return json.dumps(res)


@mcp.tool()
def get_variable_weights(case_path: str) -> str:
    """Subtask-3 Joint ExtraTrees + SHAP: returns the preliminary per-variable
    weights plus decisive and important factor lists."""
    res = _load_models()["extratrees"].predict_raw(case_path)
    return json.dumps(res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run(transport="stdio")
