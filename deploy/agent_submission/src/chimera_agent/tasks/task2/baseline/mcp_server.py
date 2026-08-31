"""MCP server (Task 2): three fixed subtask tools over the trained Task 2 pickles.

Each tool wraps one self-contained ``predict_raw`` artifact. The server
runs as a stdio subprocess; tools are discovered via ``tools/list`` and
invoked by the orchestrator in fixed order — never by the LLM.

  - get_treatment_decision_support : Subtask1FuzzySVREnsemble
        -> preliminary treatment decision, modality/fused probabilities,
           and union of KNN neighbor cases with their reference reasoning texts.
  - get_confidence_estimate        : Subtask2FuzzyKNNClassifier
        -> preliminary confidence (uncertain/borderline/clear) with probabilities.
  - get_variable_weights           : Subtask3MultiVariableEnsemble
        -> preliminary per-variable weights for 11 variables + decisive/important lists.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# The pickles were serialized from deploy.tasks.task2.model_definitions;
# joblib.load requires that exact import path to resolve at unpickling time.
_REPO_ROOT = Path(__file__).resolve().parents[7]  # .../<repo root>
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib  # noqa: E402

log = logging.getLogger(__name__)

mcp = FastMCP("chimera-task2-submodels")

_MODELS: dict[str, object] = {}


def _load_models() -> dict[str, object]:
    if _MODELS:
        return _MODELS
    models_dir = Path(os.environ.get("SUBMODELS_DIR", "models"))
    # Task 2 uses an isolated model directory. These canonical names are
    # intentionally resolved only inside that directory, never in Task 1's
    # shared model root.
    mapping = {
        "ensemble": ["subtask1_ensemble.pkl"],
        "knn": ["subtask2_knn.pkl"],
        "extratrees": ["subtask3_extratrees.pkl"],
    }
    for key, fnames in mapping.items():
        found = None
        for fn in fnames:
            candidate = models_dir / fn
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(
                f"Required Task-2 model is missing in {models_dir} (tried {fnames}). "
                "Mount the canonical deploy/tasks/task2/training/models artifacts "
                "at SUBMODELS_DIR."
            )
        log.info("Loading %s from %s", key, found)
        _MODELS[key] = joblib.load(found)
    return _MODELS


@mcp.tool()
def get_treatment_decision_support(case_path: str) -> str:
    """Subtask-1 Fuzzy-SVR tuned ensemble: returns the preliminary treatment decision,
    fused probabilities, modality probabilities, and the union of KNN
    neighbor cases with their reference reasoning free_texts."""
    res = _load_models()["ensemble"].predict_raw(case_path)  # type: ignore[attr-defined]
    return json.dumps(res)


@mcp.tool()
def get_confidence_estimate(case_path: str) -> str:
    """Subtask-2 Fuzzy-KNN classifier: returns the preliminary confidence
    label (uncertain / borderline / clear) with class probabilities."""
    res = _load_models()["knn"].predict_raw(case_path)  # type: ignore[attr-defined]
    return json.dumps(res)


@mcp.tool()
def get_variable_weights(case_path: str) -> str:
    """Subtask-3 98D ExtraTrees/RF ensemble: returns the preliminary per-variable
    weights for the 11 clinical variables plus decisive and important factor lists."""
    res = _load_models()["extratrees"].predict_raw(case_path)  # type: ignore[attr-defined]
    return json.dumps(res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run(transport="stdio")
