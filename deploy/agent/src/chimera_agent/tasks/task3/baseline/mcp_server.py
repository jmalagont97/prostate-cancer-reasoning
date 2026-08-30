"""MCP server exposing the production Task 3 Cox predictor."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "deploy" / "tasks").is_dir()),
    Path(__file__).resolve().parents[5],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.tasks.task3.inference import predict  # noqa: E402

log = logging.getLogger(__name__)
mcp = FastMCP("chimera-task3-cox")
_MODEL_NAME = "task3_cox_pca5.pkl"


def _model_path() -> Path:
    path = Path(os.environ.get("SUBMODELS_DIR", "/opt/ml/model")) / _MODEL_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Required Task-3 model is missing: {path}")
    return path


@mcp.tool()
def predict_bcr_task3(case_path: str) -> str:
    """Predict months to recurrence and binary event from a complete case directory."""
    case_dir = Path(case_path)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Task-3 case directory is missing: {case_dir}")
    result = predict(case_dir, _model_path())
    if set(result) != {"months_to_recurrence", "event"}:
        raise ValueError(f"Unexpected Task-3 predictor output keys: {set(result)}")
    return json.dumps(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run(transport="stdio")
