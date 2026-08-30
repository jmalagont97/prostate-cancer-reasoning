#!/usr/bin/env python3
"""Task 3 production inference from a raw case directory or JSON bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.tasks.task3.model_definitions import Task3CoxPCA5
from deploy.tasks.task3.preprocessing import transform_case

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "deploy" / "tasks" / "task3" / "training" / "models" / "task3_cox_pca5.pkl"


def predict(raw_case: str | Path | dict, model_path: str | Path = DEFAULT_MODEL) -> dict:
    features, embeddings, _ = transform_case(raw_case, require_label=False)
    case_id = str(features.case_id.iloc[0])
    return Task3CoxPCA5.load(model_path).predict(features, {case_id: embeddings["prostatectomy"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = predict(args.case, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
