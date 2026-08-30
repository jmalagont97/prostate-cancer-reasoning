#!/usr/bin/env python3
"""Train Task 3 Cox-PCA5 on every annotated raw case."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

from deploy.tasks.task3.model_definitions import Task3CoxPCA5
from deploy.tasks.task3.preprocessing import transform_case

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data" / "raw" / "train_release" / "task3"
MODEL_DIR = ROOT / "deploy" / "tasks" / "task3" / "training" / "models"


def load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    features, labels, embeddings = [], [], {}
    for case_dir in sorted(path for path in RAW.iterdir() if path.is_dir()):
        row, emb, label = transform_case(case_dir, require_label=True)
        assert label is not None
        features.append(row.iloc[0].to_dict())
        labels.append({"case_id": row.case_id.iloc[0], "months_to_recurrence": label["months_to_recurrence"], "event": label["event"]})
        embeddings[str(row.case_id.iloc[0])] = emb["prostatectomy"]
    return pd.DataFrame(features), pd.DataFrame(labels), embeddings


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features, labels, embeddings = load_cohort()
    if len(features) != 75 or labels.event.sum() != 19:
        raise ValueError(f"Unexpected cohort: {len(features)} cases, {labels.event.sum()} events")
    model = Task3CoxPCA5().fit(features, labels, embeddings)
    path = MODEL_DIR / "task3_cox_pca5.pkl"
    model.save(path)
    metadata = {
        "model": "Task3CoxPCA5", "n_cases": len(features), "n_events": int(labels.event.sum()),
        "penalizer": 0.3, "pca_components": 5, "time_percentile": 0.90,
        "event_horizon_months": 60.0, "survival_threshold": 0.7,
        "features": model.feature_columns,
    }
    (MODEL_DIR / "task3_cox_pca5.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"TRAINED: {len(features)} cases, {int(labels.event.sum())} events")
    print(f"SAVED: {path}")


if __name__ == "__main__":
    main()
