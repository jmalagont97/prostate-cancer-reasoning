"""exp_2: confidence model, 4 conditions = {logistic, kdm} x {count, flags}.

Reuses chimera_task1.train_reasoning.repeated_out_of_fold_predict (logistic)
and chimera_task1.train_confidence_kdm.fit_predict_kdm (KDM) -- no changes
to either module. Feature frame comes from
features.select_official_feature_frame (11-variable, schema-faithful path).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_2/scripts/run_confidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from chimera_task1.features import build_preprocessor, select_official_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_reasoning import N_SPLITS, RANDOM_STATE, load_annotated, repeated_out_of_fold_predict

N_REPEATS_KDM = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_logistic_condition(X: pd.DataFrame, y_labels: np.ndarray, preprocessor) -> dict:
    dists = []
    for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
        dists.append(ordinal_distance(list(y_labels), list(preds), CONFIDENCE_RANK))
    return {
        "model": "OvR logistic regression",
        "ordinal_distance_mean": round(float(np.mean(dists)), 3),
        "ordinal_distance_std": round(float(np.std(dists)), 3),
    }


def run_kdm_condition(X_frame: pd.DataFrame, y_labels: np.ndarray) -> dict:
    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)

    y = np.array([CONFIDENCE_RANK[label] for label in y_labels])
    n_classes = len(CONFIDENCE_LEVELS)

    dists, entropies = [], []
    for repeat in range(N_REPEATS_KDM):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=object)
        probs_all = np.zeros((len(y), n_classes))
        for train_idx, test_idx in kf.split(X):
            test_probs = fit_predict_kdm(X[train_idx], y[train_idx], X[test_idx], n_classes)
            probs_all[test_idx] = test_probs
            preds[test_idx] = [CONFIDENCE_LEVELS[i] for i in test_probs.argmax(axis=1)]
        dists.append(ordinal_distance(list(y_labels), list(preds), CONFIDENCE_RANK))
        entropy = -(probs_all * np.log(np.clip(probs_all, 1e-7, 1))).sum(axis=1)
        entropies.append(entropy.mean())

    return {
        "model": "KDM (memory-based, sigma-only trained)",
        "ordinal_distance_mean": round(float(np.mean(dists)), 3),
        "ordinal_distance_std": round(float(np.std(dists)), 3),
        "mean_predictive_entropy": round(float(np.mean(entropies)), 3),
        "max_possible_entropy": round(float(np.log(n_classes)), 3),
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "confidence", "features": "11 official Task-1 variables", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] {metrics}")


def main() -> None:
    ann, inp_ann = load_annotated()
    y_labels = ann["target_confidence"].values
    majority = pd.Series(y_labels).mode()[0]
    print(f"n annotated = {len(ann)}, majority = '{majority}'\n")

    for treatment in ("count", "flags"):
        X = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)
        preprocessor = build_preprocessor(X)

        write_metrics(
            f"confidence_logistic_{treatment}",
            {"comorbidity_treatment": treatment, **run_logistic_condition(X, y_labels, preprocessor)},
        )
        write_metrics(
            f"confidence_kdm_{treatment}",
            {"comorbidity_treatment": treatment, **run_kdm_condition(X, y_labels)},
        )


if __name__ == "__main__":
    main()
