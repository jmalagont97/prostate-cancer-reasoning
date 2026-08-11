"""exp_3: confidence model, 8 conditions = 7 sklearn/xgboost models + KDM, on the 19-column frame.

cv_utils.repeated_cv_proba is already generic over n_classes (inferred from y), so it directly
covers this 3-class target with no changes -- the same helper run_decision.py uses.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_3/scripts/run_confidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_utils import N_SPLITS, RANDOM_STATE, repeated_cv_proba  # noqa: E402
from models import NO_IMBALANCE_HANDLING, build_sklearn_models  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_sklearn_condition(feature_frame, y: np.ndarray, y_labels, model_name: str, clf, use_sample_weight: bool) -> dict:
    all_probas = repeated_cv_proba(feature_frame, y, clf, use_sample_weight, N_REPEATS, stratified=True)

    dists = []
    for proba in all_probas:
        preds = [CONFIDENCE_LEVELS[i] for i in proba.argmax(axis=1)]
        dists.append(ordinal_distance(list(y_labels), preds, CONFIDENCE_RANK))

    result = {
        "model": model_name,
        "ordinal_distance_mean": round(float(np.mean(dists)), 3),
        "ordinal_distance_std": round(float(np.std(dists)), 3),
    }
    if model_name in NO_IMBALANCE_HANDLING:
        result["class_imbalance_handling"] = "none"
    elif use_sample_weight:
        result["class_imbalance_handling"] = "sample_weight (balanced)"
    else:
        result["class_imbalance_handling"] = "class_weight (balanced)"
    return result


def run_kdm_condition(feature_frame, y: np.ndarray, y_labels) -> dict:
    from sklearn.preprocessing import StandardScaler

    preprocessor = build_preprocessor(feature_frame)
    X_pre = preprocessor.fit_transform(feature_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)
    n_classes = len(CONFIDENCE_LEVELS)

    dists, entropies = [], []
    for repeat in range(N_REPEATS):
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
        "class_imbalance_handling": "none",
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "confidence", "features": "exp_3 19-column frame (PSA-reduced + MRI-PCA)", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] {metrics}")


def main() -> None:
    ann, inp_ann = load_annotated()
    y_labels = ann["target_confidence"].values
    y = np.array([CONFIDENCE_RANK[label] for label in y_labels])
    majority = pd.Series(y_labels).mode()[0]
    print(f"n annotated = {len(ann)}, majority = '{majority}'\n")

    # Fit MRI-PCA on the full 195-case embedding population (same transform decision uses), then
    # align to the 91 annotated cases by case_id -- inp_ann's row order/index doesn't match the
    # full inputs.csv's positional index, so a plain .join() on raw index would silently misalign.
    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)

    X = select_exp3_feature_frame(inp_ann, mri_pca)
    print(f"feature frame: {X.shape}\n")

    for name, (clf, use_sample_weight) in build_sklearn_models().items():
        write_metrics(f"confidence_{name}", run_sklearn_condition(X, y, y_labels, name, clf, use_sample_weight))

    write_metrics("confidence_kdm", run_kdm_condition(X, y, y_labels))


if __name__ == "__main__":
    main()
