"""Backfill macro-F1 into exp_5's already-completed results, per this project's cross-experiment
macro-F1 reporting initiative, extended back to exp_1-exp_5 on 2026-08-13.

Reproduces experiments/exp_5/scripts/run_weights.py's exact model/CV configuration (unchanged),
reusing cv_utils.repeated_cv_proba directly -- same per-factor skip handling for classes with too
few examples for a given fold's training split (ValueError -> recorded as skipped, matching the
original script's discipline exactly, not silently forced past).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_5/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cv_utils import N_SPLITS, RANDOM_STATE, repeated_cv_proba  # noqa: E402
from models import build_sklearn_models  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_exp3_feature_frame
from chimera_task1.reasoning_labels import TASK1_FACTORS, WEIGHT_RANK, weight_col
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

N_REPEATS = 5  # matches exp_5's own reduced N_REPEATS (higher fit count: 8 models x 9 factors x 2 scopes)
RESULTS_DIR = Path(__file__).parent.parent / "results"
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]


def merge_metrics(condition: str, new_fields: dict) -> None:
    path = RESULTS_DIR / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


def factor_macro_f1_sklearn(X, y_labels, clf, use_sample_weight: bool) -> float | None:
    y = np.array([WEIGHT_RANK[label] for label in y_labels])
    global_classes = np.unique(y)
    try:
        all_probas = repeated_cv_proba(X, y, clf, use_sample_weight, N_REPEATS, stratified=False)
    except ValueError:
        return None
    per_repeat = []
    for proba in all_probas:
        pred_dense = proba.argmax(axis=1)
        pred_rank = global_classes[pred_dense]
        per_repeat.append(f1_score(y, pred_rank, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    return float(np.mean(per_repeat))


def factor_macro_f1_kdm(X, y_labels) -> float | None:
    y = np.array([WEIGHT_RANK[label] for label in y_labels])
    n_classes = 4
    preprocessor = build_preprocessor(X)
    X_pre = preprocessor.fit_transform(X)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X_scaled = StandardScaler().fit_transform(X_pre)

    per_repeat = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in kf.split(X_scaled):
            test_probs = fit_predict_kdm(X_scaled[train_idx], y[train_idx], X_scaled[test_idx], n_classes)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    return float(np.mean(per_repeat))


def run_condition(condition: str, ann, X_full_or_groups, model_name: str, clf, use_sample_weight: bool, restricted: bool) -> None:
    per_factor = {}
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        if restricted:
            cols = restricted_feature_group(factor, "flags")
            X = X_full_or_groups[cols]
        else:
            X = X_full_or_groups

        try:
            if model_name == "kdm":
                f1 = factor_macro_f1_kdm(X, y_labels)
            else:
                f1 = factor_macro_f1_sklearn(X, y_labels, clf, use_sample_weight)
        except ValueError as e:
            # Same discipline as the original run_weights.py: a restricted single-column frame can
            # have near-zero nearest-neighbor distances for some factor, making init_kdm_layer's
            # KNN-based sigma estimate degenerate (sigma <= min_sigma). Recorded as a skip, not
            # forced past -- a genuine data-scarcity/degeneracy limit for this factor, not a bug.
            print(f"  [SKIP] {model_name} / {factor}: {type(e).__name__}: {e}")
            continue

        if f1 is None:
            print(f"  [SKIP] {model_name} / {factor}")
            continue
        per_factor[factor] = round(f1, 3)

    merge_metrics(condition, {
        "mean_macro_f1": round(float(np.mean(list(per_factor.values()))), 3) if per_factor else None,
        "per_factor_macro_f1": per_factor,
    })


def main() -> None:
    ann, inp_ann = load_annotated()

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_full = select_exp3_feature_frame(inp_ann, mri_pca)

    models = build_sklearn_models()
    models["kdm"] = (None, False)

    print("=== official scope ===")
    for name, (clf, use_sw) in models.items():
        run_condition(f"weights_official_{name}", ann, X_full, name, clf, use_sw, restricted=False)

    print("\n=== restricted scope ===")
    for name, (clf, use_sw) in models.items():
        run_condition(f"weights_restricted_{name}", ann, X_full, name, clf, use_sw, restricted=True)


if __name__ == "__main__":
    main()
