"""exp_11: leave-one-out check for the directly-trained confidence ARD-KDM, both frames --
first-ever LOO check for a non-decision subtask (experiments/INDEX.md's "leave-one-out (LOO)
evaluation protocol" note), following experiments/exp_10/scripts/
verify_decision_loo_repeated_holdout.py's run_loo() shape, generalized from 2 to 3 classes.

Uses the stricter "held-out discipline" MRI-PCA convention (fit PCA on each fold's train rows
only, per experiments/exp_3/scripts/holdout_eval.py's mri_pca_train_only) -- same choice exp_10's
verification script made, treating LOO as part of the held-out family, not the CV family.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_11/scripts/loo_confidence_direct_ard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import mri_pca_train_only  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_CLASSES = 3
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"

FRAMES = {
    "19col": select_exp3_feature_frame,
    "23col": select_exp8_feature_frame,
}


def run_loo(inp_ann: pd.DataFrame, y_rank: np.ndarray, frame_fn) -> dict:
    n = len(y_rank)
    oof_probs = np.zeros((n, N_CLASSES))
    idx_all = np.arange(n)

    for train_idx, test_idx in LeaveOneOut().split(idx_all):
        inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
        mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
        X_all = frame_fn(inp_ann, mri_pca_aligned)

        X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

        preprocessor = build_preprocessor(X_train_raw)
        X_train_pre = preprocessor.fit_transform(X_train_raw)
        X_test_pre = preprocessor.transform(X_test_raw)
        X_train_pre = X_train_pre.toarray() if hasattr(X_train_pre, "toarray") else X_train_pre
        X_test_pre = X_test_pre.toarray() if hasattr(X_test_pre, "toarray") else X_test_pre
        scaler = StandardScaler().fit(X_train_pre)
        X_train = scaler.transform(X_train_pre)
        X_test = scaler.transform(X_test_pre)

        model = fit_kdm_backbone_ard(X_train, y_rank[train_idx], n_classes=N_CLASSES, **ARD_CONFIG)
        sig = compute_signals_ard(model, X_test)
        oof_probs[test_idx] = sig["probs"]

        if (test_idx[0] + 1) % 10 == 0:
            print(f"    LOO fold {test_idx[0] + 1}/{n} done")

    preds = oof_probs.argmax(axis=1)
    pred_labels = [CONFIDENCE_LEVELS[r] for r in preds]
    y_labels = [CONFIDENCE_LEVELS[r] for r in y_rank]

    return {
        "accuracy": round(float(accuracy_score(y_rank, preds)), 3),
        "macro_f1": round(float(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0)), 3),
        "ordinal_distance": round(float(ordinal_distance(y_labels, pred_labels, CONFIDENCE_RANK)), 3),
        "roc_auc": (lambda v: round(v, 3) if v is not None else None)(safe_multiclass_auroc(y_rank, oof_probs, labels=[0, 1, 2])),
        "brier_score": round(float(multiclass_brier_score(y_rank, oof_probs, N_CLASSES)), 3),
        "n_folds": n,
    }


def main() -> None:
    ann, inp_ann = load_annotated()
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])

    results = {}
    for frame_tag, frame_fn in FRAMES.items():
        print(f"\n=== {frame_tag} ===")
        result = run_loo(inp_ann, y_conf_rank, frame_fn)
        print(f"  LOO (91-fold, pooled): {result}")
        results[frame_tag] = result

    out_dir = RESULTS_DIR / "loo_confidence_direct_ard"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for frame_tag, r in results.items():
        print(f"{frame_tag}: accuracy={r['accuracy']} macro_f1={r['macro_f1']} "
              f"ordinal_distance={r['ordinal_distance']} roc_auc={r['roc_auc']} brier={r['brier_score']}")


if __name__ == "__main__":
    main()
