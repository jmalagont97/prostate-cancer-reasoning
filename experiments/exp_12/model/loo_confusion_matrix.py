"""Confusion matrix + classification report for exp_12's deployed model
(confidence_kdm_direct_scalar_23col), scored via leave-one-out (91-fold, pooled) -- the same
protocol exp_12/scripts/loo_confidence_direct_scalar.py already used to report LOO ordinal
distance/macro-F1/AUROC/Brier, but restricted to the 23-col frame only (the frame actually
deployed in confidence_kdm_23col.pkl) and additionally capturing the pooled predictions needed
for a confusion matrix / classification report, which the original script didn't persist.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_12/model/loo_confusion_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import mri_pca_train_only  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK
from chimera_task1.train_reasoning import load_annotated

N_CLASSES = 3
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    ann, inp_ann = load_annotated()
    y_conf_labels = ann["target_confidence"].values
    y_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])
    n = len(y_rank)

    oof_probs = np.zeros((n, N_CLASSES))
    idx_all = np.arange(n)

    for train_idx, test_idx in LeaveOneOut().split(idx_all):
        inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
        mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
        X_all = select_exp8_feature_frame(inp_ann, mri_pca_aligned)

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

        model = fit_kdm_backbone(X_train, y_rank[train_idx], n_classes=N_CLASSES)
        sig = compute_signals(model, X_test)
        oof_probs[test_idx] = sig["probs"]

        if (test_idx[0] + 1) % 15 == 0:
            print(f"  LOO fold {test_idx[0] + 1}/{n} done")

    preds = oof_probs.argmax(axis=1)

    cm = confusion_matrix(y_rank, preds, labels=[0, 1, 2])
    report = classification_report(
        y_rank, preds, labels=[0, 1, 2], target_names=CONFIDENCE_LEVELS, digits=3, zero_division=0
    )

    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in CONFIDENCE_LEVELS))
    for i, row in enumerate(cm):
        print(f"{CONFIDENCE_LEVELS[i]:>12}" + "".join(f"{v:>12}" for v in row))

    print("\nClassification report")
    print(report)

    out = {
        "condition": "confidence_kdm_direct_scalar_23col",
        "protocol": "leave-one-out (91-fold, pooled)",
        "labels": CONFIDENCE_LEVELS,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report,
        "classification_report": classification_report(
            y_rank, preds, labels=[0, 1, 2], target_names=CONFIDENCE_LEVELS, digits=3,
            zero_division=0, output_dict=True,
        ),
        "n": n,
    }
    out_dir = RESULTS_DIR / "loo_confusion_matrix_confidence_direct_scalar_23col"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
