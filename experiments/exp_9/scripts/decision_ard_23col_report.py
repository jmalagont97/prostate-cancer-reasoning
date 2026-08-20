"""Follow-up (2026-08-18): confusion matrix + classification_report for exp_9's decision result
on the 23-column ARD-KDM frame -- the project's best-validated decision configuration (see
experiments/exp_9/reports/summary.md #3, confirmed in experiments/exp_10/reports/summary.md).

Both `holdout_eval_ard.py` (exp_9) and `verify_decision_loo_repeated_holdout.py` (exp_10) already
computed these predictions but only ever printed the classification_report to stdout -- neither
script persisted it. This script re-fits the exact same pipeline (fit_kdm_backbone_ard,
select_exp8_feature_frame, mri_pca_train_only, ARD_CONFIG identical to both source scripts) and
saves the confusion matrix + classification_report for:
  (a) held-out split (seed=0, n_test=19, matches exp_9/exp_10's already-reported macro_f1=0.680)
  (b) LOO (91-fold pooled predictions, matches exp_10's already-reported macro_f1=0.639)

Does not touch or re-run any existing results -- purely additive, restricted to the 23-column ARD
condition already established as the reference. Numbers should reproduce exp_9/exp_10's existing
metrics.json exactly (same seed, same config) -- verified via macro_f1 match before trusting the
new confusion-matrix/report output.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_9/scripts/decision_ard_23col_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402

from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
TARGET_NAMES = ["no", "yes"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def fit_predict(inp_ann, y_decision, train_idx, test_idx):
    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_all = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)
    model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
    sig = compute_signals_ard(model, X_test)
    return sig["probs"]


def report_block(y_true, preds, label, reference_macro_f1):
    macro_f1 = round(float(f1_score(y_true, preds, average="macro")), 3)
    cm = confusion_matrix(y_true, preds).tolist()
    cr_text = classification_report(y_true, preds, target_names=TARGET_NAMES, digits=3, zero_division=0)
    cr_dict = classification_report(y_true, preds, target_names=TARGET_NAMES, digits=3, zero_division=0, output_dict=True)

    match = "MATCH" if abs(macro_f1 - reference_macro_f1) < 0.005 else "MISMATCH -- INVESTIGATE"
    print(f"\n--- {label} ---  macro-F1={macro_f1:.3f} (reference={reference_macro_f1}, {match})")
    print(f"Confusion matrix (rows=true, cols=pred, order={TARGET_NAMES}):")
    print(np.array(cm))
    print(cr_text)

    return {
        "macro_f1": macro_f1,
        "reference_macro_f1": reference_macro_f1,
        "reference_match": match == "MATCH",
        "confusion_matrix": cm,
        "confusion_matrix_labels": TARGET_NAMES,
        "classification_report_text": cr_text,
        "classification_report": cr_dict,
    }


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)
    n = len(y_decision)

    # (a) held-out, seed=0 -- reproduces exp_9's holdout_eval_ard.py "(c) ARD, 23-col"
    idx_all = np.arange(n)
    train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    probs_holdout = fit_predict(inp_ann, y_decision, train_idx, test_idx)
    preds_holdout = probs_holdout.argmax(axis=1)
    holdout_result = report_block(y_decision[test_idx], preds_holdout, "HELD-OUT (seed=0, n=19)", 0.680)

    # (b) LOO, 91 folds pooled -- reproduces exp_10's verify_decision_loo_repeated_holdout.py "23col_exp9"
    oof_probs = np.zeros((n, 2))
    for i, (train_idx_loo, test_idx_loo) in enumerate(LeaveOneOut().split(idx_all)):
        probs = fit_predict(inp_ann, y_decision, train_idx_loo, test_idx_loo)
        oof_probs[test_idx_loo] = probs
        if (i + 1) % 10 == 0:
            print(f"    LOO fold {i + 1}/{n} done")
    preds_loo = oof_probs.argmax(axis=1)
    loo_result = report_block(y_decision, preds_loo, "LEAVE-ONE-OUT (91 folds, pooled)", 0.639)

    out = {
        "condition": "decision_kdm_ard_23col",
        "note": "confusion matrix + classification_report follow-up; macro_f1 must match "
                "exp_9/exp_10's already-reported values (0.680 held-out, 0.639 LOO) -- see "
                "reference_match field in each block",
        "held_out_seed0": holdout_result,
        "loo": loo_result,
    }
    out_dir = RESULTS_DIR / "decision_kdm_ard_23col_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
