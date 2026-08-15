"""Backfill AUROC + Brier score into exp_9's decision-target results (CV conditions + the
mandatory held-out check), per this session's request. Decision-only -- confidence/weights are out
of scope, so this reproduces just the fit+probs half of run_signals_{19,23}col.py's and
holdout_eval_ard.py's loops (no signal computation, no per-factor occlusion), much cheaper than the
original full runs.

Aggregation matches macro-F1's existing convention: for CV, per-repeat pooled AUROC/Brier (all 91
out-of-fold predictions for that repeat), then mean+/-std across the 10 repeats -- not a per-fold
average, which would be noisy/undefined-in-parts for AUROC on ~14-18-row folds. For the held-out
check, a single value (n=19, no repeats).

Reproduces the EXACT same CV loop / held-out split as the original scripts (same RANDOM_STATE,
config, frames) -- read-modify-write into the existing metrics.json files, original fields
untouched.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_9/scripts/backfill_decision_auroc_brier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from kdm_backbone import compute_signals as compute_signals_scalar, fit_kdm_backbone as fit_kdm_backbone_scalar  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
TEST_SIZE = 0.2
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"


def merge_metrics(condition: str, new_fields: dict) -> None:
    path = RESULTS_DIR / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


def cv_auroc_brier(X_frame, y_decision) -> tuple[float, float, float, float]:
    """Fixed 5-fold x 10-repeat ARD-KDM decision CV, decision-only (no signals/weights). Returns
    (auroc_mean, auroc_std, brier_mean, brier_std), each per-repeat-pooled then averaged."""
    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    n = len(y_decision)
    aurocs, briers = [], []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_probs = np.zeros((n, 2))
        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])
            model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
            sig = compute_signals_ard(model, X_test)
            oof_probs[test_idx] = sig["probs"]
        aurocs.append(roc_auc_score(y_decision, oof_probs[:, 1]))
        briers.append(brier_score_loss(y_decision, oof_probs[:, 1]))
        print(f"  repeat {repeat} done  auroc={aurocs[-1]:.3f} brier={briers[-1]:.3f}")

    return (
        round(float(np.mean(aurocs)), 3), round(float(np.std(aurocs)), 3),
        round(float(np.mean(briers)), 3), round(float(np.std(briers)), 3),
    )


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)

    print("=== decision_kdm_ard_19col ===")
    X19 = select_exp3_feature_frame(inp_ann, mri_pca)
    am, asd, bm, bsd = cv_auroc_brier(X19, y_decision)
    merge_metrics("decision_kdm_ard_19col", {
        "roc_auc_mean": am, "roc_auc_std": asd, "brier_score_mean": bm, "brier_score_std": bsd,
    })

    print("\n=== decision_kdm_ard_23col ===")
    X23 = select_exp8_feature_frame(inp_ann, mri_pca)
    am, asd, bm, bsd = cv_auroc_brier(X23, y_decision)
    merge_metrics("decision_kdm_ard_23col", {
        "roc_auc_mean": am, "roc_auc_std": asd, "brier_score_mean": bm, "brier_score_std": bsd,
    })

    print("\n=== holdout_eval_ard (3-way) ===")
    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)

    X_all_19 = select_exp3_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_19 = X_all_19.iloc[train_idx].reset_index(drop=True)
    X_test_raw_19 = X_all_19.iloc[test_idx].reset_index(drop=True)
    X_train_19, X_test_19 = fit_transform_features(X_train_raw_19, X_test_raw_19)

    model_a = fit_kdm_backbone_scalar(X_train_19, y_dec_train, n_classes=2)
    sig_a = compute_signals_scalar(model_a, X_test_19)
    model_b = fit_kdm_backbone_ard(X_train_19, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig_b = compute_signals_ard(model_b, X_test_19)

    X_all_23 = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_23 = X_all_23.iloc[train_idx].reset_index(drop=True)
    X_test_raw_23 = X_all_23.iloc[test_idx].reset_index(drop=True)
    X_train_23, X_test_23 = fit_transform_features(X_train_raw_23, X_test_raw_23)
    model_c = fit_kdm_backbone_ard(X_train_23, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig_c = compute_signals_ard(model_c, X_test_23)

    def score(sig):
        p = sig["probs"][:, 1]
        return {
            "roc_auc": round(float(roc_auc_score(y_dec_test, p)), 3),
            "brier_score": round(float(brier_score_loss(y_dec_test, p)), 3),
        }

    new_fields = {
        "exp6_scalar_19col": score(sig_a),
        "ard_19col": score(sig_b),
        "ard_23col": score(sig_c),
    }
    for k, v in new_fields.items():
        print(f"  {k}: {v}")

    path = RESULTS_DIR / "holdout_eval_ard" / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    for key, vals in new_fields.items():
        payload[key].update(vals)  # merge into the existing per-condition sub-dict
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("[holdout_eval_ard] merged roc_auc/brier_score into each of the 3 sub-results")


if __name__ == "__main__":
    main()
