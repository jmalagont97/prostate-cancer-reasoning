"""Backfill AUROC + Brier score into exp_10's decision-target results (the CV condition +
the mandatory held-out check), per this session's request -- decision-only, same discipline as
experiments/exp_9/scripts/backfill_decision_auroc_brier.py (per-repeat-pooled AUROC/Brier for CV,
mean+/-std across repeats; single value for the one-shot held-out check).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_10/scripts/backfill_decision_auroc_brier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_fullschema import fit_transform_fullschema, select_exp10_feature_frame  # noqa: E402
from holdout_eval import mri_pca_train_only  # noqa: E402

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


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp10_feature_frame(inp_ann, mri_pca)

    print("=== decision_kdm_ard_fullschema (CV) ===")
    n = len(y_decision)
    aurocs, briers = [], []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_probs = np.zeros((n, 2))
        for train_idx, test_idx in kf.split(X_frame):
            X_train_raw = X_frame.iloc[train_idx].reset_index(drop=True)
            X_test_raw = X_frame.iloc[test_idx].reset_index(drop=True)
            X_train, X_test = fit_transform_fullschema(X_train_raw, X_test_raw)
            model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
            sig = compute_signals_ard(model, X_test)
            oof_probs[test_idx] = sig["probs"]
        aurocs.append(roc_auc_score(y_decision, oof_probs[:, 1]))
        briers.append(brier_score_loss(y_decision, oof_probs[:, 1]))
        print(f"  repeat {repeat} done  auroc={aurocs[-1]:.3f} brier={briers[-1]:.3f}")

    merge_metrics("decision_kdm_ard_fullschema", {
        "roc_auc_mean": round(float(np.mean(aurocs)), 3), "roc_auc_std": round(float(np.std(aurocs)), 3),
        "brier_score_mean": round(float(np.mean(briers)), 3), "brier_score_std": round(float(np.std(briers)), 3),
    })

    print("\n=== holdout_eval_fullschema ===")
    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_all = select_exp10_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_fullschema(X_train_raw, X_test_raw)

    model = fit_kdm_backbone_ard(X_train, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig = compute_signals_ard(model, X_test)
    p = sig["probs"][:, 1]
    new_fields = {
        "roc_auc": round(float(roc_auc_score(y_dec_test, p)), 3),
        "brier_score": round(float(brier_score_loss(y_dec_test, p)), 3),
    }
    print(f"  {new_fields}")

    path = RESULTS_DIR / "holdout_eval_fullschema" / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload["ard_fullschema"].update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("[holdout_eval_fullschema] merged roc_auc/brier_score")


if __name__ == "__main__":
    main()
