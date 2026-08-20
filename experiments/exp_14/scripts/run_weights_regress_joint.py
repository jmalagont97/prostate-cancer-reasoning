"""exp_14: KDM regression trained on ALL 9 factors' weight ranks AT ONCE (dim_y=9, one shared
prototype pool) -- untried by any prior weights condition, including the SVM incumbent, which
also fits 9 independent per-factor models. Tests whether sharing statistical strength across
factors helps, especially the 4 data-scarce factors (cspca/comorbidity/psad/vol) stuck near
decisive-set F1 ~ 0 in every prior experiment. 23-column frame only, scalar backbone only.

Per-factor metrics are sliced out of the single joint fit's (n, 9) predictions per fold, so this
is directly comparable to run_weights_regress_per_factor.py's per-factor breakdown -- only the
fit itself differs (1 fit per fold here, vs. 9 fits per fold there).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_14/scripts/run_weights_regress_joint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_regress_backbone import compute_signals_regress, fit_kdm_regress  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import (
    TASK1_FACTORS,
    WEIGHT_LEVELS,
    WEIGHT_RANK,
    decisive_set_f1,
    ordinal_distance,
    weight_col,
)
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
N_CLASSES = 4
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
DIM_Y = len(IN_SCOPE_FACTORS)
RESULTS_DIR = Path(__file__).parent.parent / "results"


def fit_predict_oof(X_pre: np.ndarray, Y_all: np.ndarray, repeat: int):
    """One joint fit per fold, covering all 9 factors -- fit-call count equals n_folds, not
    n_folds x 9 (verified below in main() via a fit counter)."""
    n = len(X_pre)
    oof_pred = np.full((n, DIM_Y), -1, dtype=int)
    oof_pseudo_probs = np.zeros((n, DIM_Y, N_CLASSES))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
    n_fits = 0
    for train_idx, test_idx in kf.split(X_pre):
        scaler = StandardScaler().fit(X_pre[train_idx])
        X_train = scaler.transform(X_pre[train_idx])
        X_test = scaler.transform(X_pre[test_idx])
        try:
            model = fit_kdm_regress(X_train, Y_all[train_idx], dim_y=DIM_Y)
        except ValueError:
            return None, None
        n_fits += 1
        sig = compute_signals_regress(model, X_test)
        oof_pred[test_idx] = sig["pred_rank"]
        oof_pseudo_probs[test_idx] = sig["pseudo_probs"]
    return (oof_pred, oof_pseudo_probs), n_fits


def main() -> None:
    ann, inp_ann = load_annotated()
    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca)
    print(f"n={len(ann)}, feature frame: {X_full.shape}, in-scope factors: {IN_SCOPE_FACTORS}\n")

    preprocessor = build_preprocessor(X_full)
    X_pre = preprocessor.fit_transform(X_full)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)  # (91, 9)

    per_repeat = {f: {"accuracy": [], "macro_f1": [], "ordinal_error": [], "decisive_set_f1": [], "roc_auc": [], "brier_score": []} for f in IN_SCOPE_FACTORS}
    total_fits = 0
    total_expected_folds = N_REPEATS * N_SPLITS

    for repeat in range(N_REPEATS):
        result, n_fits = fit_predict_oof(X_pre, Y_all, repeat)
        if result is None:
            print(f"  [SKIP] joint model: degenerate fit at repeat {repeat}")
            continue
        total_fits += n_fits
        oof_pred, oof_pseudo_probs = result

        for j, factor in enumerate(IN_SCOPE_FACTORS):
            y_rank = Y_all[:, j]
            y_labels = ann[weight_col(factor)].values
            preds = oof_pred[:, j]
            pred_labels = [WEIGHT_LEVELS[r] for r in preds]

            per_repeat[factor]["accuracy"].append(accuracy_score(y_rank, preds))
            per_repeat[factor]["macro_f1"].append(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0))
            per_repeat[factor]["ordinal_error"].append(ordinal_distance(list(y_labels), pred_labels, WEIGHT_RANK))
            per_repeat[factor]["decisive_set_f1"].append(decisive_set_f1(list(y_labels), pred_labels))
            auroc = safe_multiclass_auroc(y_rank, oof_pseudo_probs[:, j, :], labels=[0, 1, 2, 3])
            if auroc is not None:
                per_repeat[factor]["roc_auc"].append(auroc)
            per_repeat[factor]["brier_score"].append(multiclass_brier_score(y_rank, oof_pseudo_probs[:, j, :], N_CLASSES))

    print(f"Fit-count check: {total_fits} joint fits total (expected {total_expected_folds} = "
          f"{N_REPEATS} repeats x {N_SPLITS} folds, NOT x{DIM_Y} factors -- confirms one shared "
          f"fit per fold, not a per-factor refit).\n")

    per_factor = {}
    for factor in IN_SCOPE_FACTORS:
        pr = per_repeat[factor]
        if not pr["accuracy"]:
            per_factor[factor] = {"skipped": True}
            continue
        per_factor[factor] = {
            "accuracy": round(float(np.mean(pr["accuracy"])), 3),
            "macro_f1": round(float(np.mean(pr["macro_f1"])), 3),
            "ordinal_error": round(float(np.mean(pr["ordinal_error"])), 3),
            "decisive_set_f1": round(float(np.mean(pr["decisive_set_f1"])), 3),
            "roc_auc": round(float(np.mean(pr["roc_auc"])), 3) if pr["roc_auc"] else None,
            "brier_score": round(float(np.mean(pr["brier_score"])), 3),
        }
        print(f"  {factor}: ordinal_error={per_factor[factor]['ordinal_error']} "
              f"macro_f1={per_factor[factor]['macro_f1']}")

    included = {f: v for f, v in per_factor.items() if "skipped" not in v}
    condition = "weights_kdm_regress_joint"
    payload = {
        "condition": condition,
        "target": "variable_weights",
        "features": "exp_8 23-column frame (regression training, scalar backbone, joint dim_y=9)",
        "model": "KDMRegressModel trained on all 9 factors' weight ranks at once, one shared prototype pool (dm_rbf_loglik loss)",
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in included.values()])), 3) if included else None,
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in included.values()])), 3) if included else None,
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in included.values()])), 3) if included else None,
        "n_factors_included": len(included),
        "n_factors_skipped": len(IN_SCOPE_FACTORS) - len(included),
        "n_repeats": N_REPEATS,
        "n_fits_total": total_fits,
        "n_fits_expected": total_expected_folds,
        "per_factor": per_factor,
        "auroc_brier_note": "AUROC/Brier use regression-derived pseudo-probabilities; the joint "
                             "condition additionally shares ONE variance across all 9 factors per "
                             "case (predict_reg's variance is scalar-per-case, not per-dimension) "
                             "-- only the per-factor mean differs. See DESIGN.md Section 2c.",
        "note": "compare against weights_kdm_occlusion (0.405, best KDM ever, primary bar), "
                "baseline (0.413), weights_svm incumbent (0.382, reported for transparency), and "
                "run_weights_regress_per_factor.py's per-factor regression (same objective, 9 "
                "independent fits instead of 1 shared fit)",
    }
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[{condition}] mean_ordinal_error={payload['mean_ordinal_error']} "
          f"mean_decisive_set_f1={payload['mean_decisive_set_f1']} mean_macro_f1={payload['mean_macro_f1']} "
          f"({len(included)}/{len(IN_SCOPE_FACTORS)} factors included)\n")


if __name__ == "__main__":
    main()
