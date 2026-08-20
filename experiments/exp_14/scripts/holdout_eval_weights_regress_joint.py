"""exp_14: mandatory held-out check for joint (dim_y=9) KDM regression weights, 23-col frame.
One shared fit covering all 9 factors, sliced per factor for reporting -- same shape as
holdout_eval_weights_regress_per_factor.py except the fit itself.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_14/scripts/holdout_eval_weights_regress_joint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_regress_backbone import compute_signals_regress, fit_kdm_regress  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.reasoning_labels import (
    TASK1_FACTORS,
    WEIGHT_LEVELS,
    WEIGHT_RANK,
    decisive_set_f1,
    ordinal_distance,
    weight_col,
)
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
N_CLASSES = 4
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
DIM_Y = len(IN_SCOPE_FACTORS)
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_full.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_full.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)

    model = fit_kdm_regress(X_train, Y_all[train_idx], dim_y=DIM_Y)
    sig = compute_signals_regress(model, X_test)
    preds_all = sig["pred_rank"]  # (n_test, 9)
    pseudo_probs_all = sig["pseudo_probs"]  # (n_test, 9, 4)

    per_factor = {}
    for j, factor in enumerate(IN_SCOPE_FACTORS):
        y_labels = ann[weight_col(factor)].values
        y_test_rank = Y_all[test_idx, j]
        y_test_labels = y_labels[test_idx]
        preds = preds_all[:, j]
        pred_labels = [WEIGHT_LEVELS[r] for r in preds]

        acc = accuracy_score(y_test_rank, preds)
        macro_f1 = f1_score(y_test_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0)
        ord_err = ordinal_distance(list(y_test_labels), pred_labels, WEIGHT_RANK)
        dset_f1 = decisive_set_f1(list(y_test_labels), pred_labels)
        auroc = safe_multiclass_auroc(y_test_rank, pseudo_probs_all[:, j, :], labels=[0, 1, 2, 3])
        brier = multiclass_brier_score(y_test_rank, pseudo_probs_all[:, j, :], N_CLASSES)

        per_factor[factor] = {
            "accuracy": round(float(acc), 3),
            "macro_f1": round(float(macro_f1), 3),
            "ordinal_error": round(float(ord_err), 3),
            "decisive_set_f1": round(float(dset_f1), 3),
            "roc_auc": round(float(auroc), 3) if auroc is not None else None,
            "brier_score": round(float(brier), 3),
        }
        print(f"  {factor}: ordinal_error={per_factor[factor]['ordinal_error']} macro_f1={per_factor[factor]['macro_f1']}")

    condition = "holdout_weights_regress_joint"
    payload = {
        "condition": condition,
        "n_test": len(test_idx),
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in per_factor.values()])), 3),
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in per_factor.values()])), 3),
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in per_factor.values()])), 3),
        "n_factors_included": len(per_factor),
        "n_factors_skipped": 0,
        "per_factor": per_factor,
        "reference": {
            "weights_svm_incumbent_ordinal_error": 0.382,
            "weights_kdm_occlusion_ordinal_error": 0.405,
            "baseline_ordinal_error": 0.413,
        },
    }
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[{condition}] mean_ordinal_error={payload['mean_ordinal_error']} "
          f"mean_decisive_set_f1={payload['mean_decisive_set_f1']} mean_macro_f1={payload['mean_macro_f1']}\n")


if __name__ == "__main__":
    main()
