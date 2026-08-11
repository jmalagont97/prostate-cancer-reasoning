"""exp_7: genuine held-out comparison of exp_6's plain KDM vs. exp_7's tuned+preprocessed KDM.

Same held-out split as experiments/exp_3/scripts/holdout_eval.py (~20% of the 91 labeled cases,
stratified by decision, RANDOM_STATE=0, never used for fitting or feature-engineering choices --
MRI-PCA fit on the train portion only). Fits both models on the train portion, scores both once
on the untouched test portion -- the out-of-sample check DESIGN.md Section 2/9 requires before
calling the hyperparameter search's winner a genuine improvement rather than CV noise.

Requires results/hyperparameter_search/winner.json to exist (run search_hyperparameters.py first).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_7/scripts/holdout_eval_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from kdm_backbone import compute_signals as compute_signals_v1, fit_kdm_backbone as fit_kdm_backbone_v1  # noqa: E402
from kdm_backbone_v2 import LOG1P_COLUMNS, apply_log1p_transform, compute_signals as compute_signals_v2, fit_kdm_backbone as fit_kdm_backbone_v2  # noqa: E402

from chimera_task1.features import select_exp3_feature_frame
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    with open(RESULTS_DIR / "hyperparameter_search" / "winner.json") as f:
        winner = json.load(f)
    winning_config = {
        "n_epochs": winner["n_epochs"], "lr": winner["lr"], "sigma_mult": winner["sigma_mult"],
        "optimizer": winner["optimizer"], "weight_decay": winner["weight_decay"],
    }
    print(f"Winning config: {winning_config}\n")

    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train, inp_test = inp_ann.iloc[train_idx].reset_index(drop=True), inp_ann.iloc[test_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]

    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_all = select_exp3_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

    print(f"held-out decision positive rate: {y_dec_test.mean():.2%} (train: {y_dec_train.mean():.2%})\n")

    # --- (a) exp_6's plain KDM, original features ---
    X_train_v1, X_test_v1 = fit_transform_features(X_train_raw, X_test_raw)
    model_v1 = fit_kdm_backbone_v1(X_train_v1, y_dec_train, n_classes=2)
    sig_v1 = compute_signals_v1(model_v1, X_test_v1)
    preds_v1 = sig_v1["probs"].argmax(axis=1)
    f1_v1 = f1_score(y_dec_test, preds_v1)
    macro_f1_v1 = f1_score(y_dec_test, preds_v1, average="macro")

    # --- (b) exp_7's tuned KDM, log1p-transformed features (train-only fit, applied to both) ---
    log1p_idx = [X_all.columns.get_loc(c) for c in LOG1P_COLUMNS]
    X_train_raw_log = X_train_raw.copy()
    X_test_raw_log = X_test_raw.copy()
    for c in LOG1P_COLUMNS:
        X_train_raw_log[c] = np.log1p(X_train_raw_log[c])
        X_test_raw_log[c] = np.log1p(X_test_raw_log[c])
    X_train_v2, X_test_v2 = fit_transform_features(X_train_raw_log, X_test_raw_log)
    model_v2 = fit_kdm_backbone_v2(X_train_v2, y_dec_train, n_classes=2, **winning_config)
    sig_v2 = compute_signals_v2(model_v2, X_test_v2)
    preds_v2 = sig_v2["probs"].argmax(axis=1)
    f1_v2 = f1_score(y_dec_test, preds_v2)
    macro_f1_v2 = f1_score(y_dec_test, preds_v2, average="macro")

    print("=" * 70)
    print(f"HELD-OUT COMPARISON (n={len(test_idx)}, never used for any model selection)")
    print("=" * 70)
    print(f"\n--- (a) exp_6 plain KDM ---  F1={f1_v1:.3f}  macro-F1={macro_f1_v1:.3f}")
    print(classification_report(y_dec_test, preds_v1, target_names=["no", "yes"], digits=3, zero_division=0))
    print(f"\n--- (b) exp_7 tuned+preprocessed KDM ---  F1={f1_v2:.3f}  macro-F1={macro_f1_v2:.3f}")
    print(classification_report(y_dec_test, preds_v2, target_names=["no", "yes"], digits=3, zero_division=0))

    print("=" * 70)
    print(f"macro-F1 delta (b - a): {macro_f1_v2 - macro_f1_v1:+.3f}")
    print("=" * 70)

    out = {
        "n_test": len(test_idx),
        "exp6_plain_kdm": {"f1": round(float(f1_v1), 3), "macro_f1": round(float(macro_f1_v1), 3)},
        "exp7_tuned_kdm": {"f1": round(float(f1_v2), 3), "macro_f1": round(float(macro_f1_v2), 3), "config": winning_config},
        "macro_f1_delta": round(float(macro_f1_v2 - macro_f1_v1), 3),
    }
    out_dir = RESULTS_DIR / "holdout_eval_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
