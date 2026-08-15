"""exp_8: genuine held-out comparison of exp_6's plain KDM (19-col, original hyperparameters) vs.
exp_8's combined config (23-col expanded frame + winning hyperparameters).

Same held-out split as exp_3/exp_7's holdout_eval scripts (~20% of the 91 labeled cases,
stratified by decision, RANDOM_STATE=0, never used for fitting or feature-engineering choices).
This is the MANDATORY check (DESIGN.md Section 7/exp_7's own lesson) before any CV-measured
improvement gets reported as genuine.

Requires results/hyperparameter_search/winner.json to exist (run search_hyperparameters_v3.py first).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_8/scripts/holdout_eval_v3.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_7" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from kdm_backbone import compute_signals as compute_signals_v1, fit_kdm_backbone as fit_kdm_backbone_v1  # noqa: E402
from kdm_backbone_v2 import compute_signals as compute_signals_v2, fit_kdm_backbone as fit_kdm_backbone_v2  # noqa: E402

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

    # --- (a) exp_6's plain KDM, original 19-column frame ---
    X_all_v1 = select_exp3_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_v1 = X_all_v1.iloc[train_idx].reset_index(drop=True)
    X_test_raw_v1 = X_all_v1.iloc[test_idx].reset_index(drop=True)
    X_train_v1, X_test_v1 = fit_transform_features(X_train_raw_v1, X_test_raw_v1)
    model_v1 = fit_kdm_backbone_v1(X_train_v1, y_dec_train, n_classes=2)
    sig_v1 = compute_signals_v1(model_v1, X_test_v1)
    preds_v1 = sig_v1["probs"].argmax(axis=1)
    f1_v1 = f1_score(y_dec_test, preds_v1)
    macro_f1_v1 = f1_score(y_dec_test, preds_v1, average="macro")

    # --- (b) exp_8's combined config: 23-column frame + winning hyperparameters ---
    X_all_v3 = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_v3 = X_all_v3.iloc[train_idx].reset_index(drop=True)
    X_test_raw_v3 = X_all_v3.iloc[test_idx].reset_index(drop=True)
    X_train_v3, X_test_v3 = fit_transform_features(X_train_raw_v3, X_test_raw_v3)
    model_v3 = fit_kdm_backbone_v2(X_train_v3, y_dec_train, n_classes=2, **winning_config)
    sig_v3 = compute_signals_v2(model_v3, X_test_v3)
    preds_v3 = sig_v3["probs"].argmax(axis=1)
    f1_v3 = f1_score(y_dec_test, preds_v3)
    macro_f1_v3 = f1_score(y_dec_test, preds_v3, average="macro")

    print("=" * 70)
    print(f"HELD-OUT COMPARISON (n={len(test_idx)}, never used for any model selection)")
    print("=" * 70)
    print(f"\n--- (a) exp_6 plain KDM, 19-col frame ---  F1={f1_v1:.3f}  macro-F1={macro_f1_v1:.3f}")
    print(classification_report(y_dec_test, preds_v1, target_names=["no", "yes"], digits=3, zero_division=0))
    print(f"\n--- (b) exp_8 combined: 23-col frame + tuned hyperparameters ---  F1={f1_v3:.3f}  macro-F1={macro_f1_v3:.3f}")
    print(classification_report(y_dec_test, preds_v3, target_names=["no", "yes"], digits=3, zero_division=0))

    print("=" * 70)
    print(f"macro-F1 delta (b - a): {macro_f1_v3 - macro_f1_v1:+.3f}")
    print("=" * 70)

    out = {
        "n_test": len(test_idx),
        "exp6_plain_kdm_19col": {"f1": round(float(f1_v1), 3), "macro_f1": round(float(macro_f1_v1), 3)},
        "exp8_combined_23col": {"f1": round(float(f1_v3), 3), "macro_f1": round(float(macro_f1_v3), 3), "config": winning_config},
        "macro_f1_delta": round(float(macro_f1_v3 - macro_f1_v1), 3),
    }
    out_dir = RESULTS_DIR / "holdout_eval_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
