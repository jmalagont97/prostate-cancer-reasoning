"""exp_9: mandatory held-out check (per exp_7/exp_8's established discipline) -- a 3-way
comparison on the same 19 held-out cases exp_3/exp_7/exp_8 have all used:
  (a) exp_6's original scalar-sigma KDM, 19-column frame
  (b) ARD-KDM, 19-column frame
  (c) ARD-KDM, 23-column frame (exp_8's expanded frame)

This is a 3-way comparison, not 2-way, because exp_9's central question is specifically about the
frame-crossed-with-architecture interaction (does ARD rescue the 23-column frame's regression),
not just "is ARD better than scalar" in isolation.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_9/scripts/holdout_eval_ard.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from kdm_backbone import compute_signals as compute_signals_scalar, fit_kdm_backbone as fit_kdm_backbone_scalar  # noqa: E402

from chimera_task1.features import select_exp3_feature_frame
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
RESULTS_DIR = Path(__file__).parent.parent / "results"
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}


def score(y_true, probs, label):
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    print(f"\n--- {label} ---  F1={f1:.3f}  macro-F1={macro_f1:.3f}")
    print(classification_report(y_true, preds, target_names=["no", "yes"], digits=3, zero_division=0))
    return {"f1": round(float(f1), 3), "macro_f1": round(float(macro_f1), 3)}


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train, inp_test = inp_ann.iloc[train_idx].reset_index(drop=True), inp_ann.iloc[test_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]

    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)

    # --- (a) exp_6 scalar-sigma KDM, 19-column frame ---
    X_all_19 = select_exp3_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_19 = X_all_19.iloc[train_idx].reset_index(drop=True)
    X_test_raw_19 = X_all_19.iloc[test_idx].reset_index(drop=True)
    X_train_19, X_test_19 = fit_transform_features(X_train_raw_19, X_test_raw_19)
    model_a = fit_kdm_backbone_scalar(X_train_19, y_dec_train, n_classes=2)
    sig_a = compute_signals_scalar(model_a, X_test_19)

    # --- (b) ARD-KDM, 19-column frame ---
    model_b = fit_kdm_backbone_ard(X_train_19, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig_b = compute_signals_ard(model_b, X_test_19)

    # --- (c) ARD-KDM, 23-column frame ---
    X_all_23 = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw_23 = X_all_23.iloc[train_idx].reset_index(drop=True)
    X_test_raw_23 = X_all_23.iloc[test_idx].reset_index(drop=True)
    X_train_23, X_test_23 = fit_transform_features(X_train_raw_23, X_test_raw_23)
    model_c = fit_kdm_backbone_ard(X_train_23, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig_c = compute_signals_ard(model_c, X_test_23)

    print("=" * 70)
    print(f"HELD-OUT 3-WAY COMPARISON (n={len(test_idx)}, never used for any model selection)")
    print("=" * 70)
    result_a = score(y_dec_test, sig_a["probs"], "(a) exp_6 scalar-sigma, 19-col")
    result_b = score(y_dec_test, sig_b["probs"], "(b) ARD, 19-col")
    result_c = score(y_dec_test, sig_c["probs"], "(c) ARD, 23-col")

    print("=" * 70)
    print(f"(b) - (a) macro-F1 delta [ARD's own effect, 19-col]: {result_b['macro_f1'] - result_a['macro_f1']:+.3f}")
    print(f"(c) - (a) macro-F1 delta [ARD + expanded frame vs. exp_6 reference]: {result_c['macro_f1'] - result_a['macro_f1']:+.3f}")
    print(f"(c) - (b) macro-F1 delta [does the expanded frame still hurt, even under ARD?]: {result_c['macro_f1'] - result_b['macro_f1']:+.3f}")
    print("=" * 70)

    out = {
        "n_test": len(test_idx),
        "exp6_scalar_19col": result_a,
        "ard_19col": result_b,
        "ard_23col": result_c,
        "ard_effect_19col_delta": round(result_b["macro_f1"] - result_a["macro_f1"], 3),
        "ard_plus_expansion_delta": round(result_c["macro_f1"] - result_a["macro_f1"], 3),
        "expansion_under_ard_delta": round(result_c["macro_f1"] - result_b["macro_f1"], 3),
        "config": ARD_CONFIG,
    }
    out_dir = RESULTS_DIR / "holdout_eval_ard"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
