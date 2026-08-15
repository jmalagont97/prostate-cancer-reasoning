"""exp_11: mandatory held-out check for the directly-trained confidence ARD-KDM, both frames.

Same fixed 19-case split used since exp_3 (stratified on y_decision, not confidence -- "the same
split" means the same row indices; exp_3's own original holdout_eval.py scored confidence_svm/
confidence_kdm on this exact decision-stratified split too, so this isn't a new convention). Note:
exp_3's original held-out confidence numbers were only ever printed to console, never persisted to
a metrics.json -- there is no historical file to cite here, only exp_3's CV number (0.530 ordinal
distance / 0.508 macro-F1, cited in DESIGN.md Section 4).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_11/scripts/holdout_eval_confidence_direct_ard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
N_CLASSES = 3
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"

FRAMES = {
    "19col": select_exp3_feature_frame,
    "23col": select_exp8_feature_frame,
}


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)

    results = {}
    for frame_tag, frame_fn in FRAMES.items():
        X_all = frame_fn(inp_ann, mri_pca_aligned)
        X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
        X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

        model = fit_kdm_backbone_ard(X_train, y_conf_rank[train_idx], n_classes=N_CLASSES, **ARD_CONFIG)
        sig = compute_signals_ard(model, X_test)
        probs = sig["probs"]
        preds = probs.argmax(axis=1)
        pred_labels = [CONFIDENCE_LEVELS[r] for r in preds]
        y_test_rank = y_conf_rank[test_idx]
        y_test_labels = y_conf_labels[test_idx]

        acc = accuracy_score(y_test_rank, preds)
        macro_f1 = f1_score(y_test_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0)
        ord_dist = ordinal_distance(list(y_test_labels), pred_labels, CONFIDENCE_RANK)
        auroc = safe_multiclass_auroc(y_test_rank, probs, labels=[0, 1, 2])
        brier = multiclass_brier_score(y_test_rank, probs, N_CLASSES)

        print(f"=== {frame_tag} ===  probs_check_ok={sig['probs_check_ok']}")
        print(f"accuracy={acc:.3f} macro_f1={macro_f1:.3f} ordinal_distance={ord_dist:.3f} "
              f"roc_auc={auroc} brier={brier:.3f}")
        print(classification_report(y_test_rank, preds, target_names=CONFIDENCE_LEVELS, digits=3, zero_division=0))

        results[frame_tag] = {
            "accuracy": round(float(acc), 3),
            "macro_f1": round(float(macro_f1), 3),
            "ordinal_distance": round(float(ord_dist), 3),
            "roc_auc": round(float(auroc), 3) if auroc is not None else None,
            "brier_score": round(float(brier), 3),
            "probs_check_ok": sig["probs_check_ok"],
        }

    out = {
        "n_test": len(test_idx),
        **results,
        "reference": {
            "exp3_direct_confidence_kdm_cv_ordinal_distance": 0.530,
            "exp3_direct_confidence_kdm_cv_macro_f1": 0.508,
            "note": "exp_3's original held-out confidence numbers were only printed, never persisted -- CV is the only historical reference available",
        },
        "config": ARD_CONFIG,
    }
    out_dir = RESULTS_DIR / "holdout_confidence_direct_ard"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
