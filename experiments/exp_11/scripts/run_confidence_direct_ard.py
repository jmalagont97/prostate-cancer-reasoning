"""exp_11: ARD-KDM trained DIRECTLY on the confidence label (not derived from a decision-trained
backbone, unlike every confidence condition since exp_6). Reviving exp_2/exp_3's pre-exp_6
approach with exp_9's ARD backbone -- see DESIGN.md Section 1.

No recalibration step needed, unlike exp_9/exp_10's derived-signal scripts: the model's own
argmax(probs) IS the prediction. Full metric suite (accuracy, macro-F1, ordinal distance, one-vs-
rest AUROC, multiclass Brier), per-repeat-pooled then averaged across repeats -- first experiment
run natively under both new reporting conventions (experiments/INDEX.md).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_11/scripts/run_confidence_direct_ard.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
N_CLASSES = 3
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"

FRAMES = {
    "19col": select_exp3_feature_frame,
    "23col": select_exp8_feature_frame,
}


def run_frame(frame_tag: str, frame_fn, X_frame: pd.DataFrame, y_rank: np.ndarray, y_labels: np.ndarray) -> None:
    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    n = len(y_rank)
    per_repeat = {"accuracy": [], "macro_f1": [], "ordinal_distance": [], "roc_auc": [], "brier_score": []}
    all_checks_ok = True

    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_probs = np.zeros((n, N_CLASSES))

        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])

            model = fit_kdm_backbone_ard(X_train, y_rank[train_idx], n_classes=N_CLASSES, **ARD_CONFIG)
            sig = compute_signals_ard(model, X_test)
            all_checks_ok = all_checks_ok and sig["probs_check_ok"]
            oof_probs[test_idx] = sig["probs"]

        preds = oof_probs.argmax(axis=1)
        pred_labels = [CONFIDENCE_LEVELS[r] for r in preds]

        per_repeat["accuracy"].append(accuracy_score(y_rank, preds))
        per_repeat["macro_f1"].append(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0))
        per_repeat["ordinal_distance"].append(ordinal_distance(list(y_labels), pred_labels, CONFIDENCE_RANK))
        auroc = safe_multiclass_auroc(y_rank, oof_probs, labels=[0, 1, 2])
        if auroc is not None:
            per_repeat["roc_auc"].append(auroc)
        per_repeat["brier_score"].append(multiclass_brier_score(y_rank, oof_probs, N_CLASSES))

        print(f"  [{frame_tag}] repeat {repeat} done  "
              f"acc={per_repeat['accuracy'][-1]:.3f} macro_f1={per_repeat['macro_f1'][-1]:.3f} "
              f"ord_dist={per_repeat['ordinal_distance'][-1]:.3f}")

    print(f"[{frame_tag}] probs_check_ok across every fold/repeat: {all_checks_ok}")
    if not all_checks_ok:
        raise RuntimeError("compute_signals_ard's hand-replicated normalization did not match model.forward() at least once")

    condition = f"confidence_kdm_direct_ard_{frame_tag}"
    payload = {
        "condition": condition,
        "target": "confidence",
        "features": f"exp_9/exp_10 {frame_tag} frame (direct training, not derived signal)",
        "model": f"ARD-KDM trained directly on the 3-class confidence label, fixed config: {ARD_CONFIG}",
        "accuracy_mean": round(float(np.mean(per_repeat["accuracy"])), 3),
        "accuracy_std": round(float(np.std(per_repeat["accuracy"])), 3),
        "macro_f1_mean": round(float(np.mean(per_repeat["macro_f1"])), 3),
        "macro_f1_std": round(float(np.std(per_repeat["macro_f1"])), 3),
        "ordinal_distance_mean": round(float(np.mean(per_repeat["ordinal_distance"])), 3),
        "ordinal_distance_std": round(float(np.std(per_repeat["ordinal_distance"])), 3),
        "roc_auc_mean": round(float(np.mean(per_repeat["roc_auc"])), 3) if per_repeat["roc_auc"] else None,
        "roc_auc_std": round(float(np.std(per_repeat["roc_auc"])), 3) if per_repeat["roc_auc"] else None,
        "roc_auc_n_repeats_included": len(per_repeat["roc_auc"]),
        "brier_score_mean": round(float(np.mean(per_repeat["brier_score"])), 3),
        "brier_score_std": round(float(np.std(per_repeat["brier_score"])), 3),
        "n_folds": N_REPEATS * N_SPLITS,
        "note": "compare against exp_3's original directly-trained confidence_kdm (scalar backbone): "
                "ordinal_distance=0.530, macro_f1=0.508 -- and exp_6-exp_10's best derived-signal "
                "conditions (exp_6 entropy_isotonic: 0.731/0.223; exp_9 ARD 19-col blend: 0.836/0.283)",
    }
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] accuracy={payload['accuracy_mean']} macro_f1={payload['macro_f1_mean']} "
          f"ordinal_distance={payload['ordinal_distance_mean']} roc_auc={payload['roc_auc_mean']} "
          f"brier={payload['brier_score_mean']}\n")


def main() -> None:
    ann, inp_ann = load_annotated()
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)

    print(f"n={len(y_conf_rank)}, class distribution: "
          f"{dict(zip(*np.unique(y_conf_labels, return_counts=True)))}\n")

    for frame_tag, frame_fn in FRAMES.items():
        X_frame = frame_fn(inp_ann, mri_pca)
        print(f"=== {frame_tag} frame: {X_frame.shape} ===")
        run_frame(frame_tag, frame_fn, X_frame, y_conf_rank, y_conf_labels)


if __name__ == "__main__":
    main()
