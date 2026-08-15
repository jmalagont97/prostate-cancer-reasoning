"""exp_13: scalar-sigma KDM trained DIRECTLY on each factor's weight label (not derived from a
decision-trained backbone) -- reviving exp_5's pre-exp_6 direct-training precedent for weights on
the 23-column frame, the way exp_11/exp_12 revived it for confidence. Both scopes (official,
restricted) in one script, all 9 in-scope factors, no recalibration step (the model's own
argmax(probs) is the prediction, same simplification as exp_11/exp_12). See DESIGN.md Section 2.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_13/scripts/run_weights_direct_scalar.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group
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
BACKBONE_TAG = "scalar"
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def fit_predict_oof(X_pre: np.ndarray, y_rank: np.ndarray, repeat: int) -> np.ndarray | None:
    """One repeat's out-of-fold probs for one factor/scope. Returns None (skip, logged) on a
    degenerate fit failure -- same discipline as exp_5's ValueError-catch precedent for rare
    per-factor classes / near-zero-variance restricted column groups."""
    n = len(y_rank)
    oof_probs = np.zeros((n, N_CLASSES))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
    for train_idx, test_idx in kf.split(X_pre):
        scaler = StandardScaler().fit(X_pre[train_idx])
        X_train = scaler.transform(X_pre[train_idx])
        X_test = scaler.transform(X_pre[test_idx])
        try:
            model = fit_kdm_backbone(X_train, y_rank[train_idx], n_classes=N_CLASSES)
        except ValueError:
            return None
        sig = compute_signals(model, X_test)
        if not sig["probs_check_ok"]:
            return None
        oof_probs[test_idx] = sig["probs"]
    return oof_probs


def run_factor(factor: str, X_pre: np.ndarray, y_rank: np.ndarray, y_labels: np.ndarray) -> dict | None:
    per_repeat = {"accuracy": [], "macro_f1": [], "ordinal_error": [], "decisive_set_f1": [], "roc_auc": [], "brier_score": []}
    for repeat in range(N_REPEATS):
        oof_probs = fit_predict_oof(X_pre, y_rank, repeat)
        if oof_probs is None:
            print(f"    [SKIP] {factor}: degenerate fit at repeat {repeat}")
            return None
        preds = oof_probs.argmax(axis=1)
        pred_labels = [WEIGHT_LEVELS[r] for r in preds]

        per_repeat["accuracy"].append(accuracy_score(y_rank, preds))
        per_repeat["macro_f1"].append(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        per_repeat["ordinal_error"].append(ordinal_distance(list(y_labels), pred_labels, WEIGHT_RANK))
        per_repeat["decisive_set_f1"].append(decisive_set_f1(list(y_labels), pred_labels))
        auroc = safe_multiclass_auroc(y_rank, oof_probs, labels=[0, 1, 2, 3])
        if auroc is not None:
            per_repeat["roc_auc"].append(auroc)
        per_repeat["brier_score"].append(multiclass_brier_score(y_rank, oof_probs, N_CLASSES))

    return {
        "accuracy": round(float(np.mean(per_repeat["accuracy"])), 3),
        "macro_f1": round(float(np.mean(per_repeat["macro_f1"])), 3),
        "ordinal_error": round(float(np.mean(per_repeat["ordinal_error"])), 3),
        "decisive_set_f1": round(float(np.mean(per_repeat["decisive_set_f1"])), 3),
        "roc_auc": round(float(np.mean(per_repeat["roc_auc"])), 3) if per_repeat["roc_auc"] else None,
        "brier_score": round(float(np.mean(per_repeat["brier_score"])), 3),
    }


def run_scope(scope: str, ann: pd.DataFrame, X_full: pd.DataFrame) -> None:
    preprocessor_full = build_preprocessor(X_full)
    X_full_pre = preprocessor_full.fit_transform(X_full)
    X_full_pre = X_full_pre.toarray() if hasattr(X_full_pre, "toarray") else X_full_pre

    per_factor = {}
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])

        if scope == "official":
            X_pre = X_full_pre
        else:
            cols = restricted_feature_group(factor, "flags")
            X_restricted = X_full[cols]
            preprocessor = build_preprocessor(X_restricted)
            X_pre = preprocessor.fit_transform(X_restricted)
            X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

        print(f"  [{scope}] {factor} ({X_pre.shape[1]} cols)...")
        result = run_factor(factor, X_pre, y_rank, y_labels)
        if result is None:
            per_factor[factor] = {"skipped": True}
        else:
            per_factor[factor] = result
            print(f"    ordinal_error={result['ordinal_error']} macro_f1={result['macro_f1']}")

    included = {f: v for f, v in per_factor.items() if "skipped" not in v}
    condition = f"weights_kdm_direct_{BACKBONE_TAG}_{scope}"
    payload = {
        "condition": condition,
        "target": "variable_weights",
        "features": f"exp_8 23-column frame, {scope} scope (direct training, {BACKBONE_TAG} backbone)",
        "model": f"{BACKBONE_TAG}-sigma KDM trained directly on each factor's 4-class weight label",
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in included.values()])), 3) if included else None,
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in included.values()])), 3) if included else None,
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in included.values()])), 3) if included else None,
        "n_factors_included": len(included),
        "n_factors_skipped": len(IN_SCOPE_FACTORS) - len(included),
        "n_repeats": N_REPEATS,
        "per_factor": per_factor,
        "note": "compare against weights_svm incumbent (official=0.382, restricted=0.392 mean ordinal error) "
                "and exp_5's original direct scalar-KDM on a worse frame (official=0.478, restricted=0.454)",
    }
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] mean_ordinal_error={payload['mean_ordinal_error']} "
          f"mean_decisive_set_f1={payload['mean_decisive_set_f1']} mean_macro_f1={payload['mean_macro_f1']} "
          f"({len(included)}/{len(IN_SCOPE_FACTORS)} factors included)\n")


def main() -> None:
    ann, inp_ann = load_annotated()
    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca)
    print(f"n={len(ann)}, feature frame: {X_full.shape}, in-scope factors: {IN_SCOPE_FACTORS}\n")

    for scope in ("official", "restricted"):
        print(f"=== {scope} scope ===")
        run_scope(scope, ann, X_full)


if __name__ == "__main__":
    main()
