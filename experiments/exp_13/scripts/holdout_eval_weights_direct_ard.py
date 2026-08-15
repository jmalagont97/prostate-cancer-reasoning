"""exp_13: mandatory held-out check for the directly-trained ARD-KDM weights models, both
scopes, all 9 in-scope factors. Identical to holdout_eval_weights_direct_scalar.py except the
backbone (exp_9's fit_kdm_backbone_ard/compute_signals_ard).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_13/scripts/holdout_eval_weights_direct_ard.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import restricted_feature_group
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
BACKBONE_TAG = "ard"
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def eval_factor(factor: str, X_train_raw, X_test_raw, y_labels, y_rank, train_idx, test_idx) -> dict | None:
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)
    try:
        model = fit_kdm_backbone_ard(X_train, y_rank[train_idx], n_classes=N_CLASSES, **ARD_CONFIG)
    except ValueError:
        print(f"    [SKIP] {factor}: degenerate fit")
        return None
    sig = compute_signals_ard(model, X_test)
    if not sig["probs_check_ok"]:
        print(f"    [SKIP] {factor}: probs_check_ok=False")
        return None
    probs = sig["probs"]
    preds = probs.argmax(axis=1)
    pred_labels = [WEIGHT_LEVELS[r] for r in preds]
    y_test_rank = y_rank[test_idx]
    y_test_labels = y_labels[test_idx]

    acc = accuracy_score(y_test_rank, preds)
    macro_f1 = f1_score(y_test_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    ord_err = ordinal_distance(list(y_test_labels), pred_labels, WEIGHT_RANK)
    dset_f1 = decisive_set_f1(list(y_test_labels), pred_labels)
    auroc = safe_multiclass_auroc(y_test_rank, probs, labels=[0, 1, 2, 3])
    brier = multiclass_brier_score(y_test_rank, probs, N_CLASSES)

    return {
        "accuracy": round(float(acc), 3),
        "macro_f1": round(float(macro_f1), 3),
        "ordinal_error": round(float(ord_err), 3),
        "decisive_set_f1": round(float(dset_f1), 3),
        "roc_auc": round(float(auroc), 3) if auroc is not None else None,
        "brier_score": round(float(brier), 3),
    }


def run_scope(scope: str, ann, X_full, train_idx, test_idx) -> None:
    per_factor = {}
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])

        if scope == "official":
            X_use = X_full
        else:
            cols = restricted_feature_group(factor, "flags")
            X_use = X_full[cols]

        X_train_raw = X_use.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_use.iloc[test_idx].reset_index(drop=True)

        print(f"  [{scope}] {factor} ({X_use.shape[1]} cols)...")
        result = eval_factor(factor, X_train_raw, X_test_raw, y_labels, y_rank, train_idx, test_idx)
        if result is None:
            per_factor[factor] = {"skipped": True}
        else:
            per_factor[factor] = result
            print(f"    ordinal_error={result['ordinal_error']} macro_f1={result['macro_f1']}")

    included = {f: v for f, v in per_factor.items() if "skipped" not in v}
    condition = f"holdout_weights_direct_{BACKBONE_TAG}_{scope}"
    payload = {
        "condition": condition,
        "n_test": len(test_idx),
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in included.values()])), 3) if included else None,
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in included.values()])), 3) if included else None,
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in included.values()])), 3) if included else None,
        "n_factors_included": len(included),
        "n_factors_skipped": len(IN_SCOPE_FACTORS) - len(included),
        "per_factor": per_factor,
        "reference": {
            "weights_svm_incumbent_ordinal_error": 0.382 if scope == "official" else 0.392,
            "exp5_direct_scalar_kdm_ordinal_error": 0.478 if scope == "official" else 0.454,
        },
        "config": ARD_CONFIG,
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
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca_aligned)

    for scope in ("official", "restricted"):
        print(f"=== {scope} scope ===")
        run_scope(scope, ann, X_full, train_idx, test_idx)


if __name__ == "__main__":
    main()
