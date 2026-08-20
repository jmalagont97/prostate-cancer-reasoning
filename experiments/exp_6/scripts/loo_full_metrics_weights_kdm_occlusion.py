"""Backfill: full metric suite (accuracy, macro-F1, AUROC, Brier, on top of exp_6's original
ordinal_error/decisive_set_f1) + leave-one-out evaluation + pooled confusion matrix/classification
report + per-case breakdown, for weights_kdm_occlusion -- the best KDM-based variable-weights
model in this project (0.405 ordinal error, still the primary bar exp_9/exp_13/exp_14 all measured
themselves against and never beat).

Faithfully reproduces experiments/exp_6/scripts/run_signals.py's exact pipeline (ONE decision-
trained KDM backbone shared across all 9 factors per fold; per-factor occlusion-delta signal
[Signal D]; isotonic regression recalibrates the signal into an ordinal weight rank) -- including
its one preprocessing quirk: build_preprocessor's imputer is fit once on the full 91-row pool
(only StandardScaler is refit per fold), exactly as the original script did. Only the evaluation
protocol changes here (LOO instead of 10x5 CV), the model/pipeline itself is untouched.

AUROC/Brier caveat: isotonic regression has no native calibrated-probability output (it's a
deterministic monotonic point estimate, then rounded to a class) -- unlike weights_svm's real
predict_proba or exp_14's regression variance-based pseudo-probabilities, there's no principled
uncertainty estimate to build a smooth probability distribution from here. AUROC/Brier are
computed from a ONE-HOT hard-decision "probability" (1.0 on the predicted class) as the only
honest option available, explicitly flagged as degenerate in the output -- not comparable in
richness to the other two models' AUROC/Brier, included only so no metric this project tracks is
silently omitted.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_6/scripts/loo_full_metrics_weights_kdm_occlusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone, occlusion_delta  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_exp3_feature_frame
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
N_CLASSES = 4
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def isotonic_rank(train_signal: np.ndarray, train_rank: np.ndarray, test_signal: np.ndarray, n_levels: int) -> int:
    # Identical to run_signals.py's isotonic_rank, unchanged.
    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(train_signal, train_rank)
    raw = iso.predict(test_signal)
    return int(np.clip(np.round(raw), 0, n_levels - 1).astype(int)[0])


def main() -> None:
    ann, inp_ann = load_annotated()
    n = len(ann)
    case_ids = ann["case_id"].values
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp3_feature_frame(inp_ann, mri_pca)
    print(f"n={n}, 19-col frame: {X_frame.shape}, in-scope factors: {IN_SCOPE_FACTORS}\n")

    # Matches run_signals.py exactly: imputer fit ONCE on the full 91-row pool (not per fold);
    # only StandardScaler is refit per fold below.
    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    factor_col_idx = {f: [X_frame.columns.get_loc(c) for c in restricted_feature_group(f, "flags")] for f in IN_SCOPE_FACTORS}

    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)  # (91, 9)

    oof_pred = np.full((n, len(IN_SCOPE_FACTORS)), -1, dtype=int)
    factor_failed = set()
    per_case_rows = [{"case_id": cid} for cid in case_ids]

    for fold_i, (train_idx, test_idx) in enumerate(LeaveOneOut().split(np.arange(n))):
        scaler = StandardScaler().fit(X_pre[train_idx])
        X_train = scaler.transform(X_pre[train_idx])
        X_test = scaler.transform(X_pre[test_idx])

        model = fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2)

        for j, factor in enumerate(IN_SCOPE_FACTORS):
            if factor in factor_failed:
                continue
            col_idx = factor_col_idx[factor]
            fill = np.median(X_train[:, col_idx], axis=0)
            D_tr = np.abs(occlusion_delta(model, X_train, col_idx, fill))
            D_te = np.abs(occlusion_delta(model, X_test, col_idx, fill))
            y_w_tr = Y_all[train_idx, j]
            try:
                oof_pred[test_idx[0], j] = isotonic_rank(D_tr, y_w_tr, D_te, N_CLASSES)
            except ValueError as e:
                factor_failed.add(factor)
                print(f"  [SKIP] {factor}: {e}")

        if (fold_i + 1) % 15 == 0:
            print(f"  LOO fold {fold_i + 1}/{n} done")

    included_factors = [f for f in IN_SCOPE_FACTORS if f not in factor_failed]
    per_factor = {}
    all_true_pooled, all_pred_pooled = [], []

    for j, factor in enumerate(IN_SCOPE_FACTORS):
        if factor in factor_failed:
            per_factor[factor] = {"skipped": True}
            continue
        y_labels = ann[weight_col(factor)].values
        y_rank = Y_all[:, j]
        preds = oof_pred[:, j]
        pred_labels = [WEIGHT_LEVELS[r] for r in preds]

        onehot_proba = np.eye(N_CLASSES)[preds]  # degenerate hard-decision "probability", see module docstring
        acc = accuracy_score(y_rank, preds)
        macro_f1 = f1_score(y_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0)
        ord_err = ordinal_distance(list(y_labels), pred_labels, WEIGHT_RANK)
        dset_f1 = decisive_set_f1(list(y_labels), pred_labels)
        auroc = safe_multiclass_auroc(y_rank, onehot_proba, labels=[0, 1, 2, 3])
        brier = multiclass_brier_score(y_rank, onehot_proba, N_CLASSES)

        per_factor[factor] = {
            "accuracy": round(float(acc), 3),
            "macro_f1": round(float(macro_f1), 3),
            "ordinal_error": round(float(ord_err), 3),
            "decisive_set_f1": round(float(dset_f1), 3),
            "roc_auc": round(float(auroc), 3) if auroc is not None else None,
            "brier_score": round(float(brier), 3),
        }
        print(f"  {factor}: accuracy={acc:.3f} macro_f1={macro_f1:.3f} ordinal_error={ord_err:.3f} "
              f"decisive_set_f1={dset_f1:.3f} roc_auc={auroc} brier={brier:.3f}")

        all_true_pooled.extend(int(v) for v in y_rank)
        all_pred_pooled.extend(int(v) for v in preds)
        for i, cid in enumerate(case_ids):
            per_case_rows[i][f"{factor}_true"] = WEIGHT_LEVELS[int(y_rank[i])]
            per_case_rows[i][f"{factor}_pred"] = WEIGHT_LEVELS[int(preds[i])]
            per_case_rows[i][f"{factor}_abs_error"] = int(abs(int(y_rank[i]) - int(preds[i])))

    for row in per_case_rows:
        abs_errors = [row[f"{f}_abs_error"] for f in included_factors]
        row["mean_ordinal_error"] = round(float(np.mean(abs_errors)), 3) if abs_errors else None
        row["n_exact_matches"] = int(sum(1 for e in abs_errors if e == 0)) if abs_errors else None
    per_case_df = pd.DataFrame(per_case_rows)

    all_true_pooled = np.array(all_true_pooled)
    all_pred_pooled = np.array(all_pred_pooled)
    cm = confusion_matrix(all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3])
    report = classification_report(
        all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS,
        digits=3, zero_division=0,
    )

    print(f"\nPooled across {len(included_factors)}/{len(IN_SCOPE_FACTORS)} factors x {n} cases = {len(all_true_pooled)} predictions")
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in WEIGHT_LEVELS))
    for i, row in enumerate(cm):
        print(f"{WEIGHT_LEVELS[i]:>12}" + "".join(f"{v:>12}" for v in row))
    print("\nClassification report")
    print(report)

    included = {f: v for f, v in per_factor.items() if "skipped" not in v}
    payload = {
        "condition": "weights_kdm_occlusion",
        "target": "variable_weights",
        "protocol": "leave-one-out (91-fold, pooled)",
        "features": "exp_3 19-column frame (PSA-reduced + MRI-PCA)",
        "model": "KDM decision backbone (shared, n_classes=2) + per-factor occlusion-delta signal + isotonic recalibration -- unchanged from exp_6",
        "mean_accuracy": round(float(np.mean([v["accuracy"] for v in included.values()])), 3),
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in included.values()])), 3),
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in included.values()])), 3),
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in included.values()])), 3),
        "mean_roc_auc": round(float(np.mean([v["roc_auc"] for v in included.values() if v["roc_auc"] is not None])), 3),
        "mean_brier_score": round(float(np.mean([v["brier_score"] for v in included.values()])), 3),
        "n_factors_included": len(included_factors),
        "n_factors_skipped": len(IN_SCOPE_FACTORS) - len(included_factors),
        "per_factor": per_factor,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": WEIGHT_LEVELS,
        "classification_report_text": report,
        "classification_report": classification_report(
            all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS,
            digits=3, zero_division=0, output_dict=True,
        ),
        "n_predictions": len(all_true_pooled),
        "auroc_brier_note": "AUROC/Brier are computed from a one-hot HARD-DECISION pseudo-"
                             "probability (1.0 on the predicted class), the only option available "
                             "since isotonic regression has no native calibrated-probability "
                             "output -- not directly comparable in richness to weights_svm's real "
                             "predict_proba or exp_14's regression-variance-based pseudo-probs.",
        "reference": {
            "exp6_original_10x5cv_mean_ordinal_error": 0.405,
            "exp6_original_10x5cv_mean_decisive_set_f1": 0.442,
            "baseline_ordinal_error": 0.413,
            "weights_svm_incumbent_ordinal_error": 0.382,
        },
    }
    out_dir = RESULTS_DIR / "loo_full_metrics_weights_kdm_occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    per_case_df.to_csv(out_dir / "per_case.csv", index=False)

    print(f"\n[weights_kdm_occlusion, LOO] mean_ordinal_error={payload['mean_ordinal_error']} "
          f"(10x5-CV reference 0.405) mean_decisive_set_f1={payload['mean_decisive_set_f1']} "
          f"(10x5-CV reference 0.442) mean_accuracy={payload['mean_accuracy']} "
          f"mean_macro_f1={payload['mean_macro_f1']} mean_roc_auc={payload['mean_roc_auc']} "
          f"mean_brier={payload['mean_brier_score']}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'per_case.csv'} ({len(per_case_df)} rows)")


if __name__ == "__main__":
    main()
