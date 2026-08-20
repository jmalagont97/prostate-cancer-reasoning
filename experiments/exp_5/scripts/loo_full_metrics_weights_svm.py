"""Backfill: full metric suite (accuracy, macro-F1, AUROC, Brier, on top of exp_5's original
ordinal_error/decisive_set_f1) + leave-one-out evaluation + pooled confusion matrix/classification
report + per-case breakdown, for weights_svm (restricted scope) -- this project's best-ever
variable-weights model, still unmatched as of exp_14.

exp_5 predates both the full-metric-suite convention (exp_11+) and the mandatory-LOO convention
(exp_11+), so this model has only ever been reported with 5-repeat x 5-fold CV ordinal_error/
decisive_set_f1 (0.382/0.457). This backfills the same diagnostic depth exp_12 (confidence) and
exp_14 (weights, KDM regression) already have, using the EXACT original setup from
experiments/exp_5/scripts/run_weights.py + experiments/exp_3/scripts/{models.py,cv_utils.py}
(19-column frame, restricted_feature_group per factor, SVC(kernel="rbf", C=1.0,
class_weight="balanced", probability=True), build_preprocessor + StandardScaler fit per fold) --
only the evaluation protocol changes (LOO instead of 5x5 CV), the model itself is untouched.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_5/scripts/loo_full_metrics_weights_svm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
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


def build_clf() -> SVC:
    # Identical to experiments/exp_3/scripts/models.py's "svm" entry, reused unchanged.
    return SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE)


def main() -> None:
    ann, inp_ann = load_annotated()
    n = len(ann)
    case_ids = ann["case_id"].values

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_full = select_exp3_feature_frame(inp_ann, mri_pca)
    print(f"n={n}, 19-col frame: {X_full.shape}, in-scope factors: {IN_SCOPE_FACTORS}\n")

    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)  # (91, 9)

    per_factor = {}
    all_true_pooled, all_pred_pooled = [], []
    per_case_rows = [{"case_id": cid} for cid in case_ids]

    for j, factor in enumerate(IN_SCOPE_FACTORS):
        y_labels = ann[weight_col(factor)].values
        y_rank = Y_all[:, j]
        cols = restricted_feature_group(factor, "flags")
        X = X_full[cols]
        global_classes = np.unique(y_rank)
        class_to_col = {c: i for i, c in enumerate(global_classes)}

        oof_proba_local = np.zeros((n, len(global_classes)))
        oof_pred = np.zeros(n, dtype=int)

        for train_idx, test_idx in LeaveOneOut().split(np.arange(n)):
            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            preprocessor = build_preprocessor(X)
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
            X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            clf = build_clf()
            clf.fit(X_train, y_rank[train_idx])
            fold_proba = clf.predict_proba(X_test)  # (1, n_local_classes)
            full_proba = np.zeros(len(global_classes))
            for c_i, c in enumerate(clf.classes_):
                full_proba[class_to_col[c]] = fold_proba[0, c_i]
            oof_proba_local[test_idx[0]] = full_proba
            oof_pred[test_idx[0]] = global_classes[full_proba.argmax()]

        # Expand local (present-classes-only) proba into the full 4-class [0,1,2,3] space, same
        # class_to_col remap discipline as cv_utils.repeated_cv_proba, for AUROC/Brier/pooling.
        oof_proba_full = np.zeros((n, N_CLASSES))
        for c, col in class_to_col.items():
            oof_proba_full[:, c] = oof_proba_local[:, col]

        pred_labels = [WEIGHT_LEVELS[r] for r in oof_pred]
        acc = accuracy_score(y_rank, oof_pred)
        macro_f1 = f1_score(y_rank, oof_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)
        ord_err = ordinal_distance(list(y_labels), pred_labels, WEIGHT_RANK)
        dset_f1 = decisive_set_f1(list(y_labels), pred_labels)
        auroc = safe_multiclass_auroc(y_rank, oof_proba_full, labels=[0, 1, 2, 3])
        brier = multiclass_brier_score(y_rank, oof_proba_full, N_CLASSES)

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
        all_pred_pooled.extend(int(v) for v in oof_pred)
        for i, cid in enumerate(case_ids):
            per_case_rows[i][f"{factor}_true"] = WEIGHT_LEVELS[int(y_rank[i])]
            per_case_rows[i][f"{factor}_pred"] = WEIGHT_LEVELS[int(oof_pred[i])]
            per_case_rows[i][f"{factor}_abs_error"] = int(abs(int(y_rank[i]) - int(oof_pred[i])))

    for row in per_case_rows:
        abs_errors = [row[f"{f}_abs_error"] for f in IN_SCOPE_FACTORS]
        row["mean_ordinal_error"] = round(float(np.mean(abs_errors)), 3)
        row["n_exact_matches"] = int(sum(1 for e in abs_errors if e == 0))
    per_case_df = pd.DataFrame(per_case_rows)

    all_true_pooled = np.array(all_true_pooled)
    all_pred_pooled = np.array(all_pred_pooled)
    cm = confusion_matrix(all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3])
    report = classification_report(
        all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS,
        digits=3, zero_division=0,
    )

    print(f"\nPooled across all {len(IN_SCOPE_FACTORS)} factors x {n} cases = {len(all_true_pooled)} predictions")
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in WEIGHT_LEVELS))
    for i, row in enumerate(cm):
        print(f"{WEIGHT_LEVELS[i]:>12}" + "".join(f"{v:>12}" for v in row))
    print("\nClassification report")
    print(report)

    included = per_factor
    payload = {
        "condition": "weights_svm_restricted",
        "target": "variable_weights",
        "protocol": "leave-one-out (91-fold, pooled)",
        "features": "exp_3 19-column frame, restricted scope (restricted_feature_group per factor)",
        "model": "SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True) -- unchanged from exp_5",
        "mean_accuracy": round(float(np.mean([v["accuracy"] for v in included.values()])), 3),
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in included.values()])), 3),
        "mean_ordinal_error": round(float(np.mean([v["ordinal_error"] for v in included.values()])), 3),
        "mean_decisive_set_f1": round(float(np.mean([v["decisive_set_f1"] for v in included.values()])), 3),
        "mean_roc_auc": round(float(np.mean([v["roc_auc"] for v in included.values() if v["roc_auc"] is not None])), 3),
        "mean_brier_score": round(float(np.mean([v["brier_score"] for v in included.values()])), 3),
        "per_factor": per_factor,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": WEIGHT_LEVELS,
        "classification_report_text": report,
        "classification_report": classification_report(
            all_true_pooled, all_pred_pooled, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS,
            digits=3, zero_division=0, output_dict=True,
        ),
        "n_predictions": len(all_true_pooled),
        "reference": {
            "exp5_original_5x5cv_mean_ordinal_error": 0.382,
            "exp5_original_5x5cv_mean_decisive_set_f1": 0.457,
            "baseline_ordinal_error": 0.413,
        },
    }
    out_dir = RESULTS_DIR / "loo_full_metrics_weights_svm_restricted"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    per_case_df.to_csv(out_dir / "per_case.csv", index=False)

    print(f"\n[weights_svm_restricted, LOO] mean_ordinal_error={payload['mean_ordinal_error']} "
          f"(5x5-CV reference 0.382) mean_decisive_set_f1={payload['mean_decisive_set_f1']} "
          f"(5x5-CV reference 0.457) mean_accuracy={payload['mean_accuracy']} "
          f"mean_macro_f1={payload['mean_macro_f1']} mean_roc_auc={payload['mean_roc_auc']} "
          f"mean_brier={payload['mean_brier_score']}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'per_case.csv'} ({len(per_case_df)} rows)")


if __name__ == "__main__":
    main()
