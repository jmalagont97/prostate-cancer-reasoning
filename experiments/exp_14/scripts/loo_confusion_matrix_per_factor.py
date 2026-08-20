"""exp_14: pooled confusion matrix + classification report for the per-factor KDM regression
condition, scored via leave-one-out (91 folds x 9 factors = 819 fits), pooling every (case,
factor) prediction across all 9 in-scope factors into one confusion matrix over the 4 ordinal
weight levels. LOO wasn't part of the original staged plan (neither condition cleared the
ordinal-error bar), but is run here on explicit request to see the full diagnostic picture
regardless of the mixed verdict.

Also saves a per-case breakdown (one row per case_id: each factor's true/predicted rank, plus a
per-case mean ordinal error and exact-match count out of 9) -- the pooled confusion matrix answers
"how does this model do per factor overall", the per-case table answers "how does it do on this
specific patient's record", a different and complementary cut of the same LOO predictions.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_14/scripts/loo_confusion_matrix_per_factor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_regress_backbone import compute_signals_regress, fit_kdm_regress  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import mri_pca_train_only  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import TASK1_FACTORS, WEIGHT_LEVELS, WEIGHT_RANK, weight_col
from chimera_task1.train_reasoning import load_annotated

IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    ann, inp_ann = load_annotated()
    n = len(ann)
    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)  # (91, 9)

    all_true = []
    all_pred = []
    per_case_rows = []
    idx_all = np.arange(n)
    case_ids = ann["case_id"].values

    for fold_i, (train_idx, test_idx) in enumerate(LeaveOneOut().split(idx_all)):
        inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
        mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
        X_all = select_exp8_feature_frame(inp_ann, mri_pca_aligned)

        X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
        preprocessor = build_preprocessor(X_train_raw)
        X_train_pre = preprocessor.fit_transform(X_train_raw)
        X_test_pre = preprocessor.transform(X_test_raw)
        X_train_pre = X_train_pre.toarray() if hasattr(X_train_pre, "toarray") else X_train_pre
        X_test_pre = X_test_pre.toarray() if hasattr(X_test_pre, "toarray") else X_test_pre
        scaler = StandardScaler().fit(X_train_pre)
        X_train = scaler.transform(X_train_pre)
        X_test = scaler.transform(X_test_pre)

        case_row = {"case_id": case_ids[test_idx[0]]}
        for j, factor in enumerate(IN_SCOPE_FACTORS):
            model = fit_kdm_regress(X_train, Y_all[train_idx, j].reshape(-1, 1), dim_y=1)
            sig = compute_signals_regress(model, X_test)
            true_r = int(Y_all[test_idx[0], j])
            pred_r = int(sig["pred_rank"][0, 0])
            all_true.append(true_r)
            all_pred.append(pred_r)
            case_row[f"{factor}_true"] = WEIGHT_LEVELS[true_r]
            case_row[f"{factor}_pred"] = WEIGHT_LEVELS[pred_r]
            case_row[f"{factor}_abs_error"] = abs(true_r - pred_r)
        abs_errors = [case_row[f"{f}_abs_error"] for f in IN_SCOPE_FACTORS]
        case_row["mean_ordinal_error"] = round(float(np.mean(abs_errors)), 3)
        case_row["n_exact_matches"] = int(sum(1 for e in abs_errors if e == 0))
        per_case_rows.append(case_row)

        if (fold_i + 1) % 15 == 0:
            print(f"  LOO fold {fold_i + 1}/{n} done ({(fold_i + 1) * len(IN_SCOPE_FACTORS)} predictions so far)")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    per_case_df = pd.DataFrame(per_case_rows)

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    report = classification_report(
        all_true, all_pred, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS, digits=3, zero_division=0
    )

    print(f"\nPooled across all {len(IN_SCOPE_FACTORS)} factors x {n} cases = {len(all_true)} predictions")
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in WEIGHT_LEVELS))
    for i, row in enumerate(cm):
        print(f"{WEIGHT_LEVELS[i]:>12}" + "".join(f"{v:>12}" for v in row))

    print("\nClassification report")
    print(report)

    out = {
        "condition": "weights_kdm_regress_per_factor",
        "protocol": "leave-one-out (91-fold x 9 factors, pooled)",
        "labels": WEIGHT_LEVELS,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report,
        "classification_report": classification_report(
            all_true, all_pred, labels=[0, 1, 2, 3], target_names=WEIGHT_LEVELS, digits=3,
            zero_division=0, output_dict=True,
        ),
        "n_predictions": len(all_true),
    }
    out_dir = RESULTS_DIR / "loo_confusion_matrix_weights_regress_per_factor"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    per_case_df.to_csv(out_dir / "per_case.csv", index=False)
    print(f"\nSaved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'per_case.csv'} ({len(per_case_df)} rows)")

    print("\nPer-case summary (mean ordinal error across the 9 factors, exact-match count out of 9):")
    summary_cols = ["case_id", "mean_ordinal_error", "n_exact_matches"]
    print(per_case_df[summary_cols].sort_values("mean_ordinal_error").to_string(index=False))
    print(f"\nPer-case mean_ordinal_error: mean={per_case_df['mean_ordinal_error'].mean():.3f} "
          f"std={per_case_df['mean_ordinal_error'].std():.3f} "
          f"min={per_case_df['mean_ordinal_error'].min():.3f} max={per_case_df['mean_ordinal_error'].max():.3f}")
    print(f"Per-case n_exact_matches (out of 9): mean={per_case_df['n_exact_matches'].mean():.2f} "
          f"min={per_case_df['n_exact_matches'].min()} max={per_case_df['n_exact_matches'].max()}")


if __name__ == "__main__":
    main()
