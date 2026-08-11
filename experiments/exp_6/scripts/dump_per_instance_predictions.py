"""exp_6: per-instance PREDICTIONS (not just raw signals) for all three targets, using the two
best-performing recalibrated conditions from run_signals.py's full 10-repeat evaluation:
  - decision:   decision_kdm_backbone (the backbone's own argmax)
  - confidence: confidence_kdm_entropy_isotonic (best of the 5 confidence conditions, 0.731)
  - weights:    weights_kdm_occlusion (the only weights condition beating baseline, 0.405)

Single pass (repeat=0's 5-fold split, RANDOM_STATE=0, same split as dump_per_instance.py) --
NOT averaged over 10 repeats like results/*/metrics.json's official numbers. This gives one
concrete, auditable set of predictions per case rather than a repeat-averaged aggregate; expect
its accuracy numbers to differ slightly (CV noise) from the official repeated-CV metrics.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_6/scripts/dump_per_instance_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone, occlusion_delta  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_exp3_feature_frame
from chimera_task1.reasoning_labels import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_RANK,
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
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
OUT_PATH = Path(__file__).parent.parent / "results" / "per_instance_predictions.csv"


def isotonic_rank(train_signal, train_rank, test_signal, n_levels):
    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(train_signal, train_rank)
    raw = iso.predict(test_signal)
    return np.clip(np.round(raw), 0, n_levels - 1).astype(int)


def main() -> None:
    ann, inp_ann = load_annotated()
    case_ids = ann["case_id"].values
    y_decision_labels = ann["target_biopsy_decision"].values
    y_decision = (y_decision_labels == "yes").astype(int)
    y_confidence_labels = ann["target_confidence"].values
    y_confidence_rank = np.array([CONFIDENCE_RANK[label] for label in y_confidence_labels])
    weight_labels = {f: ann[weight_col(f)].values for f in IN_SCOPE_FACTORS}
    weight_rank = {f: np.array([WEIGHT_RANK[label] for label in weight_labels[f]]) for f in IN_SCOPE_FACTORS}

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp3_feature_frame(inp_ann, mri_pca)

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    factor_col_idx = {f: [X_frame.columns.get_loc(c) for c in restricted_feature_group(f, "flags")] for f in IN_SCOPE_FACTORS}

    n = len(y_decision)
    decision_pred = np.full(n, -1, dtype=int)
    decision_p_yes = np.full(n, np.nan)
    confidence_pred_rank = np.full(n, -1, dtype=int)
    weight_pred_rank = {f: np.full(n, -1, dtype=int) for f in IN_SCOPE_FACTORS}

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kf.split(X_pre):
        scaler = StandardScaler().fit(X_pre[train_idx])
        X_train = scaler.transform(X_pre[train_idx])
        X_test = scaler.transform(X_pre[test_idx])

        model = fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2)
        sig_train = compute_signals(model, X_train)
        sig_test = compute_signals(model, X_test)
        assert sig_train["probs_check_ok"] and sig_test["probs_check_ok"]

        decision_p_yes[test_idx] = sig_test["probs"][:, 1]
        decision_pred[test_idx] = sig_test["probs"].argmax(axis=1)

        confidence_pred_rank[test_idx] = isotonic_rank(
            sig_train["entropy"], y_confidence_rank[train_idx], sig_test["entropy"], n_levels=3
        )

        for factor in IN_SCOPE_FACTORS:
            col_idx = factor_col_idx[factor]
            fill = np.median(X_train[:, col_idx], axis=0)
            D_tr = np.abs(occlusion_delta(model, X_train, col_idx, fill))
            D_te = np.abs(occlusion_delta(model, X_test, col_idx, fill))
            weight_pred_rank[factor][test_idx] = isotonic_rank(D_tr, weight_rank[factor][train_idx], D_te, n_levels=4)

    # --- assemble per-instance table ---
    decision_pred_labels = np.where(decision_pred == 1, "yes", "no")
    confidence_pred_labels = np.array([CONFIDENCE_LEVELS[r] for r in confidence_pred_rank])

    out = pd.DataFrame({
        "case_id": case_ids,
        "decision_true": y_decision_labels,
        "decision_pred": decision_pred_labels,
        "decision_p_yes": np.round(decision_p_yes, 4),
        "decision_correct": decision_pred_labels == y_decision_labels,
        "confidence_true": y_confidence_labels,
        "confidence_pred": confidence_pred_labels,
        "confidence_correct": confidence_pred_labels == y_confidence_labels,
        "confidence_ordinal_error": np.abs(confidence_pred_rank - y_confidence_rank),
    })
    for factor in IN_SCOPE_FACTORS:
        pred_labels = np.array([WEIGHT_LEVELS[r] for r in weight_pred_rank[factor]])
        out[f"weight_true_{factor}"] = weight_labels[factor]
        out[f"weight_pred_{factor}"] = pred_labels
        out[f"weight_correct_{factor}"] = pred_labels == weight_labels[factor]
        out[f"weight_ordinal_error_{factor}"] = np.abs(weight_pred_rank[factor] - weight_rank[factor])

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns to {OUT_PATH}\n")

    # --- accuracy summary ---
    print("=== Decision ===")
    print(f"  exact-match accuracy: {out['decision_correct'].mean():.3f}  ({out['decision_correct'].sum()}/{n})")
    print(f"  macro-F1:             {f1_score(y_decision, decision_pred, average='macro'):.3f}")

    print("\n=== Confidence ===")
    print(f"  exact-match accuracy: {out['confidence_correct'].mean():.3f}  ({out['confidence_correct'].sum()}/{n})")
    print(f"  mean ordinal error:   {out['confidence_ordinal_error'].mean():.3f}")
    print(f"  ordinal_distance():   {ordinal_distance(list(y_confidence_labels), list(confidence_pred_labels), CONFIDENCE_RANK):.3f}")

    print("\n=== Variable weights (per factor) ===")
    print(f"  {'factor':14s} {'accuracy':>9s} {'mean|err|':>10s} {'decisive_f1':>12s}")
    accs, errs, f1s = [], [], []
    for factor in IN_SCOPE_FACTORS:
        acc = out[f"weight_correct_{factor}"].mean()
        err = out[f"weight_ordinal_error_{factor}"].mean()
        pred_labels = out[f"weight_pred_{factor}"].values
        f1v = decisive_set_f1(list(weight_labels[factor]), list(pred_labels))
        accs.append(acc)
        errs.append(err)
        f1s.append(f1v)
        print(f"  {factor:14s} {acc:9.3f} {err:10.3f} {f1v:12.3f}")
    print(f"  {'MEAN':14s} {np.mean(accs):9.3f} {np.mean(errs):10.3f} {np.mean(f1s):12.3f}")


if __name__ == "__main__":
    main()
