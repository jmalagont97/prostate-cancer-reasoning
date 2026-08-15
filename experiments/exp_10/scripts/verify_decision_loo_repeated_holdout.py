"""exp_10 follow-up verification: leave-one-out (LOO) CV + 10 repeated 80/20 held-out splits for
the DECISION target only, on both exp_9's 23-column ARD reference frame and exp_10's 48-column
full-schema frame -- targeted at the sharp CV/held-out disagreement exp_10's report flagged
(CV macro-F1=0.548 vs. held-out macro-F1=0.708 on the full-schema frame).

Two deliberately different verification tools, not redundant:
  - LOO: deterministic, trains each fold on 90/91 cases (vs. 5-fold CV's 72/91) -- removes the
    single-held-out-split's sampling arbitrariness entirely (every case is tested exactly once,
    pooled into one macro-F1), at the cost of the classical LOO caveat that its fold outcomes are
    highly correlated with each other (each pair of folds shares 89/90 training points), so it is
    NOT simply "more trustworthy than 5-fold CV" -- it is a different bias/variance tradeoff, run
    here as a third, independent lens, not a tiebreaker.
  - Repeated held-out (10 splits, seeds 0-9): directly answers "was the single already-reported
    held-out split (seed=0) lucky" without LOO's correlated-fold variance property. Seed 0 exactly
    reproduces the numbers already in experiments/exp_9/reports/summary.md and
    experiments/exp_10/reports/summary.md -- included deliberately as a same-script sanity check
    that this new code path is computing the same thing, before trusting seeds 1-9.

Both use the stricter "held-out discipline" MRI-PCA convention (fit PCA on the fold's train rows
only, per experiments/exp_3/scripts/holdout_eval.py's mri_pca_train_only) rather than the CV
scripts' single global-pool PCA fit -- consistent with treating this whole exercise as an
extension of the held-out family of checks, not the CV family.

LOO's macro-F1 is computed by POOLING all 91 out-of-fold predictions into one confusion matrix and
scoring once -- NOT by averaging a per-fold F1 (a single test point has no meaningful F1 alone).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_fullschema import fit_transform_fullschema, select_exp10_feature_frame  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402

from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
N_REPEATED_HOLDOUT = 10
TEST_SIZE = 0.2
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Already-established reference numbers (seed=0 held-out, already-reported CV) -- not recomputed,
# cited for comparison exactly as every prior report in this project has done.
REFERENCE = {
    "23col_exp9": {"cv_macro_f1": 0.608, "held_out_seed0_macro_f1": 0.680},
    "48col_exp10_fullschema": {"cv_macro_f1": 0.548, "held_out_seed0_macro_f1": 0.708},
}

FRAMES = {
    "23col_exp9": (select_exp8_feature_frame, fit_transform_features),
    "48col_exp10_fullschema": (select_exp10_feature_frame, fit_transform_fullschema),
}


def run_loo(inp_ann: pd.DataFrame, y_decision: np.ndarray, frame_fn, transform_fn) -> dict:
    n = len(y_decision)
    oof_probs = np.zeros((n, 2))
    idx_all = np.arange(n)

    for train_idx, test_idx in LeaveOneOut().split(idx_all):
        inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
        mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
        X_all = frame_fn(inp_ann, mri_pca_aligned)
        X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
        X_train, X_test = transform_fn(X_train_raw, X_test_raw)

        model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
        sig = compute_signals_ard(model, X_test)
        oof_probs[test_idx] = sig["probs"]

        if (test_idx[0] + 1) % 10 == 0:
            print(f"    LOO fold {test_idx[0] + 1}/{n} done")

    preds = oof_probs.argmax(axis=1)
    return {
        "macro_f1": round(float(f1_score(y_decision, preds, average="macro")), 3),
        "f1": round(float(f1_score(y_decision, preds)), 3),
        "roc_auc": round(float(roc_auc_score(y_decision, oof_probs[:, 1])), 3),
        "brier_score": round(float(brier_score_loss(y_decision, oof_probs[:, 1])), 3),
        "n_folds": n,
    }


def run_repeated_holdout(inp_ann: pd.DataFrame, y_decision: np.ndarray, frame_fn, transform_fn, n_repeats: int) -> dict:
    per_seed_macro_f1, per_seed_auroc, per_seed_brier = [], [], []
    idx_all = np.arange(len(y_decision))
    for seed in range(n_repeats):
        train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)
        inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
        mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
        X_all = frame_fn(inp_ann, mri_pca_aligned)
        X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
        X_train, X_test = transform_fn(X_train_raw, X_test_raw)

        model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
        sig = compute_signals_ard(model, X_test)
        y_test = y_decision[test_idx]
        p = sig["probs"][:, 1]
        preds = sig["probs"].argmax(axis=1)
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        # AUROC is undefined for a single-class test fold -- can't happen here since
        # train_test_split(stratify=y_decision) guarantees both classes appear in every split at
        # this positive rate (61.5%) and TEST_SIZE (0.2, n_test=19), but guarded defensively.
        auroc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan")
        brier = float(brier_score_loss(y_test, p))
        per_seed_macro_f1.append(round(macro_f1, 3))
        per_seed_auroc.append(round(auroc, 3))
        per_seed_brier.append(round(brier, 3))
        print(f"    seed={seed} macro_f1={macro_f1:.3f} auroc={auroc:.3f} brier={brier:.3f}")

    return {
        "macro_f1_mean": round(float(np.mean(per_seed_macro_f1)), 3),
        "macro_f1_std": round(float(np.std(per_seed_macro_f1)), 3),
        "roc_auc_mean": round(float(np.nanmean(per_seed_auroc)), 3),
        "roc_auc_std": round(float(np.nanstd(per_seed_auroc)), 3),
        "brier_score_mean": round(float(np.mean(per_seed_brier)), 3),
        "brier_score_std": round(float(np.std(per_seed_brier)), 3),
        "per_seed_macro_f1": per_seed_macro_f1,
        "per_seed_roc_auc": per_seed_auroc,
        "per_seed_brier_score": per_seed_brier,
        "seed0_matches_reference": per_seed_macro_f1[0],
    }


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    results = {}
    for frame_name, (frame_fn, transform_fn) in FRAMES.items():
        print(f"\n{'=' * 70}\n{frame_name}\n{'=' * 70}")

        print("--- Leave-one-out (91 folds) ---")
        loo_result = run_loo(inp_ann, y_decision, frame_fn, transform_fn)
        print(f"  LOO macro_f1={loo_result['macro_f1']}  f1={loo_result['f1']}  "
              f"roc_auc={loo_result['roc_auc']}  brier={loo_result['brier_score']}")

        print("--- Repeated held-out (10 seeds, seed 0 = already-reported split) ---")
        holdout_result = run_repeated_holdout(inp_ann, y_decision, frame_fn, transform_fn, N_REPEATED_HOLDOUT)
        print(f"  repeated held-out macro_f1={holdout_result['macro_f1_mean']} +/- {holdout_result['macro_f1_std']}  "
              f"roc_auc={holdout_result['roc_auc_mean']} +/- {holdout_result['roc_auc_std']}  "
              f"brier={holdout_result['brier_score_mean']} +/- {holdout_result['brier_score_std']}")
        ref_seed0 = REFERENCE[frame_name]["held_out_seed0_macro_f1"]
        seed0_ok = abs(holdout_result["seed0_matches_reference"] - ref_seed0) < 0.005
        print(f"  seed=0 sanity check: {holdout_result['seed0_matches_reference']} vs. reference {ref_seed0} -- {'MATCH' if seed0_ok else 'MISMATCH -- INVESTIGATE'}")

        results[frame_name] = {
            "reference": REFERENCE[frame_name],
            "loo": loo_result,
            "repeated_holdout": holdout_result,
            "seed0_sanity_check_ok": seed0_ok,
        }

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for frame_name, r in results.items():
        print(f"{frame_name}:")
        print(f"  CV (already reported)        = macro_f1 {r['reference']['cv_macro_f1']}")
        print(f"  held-out, seed=0 (reported)  = macro_f1 {r['reference']['held_out_seed0_macro_f1']}")
        print(f"  held-out, 10-seed mean+/-std = macro_f1 {r['repeated_holdout']['macro_f1_mean']} +/- {r['repeated_holdout']['macro_f1_std']}"
              f"  |  roc_auc {r['repeated_holdout']['roc_auc_mean']} +/- {r['repeated_holdout']['roc_auc_std']}"
              f"  |  brier {r['repeated_holdout']['brier_score_mean']} +/- {r['repeated_holdout']['brier_score_std']}")
        print(f"  LOO (91-fold, pooled)        = macro_f1 {r['loo']['macro_f1']}"
              f"  |  roc_auc {r['loo']['roc_auc']}  |  brier {r['loo']['brier_score']}")

    out_dir = RESULTS_DIR / "verify_decision_loo_repeated_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
