"""exp_14 follow-up: 10 repeated 80/20 held-out splits for both regression conditions -- the
single seed=0 held-out check showed decisive-set F1 EVEN HIGHER than CV's already-striking
0.506-0.507 (0.558 per-factor, 0.544 joint), the "suspiciously good single split" pattern this
project's standing rule requires checking before trusting. Ordinal error, by contrast, agreed
closely between CV and held-out already, so it isn't re-checked here for the same reason exp_13
didn't re-check official-scope's held-out numbers that already matched CV.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_14/scripts/repeated_holdout_weights_regress.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_regress_backbone import compute_signals_regress, fit_kdm_regress  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402

from chimera_task1.reasoning_labels import (
    TASK1_FACTORS,
    WEIGHT_LEVELS,
    WEIGHT_RANK,
    decisive_set_f1,
    ordinal_distance,
    weight_col,
)
from chimera_task1.train_reasoning import load_annotated

TEST_SIZE = 0.2
N_REPEATS = 10
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
DIM_Y = len(IN_SCOPE_FACTORS)
RESULTS_DIR = Path(__file__).parent.parent / "results"


def eval_seed_per_factor(ann, inp_ann, seed, y_decision) -> tuple[float, float]:
    idx_all = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)
    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_full.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_full.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

    ord_errs, dset_f1s = [], []
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
        model = fit_kdm_regress(X_train, y_rank[train_idx].reshape(-1, 1), dim_y=1)
        sig = compute_signals_regress(model, X_test)
        preds = sig["pred_rank"].squeeze(axis=1)
        pred_labels = [WEIGHT_LEVELS[r] for r in preds]
        y_test_labels = y_labels[test_idx]
        ord_errs.append(ordinal_distance(list(y_test_labels), pred_labels, WEIGHT_RANK))
        dset_f1s.append(decisive_set_f1(list(y_test_labels), pred_labels))
    return float(np.mean(ord_errs)), float(np.mean(dset_f1s))


def eval_seed_joint(ann, inp_ann, seed, y_decision, Y_all) -> tuple[float, float]:
    idx_all = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)
    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_full = select_exp8_feature_frame(inp_ann, mri_pca_aligned)
    X_train_raw = X_full.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_full.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

    model = fit_kdm_regress(X_train, Y_all[train_idx], dim_y=DIM_Y)
    sig = compute_signals_regress(model, X_test)
    preds_all = sig["pred_rank"]

    ord_errs, dset_f1s = [], []
    for j, factor in enumerate(IN_SCOPE_FACTORS):
        y_labels = ann[weight_col(factor)].values
        pred_labels = [WEIGHT_LEVELS[r] for r in preds_all[:, j]]
        y_test_labels = y_labels[test_idx]
        ord_errs.append(ordinal_distance(list(y_test_labels), pred_labels, WEIGHT_RANK))
        dset_f1s.append(decisive_set_f1(list(y_test_labels), pred_labels))
    return float(np.mean(ord_errs)), float(np.mean(dset_f1s))


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values
    Y_all = np.stack([
        np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS
    ], axis=1)

    results = {}
    for tag, eval_fn in [("per_factor", None), ("joint", None)]:
        print(f"=== {tag} ===")
        ord_errs, dset_f1s = [], []
        for seed in range(N_REPEATS):
            if tag == "per_factor":
                oe, df1 = eval_seed_per_factor(ann, inp_ann, seed, y_decision)
            else:
                oe, df1 = eval_seed_joint(ann, inp_ann, seed, y_decision, Y_all)
            ord_errs.append(round(oe, 3))
            dset_f1s.append(round(df1, 3))
            print(f"  seed={seed} ordinal_error={oe:.3f} decisive_set_f1={df1:.3f}")

        results[tag] = {
            "ordinal_error_mean": round(float(np.mean(ord_errs)), 3),
            "ordinal_error_std": round(float(np.std(ord_errs)), 3),
            "decisive_set_f1_mean": round(float(np.mean(dset_f1s)), 3),
            "decisive_set_f1_std": round(float(np.std(dset_f1s)), 3),
            "per_seed_ordinal_error": ord_errs,
            "per_seed_decisive_set_f1": dset_f1s,
            "seed0_decisive_set_f1": dset_f1s[0],
        }
        print(f"  10-seed mean: ordinal_error={results[tag]['ordinal_error_mean']}"
              f" +/- {results[tag]['ordinal_error_std']}, decisive_set_f1="
              f"{results[tag]['decisive_set_f1_mean']} +/- {results[tag]['decisive_set_f1_std']}\n")

    out_dir = RESULTS_DIR / "repeated_holdout_weights_regress"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== SUMMARY ===")
    print("CV reference: per_factor decisive_set_f1=0.507, joint=0.506 (both ordinal_error ~0.46-0.47)")
    for tag, r in results.items():
        print(f"{tag}: seed=0 decisive_set_f1={r['seed0_decisive_set_f1']} vs. 10-seed mean "
              f"{r['decisive_set_f1_mean']} +/- {r['decisive_set_f1_std']}")


if __name__ == "__main__":
    main()
