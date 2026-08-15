"""exp_13 follow-up: 10 repeated 80/20 held-out splits for weights direct-KDM, RESTRICTED scope
only, both backbones -- the single seed=0 held-out check gave a suspiciously good number for this
scope (scalar: 0.413, exactly tying baseline; ARD: 0.421, close to it) while CV for the same scope
showed both backbones sitting at 0.454, clearly worse than baseline -- the same "suspiciously clean
single split" pattern flagged in exp_10/exp_11/exp_12. Official scope is NOT re-checked here: its
held-out numbers (scalar 0.462, ARD 0.456) already agree with CV's negative verdict, so there is
nothing suspicious to verify.

Per DESIGN.md Section 2d's staged-execution logic, this replaces committing to full LOO for all 4
conditions -- a repeated-holdout check is far cheaper (10 seeds x 7 non-skip factors x 1 fit each
per backbone, ~140 fits total) and directly answers whether the tie is real or a lucky split before
deciding whether restricted scope deserves the expensive LOO treatment.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_13/scripts/repeated_holdout_weights_direct_restricted.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_11" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone  # noqa: E402
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

N_CLASSES = 4
TEST_SIZE = 0.2
N_REPEATS = 10
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"

BACKBONES = {
    "scalar": (lambda X_train, y_train: fit_kdm_backbone(X_train, y_train, n_classes=N_CLASSES), compute_signals),
    "ard": (lambda X_train, y_train: fit_kdm_backbone_ard(X_train, y_train, n_classes=N_CLASSES, **ARD_CONFIG), compute_signals_ard),
}


def eval_seed(backbone_tag, ann, X_full, seed, y_decision) -> dict | None:
    fit_fn, signals_fn = BACKBONES[backbone_tag]
    idx_all = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)

    ord_errs, dset_f1s, macro_f1s = [], [], []
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
        cols = restricted_feature_group(factor, "flags")
        X_use = X_full[cols]
        X_train_raw = X_use.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_use.iloc[test_idx].reset_index(drop=True)
        X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

        try:
            model = fit_fn(X_train, y_rank[train_idx])
        except ValueError:
            continue
        sig = signals_fn(model, X_test)
        if not sig["probs_check_ok"]:
            continue
        probs = sig["probs"]
        preds = probs.argmax(axis=1)
        pred_labels = [WEIGHT_LEVELS[r] for r in preds]
        y_test_rank = y_rank[test_idx]
        y_test_labels = y_labels[test_idx]

        from sklearn.metrics import f1_score
        ord_errs.append(ordinal_distance(list(y_test_labels), pred_labels, WEIGHT_RANK))
        dset_f1s.append(decisive_set_f1(list(y_test_labels), pred_labels))
        macro_f1s.append(f1_score(y_test_rank, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0))

    if not ord_errs:
        return None
    return {
        "mean_ordinal_error": round(float(np.mean(ord_errs)), 3),
        "mean_decisive_set_f1": round(float(np.mean(dset_f1s)), 3),
        "mean_macro_f1": round(float(np.mean(macro_f1s)), 3),
        "n_factors_included": len(ord_errs),
    }


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values
    idx_all = np.arange(len(ann))

    results = {}
    for backbone_tag in ("scalar", "ard"):
        print(f"=== {backbone_tag} (restricted scope) ===")
        per_seed = {"mean_ordinal_error": [], "mean_decisive_set_f1": [], "mean_macro_f1": []}

        for seed in range(N_REPEATS):
            train_idx, _ = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)
            inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
            mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
            X_full = select_exp8_feature_frame(inp_ann, mri_pca_aligned)

            result = eval_seed(backbone_tag, ann, X_full, seed, y_decision)
            if result is None:
                print(f"  seed={seed} [SKIP] all factors degenerate")
                continue
            per_seed["mean_ordinal_error"].append(result["mean_ordinal_error"])
            per_seed["mean_decisive_set_f1"].append(result["mean_decisive_set_f1"])
            per_seed["mean_macro_f1"].append(result["mean_macro_f1"])
            print(f"  seed={seed} ordinal_error={result['mean_ordinal_error']} "
                  f"decisive_set_f1={result['mean_decisive_set_f1']} macro_f1={result['mean_macro_f1']} "
                  f"({result['n_factors_included']}/9 factors)")

        results[backbone_tag] = {
            "ordinal_error_mean": round(float(np.mean(per_seed["mean_ordinal_error"])), 3),
            "ordinal_error_std": round(float(np.std(per_seed["mean_ordinal_error"])), 3),
            "decisive_set_f1_mean": round(float(np.mean(per_seed["mean_decisive_set_f1"])), 3),
            "decisive_set_f1_std": round(float(np.std(per_seed["mean_decisive_set_f1"])), 3),
            "macro_f1_mean": round(float(np.mean(per_seed["mean_macro_f1"])), 3),
            "macro_f1_std": round(float(np.std(per_seed["mean_macro_f1"])), 3),
            "per_seed": per_seed,
            "seed0_ordinal_error": per_seed["mean_ordinal_error"][0],
        }
        print(f"  10-seed mean: ordinal_error={results[backbone_tag]['ordinal_error_mean']}"
              f" +/- {results[backbone_tag]['ordinal_error_std']}\n")

    out_dir = RESULTS_DIR / "repeated_holdout_weights_direct_restricted"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== SUMMARY ===")
    print(f"CV restricted-scope reference: scalar=0.454, ard=0.454 (both worse than baseline 0.413)")
    for backbone_tag, r in results.items():
        print(f"{backbone_tag}: seed=0 ordinal_error={r['seed0_ordinal_error']} "
              f"vs. 10-seed mean {r['ordinal_error_mean']} +/- {r['ordinal_error_std']}")


if __name__ == "__main__":
    main()
