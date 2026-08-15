"""exp_11 follow-up: 10 repeated 80/20 held-out splits for confidence, both frames -- the single
seed=0 held-out check gave an identical 0.316 ordinal distance for BOTH frames, a striking
coincidence that matches exp_10's already-documented lucky-split pattern (see experiments/INDEX.md
and project memory: "any future single-split held-out result this striking should get the same
LOO/repeated-holdout treatment before being reported as a finding"). CV and LOO already agree with
each other (~0.51-0.55) and disagree with the single held-out split (0.316) -- this resolves which
one was right.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_11/scripts/repeated_holdout_confidence_direct_ard.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from holdout_eval import fit_transform_features, mri_pca_train_only  # noqa: E402
from metrics_multiclass import multiclass_brier_score, safe_multiclass_auroc  # noqa: E402

from chimera_task1.features import select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_reasoning import load_annotated

N_CLASSES = 3
TEST_SIZE = 0.2
N_REPEATS = 10
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
RESULTS_DIR = Path(__file__).parent.parent / "results"

FRAMES = {
    "19col": select_exp3_feature_frame,
    "23col": select_exp8_feature_frame,
}


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])
    idx_all = np.arange(len(ann))

    results = {}
    for frame_tag, frame_fn in FRAMES.items():
        print(f"=== {frame_tag} ===")
        per_seed = {"accuracy": [], "macro_f1": [], "ordinal_distance": [], "roc_auc": [], "brier_score": []}

        for seed in range(N_REPEATS):
            train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_decision, random_state=seed)
            inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
            mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
            X_all = frame_fn(inp_ann, mri_pca_aligned)
            X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
            X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
            X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)

            model = fit_kdm_backbone_ard(X_train, y_conf_rank[train_idx], n_classes=N_CLASSES, **ARD_CONFIG)
            sig = compute_signals_ard(model, X_test)
            probs = sig["probs"]
            preds = probs.argmax(axis=1)
            pred_labels = [CONFIDENCE_LEVELS[r] for r in preds]
            y_test_rank = y_conf_rank[test_idx]
            y_test_labels = y_conf_labels[test_idx]

            acc = accuracy_score(y_test_rank, preds)
            macro_f1 = f1_score(y_test_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0)
            ord_dist = ordinal_distance(list(y_test_labels), pred_labels, CONFIDENCE_RANK)
            auroc = safe_multiclass_auroc(y_test_rank, probs, labels=[0, 1, 2])
            brier = multiclass_brier_score(y_test_rank, probs, N_CLASSES)

            per_seed["accuracy"].append(round(float(acc), 3))
            per_seed["macro_f1"].append(round(float(macro_f1), 3))
            per_seed["ordinal_distance"].append(round(float(ord_dist), 3))
            if auroc is not None:
                per_seed["roc_auc"].append(round(float(auroc), 3))
            per_seed["brier_score"].append(round(float(brier), 3))
            print(f"  seed={seed} acc={acc:.3f} macro_f1={macro_f1:.3f} ord_dist={ord_dist:.3f}")

        results[frame_tag] = {
            "accuracy_mean": round(float(np.mean(per_seed["accuracy"])), 3),
            "accuracy_std": round(float(np.std(per_seed["accuracy"])), 3),
            "macro_f1_mean": round(float(np.mean(per_seed["macro_f1"])), 3),
            "macro_f1_std": round(float(np.std(per_seed["macro_f1"])), 3),
            "ordinal_distance_mean": round(float(np.mean(per_seed["ordinal_distance"])), 3),
            "ordinal_distance_std": round(float(np.std(per_seed["ordinal_distance"])), 3),
            "roc_auc_mean": round(float(np.mean(per_seed["roc_auc"])), 3) if per_seed["roc_auc"] else None,
            "brier_score_mean": round(float(np.mean(per_seed["brier_score"])), 3),
            "brier_score_std": round(float(np.std(per_seed["brier_score"])), 3),
            "per_seed": per_seed,
            "seed0_ordinal_distance": per_seed["ordinal_distance"][0],
        }
        print(f"  10-seed mean: ordinal_distance={results[frame_tag]['ordinal_distance_mean']}"
              f" +/- {results[frame_tag]['ordinal_distance_std']}"
              f"  macro_f1={results[frame_tag]['macro_f1_mean']} +/- {results[frame_tag]['macro_f1_std']}\n")

    out_dir = RESULTS_DIR / "repeated_holdout_confidence_direct_ard"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== SUMMARY ===")
    for frame_tag, r in results.items():
        print(f"{frame_tag}: seed=0 ord_dist={r['seed0_ordinal_distance']} (reported earlier: 0.316) "
              f"vs. 10-seed mean {r['ordinal_distance_mean']} +/- {r['ordinal_distance_std']}")


if __name__ == "__main__":
    main()
