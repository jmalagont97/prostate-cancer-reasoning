"""exp_8: isolate whether the expanded 23-column frame or the tuned hyperparameters is doing the
work (DESIGN.md Section 2d's three-way ablation), mirroring exp_7's run_ablations.py pattern.

  decision_kdm_features_only:  23-column frame (exp_8) + exp_6's ORIGINAL fixed hyperparameters
  decision_kdm_tuned_only:     19-column frame (exp_3, original) + winning hyperparameters

The third leg of the ablation -- both combined -- is decision_kdm_v3, produced by
run_signals_v3.py's own unified loop, not duplicated here.

Requires results/hyperparameter_search/winner.json to exist (run search_hyperparameters_v3.py first).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_8/scripts/run_ablations_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_7" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from features_v3 import select_exp8_feature_frame  # noqa: E402
from kdm_backbone import compute_signals  # noqa: E402
from kdm_backbone_v2 import fit_kdm_backbone  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEARCH_DIR = RESULTS_DIR / "hyperparameter_search"

EXP6_DEFAULTS = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0, "optimizer": "adam", "weight_decay": 0.0}


def evaluate(X_pre: np.ndarray, y: np.ndarray, **fit_kwargs) -> tuple[float, float]:
    f1_scores = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])
            model = fit_kdm_backbone(X_train, y[train_idx], n_classes=2, **fit_kwargs)
            sig = compute_signals(model, X_test)
            preds[test_idx] = sig["probs"].argmax(axis=1)
        f1_scores.append(f1_score(y, preds, average="macro"))
    return float(np.mean(f1_scores)), float(np.std(f1_scores))


def write_metrics(condition: str, features_desc: str, fit_kwargs: dict, mean_f1: float, std_f1: float) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition": condition,
        "target": "decision",
        "features": features_desc,
        "model": "KDM (memory-based, sigma-only trained), exp_8 ablation",
        "fit_kwargs": fit_kwargs,
        "macro_f1_mean": round(mean_f1, 3),
        "macro_f1_std": round(std_f1, 3),
        "n_folds": N_REPEATS * N_SPLITS,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] macro_f1={mean_f1:.3f} +/- {std_f1:.3f}")


def main() -> None:
    with open(SEARCH_DIR / "winner.json") as f:
        winner = json.load(f)
    winning_config = {
        "n_epochs": winner["n_epochs"], "lr": winner["lr"], "sigma_mult": winner["sigma_mult"],
        "optimizer": winner["optimizer"], "weight_decay": winner["weight_decay"],
    }
    print(f"Winning config from search: {winning_config}\n")

    ann, inp_ann = load_annotated()
    y = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)

    X_frame_8 = select_exp8_feature_frame(inp_ann, mri_pca)
    X_pre_8 = build_preprocessor(X_frame_8).fit_transform(X_frame_8)
    X_pre_8 = X_pre_8.toarray() if hasattr(X_pre_8, "toarray") else X_pre_8

    X_frame_3 = select_exp3_feature_frame(inp_ann, mri_pca)
    X_pre_3 = build_preprocessor(X_frame_3).fit_transform(X_frame_3)
    X_pre_3 = X_pre_3.toarray() if hasattr(X_pre_3, "toarray") else X_pre_3

    print("=== decision_kdm_features_only (23-col exp_8 frame + exp_6 original hyperparameters) ===")
    mean_f1, std_f1 = evaluate(X_pre_8, y, **EXP6_DEFAULTS)
    write_metrics("decision_kdm_features_only", "exp_8 23-column frame", EXP6_DEFAULTS, mean_f1, std_f1)

    print("\n=== decision_kdm_tuned_only (19-col exp_3 original frame + winning hyperparameters) ===")
    mean_f1, std_f1 = evaluate(X_pre_3, y, **winning_config)
    write_metrics("decision_kdm_tuned_only", "exp_3 19-column frame (original)", winning_config, mean_f1, std_f1)


if __name__ == "__main__":
    main()
