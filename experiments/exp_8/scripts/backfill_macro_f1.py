"""Backfill macro-F1 into exp_8's already-completed confidence/weights results (decision already
reports macro_f1; reveal handled separately in backfill_macro_f1_reveal.py), per this project's
cross-experiment macro-F1 reporting initiative (2026-08-12).

Reproduces the EXACT same CV loop as run_signals_v3.py (23-column frame, winning hyperparameters
from results/hyperparameter_search/winner.json), merging the new metric into existing files.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_8/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_7" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from features_v3 import select_exp8_feature_frame  # noqa: E402
from kdm_backbone_v2 import compute_signals, fit_kdm_backbone, kernel_distance_contribution, occlusion_delta  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group
from chimera_task1.reasoning_labels import CONFIDENCE_RANK, TASK1_FACTORS, WEIGHT_RANK, weight_col
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated, make_classifier

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

CONFIDENCE_CONDITIONS = [
    "confidence_kdm_entropy_zeroshot",
    "confidence_kdm_entropy_isotonic",
    "confidence_kdm_dispersion_isotonic",
    "confidence_kdm_participation_isotonic",
    "confidence_kdm_blend",
]
WEIGHT_CONDITIONS = ["weights_kdm_occlusion", "weights_kdm_kernel_distance", "weights_kdm_blend"]


def isotonic_rank(train_signal, train_rank, test_signal, n_levels):
    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(train_signal, train_rank)
    raw = iso.predict(test_signal)
    return np.clip(np.round(raw), 0, n_levels - 1).astype(int)


def blend_rank(train_X, train_rank, test_X):
    clf = make_classifier()
    clf.fit(train_X, train_rank)
    return clf.predict(test_X).astype(int)


def merge_metrics(condition: str, new_fields: dict) -> None:
    path = RESULTS_DIR / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


def main() -> None:
    with open(RESULTS_DIR / "hyperparameter_search" / "winner.json") as f:
        winner = json.load(f)
    winning_config = {
        "n_epochs": winner["n_epochs"], "lr": winner["lr"], "sigma_mult": winner["sigma_mult"],
        "optimizer": winner["optimizer"], "weight_decay": winner["weight_decay"],
    }
    print(f"Using winning config: {winning_config}\n")

    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)
    y_confidence_labels = ann["target_confidence"].values
    y_confidence_rank = np.array([CONFIDENCE_RANK[label] for label in y_confidence_labels])
    weight_rank = {f: np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values]) for f in IN_SCOPE_FACTORS}

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp8_feature_frame(inp_ann, mri_pca)

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    factor_col_idx = {f: [X_frame.columns.get_loc(c) for c in restricted_feature_group(f, "flags")] for f in IN_SCOPE_FACTORS}

    n = len(y_decision)
    conf_macro_f1 = {c: [] for c in CONFIDENCE_CONDITIONS}
    weights_macro_f1 = {c: {f: [] for f in IN_SCOPE_FACTORS} for c in WEIGHT_CONDITIONS}

    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_conf = {c: np.full(n, -1, dtype=int) for c in CONFIDENCE_CONDITIONS}
        oof_weights = {c: {f: np.full(n, -1, dtype=int) for f in IN_SCOPE_FACTORS} for c in WEIGHT_CONDITIONS}
        factor_failed = {c: set() for c in WEIGHT_CONDITIONS}

        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])

            model = fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2, **winning_config)
            sig_train = compute_signals(model, X_train)
            sig_test = compute_signals(model, X_test)

            A_tr, A_te = sig_train["entropy"], sig_test["entropy"]
            B_tr, B_te = sig_train["dispersion"], sig_test["dispersion"]
            C_tr, C_te = sig_train["participation"], sig_test["participation"]
            y_conf_tr = y_confidence_rank[train_idx]

            t1, t2 = np.percentile(A_tr, [100 / 3, 200 / 3])
            oof_conf["confidence_kdm_entropy_zeroshot"][test_idx] = np.where(A_te <= t1, 2, np.where(A_te <= t2, 1, 0))
            oof_conf["confidence_kdm_entropy_isotonic"][test_idx] = isotonic_rank(A_tr, y_conf_tr, A_te, 3)
            oof_conf["confidence_kdm_dispersion_isotonic"][test_idx] = isotonic_rank(B_tr, y_conf_tr, B_te, 3)
            oof_conf["confidence_kdm_participation_isotonic"][test_idx] = isotonic_rank(C_tr, y_conf_tr, C_te, 3)
            blend_train_X = np.column_stack([A_tr, B_tr, C_tr])
            blend_test_X = np.column_stack([A_te, B_te, C_te])
            oof_conf["confidence_kdm_blend"][test_idx] = blend_rank(blend_train_X, y_conf_tr, blend_test_X)

            for factor in IN_SCOPE_FACTORS:
                col_idx = factor_col_idx[factor]
                fill = np.median(X_train[:, col_idx], axis=0)
                D_tr = np.abs(occlusion_delta(model, X_train, col_idx, fill))
                D_te = np.abs(occlusion_delta(model, X_test, col_idx, fill))
                E_tr = kernel_distance_contribution(model, X_train, col_idx)
                E_te = kernel_distance_contribution(model, X_test, col_idx)
                y_w_tr = weight_rank[factor][train_idx]

                try:
                    oof_weights["weights_kdm_occlusion"][factor][test_idx] = isotonic_rank(D_tr, y_w_tr, D_te, 4)
                except ValueError:
                    factor_failed["weights_kdm_occlusion"].add(factor)
                try:
                    oof_weights["weights_kdm_kernel_distance"][factor][test_idx] = isotonic_rank(E_tr, y_w_tr, E_te, 4)
                except ValueError:
                    factor_failed["weights_kdm_kernel_distance"].add(factor)
                try:
                    blend_w_train_X = np.column_stack([D_tr, E_tr])
                    blend_w_test_X = np.column_stack([D_te, E_te])
                    oof_weights["weights_kdm_blend"][factor][test_idx] = blend_rank(blend_w_train_X, y_w_tr, blend_w_test_X)
                except ValueError:
                    factor_failed["weights_kdm_blend"].add(factor)

        for cond in CONFIDENCE_CONDITIONS:
            conf_macro_f1[cond].append(f1_score(y_confidence_rank, oof_conf[cond], average="macro", labels=[0, 1, 2], zero_division=0))

        for cond in WEIGHT_CONDITIONS:
            for factor in IN_SCOPE_FACTORS:
                if factor in factor_failed[cond] or (oof_weights[cond][factor] == -1).any():
                    continue
                f1 = f1_score(weight_rank[factor], oof_weights[cond][factor], average="macro", labels=[0, 1, 2, 3], zero_division=0)
                weights_macro_f1[cond][factor].append(f1)

        print(f"repeat {repeat} done")

    for cond in CONFIDENCE_CONDITIONS:
        merge_metrics(f"{cond}_v3", {
            "macro_f1_mean": round(float(np.mean(conf_macro_f1[cond])), 3),
            "macro_f1_std": round(float(np.std(conf_macro_f1[cond])), 3),
        })

    for cond in WEIGHT_CONDITIONS:
        per_factor_macro_f1 = {}
        factor_means = []
        for factor in IN_SCOPE_FACTORS:
            vals = weights_macro_f1[cond][factor]
            if vals:
                m = round(float(np.mean(vals)), 3)
                per_factor_macro_f1[factor] = m
                factor_means.append(m)
            else:
                per_factor_macro_f1[factor] = None
        merge_metrics(f"{cond}_v3", {
            "mean_macro_f1": round(float(np.mean(factor_means)), 3) if factor_means else None,
            "per_factor_macro_f1": per_factor_macro_f1,
        })


if __name__ == "__main__":
    main()
