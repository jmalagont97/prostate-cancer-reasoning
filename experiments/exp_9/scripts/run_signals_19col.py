"""exp_9: ARD backbone on exp_3's original 19-column frame -- decision + confidence + weights.

Same recalibration logic as exp_6/exp_7/exp_8's run_signals scripts (isotonic increasing="auto",
make_classifier() blend, per-factor try/except, per-factor weights breakdown), unchanged. Only
the backbone differs: fit_kdm_backbone_ard/compute_signals_ard from ard_kernel.py instead of the
scalar-sigma versions, fixed hyperparameters (n_epochs=300, lr=1e-2, sigma_mult=1.0 -- exp_6's
original defaults, no search this round per DESIGN.md's guardrail), also reports macro-F1
natively for confidence/weights per this project's cross-experiment reporting initiative.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_9/scripts/run_signals_19col.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard, kernel_distance_contribution, occlusion_delta_ard  # noqa: E402

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
from chimera_task1.train_reasoning import load_annotated, make_classifier

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"
FRAME_TAG = "19col"

# Fixed hyperparameters -- exp_6's original defaults, no search this round (DESIGN.md guardrail).
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}

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


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision_labels = ann["target_biopsy_decision"].values
    assert pd.notna(y_decision_labels).all()
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
    print(f"n={len(y_decision)}, positive rate={y_decision.mean():.4f}, feature frame: {X_frame.shape} ({FRAME_TAG})\n")

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    factor_col_idx = {f: [X_frame.columns.get_loc(c) for c in restricted_feature_group(f, "flags")] for f in IN_SCOPE_FACTORS}

    n = len(y_decision)
    repeat_metrics = {
        "decision_kdm_ard": {"macro_f1": []},
        **{c: {"ordinal_distance": [], "macro_f1": []} for c in CONFIDENCE_CONDITIONS},
        **{c: {"ordinal_error": [], "decisive_set_f1": [], "n_factors_included": [], "n_factors_skipped": []} for c in WEIGHT_CONDITIONS},
    }
    per_factor_metrics = {c: {f: {"ordinal_error": [], "decisive_set_f1": [], "macro_f1": []} for f in IN_SCOPE_FACTORS} for c in WEIGHT_CONDITIONS}
    all_checks_ok = True

    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)

        oof_decision_probs = np.zeros((n, 2))
        oof_conf = {c: np.full(n, -1, dtype=int) for c in CONFIDENCE_CONDITIONS}
        oof_weights = {c: {f: np.full(n, -1, dtype=int) for f in IN_SCOPE_FACTORS} for c in WEIGHT_CONDITIONS}
        factor_failed = {c: set() for c in WEIGHT_CONDITIONS}

        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])

            model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)
            sig_train = compute_signals_ard(model, X_train)
            sig_test = compute_signals_ard(model, X_test)
            all_checks_ok = all_checks_ok and sig_train["probs_check_ok"] and sig_test["probs_check_ok"]

            oof_decision_probs[test_idx] = sig_test["probs"]

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
                D_tr = np.abs(occlusion_delta_ard(model, X_train, col_idx, fill))
                D_te = np.abs(occlusion_delta_ard(model, X_test, col_idx, fill))
                E_tr = kernel_distance_contribution(model, X_train, col_idx)
                E_te = kernel_distance_contribution(model, X_test, col_idx)
                y_w_tr = weight_rank[factor][train_idx]

                try:
                    oof_weights["weights_kdm_occlusion"][factor][test_idx] = isotonic_rank(D_tr, y_w_tr, D_te, 4)
                except ValueError as e:
                    factor_failed["weights_kdm_occlusion"].add(factor)
                    print(f"  [SKIP] weights_kdm_occlusion / {factor} (repeat {repeat}): {e}")

                try:
                    oof_weights["weights_kdm_kernel_distance"][factor][test_idx] = isotonic_rank(E_tr, y_w_tr, E_te, 4)
                except ValueError as e:
                    factor_failed["weights_kdm_kernel_distance"].add(factor)
                    print(f"  [SKIP] weights_kdm_kernel_distance / {factor} (repeat {repeat}): {e}")

                try:
                    blend_w_train_X = np.column_stack([D_tr, E_tr])
                    blend_w_test_X = np.column_stack([D_te, E_te])
                    oof_weights["weights_kdm_blend"][factor][test_idx] = blend_rank(blend_w_train_X, y_w_tr, blend_w_test_X)
                except ValueError as e:
                    factor_failed["weights_kdm_blend"].add(factor)
                    print(f"  [SKIP] weights_kdm_blend / {factor} (repeat {repeat}): {e}")

        decision_preds = oof_decision_probs.argmax(axis=1)
        repeat_metrics["decision_kdm_ard"]["macro_f1"].append(f1_score(y_decision, decision_preds, average="macro"))

        for cond in CONFIDENCE_CONDITIONS:
            pred_labels = [CONFIDENCE_LEVELS[r] for r in oof_conf[cond]]
            repeat_metrics[cond]["ordinal_distance"].append(
                ordinal_distance(list(y_confidence_labels), pred_labels, CONFIDENCE_RANK)
            )
            repeat_metrics[cond]["macro_f1"].append(
                f1_score(y_confidence_rank, oof_conf[cond], average="macro", labels=[0, 1, 2], zero_division=0)
            )

        for cond in WEIGHT_CONDITIONS:
            dists, f1s = [], []
            for factor in IN_SCOPE_FACTORS:
                if factor in factor_failed[cond] or (oof_weights[cond][factor] == -1).any():
                    continue
                pred_labels = [WEIGHT_LEVELS[r] for r in oof_weights[cond][factor]]
                d = ordinal_distance(list(weight_labels[factor]), pred_labels, WEIGHT_RANK)
                f1v = decisive_set_f1(list(weight_labels[factor]), pred_labels)
                mf1 = f1_score(weight_rank[factor], oof_weights[cond][factor], average="macro", labels=[0, 1, 2, 3], zero_division=0)
                dists.append(d)
                f1s.append(f1v)
                per_factor_metrics[cond][factor]["ordinal_error"].append(d)
                per_factor_metrics[cond][factor]["decisive_set_f1"].append(f1v)
                per_factor_metrics[cond][factor]["macro_f1"].append(mf1)
            repeat_metrics[cond]["ordinal_error"].append(float(np.mean(dists)) if dists else None)
            repeat_metrics[cond]["decisive_set_f1"].append(float(np.mean(f1s)) if f1s else None)
            repeat_metrics[cond]["n_factors_included"].append(len(dists))
            repeat_metrics[cond]["n_factors_skipped"].append(len(IN_SCOPE_FACTORS) - len(dists))

        print(f"repeat {repeat} done")

    print(f"\nprobs_check_ok across every fold/repeat: {all_checks_ok}")
    if not all_checks_ok:
        raise RuntimeError("compute_signals_ard's hand-replicated normalization did not match model.forward() at least once")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    d = RESULTS_DIR / f"decision_kdm_ard_{FRAME_TAG}"
    d.mkdir(exist_ok=True)
    macro_f1s = repeat_metrics["decision_kdm_ard"]["macro_f1"]
    payload = {
        "condition": f"decision_kdm_ard_{FRAME_TAG}",
        "target": "decision",
        "features": f"exp_3 19-column frame ({FRAME_TAG})",
        "model": f"ARD-KDM (per-dimension sigma, {len(X_frame.columns)} dims), fixed config: {ARD_CONFIG}",
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 3),
        "macro_f1_std": round(float(np.std(macro_f1s)), 3),
        "n_folds": N_REPEATS * N_SPLITS,
        "note": "compare against exp_6's decision_kdm_backbone (macro_f1=0.593, scalar sigma, same frame/hyperparameters)",
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[decision_kdm_ard_{FRAME_TAG}] macro_f1={payload['macro_f1_mean']} (exp_6 scalar-sigma was 0.593)")

    for cond in CONFIDENCE_CONDITIONS:
        d = RESULTS_DIR / f"{cond}_ard_{FRAME_TAG}"
        d.mkdir(exist_ok=True)
        dists = repeat_metrics[cond]["ordinal_distance"]
        mf1s = repeat_metrics[cond]["macro_f1"]
        payload = {
            "condition": f"{cond}_ard_{FRAME_TAG}",
            "target": "confidence",
            "features": f"exp_3 19-column frame ({FRAME_TAG})",
            "model": f"ARD-KDM backbone + signal recalibration, config: {ARD_CONFIG}",
            "ordinal_distance_mean": round(float(np.mean(dists)), 3),
            "ordinal_distance_std": round(float(np.std(dists)), 3),
            "macro_f1_mean": round(float(np.mean(mf1s)), 3),
            "macro_f1_std": round(float(np.std(mf1s)), 3),
            "n_folds": N_REPEATS * N_SPLITS,
        }
        with open(d / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{cond}_ard_{FRAME_TAG}] ordinal_distance={payload['ordinal_distance_mean']} macro_f1={payload['macro_f1_mean']}")

    for cond in WEIGHT_CONDITIONS:
        d = RESULTS_DIR / f"{cond}_ard_{FRAME_TAG}"
        d.mkdir(exist_ok=True)
        errs = [e for e in repeat_metrics[cond]["ordinal_error"] if e is not None]
        f1s = [x for x in repeat_metrics[cond]["decisive_set_f1"] if x is not None]
        per_factor = {}
        factor_macro_f1_means = []
        for factor in IN_SCOPE_FACTORS:
            fd = per_factor_metrics[cond][factor]["ordinal_error"]
            ff = per_factor_metrics[cond][factor]["decisive_set_f1"]
            fm = per_factor_metrics[cond][factor]["macro_f1"]
            if fd:
                m = round(float(np.mean(fm)), 3)
                per_factor[factor] = {
                    "ordinal_error": round(float(np.mean(fd)), 3),
                    "decisive_set_f1": round(float(np.mean(ff)), 3),
                    "macro_f1": m,
                    "n_repeats_included": len(fd),
                }
                factor_macro_f1_means.append(m)
            else:
                per_factor[factor] = {"skipped": True, "n_repeats_included": 0}
        payload = {
            "condition": f"{cond}_ard_{FRAME_TAG}",
            "target": "variable_weights",
            "features": f"exp_3 19-column frame ({FRAME_TAG})",
            "model": f"ARD-KDM backbone + signal recalibration, config: {ARD_CONFIG}",
            "mean_ordinal_error": round(float(np.mean(errs)), 3) if errs else None,
            "mean_decisive_set_f1": round(float(np.mean(f1s)), 3) if f1s else None,
            "mean_macro_f1": round(float(np.mean(factor_macro_f1_means)), 3) if factor_macro_f1_means else None,
            "mean_n_factors_included": round(float(np.mean(repeat_metrics[cond]["n_factors_included"])), 2),
            "mean_n_factors_skipped": round(float(np.mean(repeat_metrics[cond]["n_factors_skipped"])), 2),
            "n_repeats": N_REPEATS,
            "per_factor": per_factor,
        }
        with open(d / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{cond}_ard_{FRAME_TAG}] mean_ordinal_error={payload['mean_ordinal_error']} mean_macro_f1={payload['mean_macro_f1']}")


if __name__ == "__main__":
    main()
