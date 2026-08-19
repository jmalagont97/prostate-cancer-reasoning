#!/usr/bin/env python3
"""
exp_20: Fast Vectorized Permutation SHAP Significance Thresholding (Subtask 1.3)

Predicts 10 clinical relevance weights (target_code_weight_* ∈ {0, 1, 2, 3})
and section reveal sequences (target_reveal_sequence_json) using:
  1. Vectorized non-parametric permutation testing (B=500) on SHAP attributions.
  2. Sample-level p-values (p_i,k), significance descriptors (S_i,k = 1 - p_i,k), and Z-scores (Z_i,k).
  3. P-value gating (p >= 0.05 => 0 = not_used) + Global 3-Threshold Grid Search.

Execution time: < 15 seconds.
"""

import os
import sys
import time
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"
RESULTS_DIR = ROOT / "experiments" / "exp_20" / "results"
REPORTS_DIR = ROOT / "experiments" / "exp_20" / "reports"

CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]

FROZEN_21_VARS = [
    "cli_age", "cli_allergies_count", "cli_bx", "cli_comorbidity_count",
    "cli_cspca", "cli_dre", "cli_fh_binary", "cli_ipss_score", "cli_months",
    "cli_pirads", "cli_psa", "cli_psad", "cli_psav", "cli_vol",
    "vit_bp_diastolic", "vit_bp_systolic", "vit_heart_rate_bpm",
    "vit_height_cm", "vit_smoking_pack_years", "vit_smoking_status",
    "vit_weight_kg",
]

TARGET_VARS = ["age", "fh", "cspca", "pirads", "vol", "psa", "comorbidity", "psad", "dre", "bx"]
VAR_TO_TABULAR_NAME = {
    "age": "cli_age", "fh": "cli_fh_binary", "cspca": "cli_cspca",
    "pirads": "cli_pirads", "vol": "cli_vol", "psa": "cli_psa",
    "comorbidity": "cli_comorbidity_count", "psad": "cli_psad",
    "dre": "cli_dre", "bx": "cli_bx"
}

SECTIONS = [
    "radiology_report", "laboratory_results", "psa_trend",
    "previous_notes", "family_history", "pathology_report"
]

ORD_NAMES = ["not_used", "noted", "important", "decisive"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}

TOTAL_MCCV_FOLDS = 50
TOTAL_LOO_FOLDS = 88
N_PERMUTATIONS = 500


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-Weighted KNN (Subtask 1.1 Model)
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceWeightedKNN:
    def __init__(self, n_neighbors=1, metric="cosine", epsilon=1e-10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.epsilon = epsilon

    def fit(self, X, y, conf_weights):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        self.conf_weights = np.array(conf_weights, dtype=np.float64)

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        from numpy.linalg import norm
        X_norm = X / (norm(X, axis=1, keepdims=True) + self.epsilon)
        T_norm = self.X_train / (norm(self.X_train, axis=1, keepdims=True) + self.epsilon)
        dists = 1 - X_norm @ T_norm.T
        dists = np.clip(dists, 0, 2)

        proba = np.zeros(len(X))
        for i in range(len(X)):
            nn_idx = np.argsort(dists[i])[:self.n_neighbors]
            y_nn = self.y_train[nn_idx]
            c_nn = self.conf_weights[nn_idx]
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.mean(q)
        return proba

    def get_nearest_neighbors(self, X):
        X = np.array(X, dtype=np.float64)
        from numpy.linalg import norm
        X_norm = X / (norm(X, axis=1, keepdims=True) + self.epsilon)
        T_norm = self.X_train / (norm(self.X_train, axis=1, keepdims=True) + self.epsilon)
        dists = 1 - X_norm @ T_norm.T
        return np.argmin(dists, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Feature Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_features_train(X_raw, train_idx, cat_cols):
    X = X_raw.iloc[train_idx].copy()
    cat_cols_in = [c for c in cat_cols if c in X.columns]
    drop_cols = [c for c in X.columns if X[c].isna().mean() > 0.5]

    for col in X.columns:
        if col in drop_cols:
            continue
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols_in:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    keep_cols = []
    for v in FROZEN_21_VARS:
        if v in drop_cols:
            continue
        if v in X.columns:
            keep_cols.append(v)
        ind = f"{v}__is_missing"
        if ind in X.columns:
            keep_cols.append(ind)
    X = X[keep_cols]

    cat_cols_kept = [c for c in cat_cols_in if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols_kept]

    if cat_cols_kept:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        ohe.fit(X[cat_cols_kept])
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols_kept]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols_kept),
        )
    else:
        ohe = None
        X_cat = pd.DataFrame(index=X.index)

    scaler = MinMaxScaler()
    X_num = X[num_cols].copy()
    scaler.fit(X_num.values.astype(np.float64))
    X_num_scaled = pd.DataFrame(scaler.transform(X_num.values.astype(np.float64)), index=X.index, columns=num_cols)

    X_proc = pd.concat([X_num_scaled, X_cat], axis=1)
    return X_proc, drop_cols, ohe, scaler, cat_cols_kept, num_cols


def transform_features_eval(X_raw, eval_idx, cat_cols, drop_cols, ohe, scaler, cat_cols_kept, num_cols):
    X = X_raw.iloc[eval_idx].copy()
    for col in X_raw.columns:
        if col in drop_cols:
            continue
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    keep_cols = []
    for v in FROZEN_21_VARS:
        if v in drop_cols:
            continue
        if v in X.columns:
            keep_cols.append(v)
        ind = f"{v}__is_missing"
        if ind in X.columns:
            keep_cols.append(ind)
    X = X[keep_cols]

    if cat_cols_kept and ohe is not None:
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols_kept]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols_kept),
        )
    else:
        X_cat = pd.DataFrame(index=X.index)

    X_num = X[num_cols].copy()
    X_num_scaled = pd.DataFrame(scaler.transform(X_num.values.astype(np.float64)), index=X.index, columns=num_cols)

    X_proc = pd.concat([X_num_scaled, X_cat], axis=1)
    return X_proc


# ══════════════════════════════════════════════════════════════════════════════
# Fast Vectorized Permutation Engine
# ══════════════════════════════════════════════════════════════════════════════

def extract_vectorized_permuted_descriptors(knn_model, X_tr_df, X_ev_df, num_permutations=500):
    import shap
    background = shap.sample(X_tr_df.values, min(5, len(X_tr_df)), random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(knn_model.predict_proba, background, silent=True)
        tr_shap_raw = np.abs(explainer.shap_values(X_tr_df.values, nsamples=20, l1_reg="num_features(10)"))
        ev_shap_raw = np.abs(explainer.shap_values(X_ev_df.values, nsamples=20, l1_reg="num_features(10)"))

    desc_tr = {}
    desc_ev = {}

    rng = np.random.RandomState(42)
    B = num_permutations
    N_tr = len(X_tr_df)
    perm_matrix = np.array([rng.permutation(N_tr) for _ in range(B)])

    for var in TARGET_VARS:
        raw_col = VAR_TO_TABULAR_NAME[var]
        matched_cols = [c for c in X_tr_df.columns if c == raw_col or c.startswith(f"{raw_col}_")]
        if matched_cols:
            col_idx = [X_tr_df.columns.get_loc(c) for c in matched_cols]
            tr_vals = np.mean(tr_shap_raw[:, col_idx], axis=1)
            ev_vals = np.mean(ev_shap_raw[:, col_idx], axis=1)
        else:
            tr_vals = np.zeros(len(X_tr_df))
            ev_vals = np.zeros(len(X_ev_df))

        max_tr = np.max(tr_vals) + 1e-10
        tr_norm = tr_vals / max_tr
        ev_norm = ev_vals / max_tr

        null_tr = tr_norm[perm_matrix]  # shape (B, N_tr)
        null_flat = null_tr.ravel()     # shape (B * N_tr,)
        null_mean = np.mean(null_flat)
        null_std = np.std(null_flat) + 1e-8

        # Vectorized p-values & Z-scores against empirical null pool
        p_val_tr = (np.sum(null_flat[None, :] >= tr_norm[:, None], axis=1) + 1.0) / (len(null_flat) + 1.0)
        p_val_ev = (np.sum(null_flat[None, :] >= ev_norm[:, None], axis=1) + 1.0) / (len(null_flat) + 1.0)

        z_score_tr = (tr_norm - null_mean) / null_std
        z_score_ev = (ev_norm - null_mean) / null_std

        s_desc_tr = 1.0 - p_val_tr
        s_desc_ev = 1.0 - p_val_ev

        desc_tr[var] = {
            "raw_shap": tr_norm,
            "z_score": z_score_tr,
            "p_val": p_val_tr,
            "sig_desc": s_desc_tr,
        }
        desc_ev[var] = {
            "raw_shap": ev_norm,
            "z_score": z_score_ev,
            "p_val": p_val_ev,
            "sig_desc": s_desc_ev,
        }

    return desc_tr, desc_ev


# ══════════════════════════════════════════════════════════════════════════════
# Threshold Search & Sequence Mapping
# ══════════════════════════════════════════════════════════════════════════════

def compute_moe_abs_weights(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present_classes = np.unique(y_true)
    errors = []
    for c in present_classes:
        mask = (y_true == c)
        diff = np.abs(y_pred[mask] - c) / 3.0
        errors.append(np.mean(diff))
    return float(np.mean(errors)) if errors else 0.0


def fit_exhaustive_3thresholds(scores_train, y_train):
    sorted_scores = np.sort(np.unique(scores_train))
    if len(sorted_scores) < 3:
        return 0.25, 0.50, 0.75

    candidates = [float(np.min(scores_train) - 1e-3)]
    for i in range(len(sorted_scores) - 1):
        candidates.append((sorted_scores[i] + sorted_scores[i+1]) / 2.0)
    candidates.append(float(np.max(scores_train) + 1e-3))
    candidates = np.array(candidates, dtype=np.float64)

    tuples_list = []
    n_cand = len(candidates)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            for k in range(j + 1, n_cand):
                tuples_list.append((candidates[i], candidates[j], candidates[k]))

    if not tuples_list:
        return 0.25, 0.50, 0.75

    tuples = np.array(tuples_list, dtype=np.float64)
    tau1 = tuples[:, 0:1]
    tau2 = tuples[:, 1:2]
    tau3 = tuples[:, 2:3]

    scores_mat = np.asarray(scores_train, dtype=np.float64).reshape(1, -1)
    y_mat = np.asarray(y_train, dtype=int).reshape(1, -1)

    preds = np.zeros((len(tuples), scores_mat.shape[1]), dtype=int)
    preds[scores_mat >= tau1] = 1
    preds[scores_mat >= tau2] = 2
    preds[scores_mat >= tau3] = 3

    present_classes = np.unique(y_mat[0])
    class_errors = []

    for c in present_classes:
        mask = (y_mat == c)
        err_c = np.mean(np.abs(preds[:, mask[0]] - c) / 3.0, axis=1) if np.any(mask) else np.zeros(len(tuples))
        class_errors.append(err_c)

    moe_vec = np.mean(class_errors, axis=0)
    best_idx = np.argmin(moe_vec)
    best_tuple = tuples[best_idx]
    return float(best_tuple[0]), float(best_tuple[1]), float(best_tuple[2])


def predict_3thresholds(scores, tau1, tau2, tau3, p_values=None, apply_gating=False):
    scores = np.asarray(scores, dtype=np.float64)
    preds = np.zeros(len(scores), dtype=int)
    preds[scores >= tau1] = 1
    preds[scores >= tau2] = 2
    preds[scores >= tau3] = 3

    if apply_gating and p_values is not None:
        p_values = np.asarray(p_values, dtype=np.float64)
        preds[p_values >= 0.05] = 0

    return preds


def map_weights_to_reveal_sequence(pred_weights_dict):
    sections = []
    rad_max = max(pred_weights_dict["pirads"], pred_weights_dict["psad"], pred_weights_dict["vol"], pred_weights_dict["cspca"])
    if rad_max >= 1:
        sections.append("radiology_report")
    if pred_weights_dict["dre"] >= 1:
        sections.append("laboratory_results")
    if pred_weights_dict["psa"] >= 1:
        sections.append("psa_trend")
    if pred_weights_dict["fh"] >= 1:
        sections.append("family_history")
    if pred_weights_dict["bx"] >= 1:
        sections.append("pathology_report")
    if pred_weights_dict["comorbidity"] >= 1 or pred_weights_dict["age"] >= 2:
        sections.append("previous_notes")
    return sections


def compute_sequence_f1(gt_sequences, pred_sequences):
    all_sections = sorted(SECTIONS)
    gt_bin = np.zeros((len(gt_sequences), len(all_sections)), dtype=int)
    pred_bin = np.zeros((len(pred_sequences), len(all_sections)), dtype=int)

    for i in range(len(gt_sequences)):
        for sec in gt_sequences[i]:
            if sec in all_sections:
                gt_bin[i, all_sections.index(sec)] = 1
        for sec in pred_sequences[i]:
            if sec in all_sections:
                pred_bin[i, all_sections.index(sec)] = 1

    f1s = []
    for j in range(len(all_sections)):
        f1s.append(f1_score(gt_bin[:, j], pred_bin[:, j], zero_division=0))
    return float(np.mean(f1s))


# ══════════════════════════════════════════════════════════════════════════════
# Main Sweep & LOO Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("exp_20: Fast Permutation SHAP Significance Thresholding", flush=True)
    print("=" * 80, flush=True)

    # Git commit logging
    git_commit_file = RESULTS_DIR / "git_commit.txt"
    try:
        res = subprocess.run(["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True, check=True)
        commit_str = res.stdout.strip()
    except Exception as e:
        commit_str = f"UNKNOWN ({e})"
    git_commit_file.write_text(commit_str + "\n")
    print(f"  Git commit: {commit_str}", flush=True)

    # Load data
    inputs_df = pd.read_csv(DATA / "inputs.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable_df = splits_df[splits_df["cohort_status"] == "usable_labeled"].copy()
    usable_df = usable_df.sort_values("case_id").reset_index(drop=True)
    n_usable = len(usable_df)
    print(f"  Usable cohort size: {n_usable}", flush=True)

    usable_case_ids = usable_df["case_id"].tolist()
    inputs_df = inputs_df.set_index("case_id").loc[usable_case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[usable_case_ids].reset_index()

    y_binary = gt_df["target_biopsy_decision_binary"].values.astype(int)
    y_conf_labels = gt_df["target_confidence"].tolist()
    conf_weights = np.array([CONFIDENCE_MAP[c] for c in y_conf_labels], dtype=np.float64)

    gt_weights = {var: gt_df[f"target_code_weight_{var}"].values.astype(int) for var in TARGET_VARS}
    gt_reveal_sequences = [json.loads(s) for s in gt_df["target_reveal_sequence_json"].tolist()]

    X_tab_raw = inputs_df[[c for c in inputs_df.columns if c not in ["case_id", "path_hist_bx_gl_tert"] and not c.startswith("mri_emb_") and not c.startswith("txt_")]].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Precompute MCCV Permuted Descriptors (50 splits)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[Phase 1/4] Pre-computing Permutation Descriptors for MCCV (50 splits)...", flush=True)
    mccv_descriptors = []

    for split_idx in range(TOTAL_MCCV_FOLDS):
        col = f"mccv_split_{split_idx:02d}"
        split_vals = usable_df[col].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(X_tab_raw, train_idx, CATEGORICAL_COLS)
        X_ev_tab = transform_features_eval(X_tab_raw, val_idx, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols)

        knn = ConfidenceWeightedKNN(n_neighbors=1, metric="cosine")
        knn.fit(X_tr_tab, y_binary[train_idx], conf_weights[train_idx])

        desc_tr, desc_ev = extract_vectorized_permuted_descriptors(knn, X_tr_tab, X_ev_tab, num_permutations=N_PERMUTATIONS)

        mccv_descriptors.append({
            "train_idx": train_idx, "val_idx": val_idx,
            "desc_tr": desc_tr, "desc_ev": desc_ev,
        })
        if (split_idx + 1) % 10 == 0 or (split_idx + 1) == TOTAL_MCCV_FOLDS:
            print(f"  Processed {split_idx + 1}/{TOTAL_MCCV_FOLDS} MCCV splits ({((split_idx + 1)/TOTAL_MCCV_FOLDS)*100:.0f}%)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Precompute LOO Permuted Descriptors (88 folds)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[Phase 2/4] Pre-computing Permutation Descriptors for LOO (88 folds)...", flush=True)
    loo_descriptors = []

    for loo_idx in range(TOTAL_LOO_FOLDS):
        val_idx = np.array([loo_idx])
        train_idx = np.array([j for j in range(TOTAL_LOO_FOLDS) if j != loo_idx])

        X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(X_tab_raw, train_idx, CATEGORICAL_COLS)
        X_ev_tab = transform_features_eval(X_tab_raw, val_idx, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols)

        knn = ConfidenceWeightedKNN(n_neighbors=1, metric="cosine")
        knn.fit(X_tr_tab, y_binary[train_idx], conf_weights[train_idx])

        desc_tr, desc_ev = extract_vectorized_permuted_descriptors(knn, X_tr_tab, X_ev_tab, num_permutations=N_PERMUTATIONS)

        loo_descriptors.append({
            "train_idx": train_idx, "val_idx": val_idx,
            "desc_tr": desc_tr, "desc_ev": desc_ev,
        })
        if (loo_idx + 1) % 22 == 0 or (loo_idx + 1) == TOTAL_LOO_FOLDS:
            print(f"  Processed {loo_idx + 1}/{TOTAL_LOO_FOLDS} LOO folds ({((loo_idx + 1)/TOTAL_LOO_FOLDS)*100:.0f}%)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: MCCV Grid Search across 5 Experimental Conditions
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[Phase 3/4] Evaluating Experimental Conditions across MCCV...", flush=True)

    CONDITIONS = [
        ("cond_1_raw_shap", "raw_shap", False),
        ("cond_2_zscore_nogate", "z_score", False),
        ("cond_3_zscore_gated", "z_score", True),
        ("cond_4_pvalue_desc", "sig_desc", False),
        ("cond_5_pvalue_gated", "sig_desc", True),
    ]

    mccv_results = {}

    for cond_name, desc_key, apply_gating in CONDITIONS:
        split_f1_sections = []
        split_moe_weights = []

        for split_idx in range(TOTAL_MCCV_FOLDS):
            fold_data = mccv_descriptors[split_idx]
            train_idx = fold_data["train_idx"]
            val_idx = fold_data["val_idx"]
            desc_tr = fold_data["desc_tr"]
            desc_ev = fold_data["desc_ev"]

            pred_val_weights_dict = {}
            for var in TARGET_VARS:
                s_tr = desc_tr[var][desc_key]
                s_ev = desc_ev[var][desc_key]
                p_ev = desc_ev[var]["p_val"] if apply_gating else None
                y_tr = gt_weights[var][train_idx]

                tau1, tau2, tau3 = fit_exhaustive_3thresholds(s_tr, y_tr)
                preds = predict_3thresholds(s_ev, tau1, tau2, tau3, p_values=p_ev, apply_gating=apply_gating)
                pred_val_weights_dict[var] = preds

            # Evaluate section reveal sequence F1 on validation
            pred_seqs = [map_weights_to_reveal_sequence({var: pred_val_weights_dict[var][i] for var in TARGET_VARS}) for i in range(len(val_idx))]
            gt_seqs_val = [gt_reveal_sequences[i] for i in val_idx]
            f1_sec = compute_sequence_f1(gt_seqs_val, pred_seqs)
            split_f1_sections.append(f1_sec)

            # Evaluate relevance weights MOE_abs
            moes = [compute_moe_abs_weights(gt_weights[var][val_idx], pred_val_weights_dict[var]) for var in TARGET_VARS]
            split_moe_weights.append(np.mean(moes))

        mccv_results[cond_name] = {
            "mean_f1_sections": float(np.mean(split_f1_sections)),
            "std_f1_sections": float(np.std(split_f1_sections)),
            "mean_moe_weights": float(np.mean(split_moe_weights)),
            "std_moe_weights": float(np.std(split_moe_weights)),
        }
        print(f"  Condition '{cond_name}': MCCV F1_sections = {np.mean(split_f1_sections):.4f} ± {np.std(split_f1_sections):.4f} | MOE_weights = {np.mean(split_moe_weights):.4f}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: LOO Out-of-Fold Evaluation for all conditions
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[Phase 4/4] Conducting 88-Fold LOO Final Audit...", flush=True)
    loo_results = {}

    for cond_name, desc_key, apply_gating in CONDITIONS:
        loo_pred_weights = {var: np.zeros(TOTAL_LOO_FOLDS, dtype=int) for var in TARGET_VARS}
        loo_pred_seqs = []

        for loo_idx in range(TOTAL_LOO_FOLDS):
            fold_data = loo_descriptors[loo_idx]
            train_idx = fold_data["train_idx"]
            val_idx = fold_data["val_idx"][0]
            desc_tr = fold_data["desc_tr"]
            desc_ev = fold_data["desc_ev"]

            pred_single_dict = {}
            for var in TARGET_VARS:
                s_tr = desc_tr[var][desc_key]
                s_ev = desc_ev[var][desc_key]
                p_ev = desc_ev[var]["p_val"] if apply_gating else None
                y_tr = gt_weights[var][train_idx]

                tau1, tau2, tau3 = fit_exhaustive_3thresholds(s_tr, y_tr)
                p_val = predict_3thresholds(s_ev, tau1, tau2, tau3, p_values=p_ev, apply_gating=apply_gating)[0]
                loo_pred_weights[var][loo_idx] = p_val
                pred_single_dict[var] = p_val

            loo_pred_seqs.append(map_weights_to_reveal_sequence(pred_single_dict))

        f1_sec_loo = compute_sequence_f1(gt_reveal_sequences, loo_pred_seqs)
        moes_loo = [compute_moe_abs_weights(gt_weights[var], loo_pred_weights[var]) for var in TARGET_VARS]
        mean_moe_loo = float(np.mean(moes_loo))

        # Macro F1 across weights
        f1_weights_list = [f1_score(gt_weights[var], loo_pred_weights[var], average="macro", zero_division=0) for var in TARGET_VARS]
        mean_f1_weights_loo = float(np.mean(f1_weights_list))

        loo_results[cond_name] = {
            "loo_f1_sections": f1_sec_loo,
            "loo_moe_weights": mean_moe_loo,
            "loo_f1_weights_macro": mean_f1_weights_loo,
        }
        print(f"  Condition '{cond_name}': LOO F1_sections = {f1_sec_loo:.4f} | MOE_weights = {mean_moe_loo:.4f} | F1_weights = {mean_f1_weights_loo:.4f}", flush=True)

    elapsed_time = time.time() - t0

    # Save summary JSON
    summary_data = {
        "experiment": "exp_20",
        "description": "Fast Vectorized Permutation SHAP Significance Thresholding",
        "n_permutations": N_PERMUTATIONS,
        "elapsed_seconds": elapsed_time,
        "mccv_metrics": mccv_results,
        "loo_metrics": loo_results,
    }

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2) + "\n")

    print("\n" + "=" * 80, flush=True)
    print(f"COMPLETED exp_20 IN {elapsed_time:.2f} SECONDS!", flush=True)
    print(f"Summary saved to: {summary_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
