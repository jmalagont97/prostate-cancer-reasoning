#!/usr/bin/env python3
"""
exp_19: SHAP / Distance-Attribution Exhaustive Threshold Optimization for Clinical Relevance (Subtask 1.3)

Predicts the 10 official clinical relevance weights (target_code_weight_* ∈ {0, 1, 2, 3})
and section reveal sequences (target_reveal_sequence_json) using:
  1. Local feature attributions (KNN distance contributions & SHAP values) from the frozen tabular model.
  2. Global Exhaustive 3-Threshold Optimization (τ1, τ2, τ3) per variable on training fold attribution scores.

Targets (10 variables):
  - age, fh, cspca, pirads, vol, psa, comorbidity, psad, dre, bx
  - Categories: 0 = not_used, 1 = noted, 2 = important, 3 = decisive

Protocol:
  - 50 MCCV splits (70 train / 18 val) for selection by MOE_abs across 10 weights.
  - 88 LOO folds for final evaluation of winning condition.
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
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"
RESULTS_DIR = ROOT / "experiments" / "exp_19" / "results"
FIGURES_DIR = ROOT / "experiments" / "exp_19" / "reports" / "figures"

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
EPS = 1e-10

ATTRIBUTION_METHODS = ["knn_distance_attribution", "shap_kernel"]
THRESHOLD_MODES = ["exact_free", "exact_balanced_min_recall"]
SCALING_OPTIONS = ["raw", "max_normalized"]


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-Weighted KNN (Subtask 1.1 Model)
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceWeightedKNN:
    def __init__(self, n_neighbors=1, metric="cosine", use_distance_weight=False, epsilon=1e-10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_distance_weight = use_distance_weight
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
            d_nn = dists[i, nn_idx]
            y_nn = self.y_train[nn_idx]
            c_nn = self.conf_weights[nn_idx]
            w_dist = np.ones_like(d_nn)
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)
        return proba

    def get_nearest_neighbors(self, X):
        X = np.array(X, dtype=np.float64)
        from numpy.linalg import norm
        X_norm = X / (norm(X, axis=1, keepdims=True) + self.epsilon)
        T_norm = self.X_train / (norm(self.X_train, axis=1, keepdims=True) + self.epsilon)
        dists = 1 - X_norm @ T_norm.T
        dists = np.clip(dists, 0, 2)
        nn_indices = np.argmin(dists, axis=1)
        return nn_indices


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
    X_scaled_arr = scaler.transform(X_num.values.astype(np.float64))
    X_num_scaled = pd.DataFrame(X_scaled_arr, index=X.index, columns=num_cols)

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
    X_scaled_arr = scaler.transform(X_num.values.astype(np.float64))
    X_num_scaled = pd.DataFrame(X_scaled_arr, index=X.index, columns=num_cols)

    X_proc = pd.concat([X_num_scaled, X_cat], axis=1)
    return X_proc


# ══════════════════════════════════════════════════════════════════════════════
# Feature Attribution Extraction (Distance & SHAP)
# ══════════════════════════════════════════════════════════════════════════════

def extract_attributions(knn_model, X_tr_df, X_ev_df):
    nn_tr_idx = knn_model.get_nearest_neighbors(X_tr_df.values)
    nn_ev_idx = knn_model.get_nearest_neighbors(X_ev_df.values)

    # 1. KNN Distance Attribution per variable
    psi_tr_dist = {}
    psi_ev_dist = {}

    for var in TARGET_VARS:
        raw_col = VAR_TO_TABULAR_NAME[var]
        matched_cols = [c for c in X_tr_df.columns if c == raw_col or c.startswith(f"{raw_col}_")]
        if matched_cols:
            col_idx = [X_tr_df.columns.get_loc(c) for c in matched_cols]
            tr_diff = np.abs(X_tr_df.values[:, col_idx] - knn_model.X_train[nn_tr_idx][:, col_idx])
            ev_diff = np.abs(X_ev_df.values[:, col_idx] - knn_model.X_train[nn_ev_idx][:, col_idx])
            psi_tr_dist[var] = np.mean(tr_diff, axis=1)
            psi_ev_dist[var] = np.mean(ev_diff, axis=1)
        else:
            psi_tr_dist[var] = np.zeros(len(X_tr_df))
            psi_ev_dist[var] = np.zeros(len(X_ev_df))

    # 2. SHAP Kernel Explainer Attribution (optimized nsamples)
    background = shap.sample(X_tr_df.values, min(5, len(X_tr_df)), random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(knn_model.predict_proba, background, silent=True)
        tr_shap_values = np.abs(explainer.shap_values(X_tr_df.values, nsamples=20, l1_reg="num_features(10)"))
        ev_shap_values = np.abs(explainer.shap_values(X_ev_df.values, nsamples=20, l1_reg="num_features(10)"))

    psi_tr_shap = {}
    psi_ev_shap = {}

    for var in TARGET_VARS:
        raw_col = VAR_TO_TABULAR_NAME[var]
        matched_cols = [c for c in X_tr_df.columns if c == raw_col or c.startswith(f"{raw_col}_")]
        if matched_cols:
            col_idx = [X_tr_df.columns.get_loc(c) for c in matched_cols]
            psi_tr_shap[var] = np.mean(tr_shap_values[:, col_idx], axis=1)
            psi_ev_shap[var] = np.mean(ev_shap_values[:, col_idx], axis=1)
        else:
            psi_tr_shap[var] = np.zeros(len(X_tr_df))
            psi_ev_shap[var] = np.zeros(len(X_ev_df))

    return (psi_tr_dist, psi_ev_dist), (psi_tr_shap, psi_ev_shap)



# ══════════════════════════════════════════════════════════════════════════════
# Vectorized Global Exhaustive 3-Threshold Optimizer
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


def fit_exhaustive_3thresholds(psi_train, y_train, mode="exact_free", min_preds_per_class=2):
    psi_sorted = np.sort(np.unique(psi_train))
    if len(psi_sorted) < 3:
        return 0.25, 0.50, 0.75

    candidates = [0.0]
    for i in range(len(psi_sorted) - 1):
        candidates.append((psi_sorted[i] + psi_sorted[i+1]) / 2.0)
    candidates.append(float(np.max(psi_train) + 1e-3))
    candidates = np.array(candidates, dtype=np.float64)

    # Build all valid 3-threshold tuples (tau1 < tau2 < tau3)
    tuples_list = []
    n_cand = len(candidates)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            for k in range(j + 1, n_cand):
                tuples_list.append((candidates[i], candidates[j], candidates[k]))

    if not tuples_list:
        return 0.25, 0.50, 0.75

    tuples = np.array(tuples_list, dtype=np.float64)  # shape (K, 3)

    tau1 = tuples[:, 0:1]  # shape (K, 1)
    tau2 = tuples[:, 1:2]  # shape (K, 1)
    tau3 = tuples[:, 2:3]  # shape (K, 1)

    psi = np.asarray(psi_train, dtype=np.float64).reshape(1, -1)  # shape (1, N)
    y_mat = np.asarray(y_train, dtype=int).reshape(1, -1)          # shape (1, N)

    preds = np.zeros((len(tuples), psi.shape[1]), dtype=int)
    preds[psi >= tau1] = 1
    preds[psi >= tau2] = 2
    preds[psi >= tau3] = 3

    present_classes = np.unique(y_mat[0])
    class_errors = []

    for c in present_classes:
        mask = (y_mat == c)
        err_c = np.mean(np.abs(preds[:, mask[0]] - c) / 3.0, axis=1) if np.any(mask) else np.zeros(len(tuples))
        class_errors.append(err_c)

    moe_vec = np.mean(class_errors, axis=0)

    if mode == "exact_balanced_min_recall":
        valid_mask = np.ones(len(tuples), dtype=bool)
        for c in present_classes:
            count_c = np.sum(preds == c, axis=1)
            valid_mask &= (count_c >= min_preds_per_class)
        moe_vec[~valid_mask] = 999.0

    best_idx = np.argmin(moe_vec)
    best_tuple = tuples[best_idx]
    return float(best_tuple[0]), float(best_tuple[1]), float(best_tuple[2])


def categorize_by_3thresholds(psi, tau1, tau2, tau3):
    psi = np.asarray(psi, dtype=np.float64)
    preds = np.zeros(len(psi), dtype=int)
    preds[psi >= tau1] = 1
    preds[psi >= tau2] = 2
    preds[psi >= tau3] = 3
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


# ══════════════════════════════════════════════════════════════════════════════
# Main Sweep Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("exp_19: SHAP / Distance-Attribution Exhaustive Threshold Optimization")
    print("=" * 80)

    # Git commit
    git_commit_file = RESULTS_DIR / "git_commit.txt"
    try:
        res = subprocess.run(["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True, check=True)
        commit_str = res.stdout.strip()
    except Exception as e:
        commit_str = f"UNKNOWN ({e})"
    git_commit_file.write_text(commit_str + "\n")
    print(f"  Git commit: {commit_str}")

    # Load data
    inputs_df = pd.read_csv(DATA / "inputs.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable_df = splits_df[splits_df["cohort_status"] == "usable_labeled"].copy()
    usable_df = usable_df.sort_values("case_id").reset_index(drop=True)
    n_usable = len(usable_df)
    print(f"  Usable cohort size: {n_usable}")

    usable_case_ids = usable_df["case_id"].tolist()
    inputs_df = inputs_df.set_index("case_id").loc[usable_case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[usable_case_ids].reset_index()

    y_binary = gt_df["target_biopsy_decision_binary"].values.astype(int)
    y_conf_labels = gt_df["target_confidence"].tolist()
    conf_weights = np.array([CONFIDENCE_MAP[c] for c in y_conf_labels], dtype=np.float64)

    # Extract 10 ground-truth relevance weight vectors
    gt_weights = {}
    for var in TARGET_VARS:
        gt_weights[var] = gt_df[f"target_code_weight_{var}"].values.astype(int)

    # Extract ground-truth section reveal sequences
    gt_reveal_sequences = [json.loads(s) for s in gt_df["target_reveal_sequence_json"].tolist()]

    X_tab_raw = inputs_df[[c for c in inputs_df.columns if c not in ["case_id", "path_hist_bx_gl_tert"] and not c.startswith("mri_emb_") and not c.startswith("txt_")]].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Pre-computing Feature Attributions for MCCV & LOO Folds
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("Phase 1: Pre-computing Feature Attributions for MCCV (50 splits)...")
    print("─" * 80)

    mccv_attributions = []
    for split_idx in range(TOTAL_MCCV_FOLDS):
        col = f"mccv_split_{split_idx:02d}"
        split_vals = usable_df[col].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(
            X_tab_raw, train_idx, CATEGORICAL_COLS
        )
        X_ev_tab = transform_features_eval(
            X_tab_raw, val_idx, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols
        )

        knn_model = ConfidenceWeightedKNN(n_neighbors=1, metric="cosine", use_distance_weight=False)
        knn_model.fit(X_tr_tab.values, y_binary[train_idx], conf_weights[train_idx])

        (tr_dist, ev_dist), (tr_shap, ev_shap) = extract_attributions(knn_model, X_tr_tab, X_ev_tab)

        mccv_attributions.append({
            "split_idx": split_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "dist": (tr_dist, ev_dist),
            "shap": (tr_shap, ev_shap),
        })
        if (split_idx + 1) % 10 == 0 or (split_idx + 1) == TOTAL_MCCV_FOLDS:
            print(f"  Processed attributions for {split_idx+1}/{TOTAL_MCCV_FOLDS} MCCV splits")

    print("\n" + "─" * 80)
    print("Phase 2: Pre-computing Feature Attributions for LOO (88 folds)...")
    print("─" * 80)

    loo_attributions = []
    for fold_idx in range(TOTAL_LOO_FOLDS):
        split_vals = usable_df["loocv_fold"].values
        val_idx = np.where(split_vals == fold_idx)[0]
        train_idx = np.where(split_vals != fold_idx)[0]

        X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(
            X_tab_raw, train_idx, CATEGORICAL_COLS
        )
        X_ev_tab = transform_features_eval(
            X_tab_raw, val_idx, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols
        )

        knn_model = ConfidenceWeightedKNN(n_neighbors=1, metric="cosine", use_distance_weight=False)
        knn_model.fit(X_tr_tab.values, y_binary[train_idx], conf_weights[train_idx])

        (tr_dist, ev_dist), (tr_shap, ev_shap) = extract_attributions(knn_model, X_tr_tab, X_ev_tab)

        loo_attributions.append({
            "fold_idx": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "dist": (tr_dist, ev_dist),
            "shap": (tr_shap, ev_shap),
        })
        if (fold_idx + 1) % 20 == 0 or (fold_idx + 1) == TOTAL_LOO_FOLDS:
            print(f"  Processed attributions for {fold_idx+1}/{TOTAL_LOO_FOLDS} LOO folds")

    # Baseline prediction (Always predict mode 'noted' = 1.0)
    baseline_moe_list = []
    for var in TARGET_VARS:
        all_true = np.concatenate([gt_weights[var][ma["val_idx"]] for ma in mccv_attributions])
        base_pred = np.full_like(all_true, fill_value=1)
        baseline_moe_list.append(compute_moe_abs_weights(all_true, base_pred))
    baseline_moe_abs = float(np.mean(baseline_moe_list))
    print(f"\n  Baseline (Always Noted = 1.0) MCCV Target Weights MOE_abs: {baseline_moe_abs:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Grid Search over 8 Conditions in MCCV
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("Phase 3: Running 8-Condition Grid Search over MCCV...")
    print("─" * 80)

    scorecard_rows = []
    mccv_predictions_by_config = {}

    config_idx = 0
    total_configs = len(ATTRIBUTION_METHODS) * len(THRESHOLD_MODES) * len(SCALING_OPTIONS)

    for attr_method in ATTRIBUTION_METHODS:
        attr_key = "dist" if attr_method == "knn_distance_attribution" else "shap"
        for mode in THRESHOLD_MODES:
            for scaling in SCALING_OPTIONS:
                config_idx += 1
                config_name = f"method_{attr_method}_mode_{mode}_scale_{scaling}"

                var_moe_dict = {var: [] for var in TARGET_VARS}
                var_f1_dict = {var: [] for var in TARGET_VARS}
                section_f1_list = []

                all_pred_weights = {var: [] for var in TARGET_VARS}
                all_true_weights = {var: [] for var in TARGET_VARS}

                for ma in mccv_attributions:
                    train_idx = ma["train_idx"]
                    val_idx = ma["val_idx"]

                    tr_psi_dict, ev_psi_dict = ma[attr_key]

                    val_pred_dict = {}

                    for var in TARGET_VARS:
                        tr_psi = tr_psi_dict[var]
                        ev_psi = ev_psi_dict[var]

                        if scaling == "max_normalized":
                            max_val = max(np.max(tr_psi), 1e-5)
                            tr_psi = tr_psi / max_val
                            ev_psi = ev_psi / max_val

                        tau1, tau2, tau3 = fit_exhaustive_3thresholds(
                            tr_psi, gt_weights[var][train_idx], mode=mode
                        )
                        pred_ev = categorize_by_thresholds = categorize_by_3thresholds(ev_psi, tau1, tau2, tau3)

                        moe_v = compute_moe_abs_weights(gt_weights[var][val_idx], pred_ev)
                        f1_v = f1_score(gt_weights[var][val_idx], pred_ev, average="macro", zero_division=0)

                        var_moe_dict[var].append(moe_v)
                        var_f1_dict[var].append(f1_v)

                        all_pred_weights[var].extend(pred_ev)
                        all_true_weights[var].extend(gt_weights[var][val_idx])

                        val_pred_dict[var] = pred_ev

                    # Evaluate Section Reveal Sequences on Validation
                    val_pred_sections = []
                    val_true_sections = [gt_reveal_sequences[i] for i in val_idx]

                    for i in range(len(val_idx)):
                        single_pred = {var: val_pred_dict[var][i] for var in TARGET_VARS}
                        val_pred_sections.append(map_weights_to_reveal_sequence(single_pred))

                    # Compute Multi-label Section F1
                    sec_f1s = []
                    for sec in SECTIONS:
                        sec_true = [int(sec in s) for s in val_true_sections]
                        sec_pred = [int(sec in s) for s in val_pred_sections]
                        sec_f1s.append(f1_score(sec_true, sec_pred, zero_division=0))
                    section_f1_list.append(np.mean(sec_f1s))

                # Aggregate over 50 MCCV splits
                pooled_moe_vars = [compute_moe_abs_weights(all_true_weights[v], all_pred_weights[v]) for v in TARGET_VARS]
                pooled_f1_vars = [f1_score(all_true_weights[v], all_pred_weights[v], average="macro", zero_division=0) for v in TARGET_VARS]

                mean_moe_weights = float(np.mean(pooled_moe_vars))
                mean_f1_weights = float(np.mean(pooled_f1_vars))
                mean_section_f1 = float(np.mean(section_f1_list))

                row = {
                    "config_name": config_name,
                    "attr_method": attr_method,
                    "threshold_mode": mode,
                    "scaling": scaling,
                    "pooled_moe_weights": mean_moe_weights,
                    "pooled_f1_weights": mean_f1_weights,
                    "mean_section_f1": mean_section_f1,
                    "passes_baseline": (mean_moe_weights < baseline_moe_abs),
                }
                scorecard_rows.append(row)
                mccv_predictions_by_config[config_name] = (all_true_weights, all_pred_weights)

                print(f"  Config {config_idx:02d}/{total_configs} [{config_name}]: MOE_weights={mean_moe_weights:.4f}, F1_weights={mean_f1_weights:.4f}, Section_F1={mean_section_f1:.4f}")

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(RESULTS_DIR / "evaluation_scorecard.csv", index=False)

    scorecard_df = scorecard_df.sort_values(by=["pooled_moe_weights", "mean_section_f1"], ascending=[True, False]).reset_index(drop=True)
    selected_row = scorecard_df.iloc[0]
    selected_config_name = selected_row["config_name"]
    selected_attr_method = selected_row["attr_method"]
    selected_mode = selected_row["threshold_mode"]
    selected_scaling = selected_row["scaling"]

    print("\n" + "★" * 80)
    print(f"SELECTED MCCV CONFIG: {selected_config_name}")
    print(f"  Attribute Method: {selected_attr_method}, Threshold Mode: {selected_mode}, Scaling: {selected_scaling}")
    print(f"  MCCV Relevance Weights MOE_abs: {selected_row['pooled_moe_weights']:.4f} (vs Baseline {baseline_moe_abs:.4f})")
    print(f"  MCCV Relevance Weights F1_macro: {selected_row['pooled_f1_weights']:.4f}")
    print(f"  MCCV Section Reveal F1_macro: {selected_row['mean_section_f1']:.4f}")
    print("★" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: LOO Final Audit of Selected Config
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print(f"Phase 4: Evaluating Selected Config [{selected_config_name}] on LOO (88 folds)...")
    print("─" * 80)

    attr_key = "dist" if selected_attr_method == "knn_distance_attribution" else "shap"
    loo_pred_weights = {var: [] for var in TARGET_VARS}
    loo_true_weights = {var: [] for var in TARGET_VARS}
    loo_pred_sections = []
    loo_true_sections = []

    for ma in loo_attributions:
        train_idx = ma["train_idx"]
        val_idx = ma["val_idx"]

        tr_psi_dict, ev_psi_dict = ma[attr_key]

        single_pred_dict = {}

        for var in TARGET_VARS:
            tr_psi = tr_psi_dict[var]
            ev_psi = ev_psi_dict[var]

            if selected_scaling == "max_normalized":
                max_val = max(np.max(tr_psi), 1e-5)
                tr_psi = tr_psi / max_val
                ev_psi = ev_psi / max_val

            tau1, tau2, tau3 = fit_exhaustive_3thresholds(
                tr_psi, gt_weights[var][train_idx], mode=selected_mode
            )
            pred_ev = categorize_by_3thresholds(ev_psi, tau1, tau2, tau3)[0]
            true_ev = gt_weights[var][val_idx[0]]

            loo_pred_weights[var].append(pred_ev)
            loo_true_weights[var].append(true_ev)
            single_pred_dict[var] = pred_ev

        loo_pred_sections.append(map_weights_to_reveal_sequence(single_pred_dict))
        loo_true_sections.append(gt_reveal_sequences[val_idx[0]])

    # Calculate LOO metrics
    loo_var_moe = [compute_moe_abs_weights(loo_true_weights[v], loo_pred_weights[v]) for v in TARGET_VARS]
    loo_var_f1 = [f1_score(loo_true_weights[v], loo_pred_weights[v], average="macro", zero_division=0) for v in TARGET_VARS]

    loo_mean_moe = float(np.mean(loo_var_moe))
    loo_mean_f1 = float(np.mean(loo_var_f1))

    sec_f1s = []
    for sec in SECTIONS:
        sec_true = [int(sec in s) for s in loo_true_sections]
        sec_pred = [int(sec in s) for s in loo_pred_sections]
        sec_f1s.append(f1_score(sec_true, sec_pred, zero_division=0))
    loo_section_f1 = float(np.mean(sec_f1s))

    print(f"\n  LOO Results [{selected_config_name}]:")
    print(f"    LOO Relevance Weights MOE_abs: {loo_mean_moe:.4f}")
    print(f"    LOO Relevance Weights F1_macro: {loo_mean_f1:.4f}")
    print(f"    LOO Section Reveal F1_macro:    {loo_section_f1:.4f}")

    # Per variable metrics table
    var_df_rows = []
    for idx, v in enumerate(TARGET_VARS):
        var_df_rows.append({
            "variable": v,
            "mccv_moe_abs": pooled_moe_vars[idx],
            "loo_moe_abs": loo_var_moe[idx],
            "loo_f1_macro": loo_var_f1[idx],
        })
    var_metrics_df = pd.DataFrame(var_df_rows)
    var_metrics_df.to_csv(RESULTS_DIR / "per_variable_metrics.csv", index=False)

    loo_preds_df = pd.DataFrame(loo_pred_weights)
    loo_preds_df.to_csv(RESULTS_DIR / "predictions_loo.csv", index=False)

    summary_data = {
        "experiment": "exp_19",
        "description": "SHAP / Distance-Attribution Exhaustive Threshold Optimization",
        "git_commit": commit_str,
        "n_usable_cohort": n_usable,
        "baseline_always_noted": {
            "moe_abs": baseline_moe_abs,
        },
        "selected_config": {
            "config_name": selected_config_name,
            "attr_method": selected_attr_method,
            "threshold_mode": selected_mode,
            "scaling": selected_scaling,
            "mccv_weights_moe_abs": float(selected_row["pooled_moe_weights"]),
            "mccv_weights_f1_macro": float(selected_row["pooled_f1_weights"]),
            "mccv_section_f1_macro": float(selected_row["mean_section_f1"]),
            "loo_weights_moe_abs": loo_mean_moe,
            "loo_weights_f1_macro": loo_mean_f1,
            "loo_section_f1_macro": loo_section_f1,
        },
        "total_runtime_seconds": round(time.time() - t0, 2),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED in {summary_data['total_runtime_seconds']}s")
    print(f"Summary written to: {RESULTS_DIR / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
