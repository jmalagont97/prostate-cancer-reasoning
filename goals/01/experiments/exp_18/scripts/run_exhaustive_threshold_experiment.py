#!/usr/bin/env python3
"""
exp_18: Global Exhaustive Threshold Optimization on Decision Risk

Predicts target_confidence (uncertain/borderline/clear) using Global Exhaustive 2D Threshold Search
(τ1*, τ2*) directly on the continuous Decision Risk score Ω(c_fn, λ).

Formulation:
  - c_fn ∈ (0, 1), c_fp = 1 - c_fn
  - R_margen = min(p_bar * c_fn, (1 - p_bar) * c_fp) / (c_fn * c_fp)
  - R_conflicto = 2 * std(p_T, p_M, p_X)
  - Ω(c_fn, λ) = (1 - λ) * R_margen + λ * R_conflicto

Exhaustive 2D Threshold Optimizer:
  - Finds (τ1*, τ2*) with 0 < τ1 < τ2 < 1 that minimizes:
    Loss = MOE_abs(y_train, y_hat) - 0.001 * F1_macro(y_train, y_hat)
  - Mode: "exact_free" vs "exact_balanced_min_recall" (min 3 preds per class in train).

Grid search (50 conditions):
  - c_fn ∈ [0.20, 0.35, 0.50, 0.65, 0.80]
  - λ ∈ [0.00, 0.25, 0.50, 0.75, 1.00]
  - threshold_mode ∈ ["exact_free", "exact_balanced_min_recall"]

Protocol:
  - 50 MCCV splits (70 train / 18 val) for selection by Balanced Ordinal Error (MOE_abs), tiebreak F1_macro.
  - Inner 3-fold CV inside training splits for leak-safe risk calculation.
  - 88 LOO folds for final evaluation of winning condition.
"""

import os
import re
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import spacy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"
RESULTS_DIR = ROOT / "experiments" / "exp_18" / "results"
FIGURES_DIR = ROOT / "experiments" / "exp_18" / "reports" / "figures"

CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]

FROZEN_21_VARS = [
    "cli_age", "cli_allergies_count", "cli_bx", "cli_comorbidity_count",
    "cli_cspca", "cli_dre", "cli_fh_binary", "cli_ipss_score", "cli_months",
    "cli_pirads", "cli_psa", "cli_psad", "cli_psav", "cli_vol",
    "vit_bp_diastolic", "vit_bp_systolic", "vit_heart_rate_bpm",
    "vit_height_cm", "vit_smoking_pack_years", "vit_smoking_status",
    "vit_weight_kg",
]

ORD_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}
ORD_NAMES = ["uncertain", "borderline", "clear"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}

# Frozen base model hyperparameters (ConfidenceWeightedKNN)
TABULAR_K = 1
TABULAR_METRIC = "cosine"
TABULAR_WEIGHT = "uniform"

MRI_N_COMPONENTS = 1
MRI_K = 1
MRI_METRIC = "euclidean"
MRI_WEIGHT = "distance"

TEXT_MAX_FEATURES = 2000
TEXT_TFIDF_PARAMS = {
    "ngram_range": (1, 1), "min_df": 1, "max_df": 1.0,
    "use_idf": True, "smooth_idf": True, "norm": "l2",
}
TEXT_K = 3
TEXT_METRIC = "cosine"
TEXT_WEIGHT = "distance"
SPACY_MODEL = "en_core_web_sm"
NEGATION_STOPWORDS = {"no", "not", "without", "never", "neither", "nor", "none"}

INNER_N_SPLITS = 3
TOTAL_MCCV_FOLDS = 50
TOTAL_LOO_FOLDS = 88
EPS = 1e-10

# Decision Risk Grid Parameters
C_FN_VALUES = [0.20, 0.35, 0.50, 0.65, 0.80]
LAMBDA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
THRESHOLD_MODES = ["exact_free", "exact_balanced_min_recall"]


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted KNN (fuzzy variant)
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceWeightedKNN:
    def __init__(self, n_neighbors, metric, use_distance_weight, epsilon=1e-10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_distance_weight = use_distance_weight
        self.epsilon = epsilon

    def fit(self, X, y, conf_weights):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        self.conf_weights = np.array(conf_weights, dtype=np.float64)

    def _distances(self, X):
        X = np.array(X, dtype=np.float64)
        if self.metric == "euclidean":
            from numpy.linalg import norm
            dists = np.zeros((len(X), len(self.X_train)))
            for i in range(len(X)):
                dists[i] = norm(self.X_train - X[i], axis=1)
        elif self.metric == "cosine":
            from numpy.linalg import norm
            X_norm = X / (norm(X, axis=1, keepdims=True) + self.epsilon)
            T_norm = self.X_train / (norm(self.X_train, axis=1, keepdims=True) + self.epsilon)
            dists = 1 - X_norm @ T_norm.T
            dists = np.clip(dists, 0, 2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return dists

    def predict_proba(self, X):
        dists = self._distances(X)
        proba = np.zeros(len(X))
        for i in range(len(X)):
            nn_idx = np.argsort(dists[i])[:self.n_neighbors]
            d_nn = dists[i, nn_idx]
            y_nn = self.y_train[nn_idx]
            c_nn = self.conf_weights[nn_idx]
            if self.use_distance_weight:
                w_dist = 1.0 / np.maximum(d_nn, self.epsilon)
            else:
                w_dist = np.ones_like(d_nn)
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)
        return proba


# ══════════════════════════════════════════════════════════════════════════════
# Text preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_spacy():
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    print(f"  spaCy model: {SPACY_MODEL} v{nlp.meta.get('version', '?')}")
    return nlp


def preprocess_text(texts, nlp):
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            processed.append("")
            continue
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        doc = nlp(text)
        tokens = []
        for token in doc:
            lemma = token.lemma_.strip()
            if not lemma or len(lemma) <= 1:
                continue
            if re.search(r"\d", lemma):
                continue
            if token.like_num:
                continue
            if token.is_stop and token.text not in NEGATION_STOPWORDS:
                continue
            tokens.append(lemma)
        processed.append(" ".join(tokens))
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"    Preprocessed {i+1}/{total} texts")
    return processed


# ══════════════════════════════════════════════════════════════════════════════
# Tabular preprocessing
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
# Decision Risk Math
# ══════════════════════════════════════════════════════════════════════════════

def compute_decision_risk(p_T, p_M, p_X, c_fn, lam):
    c_fp = 1.0 - c_fn
    p_T = np.asarray(p_T, dtype=np.float64)
    p_M = np.asarray(p_M, dtype=np.float64)
    p_X = np.asarray(p_X, dtype=np.float64)

    p_bar = (p_T + p_M + p_X) / 3.0
    probs_stack = np.column_stack([p_T, p_M, p_X])
    sigma = np.std(probs_stack, axis=1, ddof=0)

    risk_no_bx = p_bar * c_fn
    risk_bx = (1.0 - p_bar) * c_fp
    r_unavoidable = np.minimum(risk_no_bx, risk_bx)

    denom = c_fn * c_fp
    r_margen = r_unavoidable / denom

    r_conflicto = 2.0 * sigma

    omega = (1.0 - lam) * r_margen + lam * r_conflicto
    return np.clip(omega, 0.0, 1.0)


def compute_moe_abs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    errors = []
    for c in [0, 1, 2]:
        mask = (y_true == c)
        if np.sum(mask) == 0:
            errors.append(0.0)
        else:
            diff = np.abs(y_pred[mask] - c) / 2.0
            errors.append(np.mean(diff))
    return float(np.mean(errors))


# ══════════════════════════════════════════════════════════════════════════════
# Global Exhaustive 2D Threshold Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def categorize_by_thresholds(omega, tau1, tau2):
    omega = np.asarray(omega, dtype=np.float64)
    preds = np.zeros(len(omega), dtype=int)
    preds[omega < tau1] = 2                  # clear
    preds[(omega >= tau1) & (omega < tau2)] = 1  # borderline
    preds[omega >= tau2] = 0                 # uncertain
    return preds


def fit_exhaustive_thresholds(omega_train, y_train, mode="exact_free", min_preds_per_class=3):
    """Finds optimal thresholds (tau1*, tau2*) minimizing MOE_abs on training set using vectorized NumPy broadcast."""
    omega_sorted = np.sort(np.unique(omega_train))
    if len(omega_sorted) < 2:
        return 0.33, 0.67

    candidates = [0.0]
    for i in range(len(omega_sorted) - 1):
        candidates.append((omega_sorted[i] + omega_sorted[i+1]) / 2.0)
    candidates.append(1.0)
    candidates = np.array(candidates, dtype=np.float64)

    pairs_list = []
    n_cand = len(candidates)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            pairs_list.append((candidates[i], candidates[j]))
    pairs = np.array(pairs_list, dtype=np.float64)  # shape (K, 2)

    tau1 = pairs[:, 0:1]  # shape (K, 1)
    tau2 = pairs[:, 1:2]  # shape (K, 1)

    omega = np.asarray(omega_train, dtype=np.float64).reshape(1, -1)  # shape (1, N)
    y_mat = np.asarray(y_train, dtype=int).reshape(1, -1)              # shape (1, N)

    preds = np.zeros((len(pairs), omega.shape[1]), dtype=int)
    preds[omega < tau1] = 2
    preds[(omega >= tau1) & (omega < tau2)] = 1
    preds[omega >= tau2] = 0

    mask0 = (y_mat == 0)
    mask1 = (y_mat == 1)
    mask2 = (y_mat == 2)

    err0 = np.mean(np.abs(preds[:, mask0[0]] - 0) / 2.0, axis=1) if np.any(mask0) else np.zeros(len(pairs))
    err1 = np.mean(np.abs(preds[:, mask1[0]] - 1) / 2.0, axis=1) if np.any(mask1) else np.zeros(len(pairs))
    err2 = np.mean(np.abs(preds[:, mask2[0]] - 2) / 2.0, axis=1) if np.any(mask2) else np.zeros(len(pairs))

    moe_vec = (err0 + err1 + err2) / 3.0

    if mode == "exact_balanced_min_recall":
        count0 = np.sum(preds == 0, axis=1)
        count1 = np.sum(preds == 1, axis=1)
        count2 = np.sum(preds == 2, axis=1)
        valid_mask = (count0 >= min_preds_per_class) & (count1 >= min_preds_per_class) & (count2 >= min_preds_per_class)
        moe_vec[~valid_mask] = 999.0

    best_idx = np.argmin(moe_vec)
    best_tau1, best_tau2 = float(pairs[best_idx, 0]), float(pairs[best_idx, 1])
    return best_tau1, best_tau2



# ══════════════════════════════════════════════════════════════════════════════
# Outer Fold Pipeline (fit 3 base models, produce p_T, p_M, p_X)
# ══════════════════════════════════════════════════════════════════════════════

def process_outer_fold(train_idx, eval_idx, X_tab_raw, X_mri_raw, preprocessed_texts, y_binary, conf_weights):
    skf = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=42)
    inner_p_T = np.zeros(len(train_idx))
    inner_p_M = np.zeros(len(train_idx))
    inner_p_X = np.zeros(len(train_idx))

    y_train_binary = y_binary[train_idx]

    for in_tr_local, in_val_local in skf.split(train_idx, y_train_binary):
        in_tr_global = train_idx[in_tr_local]
        in_val_global = train_idx[in_val_local]

        # Tabular
        X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(
            X_tab_raw, in_tr_global, CATEGORICAL_COLS
        )
        X_val_tab = transform_features_eval(
            X_tab_raw, in_val_global, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols
        )
        knn_t = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
        knn_t.fit(X_tr_tab.values, y_binary[in_tr_global], conf_weights[in_tr_global])
        inner_p_T[in_val_local] = knn_t.predict_proba(X_val_tab.values)

        # MRI
        pca = PCA(n_components=MRI_N_COMPONENTS, random_state=42)
        X_tr_mri = pca.fit_transform(X_mri_raw[in_tr_global])
        X_val_mri = pca.transform(X_mri_raw[in_val_global])
        knn_m = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, True)
        knn_m.fit(X_tr_mri, y_binary[in_tr_global], conf_weights[in_tr_global])
        inner_p_M[in_val_local] = knn_m.predict_proba(X_val_mri)

        # Text
        tr_texts = [preprocessed_texts[i] for i in in_tr_global]
        val_texts = [preprocessed_texts[i] for i in in_val_global]
        tfidf = TfidfVectorizer(max_features=TEXT_MAX_FEATURES, **TEXT_TFIDF_PARAMS)
        X_tr_txt = tfidf.fit_transform(tr_texts).toarray()
        X_val_txt = tfidf.transform(val_texts).toarray()
        knn_x = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, True)
        knn_x.fit(X_tr_txt, y_binary[in_tr_global], conf_weights[in_tr_global])
        inner_p_X[in_val_local] = knn_x.predict_proba(X_val_txt)

    # Base models on FULL train_idx, predict on eval_idx
    X_tr_tab, d_cols, ohe, scaler, c_kept, n_cols = build_features_train(
        X_tab_raw, train_idx, CATEGORICAL_COLS
    )
    X_ev_tab = transform_features_eval(
        X_tab_raw, eval_idx, CATEGORICAL_COLS, d_cols, ohe, scaler, c_kept, n_cols
    )
    knn_t_full = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
    knn_t_full.fit(X_tr_tab.values, y_binary[train_idx], conf_weights[train_idx])
    eval_p_T = knn_t_full.predict_proba(X_ev_tab.values)

    pca_full = PCA(n_components=MRI_N_COMPONENTS, random_state=42)
    X_tr_mri = pca_full.fit_transform(X_mri_raw[train_idx])
    X_ev_mri = pca_full.transform(X_mri_raw[eval_idx])
    knn_m_full = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, True)
    knn_m_full.fit(X_tr_mri, y_binary[train_idx], conf_weights[train_idx])
    eval_p_M = knn_m_full.predict_proba(X_ev_mri)

    tr_texts = [preprocessed_texts[i] for i in train_idx]
    ev_texts = [preprocessed_texts[i] for i in eval_idx]
    tfidf_full = TfidfVectorizer(max_features=TEXT_MAX_FEATURES, **TEXT_TFIDF_PARAMS)
    X_tr_txt = tfidf_full.fit_transform(tr_texts).toarray()
    X_ev_txt = tfidf_full.transform(ev_texts).toarray()
    knn_x_full = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, True)
    knn_x_full.fit(X_tr_txt, y_binary[train_idx], conf_weights[train_idx])
    eval_p_X = knn_x_full.predict_proba(X_ev_txt)

    return (inner_p_T, inner_p_M, inner_p_X), (eval_p_T, eval_p_M, eval_p_X)


# ══════════════════════════════════════════════════════════════════════════════
# Main Sweep Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("exp_18: Global Exhaustive Threshold Optimization on Decision Risk")
    print("=" * 80)

    # Git commit hash
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

    y_conf_labels = gt_df["target_confidence"].tolist()
    y_conf_ord = np.array([ORD_MAP[c] for c in y_conf_labels], dtype=int)
    y_binary = gt_df["target_biopsy_decision_binary"].values.astype(int)
    conf_weights = np.array([CONFIDENCE_MAP[c] for c in y_conf_labels], dtype=np.float64)

    # Pre-extract raw features
    X_tab_raw = inputs_df[[c for c in inputs_df.columns if c not in ["case_id", "path_hist_bx_gl_tert"] and not c.startswith("mri_emb_") and not c.startswith("txt_")]].copy()
    mri_cols = [f"mri_emb_{i}" for i in range(1024)]
    X_mri_raw = inputs_df[mri_cols].values.astype(np.float64)
    raw_texts = inputs_df["txt_full_prompt_narrative"].tolist()

    nlp = load_spacy()
    print("  Preprocessing narrative text...")
    preprocessed_texts = preprocess_text(raw_texts, nlp)

    # ══════════════════════════════════════════════════════════════════════════
    # Pre-compute Base Model Probabilities for all 50 MCCV & 88 LOO Folds
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("Phase 1: Pre-computing Base Model Probabilities for MCCV (50 splits)...")
    print("─" * 80)

    mccv_base_probs = []
    for split_idx in range(TOTAL_MCCV_FOLDS):
        col = f"mccv_split_{split_idx:02d}"
        split_vals = usable_df[col].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        inner_p, eval_p = process_outer_fold(
            train_idx, val_idx, X_tab_raw, X_mri_raw, preprocessed_texts, y_binary, conf_weights
        )
        mccv_base_probs.append({
            "split_idx": split_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "inner_p": inner_p,
            "eval_p": eval_p,
        })
        if (split_idx + 1) % 10 == 0 or (split_idx + 1) == TOTAL_MCCV_FOLDS:
            print(f"  Processed {split_idx+1}/{TOTAL_MCCV_FOLDS} MCCV splits")

    print("\n" + "─" * 80)
    print("Phase 2: Pre-computing Base Model Probabilities for LOO (88 folds)...")
    print("─" * 80)

    loo_base_probs = []
    for fold_idx in range(TOTAL_LOO_FOLDS):
        split_vals = usable_df["loocv_fold"].values
        val_idx = np.where(split_vals == fold_idx)[0]
        train_idx = np.where(split_vals != fold_idx)[0]

        inner_p, eval_p = process_outer_fold(
            train_idx, val_idx, X_tab_raw, X_mri_raw, preprocessed_texts, y_binary, conf_weights
        )
        loo_base_probs.append({
            "fold_idx": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "inner_p": inner_p,
            "eval_p": eval_p,
        })
        if (fold_idx + 1) % 20 == 0 or (fold_idx + 1) == TOTAL_LOO_FOLDS:
            print(f"  Processed {fold_idx+1}/{TOTAL_LOO_FOLDS} LOO folds")

    # Baseline prediction (always predict "clear" = 2)
    mccv_all_val_y = []
    for bp in mccv_base_probs:
        mccv_all_val_y.extend(y_conf_ord[bp["val_idx"]])
    mccv_all_val_y = np.array(mccv_all_val_y, dtype=int)
    baseline_pred = np.full_like(mccv_all_val_y, fill_value=2)
    baseline_moe_abs = compute_moe_abs(mccv_all_val_y, baseline_pred)
    baseline_f1_macro = f1_score(mccv_all_val_y, baseline_pred, average="macro", zero_division=0)
    print(f"\n  Baseline (Always Clear) MCCV: MOE_abs={baseline_moe_abs:.4f}, F1_macro={baseline_f1_macro:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Grid Search over 50 Exhaustive Threshold Conditions in MCCV
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("Phase 3: Running 50-Condition Exhaustive Threshold Grid Search over MCCV...")
    print("─" * 80)

    scorecard_rows = []
    mccv_predictions_by_config = {}

    config_idx = 0
    total_configs = len(C_FN_VALUES) * len(LAMBDA_VALUES) * len(THRESHOLD_MODES)

    for c_fn in C_FN_VALUES:
        for lam in LAMBDA_VALUES:
            for mode in THRESHOLD_MODES:
                config_idx += 1
                config_name = f"c_fn_{c_fn:.2f}_lambda_{lam:.2f}_mode_{mode}"

                val_moe_list = []
                val_f1_list = []
                valid_struct_count = 0

                all_pred_y = []
                all_true_y = []

                for bp in mccv_base_probs:
                    train_idx = bp["train_idx"]
                    val_idx = bp["val_idx"]

                    tr_p_T, tr_p_M, tr_p_X = bp["inner_p"]
                    ev_p_T, ev_p_M, ev_p_X = bp["eval_p"]

                    omega_train = compute_decision_risk(tr_p_T, tr_p_M, tr_p_X, c_fn, lam)
                    omega_val = compute_decision_risk(ev_p_T, ev_p_M, ev_p_X, c_fn, lam)

                    tau1, tau2 = fit_exhaustive_thresholds(omega_train, y_conf_ord[train_idx], mode=mode)
                    if tau1 < tau2:
                        valid_struct_count += 1

                    pred_val = categorize_by_thresholds(omega_val, tau1, tau2)

                    moe_f = compute_moe_abs(y_conf_ord[val_idx], pred_val)
                    f1_f = f1_score(y_conf_ord[val_idx], pred_val, average="macro", zero_division=0)

                    val_moe_list.append(moe_f)
                    val_f1_list.append(f1_f)

                    all_pred_y.extend(pred_val)
                    all_true_y.extend(y_conf_ord[val_idx])

                all_pred_y = np.array(all_pred_y, dtype=int)
                all_true_y = np.array(all_true_y, dtype=int)

                pooled_moe = compute_moe_abs(all_true_y, all_pred_y)
                pooled_f1 = f1_score(all_true_y, all_pred_y, average="macro", zero_division=0)
                mean_moe = float(np.mean(val_moe_list))
                std_moe = float(np.std(val_moe_list))
                mean_f1 = float(np.mean(val_f1_list))
                valid_rate = float(valid_struct_count / TOTAL_MCCV_FOLDS)

                cm = confusion_matrix(all_true_y, all_pred_y, labels=[0, 1, 2])
                recalls = np.diag(cm) / (np.sum(cm, axis=1) + EPS)
                has_zero_recall = bool(np.any(recalls == 0.0))

                passes_structure = (valid_rate == 1.0)
                passes_moe = (pooled_moe < baseline_moe_abs)
                passes_recall = not has_zero_recall

                row = {
                    "config_name": config_name,
                    "c_fn": c_fn,
                    "c_fp": round(1.0 - c_fn, 2),
                    "lambda": lam,
                    "threshold_mode": mode,
                    "pooled_moe_abs": pooled_moe,
                    "mean_moe_abs": mean_moe,
                    "std_moe_abs": std_moe,
                    "pooled_f1_macro": pooled_f1,
                    "mean_f1_macro": mean_f1,
                    "valid_structure_rate": valid_rate,
                    "has_zero_recall": has_zero_recall,
                    "rec_uncertain": float(recalls[0]),
                    "rec_borderline": float(recalls[1]),
                    "rec_clear": float(recalls[2]),
                    "passes_all_gates": (passes_structure and passes_moe and passes_recall),
                }
                scorecard_rows.append(row)
                mccv_predictions_by_config[config_name] = (all_true_y, all_pred_y)

                if config_idx % 10 == 0 or config_idx == total_configs:
                    print(f"  Config {config_idx:02d}/{total_configs} [{config_name}]: pooled_moe={pooled_moe:.4f}, mean_moe={mean_moe:.4f}, pooled_f1={pooled_f1:.4f}")

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(RESULTS_DIR / "evaluation_scorecard.csv", index=False)
    print(f"\n  Saved evaluation scorecard to {RESULTS_DIR / 'evaluation_scorecard.csv'}")

    # Apply Selection Cascade
    eligible_df = scorecard_df[scorecard_df["passes_all_gates"] == True].copy()
    if eligible_df.empty:
        print("  WARNING: No config passed all gates! Relaxing recall constraint for selection...")
        eligible_df = scorecard_df[
            (scorecard_df["valid_structure_rate"] == 1.0) & (scorecard_df["pooled_moe_abs"] < baseline_moe_abs)
        ].copy()

    if eligible_df.empty:
        print("  CRITICAL: No config beat baseline! Selecting min pooled_moe_abs...")
        eligible_df = scorecard_df[scorecard_df["valid_structure_rate"] == 1.0].copy()

    # Primary: Minimize mean_moe_abs AND pooled_moe_abs; Secondary: Maximize pooled_f1_macro
    eligible_df = eligible_df.sort_values(
        by=["mean_moe_abs", "pooled_moe_abs", "pooled_f1_macro"], ascending=[True, True, False]
    ).reset_index(drop=True)

    selected_row = eligible_df.iloc[0]
    selected_config_name = selected_row["config_name"]
    selected_c_fn = float(selected_row["c_fn"])
    selected_lam = float(selected_row["lambda"])
    selected_mode = selected_row["threshold_mode"]

    print("\n" + "★" * 80)
    print(f"SELECTED MCCV CONFIG: {selected_config_name}")
    print(f"  c_fn: {selected_c_fn}, c_fp: {1.0-selected_c_fn:.2f}, lambda: {selected_lam}, threshold_mode: {selected_mode}")
    print(f"  MCCV Mean MOE_abs: {selected_row['mean_moe_abs']:.4f} ± {selected_row['std_moe_abs']:.4f}")
    print(f"  MCCV Pooled MOE_abs: {selected_row['pooled_moe_abs']:.4f} (vs Baseline {baseline_moe_abs:.4f})")
    print(f"  MCCV Pooled F1_macro: {selected_row['pooled_f1_macro']:.4f}")
    print(f"  Recalls -> unc: {selected_row['rec_uncertain']:.4f}, brd: {selected_row['rec_borderline']:.4f}, clr: {selected_row['rec_clear']:.4f}")
    print("★" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: LOO Final Audit of Selected Config
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print(f"Phase 4: Evaluating Selected Config [{selected_config_name}] on LOO (88 folds)...")
    print("─" * 80)

    loo_pred_y = []
    loo_true_y = []
    loo_per_fold = []

    for bp in loo_base_probs:
        fold_idx = bp["fold_idx"]
        train_idx = bp["train_idx"]
        val_idx = bp["val_idx"]

        tr_p_T, tr_p_M, tr_p_X = bp["inner_p"]
        ev_p_T, ev_p_M, ev_p_X = bp["eval_p"]

        omega_train = compute_decision_risk(tr_p_T, tr_p_M, tr_p_X, selected_c_fn, selected_lam)
        omega_val = compute_decision_risk(ev_p_T, ev_p_M, ev_p_X, selected_c_fn, selected_lam)

        tau1, tau2 = fit_exhaustive_thresholds(omega_train, y_conf_ord[train_idx], mode=selected_mode)
        pred_val = categorize_by_thresholds(omega_val, tau1, tau2)[0]
        true_val = y_conf_ord[val_idx[0]]

        loo_pred_y.append(pred_val)
        loo_true_y.append(true_val)

        loo_per_fold.append({
            "fold_idx": fold_idx,
            "case_id": usable_case_ids[val_idx[0]],
            "true_label": ORD_NAMES[true_val],
            "pred_label": ORD_NAMES[pred_val],
            "true_ord": int(true_val),
            "pred_ord": int(pred_val),
            "omega_val": float(omega_val[0]),
            "tau1": float(tau1),
            "tau2": float(tau2),
            "p_T": float(ev_p_T[0]),
            "p_M": float(ev_p_M[0]),
            "p_X": float(ev_p_X[0]),
        })

    loo_pred_y = np.array(loo_pred_y, dtype=int)
    loo_true_y = np.array(loo_true_y, dtype=int)

    loo_moe_abs = compute_moe_abs(loo_true_y, loo_pred_y)
    loo_f1_macro = f1_score(loo_true_y, loo_pred_y, average="macro", zero_division=0)
    loo_cm = confusion_matrix(loo_true_y, loo_pred_y, labels=[0, 1, 2])
    loo_recalls = np.diag(loo_cm) / (np.sum(loo_cm, axis=1) + EPS)

    print(f"\n  LOO Results [{selected_config_name}]:")
    print(f"    LOO MOE_abs:    {loo_moe_abs:.4f}")
    print(f"    LOO F1_macro:   {loo_f1_macro:.4f}")
    print(f"    LOO Recalls:    unc={loo_recalls[0]:.4f}, brd={loo_recalls[1]:.4f}, clr={loo_recalls[2]:.4f}")

    per_fold_df = pd.DataFrame(loo_per_fold)
    per_fold_df.to_csv(RESULTS_DIR / "per_fold.csv", index=False)

    mccv_true_sel, mccv_pred_sel = mccv_predictions_by_config[selected_config_name]
    mccv_preds_df = pd.DataFrame({"true_ord": mccv_true_sel, "pred_ord": mccv_pred_sel})
    mccv_preds_df.to_csv(RESULTS_DIR / "predictions_mccv.csv", index=False)

    per_fold_df.to_csv(RESULTS_DIR / "predictions_loo.csv", index=False)

    # Confusion Matrices & Figures
    mccv_cm = confusion_matrix(mccv_true_sel, mccv_pred_sel, labels=[0, 1, 2])
    mccv_cm_norm = mccv_cm.astype(float) / (mccv_cm.sum(axis=1, keepdims=True) + EPS)
    loo_cm_norm = loo_cm.astype(float) / (loo_cm.sum(axis=1, keepdims=True) + EPS)

    cm_data = {
        "mccv_confusion_matrix_counts": mccv_cm.tolist(),
        "mccv_confusion_matrix_normalized": mccv_cm_norm.tolist(),
        "loo_confusion_matrix_counts": loo_cm.tolist(),
        "loo_confusion_matrix_normalized": loo_cm_norm.tolist(),
    }
    with open(RESULTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(mccv_cm, annot=True, fmt="d", cmap="Blues", xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, ax=axes[0])
    axes[0].set_title(f"MCCV Pooled Counts ({selected_config_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(mccv_cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, ax=axes[1])
    axes[1].set_title("MCCV Pooled Normalized (Row)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_mccv.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(loo_cm, annot=True, fmt="d", cmap="Greens", xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, ax=axes[0])
    axes[0].set_title(f"LOO Counts ({selected_config_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(loo_cm_norm, annot=True, fmt=".3f", cmap="Greens", xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, ax=axes[1])
    axes[1].set_title("LOO Normalized (Row)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_loo_selected.png", dpi=300)
    plt.close()

    summary_data = {
        "experiment": "exp_18",
        "description": "Global Exhaustive Threshold Optimization on Decision Risk",
        "git_commit": commit_str,
        "n_usable_cohort": n_usable,
        "baseline_majority_clear": {
            "moe_abs": baseline_moe_abs,
            "f1_macro": baseline_f1_macro,
        },
        "selected_config": {
            "config_name": selected_config_name,
            "c_fn": selected_c_fn,
            "c_fp": round(1.0 - selected_c_fn, 2),
            "lambda": selected_lam,
            "threshold_mode": selected_mode,
            "mccv_mean_moe_abs": float(selected_row["mean_moe_abs"]),
            "mccv_std_moe_abs": float(selected_row["std_moe_abs"]),
            "mccv_pooled_moe_abs": float(selected_row["pooled_moe_abs"]),
            "mccv_pooled_f1_macro": float(selected_row["pooled_f1_macro"]),
            "loo_moe_abs": loo_moe_abs,
            "loo_f1_macro": loo_f1_macro,
            "loo_recalls": {
                "uncertain": float(loo_recalls[0]),
                "borderline": float(loo_recalls[1]),
                "clear": float(loo_recalls[2]),
            },
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
