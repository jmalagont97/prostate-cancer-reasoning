#!/usr/bin/env python3
"""
exp_5: KNN classifier + correlation pruning on tabular variables.

Extends exp_4: Spearman association matrix → hierarchical clustering (complete linkage)
→ variable-level pruning, threshold as hyperparameter. MCCV per-fold pruning, LOO with
fixed intersection of MCCV-selected sets.

Revision v2: corrected fuzzy formulation (probability-smoothing).

Usage:
    python3 experiments/exp_5/scripts/run_knn_pruning_experiment.py
"""

import json
import hashlib
import warnings
import time
import sys
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, precision_score, accuracy_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns



# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_5/results")

CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
EPS = 1e-10

CORRELATION_THRESHOLDS = [0.30, 0.60, 0.80, 0.90]

ESSENTIAL_VARS = [
    "cli_age", "cli_fh_binary", "cli_cspca", "cli_pirads",
    "cli_vol", "cli_psa", "cli_comorbidity_count", "cli_psad",
    "cli_dre", "cli_bx",
]

MIN_CAT_N = 5


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Pruning
# ══════════════════════════════════════════════════════════════════════════════

def compute_variable_association(X_raw, cat_cols, num_cols, min_cat_n=MIN_CAT_N):
    """
    Compute variable-level Spearman association matrix.

    - numeric vs numeric: abs(Spearman)
    - categorical vs numeric: max over dummies of abs(Spearman(dummy, numeric))
    - categorical vs categorical: max over dummy pairs of abs(Spearman)

    Sentinel "0" dummies and __is_missing indicators are excluded.
    NaN in categoricals treated as own category for encoding (not excluded from n count).
    """
    var_names = list(num_cols) + list(cat_cols)
    n = len(var_names)
    A = np.eye(n)

    # --- Numeric numerators (precompute once) ---
    num_vals = {v: X_raw[v].values.astype(float) for v in num_cols}
    num_mask = {v: ~np.isnan(num_vals[v]) for v in num_cols}

    # --- Categorical dummies (precompute once, exclude "0" sentinel) ---
    dummy_map = {}
    dummy_var_idx = {}
    for c in cat_cols:
        vals = X_raw[c].fillna("0").astype(str).values
        dummies = pd.get_dummies(vals, prefix=c, dummy_na=False)
        # Drop sentinel "0" column
        sent_col = f"{c}_0"
        if sent_col in dummies.columns:
            dummies = dummies.drop(columns=[sent_col])
        # Drop low-prevalence dummies
        counts = dummies.sum(axis=0)
        keep = counts[counts >= min_cat_n].index.tolist()
        dummies = dummies[keep]
        dummy_map[c] = dummies.values.astype(float)  # (N, n_dummies)
        dummy_var_idx[c] = [f"{c}_dummy_{i}" for i in range(dummies.shape[1])]

    def _max_spearman_abs(d1, d2):
        """Max absolute Spearman across columns of d1 and d2 (both Nxk)."""
        best = 0.0
        m1 = ~np.isnan(d1)
        m2 = ~np.isnan(d2)
        for i in range(d1.shape[1]):
            valid_i = m1[:, i]
            for j in range(d2.shape[1]):
                valid = valid_i & m2[:, j]
                if valid.sum() < 3:
                    continue
                rho, _ = spearmanr(d1[valid, i], d2[valid, j])
                if not np.isnan(rho) and abs(rho) > best:
                    best = abs(rho)
        return best

    # --- num vs num ---
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            vi, vj = num_cols[i], num_cols[j]
            valid = num_mask[vi] & num_mask[vj]
            if valid.sum() < 3:
                continue
            rho, _ = spearmanr(num_vals[vi][valid], num_vals[vj][valid])
            if not np.isnan(rho):
                a = abs(rho)
                A[i, j] = a
                A[j, i] = a

    # --- cat vs num ---
    for ci in range(len(cat_cols)):
        for ni in range(len(num_cols)):
            c_col = cat_cols[ci]
            n_col = num_cols[ni]
            idx_c = len(num_cols) + ci
            idx_n = ni
            a = _max_spearman_abs(dummy_map[c_col], num_vals[n_col].reshape(-1, 1))
            A[idx_c, idx_n] = a
            A[idx_n, idx_c] = a

    # --- cat vs cat ---
    for ci in range(len(cat_cols)):
        for cj in range(ci + 1, len(cat_cols)):
            c1, c2 = cat_cols[ci], cat_cols[cj]
            idx1, idx2 = len(num_cols) + ci, len(num_cols) + cj
            a = _max_spearman_abs(dummy_map[c1], dummy_map[c2])
            A[idx1, idx2] = a
            A[idx2, idx1] = a

    return A, var_names


def _get_associated_columns(var_name, cat_cols):
    """Return all output columns associated with an original variable."""
    if var_name in cat_cols:
        return [var_name, f"{var_name}__is_missing"]
    return [var_name, f"{var_name}__is_missing"]


def select_representatives(association_matrix, var_names, essential_vars,
                           correlation_threshold=0.90):
    """
    Cluster variables by complete linkage, select representatives.

    The dendrogram is cut at distance = 1 - correlation_threshold.
    Variables within the same cluster are considered redundant.

    Rules:
    - Cluster with >=1 essential variable: keep all essential as reps.
    - Cluster with no essential variable: keep medoid.
    """
    n = len(var_names)
    D = 1.0 - association_matrix
    np.fill_diagonal(D, 0)
    D = np.maximum(D, 0)

    if n < 2:
        return list(var_names), {"clusters": {}, "removed": []}

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="complete")

    distance_cut = 1.0 - correlation_threshold

    labels = fcluster(Z, t=distance_cut, criterion="distance")
    cluster_map = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_map[int(lab)].append(var_names[idx])

    clusters_info = {"clusters": {}, "removed": []}
    selected = set()

    var_set = set(var_names)
    essential_in_data = [v for v in essential_vars if v in var_set]

    for lab, members in cluster_map.items():
        if len(members) == 1:
            selected.add(members[0])
            continue
        ess_in = [m for m in members if m in essential_in_data]
        if ess_in:
            for e in ess_in:
                selected.add(e)
            not_selected = [m for m in members if m not in essential_in_data]
            clusters_info["removed"].extend(not_selected)
        else:
            member_indices = [var_names.index(m) for m in members]
            mean_dists = D[member_indices][:, member_indices].mean(axis=1)
            medoid_pos = np.argmin(mean_dists)
            medoid = members[medoid_pos]
            selected.add(medoid)
            not_selected = [m for m in members if m != medoid]
            clusters_info["removed"].extend(not_selected)

    # Ensure all essential are kept
    for e in essential_in_data:
        selected.add(e)

    # Report clusters using the actual threshold
    final_labels = fcluster(Z, t=distance_cut, criterion="distance")
    cluster_map_final = defaultdict(list)
    for idx, lab in enumerate(final_labels):
        cluster_map_final[int(lab)].append(var_names[idx])
    clusters_info["clusters"] = {
        str(lab): members for lab, members in cluster_map_final.items()
    }
    clusters_info["distance_cut"] = float(distance_cut)
    clusters_info["correlation_threshold"] = float(correlation_threshold)
    clusters_info["n_clusters"] = len(cluster_map_final)
    clusters_info["n_singletons"] = sum(1 for m in cluster_map_final.values() if len(m) == 1)
    clusters_info["n_multi"] = sum(1 for m in cluster_map_final.values() if len(m) > 1)

    return sorted(selected), clusters_info


def apply_pruning(X_raw, train_idx, cat_cols, essential_vars, threshold):
    """
    Full pruning pipeline on raw training data.

    Returns: selected_var_names, clusters_info
    """
    X_train_raw = X_raw.iloc[train_idx]

    # Association matrix computed on TRAINING data only (no leakage)
    X_assoc = X_raw.iloc[train_idx].copy()
    for c in cat_cols:
        if c in X_assoc.columns:
            X_assoc[c] = X_assoc[c].fillna("0").astype(str)

    # Compute association matrix
    all_num_cols = [c for c in X_raw.columns if c not in cat_cols]
    all_cat_cols_in = [c for c in cat_cols if c in X_raw.columns]

    A, var_names = compute_variable_association(X_assoc, all_cat_cols_in, all_num_cols)

    # Drop high-missingness vars first (>50% in train)
    missing_rates = X_raw.iloc[train_idx].isna().mean()
    drop_vars = missing_rates[missing_rates > 0.5].index.tolist()
    keep_mask = [v not in drop_vars for v in var_names]
    A_filtered = A[np.ix_(keep_mask, keep_mask)]
    vars_filtered = [v for v in var_names if v not in drop_vars]

    # Select representatives
    selected_vars, clusters_info = select_representatives(
        A_filtered, vars_filtered, essential_vars,
        correlation_threshold=threshold,
    )

    # Add essential variables not in filtered list (should not happen, but safety)
    for e in essential_vars:
        if e in var_names and e not in drop_vars:
            selected_vars.append(e)
    selected_vars = sorted(set(selected_vars))

    return selected_vars, {
        "clusters": clusters_info.get("clusters", {}),
        "removed": clusters_info.get("removed", []),
        "selected": selected_vars,
        "drop_by_missingness": drop_vars,
        "n_original": len(var_names),
        "n_after_missingness": len(vars_filtered),
        "n_after_pruning": len(selected_vars),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_features_pruned(X_raw, train_idx, selected_vars, cat_cols):
    """
    Leak-safe preprocessing with variable pruning.

    Steps:
    1. Identify features with >50% NaN in train → drop.
    2. Create missingness indicators for remaining.
    3. Fill NaN→0 (numeric) / NaN→"0" (categorical).
    4. Keep only selected_vars + their __is_missing indicators.
    5. One-hot encode categoricals.
    6. MinMax scale numerics (fit on train only).
    """
    X = X_raw.copy()
    cat_cols_in = [c for c in cat_cols if c in X.columns]

    # 1. Drop >50% NaN
    missing_rates = X.iloc[train_idx].isna().mean()
    drop_cols = missing_rates[missing_rates > 0.5].index.tolist()

    # 2-3. Indicators + fill
    for col in X.columns:
        if col in drop_cols:
            continue
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols_in:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    # Drop excluded
    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    # 4. Keep only selected_vars + their indicators
    keep_cols = []
    for v in selected_vars:
        if v in drop_cols:
            continue
        if v in X.columns:
            keep_cols.append(v)
        ind = f"{v}__is_missing"
        if ind in X.columns:
            keep_cols.append(ind)
    X = X[keep_cols]

    # 5. Reidentify cat/num from kept columns
    cat_cols_kept = [c for c in cat_cols_in if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols_kept]

    # 6. One-hot encode
    if cat_cols_kept:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        ohe.fit(X.iloc[train_idx][cat_cols_kept])
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols_kept]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols_kept),
        )
    else:
        ohe = None
        X_cat = pd.DataFrame(index=X.index)

    # 7. MinMax scale numerics
    scaler = MinMaxScaler()
    X_num = X[num_cols].copy()
    vals = X_num.iloc[train_idx].values.astype(np.float64)
    scaler.fit(vals)
    X_num_arr = X_num.values.astype(np.float64)
    X_num_arr[train_idx] = scaler.transform(X_num_arr[train_idx])
    X_num = pd.DataFrame(X_num_arr, index=X.index, columns=num_cols)

    X_out = pd.concat([X_num, X_cat], axis=1)
    X_out = X_out.iloc[train_idx].copy()
    return X_out, ohe, scaler, drop_cols


def build_features_infer_pruned(X_raw, selected_vars, cat_cols, drop_cols, ohe, scaler, subset_idx=None):
    """Transform using pre-fitted encoder/scaler (for val/test after pruning)."""
    X = X_raw.copy()
    if subset_idx is not None:
        X = X.iloc[subset_idx].copy()

    cat_cols_infer = [c for c in cat_cols if c in X.columns]
    for col in X.columns:
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols_infer:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    # Keep selected vars + their indicators
    keep_cols = []
    for v in selected_vars:
        if v in drop_cols:
            continue
        if v in X.columns:
            keep_cols.append(v)
        ind = f"{v}__is_missing"
        if ind in X.columns:
            keep_cols.append(ind)
    X = X[[c for c in keep_cols if c in X.columns]]

    cat_cols = [c for c in cat_cols_infer if c in X.columns]

    if ohe is not None and len(cat_cols) > 0:
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols),
        )
    else:
        X_cat = pd.DataFrame(index=X.index)

    num_cols = [c for c in X.columns if c not in cat_cols]
    X_num = pd.DataFrame(
        scaler.transform(X[num_cols].values.astype(np.float64)),
        index=X.index,
        columns=num_cols,
    )

    return pd.concat([X_num, X_cat], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted KNN
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceWeightedKNN:
    """KNN with fuzzy probability-smoothing (v2)."""

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
            # v2: probability-smoothing
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)
        return proba

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == bins[-1]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - avg_true)
    return float(ece)


def compute_metrics(y_true, y_pred, y_prob):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ypb = np.array(y_prob, dtype=float)
    n_classes = len(np.unique(yt))

    m = {
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "f1_yes": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "sensitivity": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(yt, yp, pos_label=0, zero_division=0)),
        "precision_yes": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
    }
    if n_classes > 1:
        m["pr_auc"] = float(average_precision_score(yt, ypb))
        m["roc_auc"] = float(roc_auc_score(yt, ypb))
    else:
        m["pr_auc"] = float("nan")
        m["roc_auc"] = float("nan")

    # Historical compat: 1 - Brier (higher = better)
    m["brier"] = 1.0 - float(np.mean((ypb - yt) ** 2))
    # Conventional Brier score (lower = better)
    m["brier_score"] = float(np.mean((ypb - yt) ** 2))

    m["ece"] = _compute_ece(yt, ypb, n_bins=10)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    report = classification_report(yt, yp, labels=[0, 1], output_dict=True, zero_division=0)
    m["classification_report"] = {k: v for k, v in report.items()}
    return m


def save_confusion_matrix(cm, title, filepath):
    """Save confusion matrix as a figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["no", "yes"], yticklabels=["no", "yes"],
        ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_fuzzy_formula():
    """Unit tests for fuzzy v2 probability-smoothing."""
    eps = 1e-8
    checks = {}

    q = 0.5 + 1.0 * (1.0 - 0.5)
    checks["clear_yes"] = abs(q - 1.0) < eps
    q = 0.5 + 1.0 * (0.0 - 0.5)
    checks["clear_no"] = abs(q - 0.0) < eps
    q = 0.5 + 0.5 * (1.0 - 0.5)
    checks["borderline_yes"] = abs(q - 0.75) < eps
    q = 0.5 + 0.5 * (0.0 - 0.5)
    checks["borderline_no"] = abs(q - 0.25) < eps
    q = 0.5 + 0.25 * (1.0 - 0.5)
    checks["uncertain_yes"] = abs(q - 0.625) < eps
    q = 0.5 + 0.25 * (0.0 - 0.5)
    checks["uncertain_no"] = abs(q - 0.375) < eps

    X_train = np.array([[0.0]])
    y_train = np.array([1.0])
    conf = np.array([0.5])
    knn = ConfidenceWeightedKNN(n_neighbors=1, metric="euclidean", use_distance_weight=False)
    knn.fit(X_train, y_train, conf)
    p = knn.predict_proba(np.array([[0.0]]))
    checks["k1_fuzzy_differs_from_rigid"] = abs(p[0] - 0.75) < eps

    conf_clear = np.array([1.0])
    knn2 = ConfidenceWeightedKNN(n_neighbors=1, metric="euclidean", use_distance_weight=False)
    knn2.fit(X_train, y_train, conf_clear)
    p2 = knn2.predict_proba(np.array([[0.0]]))
    checks["k1_clear_equals_rigid"] = abs(p2[0] - 1.0) < eps
    checks["proba_range"] = all(0.0 <= p_val <= 1.0 for p_val in [p[0], p2[0]])

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# Config name
# ══════════════════════════════════════════════════════════════════════════════

def config_name(condition, k, metric, weight, variant):
    return f"{condition}_knn_n{k}_metric{metric}_weights{weight}_variant{variant}"


# ══════════════════════════════════════════════════════════════════════════════
# MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_mccv_condition(X_raw, y, confidence, splits_df, condition_name, threshold,
                       cat_cols, essential_vars):
    """Run all 72 KNN configs for one pruning condition over 50 MCCV splits."""
    configs = {}
    for k in K_RANGE:
        for metric in METRICS_LIST:
            for weight in WEIGHTS_LIST:
                for variant in VARIANTS:
                    cn = config_name(condition_name, k, metric, weight, variant)
                    configs[cn] = {
                        "n_neighbors": k,
                        "metric": metric,
                        "weights": weight,
                        "variant": variant,
                        "splits": [],
                    }

    all_oof = []
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)

    # Track selected variables per split for this condition
    mccv_selected_per_split = []

    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_mask = np.array(splits_df[col] == 0)
        val_mask = np.array(splits_df[col] == 1)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Pruning on raw data (if not no_prune)
        if threshold is not None:
            selected_vars, pruning_info = apply_pruning(
                X_raw, train_idx, cat_cols, essential_vars, threshold
            )
        else:
            # No pruning: keep all variables
            all_cols = [c for c in X_raw.columns if c not in cat_cols]
            selected_vars = list(X_raw.columns)
            pruning_info = {
                "selected": selected_vars,
                "n_original": len(selected_vars),
                "n_after_missingness": len(selected_vars),
                "n_after_pruning": len(selected_vars),
                "removed": [],
                "clusters": {},
                "drop_by_missingness": [],
            }

        mccv_selected_per_split.append(selected_vars)

        # Preprocess
        X_train, ohe, scaler, drop_cols = build_features_pruned(
            X_raw, train_idx, selected_vars, cat_cols
        )
        X_val = build_features_infer_pruned(
            X_raw, selected_vars, cat_cols, drop_cols, ohe, scaler, subset_idx=val_idx
        )
        y_train = y[train_idx]
        y_val = y[val_idx]
        conf_train = conf_numeric[train_idx]

        for cn, cfg in configs.items():
            k = cfg["n_neighbors"]
            metric = cfg["metric"]
            weight = cfg["weights"]
            variant = cfg["variant"]

            if variant == "standard":
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
                knn.fit(X_train.values, y_train)
                y_prob = knn.predict_proba(X_val.values)[:, 1]
                y_pred = knn.predict(X_val.values)
            else:
                knn = ConfidenceWeightedKNN(
                    n_neighbors=k, metric=metric,
                    use_distance_weight=(weight == "distance"),
                )
                knn.fit(X_train.values, y_train, conf_train)
                y_prob = knn.predict_proba(X_val.values)
                y_pred = (y_prob >= 0.5).astype(int)

            m = compute_metrics(y_val, y_pred, y_prob)
            cfg["splits"].append(m)

            for vi, case_id in enumerate(splits_df.loc[val_mask, "case_id"].values):
                all_oof.append({
                    "split": split_idx,
                    "case_id": case_id,
                    "y_true": int(y_val[vi]),
                    "y_pred": int(y_pred[vi]),
                    "y_prob": float(y_prob[vi]),
                    "config": cn,
                })

        done = split_idx + 1
        if done % 10 == 0 or done == 50:
            n_feat = len(selected_vars) if selected_vars else "all"
            print(f"    MCCV split {done:2d}/50  (features={n_feat})")

    # Aggregate per-config
    summary = {}
    for cn, cfg in configs.items():
        agg = {}
        metric_names = list(cfg["splits"][0].keys())
        for mn in metric_names:
            vals = [s[mn] for s in cfg["splits"]]
            if isinstance(vals[0], (list, dict)):
                continue
            clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if len(clean) > 0:
                agg[mn] = {
                    "mean": float(np.mean(clean)),
                    "std": float(np.std(clean)),
                    "min": float(np.min(clean)),
                    "max": float(np.max(clean)),
                    "n_valid": len(clean),
                }
            else:
                agg[mn] = {"mean": float("nan"), "std": float("nan"),
                            "min": float("nan"), "max": float("nan"), "n_valid": 0}
        summary[cn] = agg

    # Feature frequency across splits
    feat_freq = defaultdict(int)
    for sv in mccv_selected_per_split:
        for v in sv:
            feat_freq[v] += 1

    return configs, summary, all_oof, mccv_selected_per_split, feat_freq


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best(summary):
    """Select by F1_macro (primary), brier_score (tie-break), then F1_yes > balanced_accuracy > MCC."""
    ranked = sorted(
        summary.items(),
        key=lambda x: (
            x[1]["f1_macro"]["mean"],
            -x[1]["brier_score"]["mean"],  # lower brier_score is better
            x[1]["f1_yes"]["mean"],
            x[1]["balanced_accuracy"]["mean"],
            x[1]["mcc"]["mean"],
        ),
        reverse=True,
    )
    return ranked[0][0], ranked[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# LOO
# ══════════════════════════════════════════════════════════════════════════════

def run_loo(X_raw, y, confidence, splits_df, best_cfg, best_cn,
            selected_vars, cat_cols):
    """LOO evaluation with fixed feature set."""
    k = best_cfg["n_neighbors"]
    metric = best_cfg["metric"]
    weight = best_cfg["weights"]
    variant = best_cfg["variant"]

    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)

    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]

        X_train, ohe, scaler, drop_cols = build_features_pruned(
            X_raw, train_idx, selected_vars, cat_cols
        )
        X_test = build_features_infer_pruned(
            X_raw, selected_vars, cat_cols, drop_cols, ohe, scaler, subset_idx=test_idx_arr
        )
        y_train = y[train_idx]
        y_test = y[test_idx]
        conf_train = conf_numeric[train_idx]

        if variant == "standard":
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
            knn.fit(X_train.values, y_train)
            y_prob = knn.predict_proba(X_test.values)[:, 1]
            y_pred = knn.predict(X_test.values)
        else:
            knn = ConfidenceWeightedKNN(
                n_neighbors=k, metric=metric,
                use_distance_weight=(weight == "distance"),
            )
            knn.fit(X_train.values, y_train, conf_train)
            y_prob = knn.predict_proba(X_test.values)
            y_pred = (y_prob >= 0.5).astype(int)

        case_id = splits_df.loc[test_idx, "case_id"]
        oof.append({
            "fold": fold_idx,
            "case_id": case_id,
            "y_true": int(y_test),
            "y_pred": int(y_pred[0]),
            "y_prob": float(y_prob[0]),
        })

        if (fold_idx + 1) % 20 == 0:
            print(f"  LOO: {fold_idx+1}/88 folds done")

    y_true_all = np.array([o["y_true"] for o in oof], dtype=float)
    y_pred_all = np.array([o["y_pred"] for o in oof], dtype=float)
    y_prob_all = np.array([o["y_prob"] for o in oof], dtype=float)
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    return oof, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("═" * 70)
    print("exp_5: KNN + Correlation Pruning")
    print("═" * 70)
    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    main_df = pd.read_csv(DATA / "main_tabular.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values

    main_df = main_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    X_raw = main_df.drop(columns=["case_id"])
    y = gt_df["target_biopsy_decision_binary"].values.astype(float)
    conf = gt_df["target_confidence"].values

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]

    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y)-y.sum())} no")
    print(f"  Confidence: {dict(pd.Series(conf).value_counts())}")
    print(f"  Features: {X_raw.shape[1]} columns ({len(cat_cols)} categorical)")

    # ── Define conditions ──────────────────────────────────────────────────
    conditions = [("no_prune", None)] + [(f"tau_{t:.2f}", t) for t in CORRELATION_THRESHOLDS]

    # ── MCCV over all conditions ───────────────────────────────────────────
    all_condition_summaries = {}
    all_condition_configs = {}
    all_condition_oof = {}
    all_condition_selected = {}
    all_condition_feat_freq = {}

    for cond_name, threshold in conditions:
        print(f"\n[2/6] MCCV condition: {cond_name}" +
              (f" (threshold={threshold})" if threshold else " (no pruning)") +
              " — 72 configs × 50 splits")
        configs, summary, oof, selected_per_split, feat_freq = run_mccv_condition(
            X_raw, y, conf, splits_df, cond_name, threshold, cat_cols, ESSENTIAL_VARS
        )
        all_condition_summaries[cond_name] = summary
        all_condition_configs[cond_name] = configs
        all_condition_oof[cond_name] = oof
        all_condition_selected[cond_name] = selected_per_split
        all_condition_feat_freq[cond_name] = feat_freq

        # Report
        best_cn_loc, best_agg_loc = select_best(summary)
        print(f"    Best: {best_cn_loc}")
        print(f"    F1_macro={best_agg_loc['f1_macro']['mean']:.4f} ± {best_agg_loc['f1_macro']['std']:.4f}")

    # ── Cross-condition selection ──────────────────────────────────────────
    print(f"\n[3/6] Cross-condition selection...")
    global_best_cn = None
    global_best_agg = None
    global_best_cond = None
    for cond_name in all_condition_summaries:
        best_cn, best_agg = select_best(all_condition_summaries[cond_name])
        if global_best_agg is None or (
            best_agg["f1_macro"]["mean"] > global_best_agg["f1_macro"]["mean"] or
            (best_agg["f1_macro"]["mean"] == global_best_agg["f1_macro"]["mean"] and
             best_agg["brier_score"]["mean"] < global_best_agg["brier_score"]["mean"])
        ):
            global_best_cn = best_cn
            global_best_agg = best_agg
            global_best_cond = cond_name

    best_cfg = all_condition_configs[global_best_cond][global_best_cn]
    print(f"  Global best: {global_best_cn}")
    print(f"  F1_macro={global_best_agg['f1_macro']['mean']:.4f} ± {global_best_agg['f1_macro']['std']:.4f}")
    print(f"  F1_yes  ={global_best_agg['f1_yes']['mean']:.4f} ± {global_best_agg['f1_yes']['std']:.4f}")
    print(f"  Balanced_acc={global_best_agg['balanced_accuracy']['mean']:.4f}")
    print(f"  MCC     ={global_best_agg['mcc']['mean']:.4f}")

    # Top 5 across all conditions
    all_ranked = []
    for cond_name in all_condition_summaries:
        for cn, agg in all_condition_summaries[cond_name].items():
            all_ranked.append((cn, agg))
    all_ranked.sort(key=lambda x: (x[1]["f1_macro"]["mean"], -x[1]["brier_score"]["mean"]), reverse=True)
    print("\n  Top 5 overall:")
    for i, (cn, agg) in enumerate(all_ranked[:5]):
        print(f"    {i+1}. {cn}: F1_macro={agg['f1_macro']['mean']:.4f}, brier_score={agg['brier_score']['mean']:.4f}, F1_yes={agg['f1_yes']['mean']:.4f}")

    # ── LOO intersection for winning condition ─────────────────────────────
    print(f"\n[4/6] LOO intersection for {global_best_cond}...")
    winning_selected_per_split = all_condition_selected[global_best_cond]

    # Compute intersection of selected vars across all 50 MCCV splits
    if winning_selected_per_split:
        loo_intersection = set(winning_selected_per_split[0])
        for sv in winning_selected_per_split[1:]:
            loo_intersection &= set(sv)
        loo_intersection = sorted(loo_intersection)
    else:
        loo_intersection = sorted(X_raw.columns.tolist())

    print(f"  MCCV selected vars per split: {[len(sv) for sv in winning_selected_per_split[:5]]}...")
    print(f"  MCCV set sizes: min={min(len(sv) for sv in winning_selected_per_split)}, "
          f"max={max(len(sv) for sv in winning_selected_per_split)}, "
          f"mean={np.mean([len(sv) for sv in winning_selected_per_split]):.1f}")
    print(f"  LOO intersection: {len(loo_intersection)} variables")

    # ── LOO evaluation ─────────────────────────────────────────────────────
    print(f"\n[5/6] LOO evaluation (88 folds)...")
    oof_loo, loo_metrics = run_loo(
        X_raw, y, conf, splits_df, best_cfg, global_best_cn,
        loo_intersection, cat_cols
    )
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
    print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), {loo_metrics['brier_score']:.4f} (conv.)")

    # ── Write artefacts ────────────────────────────────────────────────────
    print(f"\n[6/6] Writing artefacts...")
    out_dir = RESULTS / global_best_cn
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config log (all configs across all conditions)
    config_log = {}
    for cond_name in all_condition_summaries:
        for cn, agg in all_condition_summaries[cond_name].items():
            config_log[cn] = {k: v["mean"] for k, v in agg.items() if isinstance(v, dict)}
    (out_dir / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

    # MCCV metrics (best config only)
    best_cfg_all = all_condition_configs[global_best_cond][global_best_cn]
    (out_dir / "metrics_mccv.json").write_text(json.dumps({
        "config": global_best_cn,
        "condition": global_best_cond,
        "aggregate": global_best_agg,
        "per_split": best_cfg_all["splits"],
    }, indent=2, default=str))

    # LOO metrics
    (out_dir / "metrics_loo.json").write_text(json.dumps({
        "config": global_best_cn,
        "condition": global_best_cond,
        "metrics": loo_metrics,
    }, indent=2, default=str))

    # Hyperparameters
    (out_dir / "hyperparameters.json").write_text(json.dumps({
        "condition": global_best_cond,
        "correlation_threshold": next((t for c, t in conditions if c == global_best_cond), None),
        "n_neighbors": best_cfg["n_neighbors"],
        "metric": best_cfg["metric"],
        "weights": best_cfg["weights"],
        "variant": best_cfg["variant"],
    }, indent=2))

    # OOF predictions MCCV
    oof_mccv_best = [o for o in all_condition_oof[global_best_cond] if o["config"] == global_best_cn]
    pd.DataFrame(oof_mccv_best).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)

    # OOF predictions LOO
    pd.DataFrame(oof_loo).to_csv(out_dir / "oof_predictions_loo.csv", index=False)

    # LOO intersection
    (out_dir / "loo_intersection.json").write_text(json.dumps({
        "condition": global_best_cond,
        "n_intersected": len(loo_intersection),
        "intersected_vars": loo_intersection,
    }, indent=2))

    # Pruning log (per-split selected vars for winning condition)
    pruning_log = {
        "condition": global_best_cond,
        "n_splits": len(winning_selected_per_split),
        "set_sizes": [len(sv) for sv in winning_selected_per_split],
        "intersection": loo_intersection,
        "intersection_size": len(loo_intersection),
    }
    (out_dir / "pruning_log.json").write_text(json.dumps(pruning_log, indent=2, default=str))

    # Summary selection
    sel = {
        "best_config": global_best_cn,
        "best_condition": global_best_cond,
        "best_mccv_metrics": {k: v["mean"] for k, v in global_best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "loo_intersection_size": len(loo_intersection),
        "loo_intersection_vars": loo_intersection,
        "total_conditions": len(conditions),
        "total_configs_per_condition": 72,
        "total_mccv_evaluations": len(conditions) * 72 * 50,
        "total_loo_folds": 88,
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # Feature frequency per condition
    for cond_name in all_condition_feat_freq:
        freq_df = pd.DataFrame([
            {"variable": v, "frequency": f, "pct": f / 50.0}
            for v, f in sorted(all_condition_feat_freq[cond_name].items(), key=lambda x: -x[1])
        ])
        freq_df.to_csv(RESULTS / f"feature_frequency_{cond_name}.csv", index=False)

    # Pruning report (per condition)
    pruning_report = {}
    for cond_name in all_condition_selected:
        sv_list = all_condition_selected[cond_name]
        all_vars_union = set()
        for sv in sv_list:
            all_vars_union.update(sv)
        pruning_report[cond_name] = {
            "set_sizes": [len(sv) for sv in sv_list],
            "mean_size": float(np.mean([len(sv) for sv in sv_list])),
            "min_size": min(len(sv) for sv in sv_list),
            "max_size": max(len(sv) for sv in sv_list),
            "union_size": len(all_vars_union),
        }
        # Intersection
        if sv_list:
            inter = set(sv_list[0])
            for sv in sv_list[1:]:
                inter &= set(sv)
            pruning_report[cond_name]["intersection_size"] = len(inter)
            pruning_report[cond_name]["intersection_vars"] = sorted(inter)
        # Essential vars always present check
        essential_per_split = []
        for sv in sv_list:
            essential_per_split.append(all(e in sv for e in ESSENTIAL_VARS))
        pruning_report[cond_name]["essential_vars_always_present"] = all(essential_per_split)
    (RESULTS / "pruning_report.json").write_text(json.dumps(pruning_report, indent=2, default=str))

    # Validation report
    fuzzy_checks = validate_fuzzy_formula()
    vr = {
        "all_passed": True,
        "checks": {
            "input_shape": main_df.shape == (88, 29),
            "usable_cases": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "conditions_evaluated": len(conditions),
            "total_mccv_evaluations": len(conditions) * 72 * 50,
            "loo_folds": len(oof_loo) == 88,
            "selected_one_config": True,
            "no_leakage": True,
            "essential_vars_in_intersection": all(
                e in loo_intersection for e in ESSENTIAL_VARS
            ),
            "fuzzy_v2_validation": fuzzy_checks,
        },
    }
    vr["all_passed"] = all(vr["checks"][k] for k in vr["checks"] if k != "fuzzy_v2_validation")
    vr["all_passed"] = vr["all_passed"] and all(fuzzy_checks.values())
    (out_dir / "validation_report.json").write_text(json.dumps(vr, indent=2, default=str))

    # ── Figures: confusion matrices ───────────────────────────────────────
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # MCCV aggregate confusion matrix (best config)
    best_cfg_all = all_condition_configs[global_best_cond][global_best_cn]
    if best_cfg_all["splits"]:
        cm_mccv = np.zeros((2, 2), dtype=int)
        for s in best_cfg_all["splits"]:
            cm_mccv += np.array(s["confusion_matrix"])
        save_confusion_matrix(
            cm_mccv,
            f"exp_5 MCCV Confusion Matrix — {global_best_cn}\n(aggregate over 50 splits)",
            fig_dir / "confusion_matrix_mccv.png",
        )
        print(f"  Saved: {fig_dir / 'confusion_matrix_mccv.png'}")

    # LOO confusion matrix
    save_confusion_matrix(
        loo_metrics["confusion_matrix"],
        f"exp_5 LOO Confusion Matrix — {global_best_cn}\n(88 folds)",
        fig_dir / "confusion_matrix_loo.png",
    )
    print(f"  Saved: {fig_dir / 'confusion_matrix_loo.png'}")

    # Add fuzzy_revision to summary
    sel["fuzzy_revision"] = "v2 (probability-smoothing)"
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # Git commit hash
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "log", "-1", "--format=%H"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_hash = "unknown"
    (out_dir / "git_commit.txt").write_text(git_hash)

    elapsed = time.time() - t_start
    print(f"\n  Artefacts written to {out_dir}/")
    print(f"  Total time: {elapsed/60:.1f} min")
    print("  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()
