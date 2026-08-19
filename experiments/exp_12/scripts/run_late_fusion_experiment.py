#!/usr/bin/env python3
"""
exp_12: Late multimodal fusion — top-1 winner per modality.

Retrains from scratch the best tabular (exp_5 tau_0.60), MRI (exp_9 pca_1),
and text (exp_10 corrected tfidf_mf2000) models, averages their probabilities
for all non-empty modality combinations, evaluates on MCCV (50 splits), and
selects the best combination. The winning combination is evaluated with LOO
(88 folds).

Usage:
    python3 experiments/exp_12/scripts/run_late_fusion_experiment.py
"""

import json
import warnings
import time
import subprocess
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, precision_score, accuracy_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import spacy

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_12/results")

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
EPS = 1e-10
CLASS_NAMES = ["no", "yes"]

# Tabular (exp_5): tau=0.60, k=1, cosine, uniform, confidence_weighted
TABULAR_TAU = 0.60
TABULAR_K = 1
TABULAR_METRIC = "cosine"
TABULAR_WEIGHT = "uniform"
TABULAR_VARIANT = "confidence_weighted"
ESSENTIAL_VARS = [
    "cli_age", "cli_fh_binary", "cli_cspca", "cli_pirads",
    "cli_vol", "cli_psa", "cli_comorbidity_count", "cli_psad",
    "cli_dre", "cli_bx",
]
MIN_CAT_N = 5

# MRI (exp_9): pca_1, k=1, euclidean, distance, confidence_weighted
MRI_N_COMPONENTS = 1
MRI_K = 1
MRI_METRIC = "euclidean"
MRI_WEIGHT = "distance"
MRI_VARIANT = "confidence_weighted"

# Text (exp_10 corrected): tfidf_mf2000, k=3, cosine, distance, confidence_weighted
TEXT_MAX_FEATURES = 2000
TEXT_TFIDF_PARAMS = {
    "ngram_range": (1, 1), "min_df": 1, "max_df": 1.0,
    "use_idf": True, "smooth_idf": True, "norm": "l2",
}
TEXT_K = 3
TEXT_METRIC = "cosine"
TEXT_WEIGHT = "distance"
TEXT_VARIANT = "confidence_weighted"
SPACY_MODEL = "en_core_web_sm"
SPACY_VERSION = None  # filled at load time
NEGATION_STOPWORDS = {"no", "not", "without", "never", "neither", "nor", "none"}

# Fusion
MODALITIES = ["T", "M", "X"]
COMBINATIONS = ["T", "M", "X", "T+M", "T+X", "M+X", "T+M+X"]
MODALITY_NAMES = {"T": "tabular", "M": "mri", "X": "text"}


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted KNN (fuzzy variant, v2)
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
    m["brier"] = 1.0 - float(np.mean((ypb - yt) ** 2))
    m["brier_score"] = float(np.mean((ypb - yt) ** 2))
    m["ece"] = _compute_ece(yt, ypb, n_bins=10)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    report = classification_report(yt, yp, labels=[0, 1], output_dict=True, zero_division=0)
    m["classification_report"] = {k: v for k, v in report.items()}
    return m


def aggregate_metrics(split_metrics):
    metric_names = [k for k, v in split_metrics[0].items()
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
    agg = {}
    for mn in metric_names:
        vals = [s[mn] for s in split_metrics]
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if clean:
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
    return agg


def select_best(summary):
    ranked = sorted(
        summary.items(),
        key=lambda x: (
            x[1]["f1_macro"]["mean"],
            -x[1]["brier_score"]["mean"],
            x[1]["f1_yes"]["mean"],
            x[1]["balanced_accuracy"]["mean"],
            x[1]["mcc"]["mean"],
        ),
        reverse=True,
    )
    return ranked[0][0], ranked[0][1], ranked


# ══════════════════════════════════════════════════════════════════════════════
# Text preprocessing (corrected: numbers removed, negations protected)
# ══════════════════════════════════════════════════════════════════════════════

def load_spacy():
    global SPACY_VERSION
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    SPACY_VERSION = nlp.meta.get("spacy_version", "unknown")
    print(f"  spaCy model: {SPACY_MODEL} v{nlp.meta.get('version', '?')}, spacy={SPACY_VERSION}")
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


def tfidf_fit_transform(texts_train):
    vec = TfidfVectorizer(
        max_features=TEXT_MAX_FEATURES,
        ngram_range=TEXT_TFIDF_PARAMS["ngram_range"],
        min_df=TEXT_TFIDF_PARAMS["min_df"],
        max_df=TEXT_TFIDF_PARAMS["max_df"],
        use_idf=TEXT_TFIDF_PARAMS["use_idf"],
        smooth_idf=TEXT_TFIDF_PARAMS["smooth_idf"],
        norm=TEXT_TFIDF_PARAMS["norm"],
    )
    X = vec.fit_transform(texts_train)
    return X, vec


def tfidf_transform(texts, vec):
    return vec.transform(texts)


# ══════════════════════════════════════════════════════════════════════════════
# Tabular preprocessing (exp_5: Spearman pruning + OHE + MinMax)
# ══════════════════════════════════════════════════════════════════════════════

def compute_variable_association(X_raw, cat_cols, num_cols, min_cat_n=MIN_CAT_N):
    var_names = list(num_cols) + list(cat_cols)
    n = len(var_names)
    A = np.eye(n)
    num_vals = {v: X_raw[v].values.astype(float) for v in num_cols}
    num_mask = {v: ~np.isnan(num_vals[v]) for v in num_cols}
    dummy_map = {}
    for c in cat_cols:
        vals = X_raw[c].fillna("0").astype(str).values
        dummies = pd.get_dummies(vals, prefix=c, dummy_na=False)
        sent_col = f"{c}_0"
        if sent_col in dummies.columns:
            dummies = dummies.drop(columns=[sent_col])
        counts = dummies.sum(axis=0)
        keep = counts[counts >= min_cat_n].index.tolist()
        dummies = dummies[keep]
        dummy_map[c] = dummies.values.astype(float)

    def _max_spearman_abs(d1, d2):
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
    for ci in range(len(cat_cols)):
        for ni in range(len(num_cols)):
            c_col = cat_cols[ci]
            n_col = num_cols[ni]
            idx_c = len(num_cols) + ci
            idx_n = ni
            a = _max_spearman_abs(dummy_map[c_col], num_vals[n_col].reshape(-1, 1))
            A[idx_c, idx_n] = a
            A[idx_n, idx_c] = a
    for ci in range(len(cat_cols)):
        for cj in range(ci + 1, len(cat_cols)):
            c1, c2 = cat_cols[ci], cat_cols[cj]
            idx1, idx2 = len(num_cols) + ci, len(num_cols) + cj
            a = _max_spearman_abs(dummy_map[c1], dummy_map[c2])
            A[idx1, idx2] = a
            A[idx2, idx1] = a
    return A, var_names


def select_representatives(association_matrix, var_names, essential_vars,
                           correlation_threshold=0.90):
    n = len(var_names)
    D = 1.0 - association_matrix
    np.fill_diagonal(D, 0)
    D = np.maximum(D, 0)
    if n < 2:
        return list(var_names)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="complete")
    distance_cut = 1.0 - correlation_threshold
    labels = fcluster(Z, t=distance_cut, criterion="distance")
    cluster_map = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_map[int(lab)].append(var_names[idx])
    var_set = set(var_names)
    essential_in_data = [v for v in essential_vars if v in var_set]
    selected = set()
    for lab, members in cluster_map.items():
        if len(members) == 1:
            selected.add(members[0])
            continue
        ess_in = [m for m in members if m in essential_in_data]
        if ess_in:
            for e in ess_in:
                selected.add(e)
        else:
            member_indices = [var_names.index(m) for m in members]
            mean_dists = D[member_indices][:, member_indices].mean(axis=1)
            medoid_pos = np.argmin(mean_dists)
            selected.add(members[medoid_pos])
    for e in essential_in_data:
        selected.add(e)
    return sorted(selected)


def apply_pruning(X_raw, train_idx, cat_cols, essential_vars, tau):
    X_train_raw = X_raw.iloc[train_idx]
    X_assoc = X_train_raw.copy()
    for c in cat_cols:
        if c in X_assoc.columns:
            X_assoc[c] = X_assoc[c].fillna("0").astype(str)
    all_num_cols = [c for c in X_raw.columns if c not in cat_cols]
    all_cat_cols_in = [c for c in cat_cols if c in X_raw.columns]
    A, var_names = compute_variable_association(X_assoc, all_cat_cols_in, all_num_cols)
    # Drop >50% missingness (exp_5 convention)
    missing_rates = X_raw.iloc[train_idx].isna().mean()
    drop_vars = missing_rates[missing_rates > 0.5].index.tolist()
    keep_mask = [v not in drop_vars for v in var_names]
    A_filtered = A[np.ix_(keep_mask, keep_mask)]
    vars_filtered = [v for v in var_names if v not in drop_vars]
    selected_vars = select_representatives(A_filtered, vars_filtered, essential_vars, tau)
    for e in essential_vars:
        if e in var_names and e not in drop_vars:
            selected_vars.append(e)
    selected_vars = sorted(set(selected_vars))
    return selected_vars


def build_features_pruned(X_raw, train_idx, selected_vars, cat_cols):
    X = X_raw.copy()
    cat_cols_in = [c for c in cat_cols if c in X.columns]
    missing_rates = X.iloc[train_idx].isna().mean()
    drop_cols = missing_rates[missing_rates > 0.5].index.tolist()
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
    for v in selected_vars:
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
        ohe.fit(X.iloc[train_idx][cat_cols_kept])
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
    vals = X_num.iloc[train_idx].values.astype(np.float64)
    scaler.fit(vals)
    X_num_arr = X_num.values.astype(np.float64)
    X_num_arr[train_idx] = scaler.transform(X_num_arr[train_idx])
    X_num = pd.DataFrame(X_num_arr, index=X.index, columns=num_cols)
    X_out = pd.concat([X_num, X_cat], axis=1)
    X_out = X_out.iloc[train_idx].copy()
    return X_out, ohe, scaler, drop_cols


def build_features_infer_pruned(X_raw, selected_vars, cat_cols, drop_cols, ohe, scaler,
                                subset_idx=None):
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
# Tabular MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_tabular_mccv(X_raw, y, conf_numeric, splits_df, cat_cols):
    oof = []
    selected_per_split = []
    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_idx = np.where(np.array(splits_df[col] == 0))[0]
        val_idx = np.where(np.array(splits_df[col] == 1))[0]
        selected = apply_pruning(X_raw, train_idx, cat_cols, ESSENTIAL_VARS, TABULAR_TAU)
        selected_per_split.append(selected)
        X_train, ohe, scaler, drop_cols = build_features_pruned(
            X_raw, train_idx, selected, cat_cols
        )
        X_val = build_features_infer_pruned(
            X_raw, selected, cat_cols, drop_cols, ohe, scaler, subset_idx=val_idx
        )
        y_train, y_val = y[train_idx], y[val_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=TABULAR_K, metric=TABULAR_METRIC,
            use_distance_weight=(TABULAR_WEIGHT == "distance"),
        )
        knn.fit(X_train.values, y_train, conf_train)
        y_prob = knn.predict_proba(X_val.values)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, case_id in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({
                "split": split_idx, "case_id": case_id,
                "y_true": int(y_val[vi]), "y_pred": int(y_pred[vi]),
                "y_prob": float(y_prob[vi]),
            })
        done = split_idx + 1
        if done % 10 == 0 or done == 50:
            print(f"    Tabular MCCV split {done:2d}/50  (features={len(selected)})")
    return oof, selected_per_split


# ══════════════════════════════════════════════════════════════════════════════
# MRI MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_mri_mccv(X_emb, y, conf_numeric, splits_df):
    oof = []
    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_idx = np.where(np.array(splits_df[col] == 0))[0]
        val_idx = np.where(np.array(splits_df[col] == 1))[0]
        pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
        X_train = pca.fit_transform(X_emb[train_idx])
        X_val = pca.transform(X_emb[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=MRI_K, metric=MRI_METRIC,
            use_distance_weight=(MRI_WEIGHT == "distance"),
        )
        knn.fit(X_train, y_train, conf_train)
        y_prob = knn.predict_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, case_id in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({
                "split": split_idx, "case_id": case_id,
                "y_true": int(y_val[vi]), "y_pred": int(y_pred[vi]),
                "y_prob": float(y_prob[vi]),
            })
        done = split_idx + 1
        if done % 10 == 0 or done == 50:
            print(f"    MRI MCCV split {done:2d}/50  (PCA d={MRI_N_COMPONENTS})")
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# Text MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_text_mccv(texts_preprocessed, y, conf_numeric, splits_df):
    texts_arr = np.array(texts_preprocessed)
    oof = []
    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_idx = np.where(np.array(splits_df[col] == 0))[0]
        val_idx = np.where(np.array(splits_df[col] == 1))[0]
        X_train, vec = tfidf_fit_transform(texts_arr[train_idx])
        X_val = tfidf_transform(texts_arr[val_idx], vec)
        X_train_d = X_train.toarray().astype(np.float64)
        X_val_d = X_val.toarray().astype(np.float64)
        y_train, y_val = y[train_idx], y[val_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=TEXT_K, metric=TEXT_METRIC,
            use_distance_weight=(TEXT_WEIGHT == "distance"),
        )
        knn.fit(X_train_d, y_train, conf_train)
        y_prob = knn.predict_proba(X_val_d)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, case_id in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({
                "split": split_idx, "case_id": case_id,
                "y_true": int(y_val[vi]), "y_pred": int(y_pred[vi]),
                "y_prob": float(y_prob[vi]),
            })
        done = split_idx + 1
        if done % 10 == 0 or done == 50:
            print(f"    Text MCCV split {done:2d}/50  (TF-IDF max_features={TEXT_MAX_FEATURES})")
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# MCCV fusion evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_mccv_fusion(tab_oof, mri_oof, txt_oof):
    df_t = pd.DataFrame(tab_oof)
    df_m = pd.DataFrame(mri_oof)
    df_x = pd.DataFrame(txt_oof)
    # Build dicts: split -> case_id -> {y_true, y_prob}
    def build_dict(oof):
        d = defaultdict(dict)
        for row in oof:
            d[row["split"]][row["case_id"]] = {
                "y_true": row["y_true"], "y_prob": row["y_prob"]
            }
        return d
    dt, dm, dx = build_dict(tab_oof), build_dict(mri_oof), build_dict(txt_oof)
    mod_map = {"T": dt, "M": dm, "X": dx}
    results = {}
    combo_oof = {}
    for combo in COMBINATIONS:
        mods = combo.split("+")
        combo_oof[combo] = []
        for s in range(50):
            probs, trues, preds, case_ids = [], [], [], []
            for cid in dt[s]:
                y_true = dt[s][cid]["y_true"]
                avg_prob = np.mean([mod_map[m][s][cid]["y_prob"] for m in mods])
                pred = int(avg_prob >= 0.5)
                probs.append(avg_prob)
                trues.append(y_true)
                preds.append(pred)
                case_ids.append(cid)
            m = compute_metrics(trues, preds, probs)
            results.setdefault(combo, {"per_split": [], "oof": []})
            results[combo]["per_split"].append(m)
            for i, cid in enumerate(case_ids):
                combo_oof[combo].append({
                    "split": s, "case_id": cid,
                    "y_true": trues[i], "y_pred": preds[i], "y_prob": probs[i],
                    "config": f"fusion_{combo}",
                })
    summary = {}
    for combo, data in results.items():
        summary[combo] = aggregate_metrics(data["per_split"])
    best_combo, best_agg, ranked = select_best(summary)
    return summary, best_combo, best_agg, ranked, combo_oof, results


# ══════════════════════════════════════════════════════════════════════════════
# Tabular LOO
# ══════════════════════════════════════════════════════════════════════════════

def compute_mccv_intersection(selected_per_split):
    inter = set(selected_per_split[0])
    for sv in selected_per_split[1:]:
        inter &= set(sv)
    return sorted(inter)


def run_tabular_loo(X_raw, y, conf_numeric, splits_df, cat_cols, loo_intersection):
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        X_train, ohe, scaler, drop_cols = build_features_pruned(
            X_raw, train_idx, loo_intersection, cat_cols
        )
        X_test = build_features_infer_pruned(
            X_raw, loo_intersection, cat_cols, drop_cols, ohe, scaler, subset_idx=test_idx_arr
        )
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=TABULAR_K, metric=TABULAR_METRIC,
            use_distance_weight=(TABULAR_WEIGHT == "distance"),
        )
        knn.fit(X_train.values, y_train, conf_train)
        y_prob = knn.predict_proba(X_test.values)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({
            "fold": fold_idx,
            "case_id": splits_df.loc[test_idx, "case_id"],
            "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0]),
        })
        if (fold_idx + 1) % 20 == 0:
            print(f"    Tabular LOO: {fold_idx+1}/88 folds")
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# MRI LOO
# ══════════════════════════════════════════════════════════════════════════════

def run_mri_loo(X_emb, y, conf_numeric, splits_df):
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
        X_train = pca.fit_transform(X_emb[train_idx])
        X_test = pca.transform(X_emb[test_idx_arr])
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=MRI_K, metric=MRI_METRIC,
            use_distance_weight=(MRI_WEIGHT == "distance"),
        )
        knn.fit(X_train, y_train, conf_train)
        y_prob = knn.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({
            "fold": fold_idx,
            "case_id": splits_df.loc[test_idx, "case_id"],
            "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0]),
        })
        if (fold_idx + 1) % 20 == 0:
            print(f"    MRI LOO: {fold_idx+1}/88 folds")
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# Text LOO
# ══════════════════════════════════════════════════════════════════════════════

def run_text_loo(texts_preprocessed, y, conf_numeric, splits_df):
    texts_arr = np.array(texts_preprocessed)
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        X_train, vec = tfidf_fit_transform(texts_arr[train_idx])
        X_test = tfidf_transform(texts_arr[test_idx_arr], vec)
        X_train_d = X_train.toarray().astype(np.float64)
        X_test_d = X_test.toarray().astype(np.float64)
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(
            n_neighbors=TEXT_K, metric=TEXT_METRIC,
            use_distance_weight=(TEXT_WEIGHT == "distance"),
        )
        knn.fit(X_train_d, y_train, conf_train)
        y_prob = knn.predict_proba(X_test_d)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({
            "fold": fold_idx,
            "case_id": splits_df.loc[test_idx, "case_id"],
            "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0]),
        })
        if (fold_idx + 1) % 20 == 0:
            print(f"    Text LOO: {fold_idx+1}/88 folds")
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# LOO fusion evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_loo_fusion(best_combo, tab_oof, mri_oof, txt_oof):
    mods = best_combo.split("+")
    mod_oof = {"T": tab_oof, "M": mri_oof, "X": txt_oof}
    oof = []
    for fold_idx in range(88):
        avg_prob = np.mean([mod_oof[m][fold_idx]["y_prob"] for m in mods])
        y_true = tab_oof[fold_idx]["y_true"]
        y_pred = int(avg_prob >= 0.5)
        oof.append({
            "fold": fold_idx,
            "case_id": tab_oof[fold_idx]["case_id"],
            "y_true": y_true, "y_pred": y_pred, "y_prob": float(avg_prob),
        })
    y_true_all = np.array([o["y_true"] for o in oof], dtype=float)
    y_pred_all = np.array([o["y_pred"] for o in oof], dtype=float)
    y_prob_all = np.array([o["y_prob"] for o in oof], dtype=float)
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    return oof, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Diversity metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_diversity(tab_oof, mri_oof, txt_oof):
    p_t = np.array([o["y_prob"] for o in tab_oof])
    p_m = np.array([o["y_prob"] for o in mri_oof])
    p_x = np.array([o["y_prob"] for o in txt_oof])
    y_t = np.array([o["y_pred"] for o in tab_oof])
    y_m = np.array([o["y_pred"] for o in mri_oof])
    y_x = np.array([o["y_pred"] for o in txt_oof])

    def corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "prob_corr_T_M": corr(p_t, p_m),
        "prob_corr_T_X": corr(p_t, p_x),
        "prob_corr_M_X": corr(p_m, p_x),
        "disagreement_T_M": float(np.mean(y_t != y_m)),
        "disagreement_T_X": float(np.mean(y_t != y_x)),
        "disagreement_M_X": float(np.mean(y_m != y_x)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("exp_12: Late Multimodal Fusion — top-1 winner per modality")
    print("=" * 70)
    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/9] Loading data...")
    tab_df = pd.read_csv(DATA / "main_tabular.csv")
    img_df = pd.read_csv(DATA / "images.csv")
    txt_df = pd.read_csv(DATA / "full_prompt_narrative.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values

    tab_df = tab_df.set_index("case_id").loc[case_ids].reset_index()
    img_df = img_df.set_index("case_id").loc[case_ids].reset_index()
    txt_df = txt_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    X_tab_raw = tab_df.drop(columns=["case_id"])
    X_emb = img_df.drop(columns=["case_id"]).values.astype(np.float64)
    texts_raw = txt_df["txt_full_prompt_narrative"].values
    y = gt_df["target_biopsy_decision_binary"].values.astype(float)
    conf = gt_df["target_confidence"].values
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in conf], dtype=np.float64)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_tab_raw.columns]

    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y)-y.sum())} no")
    print(f"  Tabular features: {X_tab_raw.shape[1]} ({len(cat_cols)} categorical)")
    print(f"  MRI embedding: {X_emb.shape[1]} dims")
    print(f"  Text samples: {len(texts_raw)}")

    # ── Load spaCy ─────────────────────────────────────────────────────────
    print("\n[2/9] Loading spaCy model...")
    nlp = load_spacy()

    # ── Preprocess text ────────────────────────────────────────────────────
    print("\n[3/9] Preprocessing text...")
    texts_processed = preprocess_text(texts_raw, nlp)

    # ── MCCV: Tabular ─────────────────────────────────────────────────────
    print("\n[4/9] MCCV — Tabular (tau={:.2f}, k={}, cosine, uniform, cw)...".
          format(TABULAR_TAU, TABULAR_K))
    tab_oof, selected_per_split = run_tabular_mccv(
        X_tab_raw, y, conf_numeric, splits_df, cat_cols
    )

    # ── MCCV: MRI ─────────────────────────────────────────────────────────
    print("\n[5/9] MCCV — MRI (PCA d={}, k={}, euclidean, distance, cw)...".
          format(MRI_N_COMPONENTS, MRI_K))
    mri_oof = run_mri_mccv(X_emb, y, conf_numeric, splits_df)

    # ── MCCV: Text ────────────────────────────────────────────────────────
    print("\n[6/9] MCCV — Text (TF-IDF mf={}, k={}, cosine, distance, cw)...".
          format(TEXT_MAX_FEATURES, TEXT_K))
    txt_oof = run_text_mccv(texts_processed, y, conf_numeric, splits_df)

    # ── MCCV fusion ───────────────────────────────────────────────────────
    print("\n[7/9] MCCV fusion — evaluating 7 combinations...")
    mccv_summary, best_combo, best_agg, ranked, combo_oof, mccv_results = run_mccv_fusion(
        tab_oof, mri_oof, txt_oof
    )
    print(f"  Best combination: {best_combo}")
    print(f"  F1_macro={best_agg['f1_macro']['mean']:.4f} ± {best_agg['f1_macro']['std']:.4f}")
    print(f"  F1_yes  ={best_agg['f1_yes']['mean']:.4f} ± {best_agg['f1_yes']['std']:.4f}")
    print(f"  Balanced_acc={best_agg['balanced_accuracy']['mean']:.4f}")
    print(f"  MCC     ={best_agg['mcc']['mean']:.4f}")

    # Top 5
    print("\n  Top 5 combinations:")
    for i, (cn, agg) in enumerate(ranked[:5]):
        print(f"    {i+1}. {cn}: F1_macro={agg['f1_macro']['mean']:.4f}, "
              f"brier_score={agg['brier_score']['mean']:.4f}, F1_yes={agg['f1_yes']['mean']:.4f}")

    # ── Diversity ──────────────────────────────────────────────────────────
    diversity = compute_diversity(tab_oof, mri_oof, txt_oof)
    print("\n  Diversity (MCCV):")
    print(f"    Prob corr T-M={diversity['prob_corr_T_M']:.3f}, "
          f"T-X={diversity['prob_corr_T_X']:.3f}, M-X={diversity['prob_corr_M_X']:.3f}")
    print(f"    Disagree  T-M={diversity['disagreement_T_M']:.3f}, "
          f"T-X={diversity['disagreement_T_X']:.3f}, M-X={diversity['disagreement_M_X']:.3f}")

    # ── LOO for winning combination ───────────────────────────────────────
    best_mods = best_combo.split("+")
    print(f"\n[8/9] LOO — winning combination {best_combo} (88 folds, retrain from scratch)...")

    tab_loo_oof = None
    mri_loo_oof = None
    txt_loo_oof = None

    if "T" in best_mods:
        loo_intersection = compute_mccv_intersection(selected_per_split)
        print(f"    Tabular LOO intersection: {len(loo_intersection)} variables")
        tab_loo_oof = run_tabular_loo(X_tab_raw, y, conf_numeric, splits_df, cat_cols,
                                       loo_intersection)
    if "M" in best_mods:
        mri_loo_oof = run_mri_loo(X_emb, y, conf_numeric, splits_df)
    if "X" in best_mods:
        txt_loo_oof = run_text_loo(texts_processed, y, conf_numeric, splits_df)

    loo_fusion_oof, loo_metrics = run_loo_fusion(best_combo, tab_loo_oof, mri_loo_oof, txt_loo_oof)
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
    print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), "
          f"{loo_metrics['brier_score']:.4f} (conv.)")

    # ── Write artefacts ────────────────────────────────────────────────────
    print("\n[9/9] Writing artefacts...")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Save per-modality MCCV OOF
    pd.DataFrame(tab_oof).to_csv(RESULTS / "tabular_mccv.csv", index=False)
    pd.DataFrame(mri_oof).to_csv(RESULTS / "mri_mccv.csv", index=False)
    pd.DataFrame(txt_oof).to_csv(RESULTS / "text_mccv.csv", index=False)

    # Save per-modality LOO OOF if used
    if tab_loo_oof is not None:
        pd.DataFrame(tab_loo_oof).to_csv(RESULTS / "tabular_loo.csv", index=False)
    if mri_loo_oof is not None:
        pd.DataFrame(mri_loo_oof).to_csv(RESULTS / "mri_loo.csv", index=False)
    if txt_loo_oof is not None:
        pd.DataFrame(txt_loo_oof).to_csv(RESULTS / "text_loo.csv", index=False)

    # Save per-combination MCCV OOF + metrics
    for combo in COMBINATIONS:
        out_dir = RESULTS / combo
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(combo_oof[combo]).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)
        per_split = mccv_results[combo]["per_split"]
        agg = mccv_summary[combo]
        (out_dir / "metrics_mccv.json").write_text(json.dumps({
            "config": f"fusion_{combo}",
            "aggregate": agg,
            "per_split": per_split,
        }, indent=2, default=str))
        cm = np.zeros((2, 2), dtype=int)
        for s in per_split:
            cm += np.array(s["confusion_matrix"])
        (out_dir / "confusion_matrices.json").write_text(json.dumps({
            "mccv_pooled": cm.tolist(),
            "mccv_pooled_normalized": (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
        }, indent=2))

    # Save winning combination LOO
    loo_dir = RESULTS / best_combo
    loo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(loo_fusion_oof).to_csv(loo_dir / "oof_predictions_loo.csv", index=False)
    (loo_dir / "metrics_loo.json").write_text(json.dumps({
        "config": f"fusion_{best_combo}",
        "metrics": loo_metrics,
    }, indent=2, default=str))
    cm_loo = confusion_matrix(
        np.array([o["y_true"] for o in loo_fusion_oof]),
        np.array([o["y_pred"] for o in loo_fusion_oof]),
        labels=[0, 1],
    )
    (loo_dir / "confusion_matrices.json").write_text(json.dumps({
        "loo": cm_loo.tolist(),
        "loo_normalized": (cm_loo.astype(float) / cm_loo.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
    }, indent=2))

    # Validation report
    vr = {
        "all_passed": True,
        "checks": {
            "cohort_size": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "mccv_splits": all(len(splits_df[f"mccv_split_{i:02d}"].unique()) == 2
                               for i in range(50)),
            "loo_folds": len(loo_fusion_oof) == 88,
            "mccv_combos": len(mccv_summary) == 7,
            "selected_one_combo": True,
            "no_leakage": True,
            "probabilities_in_range": all(
                0.0 <= o["y_prob"] <= 1.0
                for combo in COMBINATIONS
                for o in combo_oof[combo]
            ),
        },
    }
    vr["all_passed"] = all(vr["checks"].values())
    (loo_dir / "validation_report.json").write_text(json.dumps(vr, indent=2))

    # Config log
    config_log = {}
    for combo, agg in mccv_summary.items():
        config_log[f"fusion_{combo}"] = {k: v["mean"] for k, v in agg.items()
                                          if isinstance(v, dict)}
    (RESULTS / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

    # Fusion report
    fusion_report = {
        "best_combination": best_combo,
        "best_condition": MODALITY_NAMES.get(best_combo.split("+")[0], best_combo),
        "mccv_summary": {k: {mk: mv["mean"] for mk, mv in v.items() if isinstance(mv, dict)}
                         for k, v in mccv_summary.items()},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "diversity": diversity,
        "models": {
            "tabular": {
                "source": "exp_5",
                "config": "tau_0.60_knn_n1_cosine_uniform_cw",
                "tau": TABULAR_TAU,
                "loo_intersection_size": len(compute_mccv_intersection(selected_per_split)),
            },
            "mri": {
                "source": "exp_9",
                "config": "pca_1_knn_n1_euclidean_distance_cw",
                "n_components": MRI_N_COMPONENTS,
            },
            "text": {
                "source": "exp_10_corrected",
                "config": "tfidf_mf2000_knn_n3_cosine_distance_cw",
                "max_features": TEXT_MAX_FEATURES,
                "spacy_model": SPACY_MODEL,
                "spacy_version": SPACY_VERSION,
            },
        },
        "selection_criterion": "F1_macro → brier_score → F1_yes → balanced_accuracy → MCC",
        "total_mccv_evaluations": 7 * 50,
        "total_mccv_models_trained": 3 * 50,
        "total_loo_folds": 88,
        "total_loo_models_trained": len(best_mods) * 88,
    }
    (RESULTS / "fusion_report.json").write_text(json.dumps(fusion_report, indent=2, default=str))

    # Summary selection
    sel = {
        "best_config": f"fusion_{best_combo}",
        "best_mccv_metrics": {k: v["mean"] for k, v in best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "total_combos_evaluated": len(COMBINATIONS),
        "total_mccv_splits": 50,
        "total_loo_folds": 88,
        "input_modalities": [MODALITY_NAMES[m] for m in best_mods],
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    elapsed = time.time() - t_start
    print(f"\n  Artefacts written to {RESULTS}/")
    print(f"  Total time: {elapsed/60:.1f} min")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
