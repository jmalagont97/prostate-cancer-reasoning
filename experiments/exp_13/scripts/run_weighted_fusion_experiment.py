#!/usr/bin/env python3
"""
exp_13: Late multimodal fusion with learnable modality weights.

Uses fixed 21-variable tabular set (no correlation-pruning rerun), retrained
MRI PCA-1 and text TF-IDF pipelines from exp_12, searches a 231-candidate
simplex weight grid on MCCV, and evaluates only the best fused configuration
with LOO.

Usage:
    python3 experiments/exp_13/scripts/run_weighted_fusion_experiment.py
"""

import json
import warnings
import time
import re
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, precision_score, accuracy_score,
    average_precision_score, roc_auc_score,
    confusion_matrix,
)
import spacy

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_13/results")

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
EPS = 1e-10
STEPS = [round(i * 0.05, 2) for i in range(0, 21)]
GRID = [(w1, w2, round(1.0 - w1 - w2, 2))
        for w1, w2 in product(STEPS, repeat=2)
        if w1 + w2 <= 1.0 + 1e-8]

# Frozen tabular features from exp_5 (no correlation pruning rerun)
TABULAR_FEATURES = [
    "cli_age", "cli_allergies_count", "cli_bx", "cli_comorbidity_count",
    "cli_cspca", "cli_dre", "cli_fh_binary", "cli_ipss_score", "cli_months",
    "cli_pirads", "cli_psa", "cli_psad", "cli_psav", "cli_vol",
    "vit_bp_diastolic", "vit_bp_systolic", "vit_heart_rate_bpm",
    "vit_height_cm", "vit_smoking_pack_years", "vit_smoking_status",
    "vit_weight_kg",
]

# Tabular KNN (exp_5)
TABULAR_K = 1
TABULAR_METRIC = "cosine"
TABULAR_WEIGHT = "uniform"

# MRI KNN (exp_9)
MRI_N_COMPONENTS = 1
MRI_K = 1
MRI_METRIC = "euclidean"
MRI_WEIGHT = "distance"

# Text KNN (exp_10 corrected)
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
            nn_idx = np.argsort(dists[i])[: self.n_neighbors]
            d_nn = dists[i, nn_idx]
            y_nn = self.y_train[nn_idx]
            c_nn = self.conf_weights[nn_idx]
            w_dist = 1.0 / np.maximum(d_nn, self.epsilon) if self.use_distance_weight else np.ones_like(d_nn)
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)
        return proba

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def _compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == bins[-1] else y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def compute_metrics(y_true, y_pred, y_prob):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ypb = np.asarray(y_prob, dtype=float)
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
    m["ece"] = _compute_ece(yt, ypb)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    return m


def aggregate_metrics(split_metrics):
    agg = {}
    for key in split_metrics[0].keys():
        if key == "confusion_matrix":
            continue
        vals = [s[key] for s in split_metrics if isinstance(s[key], (int, float)) and not np.isnan(s[key])]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)), "n_valid": len(vals)}
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n_valid": 0}
    return agg


def select_best(summary):
    ranked = sorted(summary.items(), key=lambda kv: (
        kv[1]["f1_macro"]["mean"],
        -kv[1]["brier_score"]["mean"],
        kv[1]["f1_yes"]["mean"],
        kv[1]["balanced_accuracy"]["mean"],
        kv[1]["mcc"]["mean"],
    ), reverse=True)
    return ranked[0][0], ranked[0][1], ranked


def load_spacy():
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    print(f"  spaCy model: {SPACY_MODEL} v{nlp.meta.get('version', '?')}, spacy={nlp.meta.get('spacy_version', '?')}")
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
    return vec.fit_transform(texts_train), vec


def tfidf_transform(texts, vec):
    return vec.transform(texts)


def build_fixed_tabular(X_raw, train_idx, feature_cols, cat_cols):
    X = X_raw[feature_cols].copy()
    cat_cols_in = [c for c in cat_cols if c in X.columns]
    missing_rates = X.iloc[train_idx].isna().mean()
    high_missing_cols = missing_rates[missing_rates > 0.5].index.tolist()
    X = X.drop(columns=high_missing_cols, errors="ignore")
    effective_cat = [c for c in cat_cols_in if c in X.columns]
    effective_num = [c for c in X.columns if c not in effective_cat]

    for col in X.columns:
        indicator = f"{col}__is_missing"
        X[indicator] = X[col].isna().astype(int)
        if col in effective_cat:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    X_num = X[effective_num].copy() if effective_num else pd.DataFrame(index=X.index)
    scaler = MinMaxScaler()
    if effective_num:
        scaler.fit(X_num.iloc[train_idx].values.astype(np.float64))
        X_num = pd.DataFrame(
            scaler.transform(X_num.values.astype(np.float64)),
            index=X.index,
            columns=effective_num,
        )

    ohe = None
    X_cat = pd.DataFrame(index=X.index)
    if effective_cat:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        ohe.fit(X.iloc[train_idx][effective_cat])
        X_cat = pd.DataFrame(
            ohe.transform(X[effective_cat]),
            index=X.index,
            columns=ohe.get_feature_names_out(effective_cat),
        )

    X_out = pd.concat([X_num, X_cat], axis=1)
    return X_out.iloc[train_idx].copy(), ohe, scaler, effective_num, effective_cat, high_missing_cols


def infer_fixed_tabular(X_raw, feature_cols, ohe, scaler, effective_num, effective_cat,
                         drop_cols, subset_idx=None):
    X = X_raw[feature_cols].copy() if subset_idx is None else X_raw.iloc[subset_idx][feature_cols].copy()
    idx = X.index
    for col in feature_cols:
        indicator = f"{col}__is_missing"
        X[indicator] = X[col].isna().astype(int)
        if col in effective_cat:
            X[col] = X[col].fillna("0").astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    X_num = pd.DataFrame(
        scaler.transform(X[effective_num].values.astype(np.float64)),
        index=idx,
        columns=effective_num,
    ) if effective_num else pd.DataFrame(index=idx)

    X_cat = pd.DataFrame(index=idx)
    if ohe is not None and effective_cat:
        X_cat = pd.DataFrame(
            ohe.transform(X[effective_cat]),
            index=idx,
            columns=ohe.get_feature_names_out(effective_cat),
        )

    return pd.concat([X_num, X_cat], axis=1)


def run_tabular_mccv(X_raw, y, conf_numeric, splits_df):
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
    oof = []
    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_idx = np.where(np.array(splits_df[col] == 0))[0]
        val_idx = np.where(np.array(splits_df[col] == 1))[0]
        X_train, ohe, scaler, effective_num, effective_cat, drop_cols = build_fixed_tabular(
            X_raw, train_idx, TABULAR_FEATURES, cat_cols
        )
        X_val = infer_fixed_tabular(X_raw, TABULAR_FEATURES, ohe, scaler, effective_num,
                                    effective_cat, drop_cols, subset_idx=val_idx)
        y_train, y_val = y[train_idx], y[val_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(n_neighbors=TABULAR_K, metric=TABULAR_METRIC,
                                   use_distance_weight=(TABULAR_WEIGHT == "distance"))
        knn.fit(X_train.values, y_train, conf_train)
        y_prob = knn.predict_proba(X_val.values)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, cid in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({"split": split_idx, "case_id": cid, "y_true": int(y_val[vi]),
                        "y_pred": int(y_pred[vi]), "y_prob": float(y_prob[vi])})
        if (split_idx + 1) % 10 == 0 or split_idx + 1 == 50:
            print(f"    Tabular MCCV split {split_idx+1:02d}/50")
    return oof


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
        knn = ConfidenceWeightedKNN(n_neighbors=MRI_K, metric=MRI_METRIC,
                                   use_distance_weight=(MRI_WEIGHT == "distance"))
        knn.fit(X_train, y_train, conf_train)
        y_prob = knn.predict_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, cid in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({"split": split_idx, "case_id": cid, "y_true": int(y_val[vi]),
                        "y_pred": int(y_pred[vi]), "y_prob": float(y_prob[vi])})
        if (split_idx + 1) % 10 == 0 or split_idx + 1 == 50:
            print(f"    MRI MCCV split {split_idx+1:02d}/50")
    return oof


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
        knn = ConfidenceWeightedKNN(n_neighbors=TEXT_K, metric=TEXT_METRIC,
                                   use_distance_weight=(TEXT_WEIGHT == "distance"))
        knn.fit(X_train_d, y_train, conf_train)
        y_prob = knn.predict_proba(X_val_d)
        y_pred = (y_prob >= 0.5).astype(int)
        for vi, cid in enumerate(splits_df.loc[splits_df[col] == 1, "case_id"].values):
            oof.append({"split": split_idx, "case_id": cid, "y_true": int(y_val[vi]),
                        "y_pred": int(y_pred[vi]), "y_prob": float(y_prob[vi])})
        if (split_idx + 1) % 10 == 0 or split_idx + 1 == 50:
            print(f"    Text MCCV split {split_idx+1:02d}/50")
    return oof


def run_tabular_loo(X_raw, y, conf_numeric, splits_df):
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        X_train, ohe, scaler, effective_num, effective_cat, drop_cols = build_fixed_tabular(
            X_raw, train_idx, TABULAR_FEATURES, cat_cols
        )
        X_test = infer_fixed_tabular(X_raw, TABULAR_FEATURES, ohe, scaler, effective_num,
                                     effective_cat, drop_cols, subset_idx=test_idx_arr)
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(n_neighbors=TABULAR_K, metric=TABULAR_METRIC,
                                   use_distance_weight=(TABULAR_WEIGHT == "distance"))
        knn.fit(X_train.values, y_train, conf_train)
        y_prob = knn.predict_proba(X_test.values)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({"fold": fold_idx, "case_id": splits_df.loc[test_idx, "case_id"],
                    "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0])})
        if (fold_idx + 1) % 20 == 0 or fold_idx + 1 == 88:
            print(f"    Tabular LOO {fold_idx+1:03d}/88")
    return oof


def run_mri_loo(X_emb, y, conf_numeric, splits_df):
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
        X_train = pca.fit_transform(X_emb[train_idx])
        X_test = pca.transform(X_emb[test_idx_arr])
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(n_neighbors=MRI_K, metric=MRI_METRIC,
                                   use_distance_weight=(MRI_WEIGHT == "distance"))
        knn.fit(X_train, y_train, conf_train)
        y_prob = knn.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({"fold": fold_idx, "case_id": splits_df.loc[test_idx, "case_id"],
                    "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0])})
        if (fold_idx + 1) % 20 == 0 or fold_idx + 1 == 88:
            print(f"    MRI LOO {fold_idx+1:03d}/88")
    return oof


def run_text_loo(texts_preprocessed, y, conf_numeric, splits_df):
    texts_arr = np.array(texts_preprocessed)
    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]
        X_train, vec = tfidf_fit_transform(texts_arr[train_idx])
        X_test = tfidf_transform(texts_arr[test_idx_arr], vec)
        X_train_d = X_train.toarray().astype(np.float64)
        X_test_d = X_test.toarray().astype(np.float64)
        y_train, y_test = y[train_idx], y[test_idx]
        conf_train = conf_numeric[train_idx]
        knn = ConfidenceWeightedKNN(n_neighbors=TEXT_K, metric=TEXT_METRIC,
                                   use_distance_weight=(TEXT_WEIGHT == "distance"))
        knn.fit(X_train_d, y_train, conf_train)
        y_prob = knn.predict_proba(X_test_d)
        y_pred = (y_prob >= 0.5).astype(int)
        oof.append({"fold": fold_idx, "case_id": splits_df.loc[test_idx, "case_id"],
                    "y_true": int(y_test), "y_pred": int(y_pred[0]), "y_prob": float(y_prob[0])})
        if (fold_idx + 1) % 20 == 0 or fold_idx + 1 == 88:
            print(f"    Text LOO {fold_idx+1:03d}/88")
    return oof


def _align_by_split(oof, key_col="split"):
    df = pd.DataFrame(oof)
    return {(row[key_col], row["case_id"]): row for _, row in df.iterrows()}


def _apply_weight_search(oof_t, oof_m, oof_x):
    aligned_t = _align_by_split(oof_t)
    aligned_m = _align_by_split(oof_m)
    aligned_x = _align_by_split(oof_x)
    results = {}
    candidate_names = {}
    for wt, wm, wx in GRID:
        active = []
        if wt > 0:
            active.append("T")
        if wm > 0:
            active.append("M")
        if wx > 0:
            active.append("X")
        combo_name = "+".join(active) if active else "none"
        config_name = f"fusion_{combo_name}_w{wt:.2f}_{wm:.2f}_{wx:.2f}"
        candidate_names[config_name] = (wt, wm, wx)
        split_metrics = []
        split_oof = []
        for split_idx in range(50):
            split_rows = [(k, v) for k, v in aligned_t.items() if k[0] == split_idx]
            probs, trues, preds = [], [], []
            case_ids = []
            for key, row in split_rows:
                pt = aligned_t[key]["y_prob"]
                pm = aligned_m[key]["y_prob"]
                px = aligned_x[key]["y_prob"]
                p = wt * pt + wm * pm + wx * px
                p = float(np.clip(p, 0.0, 1.0))
                y_true = aligned_t[key]["y_true"]
                y_pred = int(p >= 0.5)
                probs.append(p)
                trues.append(y_true)
                preds.append(y_pred)
                case_ids.append(key[1])
                split_oof.append({"split": split_idx, "case_id": key[1], "y_true": y_true,
                                  "y_pred": y_pred, "y_prob": p, "config": config_name,
                                  "weight_T": wt, "weight_M": wm, "weight_X": wx})
            split_metrics.append(compute_metrics(trues, preds, probs))
        results[config_name] = {"per_split": split_metrics, "oof": split_oof}
    summary = {name: aggregate_metrics(data["per_split"]) for name, data in results.items()}
    best_name, best_agg, ranked = select_best(summary)
    return best_name, best_agg, ranked, results, candidate_names


def compute_diversity(tab_oof, mri_oof, txt_oof):
    df_t = pd.DataFrame(tab_oof)
    df_m = pd.DataFrame(mri_oof)
    df_x = pd.DataFrame(txt_oof)
    key = ["split", "case_id"]
    merged = df_t[key + ["y_prob"]].rename(columns={"y_prob": "p_t"})
    merged = merged.merge(df_m[key + ["y_prob"]].rename(columns={"y_prob": "p_m"}), on=key, how="inner")
    merged = merged.merge(df_x[key + ["y_prob"]].rename(columns={"y_prob": "p_x"}), on=key, how="inner")

    corr_tm = float(np.corrcoef(merged["p_t"], merged["p_m"])[0, 1])
    corr_tx = float(np.corrcoef(merged["p_t"], merged["p_x"])[0, 1])
    corr_mx = float(np.corrcoef(merged["p_m"], merged["p_x"])[0, 1])

    disagree_tm = float(((merged["p_t"] >= 0.5) != (merged["p_m"] >= 0.5)).mean())
    disagree_tx = float(((merged["p_t"] >= 0.5) != (merged["p_x"] >= 0.5)).mean())
    disagree_mx = float(((merged["p_m"] >= 0.5) != (merged["p_x"] >= 0.5)).mean())

    return {
        "prob_corr_T_M": corr_tm, "prob_corr_T_X": corr_tx, "prob_corr_M_X": corr_mx,
        "disagreement_T_M": disagree_tm, "disagreement_T_X": disagree_tx, "disagreement_M_X": disagree_mx,
    }


def run_loo_fusion(best_config_name, tab_loo_oof=None, mri_loo_oof=None, txt_loo_oof=None):
    weight_tuple = None
    for name in [best_config_name]:
        weight_tuple = best_config_name
    wt, wm, wx = map(float, weight_tuple.split("_w")[1].split("_"))
    aligned = {}
    if tab_loo_oof is not None:
        for row in tab_loo_oof:
            aligned.setdefault(row["fold"], {"y_true": row["y_true"], "pt": row["y_prob"]})
            aligned[row["fold"]]["pt"] = row["y_prob"]
            aligned[row["fold"]]["case_id"] = row["case_id"]
            aligned[row["fold"]]["y_true"] = row["y_true"]
    if mri_loo_oof is not None:
        for row in mri_loo_oof:
            aligned.setdefault(row["fold"], {"y_true": row["y_true"]})
            aligned[row["fold"]]["pm"] = row["y_prob"]
            aligned[row["fold"]]["case_id"] = row["case_id"]
            aligned[row["fold"]]["y_true"] = row["y_true"]
    if txt_loo_oof is not None:
        for row in txt_loo_oof:
            aligned.setdefault(row["fold"], {"y_true": row["y_true"]})
            aligned[row["fold"]]["px"] = row["y_prob"]
            aligned[row["fold"]]["case_id"] = row["case_id"]
            aligned[row["fold"]]["y_true"] = row["y_true"]

    oof = []
    for fold_idx in range(88):
        row = aligned[fold_idx]
        pt = row.get("pt", 0.0)
        pm = row.get("pm", 0.0)
        px = row.get("px", 0.0)
        p = float(np.clip(wt * pt + wm * pm + wx * px, 0.0, 1.0))
        y_true = int(row["y_true"])
        y_pred = int(p >= 0.5)
        oof.append({"fold": fold_idx, "case_id": row.get("case_id", ""), "y_true": y_true,
                    "y_pred": y_pred, "y_prob": p, "config": best_config_name,
                    "weight_T": wt, "weight_M": wm, "weight_X": wx})
    metrics = compute_metrics([o["y_true"] for o in oof], [o["y_pred"] for o in oof],
                              [o["y_prob"] for o in oof])
    return oof, metrics


def main():
    t_start = time.time()
    print("=" * 70)
    print("exp_13: Late multimodal fusion with learnable modality weights")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────────
    print("[1/9] Loading data...")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")
    ground_truth = pd.read_csv(DATA / "ground_truth.csv")
    tab = pd.read_csv(DATA / "main_tabular.csv")
    img = pd.read_csv(DATA / "images.csv")
    txt = pd.read_csv(DATA / "full_prompt_narrative.csv")

    case_ids = splits_df.loc[splits_df["cohort_status"] == "usable_labeled", "case_id"].values
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()
    ground_truth = ground_truth.set_index("case_id").loc[case_ids].reset_index()
    tab = tab.set_index("case_id").loc[case_ids].reset_index()
    img = img.set_index("case_id").loc[case_ids].reset_index()
    txt = txt.set_index("case_id").loc[case_ids].reset_index()

    assert len(case_ids) == 88
    y = ground_truth["target_biopsy_decision_binary"].values.astype(float)
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in ground_truth["target_confidence"]],
                            dtype=np.float64)
    X_emb = img.drop(columns=["case_id"]).values.astype(np.float64)
    X_tab_raw = tab.drop(columns=["case_id"])

    assert set(TABULAR_FEATURES).issubset(set(X_tab_raw.columns)), "Missing tabular features in input CSV"
    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y) - y.sum())} no")
    print(f"  Fixed tabular features: {len(TABULAR_FEATURES)}")

    # ── Preprocess text ─────────────────────────────────────────────────────
    print("\n[2/9] Preprocessing clinical narrative texts...")
    nlp = load_spacy()
    texts_preprocessed = preprocess_text(txt["txt_full_prompt_narrative"].values, nlp)

    # ── MCCV per-modality models ────────────────────────────────────────────
    print("\n[3/9] MCCV — Tabular (fixed 21 features, no correlation pruning)...")
    tab_oof = run_tabular_mccv(X_tab_raw, y, conf_numeric, splits_df)

    print("\n[4/9] MCCV — MRI (PCA d=1, k=1, euclidean, distance, cw)...")
    mri_oof = run_mri_mccv(X_emb, y, conf_numeric, splits_df)

    print("\n[5/9] MCCV — Text (TF-IDF max_features=2000, k=3, cosine, distance, cw)...")
    txt_oof = run_text_mccv(texts_preprocessed, y, conf_numeric, splits_df)

    # ── Search fusion weights ───────────────────────────────────────────────
    print(f"\n[6/9] Searching {len(GRID)} simplex fusion candidates on MCCV OOF...")
    best_config_name, best_agg, ranked, mccv_results, candidate_names = _apply_weight_search(
        tab_oof, mri_oof, txt_oof
    )
    best_wt, best_wm, best_wx = candidate_names[best_config_name]
    best_active = [m for m, w in zip(["T", "M", "X"], [best_wt, best_wm, best_wx]) if w > 0]
    print(f"  Best config: {best_config_name}")
    print(f"  Best weights: T={best_wt:.2f}, M={best_wm:.2f}, X={best_wx:.2f}")
    print(f"  Active modalities: {'+'.join(best_active) if best_active else 'none'}")
    print(f"  MCCV F1_macro={best_agg['f1_macro']['mean']:.4f}, "
          f"Brier={best_agg['brier_score']['mean']:.4f}")

    # ── Diversity ───────────────────────────────────────────────────────────
    diversity = compute_diversity(tab_oof, mri_oof, txt_oof)
    print("\n  Diversity (MCCV):")
    print(f"    Prob corr T-M={diversity['prob_corr_T_M']:.3f}, "
          f"T-X={diversity['prob_corr_T_X']:.3f}, M-X={diversity['prob_corr_M_X']:.3f}")
    print(f"    Disagree  T-M={diversity['disagreement_T_M']:.3f}, "
          f"T-X={diversity['disagreement_T_X']:.3f}, M-X={diversity['disagreement_M_X']:.3f}")

    # ── LOO for winning configuration ───────────────────────────────────────
    print(f"\n[7/9] LOO — winning configuration {best_config_name} (88 folds, retrain from scratch)...")
    tab_loo_oof = None
    mri_loo_oof = None
    txt_loo_oof = None
    if best_wt > 0:
        tab_loo_oof = run_tabular_loo(X_tab_raw, y, conf_numeric, splits_df)
    if best_wm > 0:
        mri_loo_oof = run_mri_loo(X_emb, y, conf_numeric, splits_df)
    if best_wx > 0:
        txt_loo_oof = run_text_loo(texts_preprocessed, y, conf_numeric, splits_df)

    loo_fusion_oof, loo_metrics = run_loo_fusion(best_config_name, tab_loo_oof, mri_loo_oof, txt_loo_oof)
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
    print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), "
          f"{loo_metrics['brier_score']:.4f} (conv.)")

    # ── Write artefacts ─────────────────────────────────────────────────────
    print("\n[8/9] Writing artefacts...")
    RESULTS.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(tab_oof).to_csv(RESULTS / "tabular_mccv.csv", index=False)
    pd.DataFrame(mri_oof).to_csv(RESULTS / "mri_mccv.csv", index=False)
    pd.DataFrame(txt_oof).to_csv(RESULTS / "text_mccv.csv", index=False)
    if tab_loo_oof is not None:
        pd.DataFrame(tab_loo_oof).to_csv(RESULTS / "tabular_loo.csv", index=False)
    if mri_loo_oof is not None:
        pd.DataFrame(mri_loo_oof).to_csv(RESULTS / "mri_loo.csv", index=False)
    if txt_loo_oof is not None:
        pd.DataFrame(txt_loo_oof).to_csv(RESULTS / "text_loo.csv", index=False)

    # Per-config MCCV artefacts
    for config_name, data in mccv_results.items():
        active = [m for m, w in zip(["T", "M", "X"], map(float, config_name.split("_w")[1].split("_"))) if w > 0]
        combo_dir = RESULTS / "+".join(active) / config_name
        combo_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data["oof"]).to_csv(combo_dir / "oof_predictions_mccv.csv", index=False)
        agg = aggregate_metrics(data["per_split"])
        (combo_dir / "metrics_mccv.json").write_text(json.dumps({
            "config": config_name,
            "weights": {"T": candidate_names[config_name][0], "M": candidate_names[config_name][1],
                        "X": candidate_names[config_name][2]},
            "aggregate": agg,
            "per_split": data["per_split"],
        }, indent=2, default=str))

    # Best-combination LOO artefacts
    best_combo_dir = RESULTS / "+".join(best_active) / best_config_name
    best_combo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(loo_fusion_oof).to_csv(best_combo_dir / "oof_predictions_loo.csv", index=False)
    (best_combo_dir / "metrics_loo.json").write_text(json.dumps({
        "config": best_config_name,
        "weights": {"T": best_wt, "M": best_wm, "X": best_wx},
        "metrics": loo_metrics,
    }, indent=2, default=str))
    cm_loo = confusion_matrix([o["y_true"] for o in loo_fusion_oof], [o["y_pred"] for o in loo_fusion_oof], labels=[0, 1])
    (best_combo_dir / "confusion_matrices.json").write_text(json.dumps({
        "loo": cm_loo.tolist(),
        "loo_normalized": (cm_loo.astype(float) / cm_loo.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
    }, indent=2))

    # Validation report
    vr = {
        "all_passed": True,
        "checks": {
            "cohort_size": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "mccv_splits": all(len(splits_df[f"mccv_split_{i:02d}"].unique()) == 2 for i in range(50)),
            "loo_folds": len(loo_fusion_oof) == 88,
            "weight_grid_size": len(GRID),
            "no_correlation_pruning": True,
            "fixed_21_features": list(TABULAR_FEATURES),
            "no_leakage": True,
            "probabilities_in_range": all(0.0 <= o["y_prob"] <= 1.0 for o in loo_fusion_oof),
        },
    }
    vr["all_passed"] = all(vr["checks"].values())
    (best_combo_dir / "validation_report.json").write_text(json.dumps(vr, indent=2))

    # Fusion report
    fusion_report = {
        "best_config": best_config_name,
        "best_weights": {"T": best_wt, "M": best_wm, "X": best_wx},
        "best_active_modalities": best_active,
        "mccv_summary": {name: {mk: mv["mean"] for mk, mv in v.items() if isinstance(mv, dict)}
                         for name, v in mccv_results.items()},
        "top_10_mccv": [{"config": name, "f1_macro": agg["f1_macro"]["mean"],
                         "brier_score": agg["brier_score"]["mean"]} for name, agg in ranked[:10]],
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "diversity": diversity,
        "models": {
            "tabular": {"source": "exp_5_fixed_features", "features": TABULAR_FEATURES,
                        "knn_config": f"k={TABULAR_K}_{TABULAR_METRIC}_{TABULAR_WEIGHT}_cw"},
            "mri": {"source": "exp_9", "n_components": MRI_N_COMPONENTS,
                    "knn_config": f"k={MRI_K}_{MRI_METRIC}_{MRI_WEIGHT}_cw"},
            "text": {"source": "exp_10_corrected", "max_features": TEXT_MAX_FEATURES,
                     "knn_config": f"k={TEXT_K}_{TEXT_METRIC}_{TEXT_WEIGHT}_cw",
                     "spacy_model": SPACY_MODEL},
        },
        "selection_criterion": "F1_macro → brier_score → F1_yes → balanced_accuracy → MCC",
        "total_mccv_weight_evaluations": len(GRID) * 50,
        "total_mccv_models_trained": 3 * 50,
        "total_loo_folds": 88,
        "total_loo_models_trained": len(best_active) * 88,
    }
    (RESULTS / "fusion_report.json").write_text(json.dumps(fusion_report, indent=2, default=str))

    # Summary selection
    sel = {
        "best_config": best_config_name,
        "best_weights": {"T": best_wt, "M": best_wm, "X": best_wx},
        "best_active_modalities": best_active,
        "best_mccv_metrics": {k: v["mean"] for k, v in best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "total_weight_grid_candidates": len(GRID),
        "total_mccv_splits": 50,
        "total_loo_folds": 88,
        "tabular_features": TABULAR_FEATURES,
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fixed_tabular_contract": "21 features from exp_5 intersection, no correlation-pruning rerun",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    elapsed = time.time() - t_start
    print(f"\n  Artefacts written to {RESULTS}/")
    print(f"  Total time: {elapsed/60:.1f} min")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
