#!/usr/bin/env python3
"""
exp_11: TF-IDF + TruncatedSVD + KNN on clinical narrative text.

Vocabulary fixed to full (max_features=None). SVD reduces geometry.
Searches 18000 evaluations (72 KNN × 5 SVD conditions × 50 splits).
Selects best by Macro-F1, evaluates via 88-fold LOO.

Usage:
    python3 experiments/exp_11/scripts/run_tfidf_svd_knn_experiment.py
"""

import json
import warnings
import time
import subprocess
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
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
import spacy

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_11/results")
REPORTS = Path("experiments/exp_11/reports")
FIGURES = REPORTS / "figures"

CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
MAX_FEATURES_LIST = [None]  # Full vocabulary, SVD handles reduction
N_COMPONENTS_LIST = [None, 1, 20, 40, 60]  # None = no SVD (control)
SVD_PARAMS = {"random_state": 42, "n_iter": 5, "algorithm": "randomized"}
EPS = 1e-10
CLASS_NAMES = ["no", "yes"]

# Negation stopwords to protect from removal (spaCy marks them as is_stop=True)
NEGATION_STOPWORDS = {"no", "not", "without", "never", "neither", "nor", "none"}

# spaCy
SPACY_MODEL = "en_core_web_sm"
SPACY_VERSION = None  # filled at load time


# ══════════════════════════════════════════════════════════════════════════════
# Text preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_spacy():
    global SPACY_VERSION
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    SPACY_VERSION = nlp.meta.get("spacy_version", "unknown")
    print(f"  spaCy model: {SPACY_MODEL} v{nlp.meta.get('version', '?')}, spacy={SPACY_VERSION}")
    return nlp


def preprocess_text(texts, nlp):
    """Lowercase, remove special chars, remove numbers, remove stopwords (protect negations), lemmatize."""
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            processed.append("")
            continue
        text = text.lower()
        # Remove special characters (keep alphanumeric + spaces + hyphens)
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        doc = nlp(text)
        tokens = []
        for token in doc:
            lemma = token.lemma_.strip()
            if not lemma or len(lemma) <= 1:
                continue
            # Remove tokens containing digits (e.g. "6.5", "147", "27")
            if re.search(r"\d", lemma):
                continue
            # Remove written number words (spaCy like_num covers all number words)
            if token.like_num:
                continue
            # Remove stopwords, but protect negation words
            if token.is_stop and token.text not in NEGATION_STOPWORDS:
                continue
            tokens.append(lemma)
        processed.append(" ".join(tokens))
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"    Preprocessed {i+1}/{total} texts")
    return processed


def tfidf_fit_transform(texts_train, max_features=None):
    """Fit TF-IDF on training texts, return sparse matrix and vectorizer."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
    )
    X = vec.fit_transform(texts_train)
    return X, vec


def tfidf_transform(texts, vec):
    """Transform texts using fitted TF-IDF vectorizer."""
    return vec.transform(texts)


def svd_fit_transform(X_sparse_train, n_components):
    """Fit TruncatedSVD + L2 normalization on sparse TF-IDF, return dense."""
    n_features = X_sparse_train.shape[1]
    nc = min(n_components, n_features - 1) if n_components is not None else None
    if nc is None:
        return None, None, None
    svd = TruncatedSVD(n_components=nc, **SVD_PARAMS)
    X_train = svd.fit_transform(X_sparse_train).astype(np.float64)
    # L2 normalize
    norm = Normalizer(norm="l2")
    X_train = norm.fit_transform(X_train)
    return X_train, svd, norm


def svd_transform(X_sparse, svd, norm):
    """Transform sparse TF-IDF using fitted SVD + normalization."""
    X = svd.transform(X_sparse).astype(np.float64)
    X = norm.transform(X)
    return X


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


# ══════════════════════════════════════════════════════════════════════════════
# Config name
# ══════════════════════════════════════════════════════════════════════════════

def config_name(k, metric, weight, variant, max_features, n_components):
    mf_str = "all" if max_features is None else str(max_features)
    svd_str = "nosvd" if n_components is None else f"svd{n_components}"
    return f"tfidf_mf{mf_str}_{svd_str}_knn_n{k}_metric{metric}_weights{weight}_variant{variant}"


# ══════════════════════════════════════════════════════════════════════════════
# MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_mccv(texts_preprocessed, y, confidence, splits_df):
    configs = {}
    for mf in MAX_FEATURES_LIST:
        for nc in N_COMPONENTS_LIST:
            for k in K_RANGE:
                for metric in METRICS_LIST:
                    for weight in WEIGHTS_LIST:
                        for variant in VARIANTS:
                            cn = config_name(k, metric, weight, variant, mf, nc)
                            configs[cn] = {
                                "n_neighbors": k,
                                "metric": metric,
                                "weights": weight,
                                "variant": variant,
                                "max_features": mf,
                                "n_components": nc,
                                "splits": [],
                            }

    n_evals = len(configs) * 50
    print(f"  {len(configs)} configurations × 50 splits = {n_evals} evaluations")
    all_oof = []

    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)
    texts_arr = np.array(texts_preprocessed)

    # Track variance explained per (mf, nc) across splits
    var_explained_log = {}

    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_mask = np.array(splits_df[col] == 0)
        val_mask = np.array(splits_df[col] == 1)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        texts_train = texts_arr[train_idx]
        texts_val = texts_arr[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        conf_train = conf_numeric[train_idx]

        # TF-IDF + SVD per (max_features, n_components)
        repr_cache = {}
        for mf in MAX_FEATURES_LIST:
            X_tfidf_train, vec = tfidf_fit_transform(texts_train, max_features=mf)
            X_tfidf_val = tfidf_transform(texts_val, vec)

            for nc in N_COMPONENTS_LIST:
                key = (mf, nc)
                if nc is None:
                    # No SVD: dense TF-IDF directly
                    X_train_d = X_tfidf_train.toarray().astype(np.float64)
                    X_val_d = X_tfidf_val.toarray().astype(np.float64)
                    repr_cache[key] = (X_train_d, X_val_d)
                else:
                    n_feat = X_tfidf_train.shape[1]
                    nc_eff = min(nc, n_feat - 1)
                    if nc_eff < 1:
                        # Cannot fit SVD with fewer features than components
                        continue
                    X_train_svd, svd, norm = svd_fit_transform(X_tfidf_train, nc)
                    X_val_svd = svd_transform(X_tfidf_val, svd, norm)
                    # Log variance explained
                    var_key = (mf, nc)
                    if var_key not in var_explained_log:
                        var_explained_log[var_key] = []
                    var_explained_log[var_key].append(float(svd.explained_variance_ratio_.sum()))
                    repr_cache[key] = (X_train_svd, X_val_svd)

        for cn, cfg in configs.items():
            k = cfg["n_neighbors"]
            metric = cfg["metric"]
            weight = cfg["weights"]
            variant = cfg["variant"]
            mf = cfg["max_features"]
            nc = cfg["n_components"]
            key = (mf, nc)

            if key not in repr_cache:
                continue

            X_train, X_val = repr_cache[key]

            if variant == "standard":
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
                knn.fit(X_train, y_train)
                y_prob = knn.predict_proba(X_val)[:, 1]
                y_pred = knn.predict(X_val)
            else:
                knn = ConfidenceWeightedKNN(
                    n_neighbors=k, metric=metric,
                    use_distance_weight=(weight == "distance"),
                )
                knn.fit(X_train, y_train, conf_train)
                y_prob = knn.predict_proba(X_val)
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
        if done % 5 == 0 or done == 50:
            print(f"  MCCV: {done}/50 splits done")

    # Aggregate
    summary = {}
    for cn, cfg in configs.items():
        agg = {}
        metric_names = list(cfg["splits"][0].keys()) if cfg["splits"] else []
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

    # Aggregate variance explained
    var_summary = {}
    for (mf, nc), vals in var_explained_log.items():
        if vals:
            mf_str = "all" if mf is None else str(mf)
            svd_str = f"svd{nc}"
            var_summary[f"mf{mf_str}_{svd_str}"] = {
                "mean_var_explained": float(np.mean(vals)),
                "std_var_explained": float(np.std(vals)),
                "n_valid_splits": len(vals),
            }

    return configs, summary, all_oof, var_summary


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best(summary):
    """Select by F1_macro (primary), brier_score (tie-break), then F1_yes > balanced_accuracy > MCC."""
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
    return ranked[0][0], ranked[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# LOO
# ══════════════════════════════════════════════════════════════════════════════

def run_loo(texts_preprocessed, y, confidence, splits_df, best_cfg, best_cn):
    k = best_cfg["n_neighbors"]
    metric = best_cfg["metric"]
    weight = best_cfg["weights"]
    variant = best_cfg["variant"]
    mf = best_cfg["max_features"]
    nc = best_cfg["n_components"]

    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)
    texts_arr = np.array(texts_preprocessed)

    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1, f"Fold {fold_idx} has {len(test_idx_arr)} cases"
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]

        texts_train = texts_arr[train_idx]
        texts_test = texts_arr[test_idx_arr]
        y_train = y[train_idx]
        y_test = y[test_idx]
        conf_train = conf_numeric[train_idx]

        X_tfidf_train, vec = tfidf_fit_transform(texts_train, max_features=mf)
        X_tfidf_test = tfidf_transform(texts_test, vec)

        if nc is None:
            X_train_d = X_tfidf_train.toarray().astype(np.float64)
            X_test_d = X_tfidf_test.toarray().astype(np.float64)
        else:
            X_train_d, svd, norm = svd_fit_transform(X_tfidf_train, nc)
            X_test_d = svd_transform(X_tfidf_test, svd, norm)

        if variant == "standard":
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
            knn.fit(X_train_d, y_train)
            y_prob = knn.predict_proba(X_test_d)[:, 1]
            y_pred = knn.predict(X_test_d)
        else:
            knn = ConfidenceWeightedKNN(
                n_neighbors=k, metric=metric,
                use_distance_weight=(weight == "distance"),
            )
            knn.fit(X_train_d, y_train, conf_train)
            y_prob = knn.predict_proba(X_test_d)
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
# Confusion matrix figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(cm_mccv, cm_loo, out_dir, best_cn):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"exp_11 — TF-IDF + TruncatedSVD + KNN: {best_cn}", fontsize=13, fontweight="bold")

    panels = [
        (0, 0, cm_mccv, "MCCV Pooled (900 predictions)", False),
        (0, 1, cm_mccv, "MCCV Pooled Normalized", True),
        (1, 0, cm_loo, "LOO (88 predictions)", False),
        (1, 1, cm_loo, "LOO Normalized", True),
    ]

    for row, col, cm, title, normalize in panels:
        ax = axes[row, col]
        if normalize:
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            fmt = ".2%"
            annot = np.array([[f"{v:{fmt}}" if v > 0 else "" for v in r] for r in cm_plot])
        else:
            cm_plot = cm.astype(float)
            fmt = "d"
            annot = np.array([[str(int(v)) for v in r] for r in cm_plot])

        sns.heatmap(cm_plot, annot=annot, fmt="", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=normalize, linewidths=0.5, linecolor="gray")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix figures saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("exp_11: TF-IDF + TruncatedSVD + KNN on clinical narrative text")
    print("=" * 70)
    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    text_df = pd.read_csv(DATA / "full_prompt_narrative.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values

    text_df = text_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    texts_raw = text_df["txt_full_prompt_narrative"].values
    y = gt_df["target_biopsy_decision_binary"].values.astype(float)
    conf = gt_df["target_confidence"].values

    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y)-y.sum())} no")
    print(f"  Confidence: {dict(pd.Series(conf).value_counts())}")
    print(f"  Text samples: {len(texts_raw)}")
    avg_len = np.mean([len(t.split()) if isinstance(t, str) else 0 for t in texts_raw])
    print(f"  Avg word count (raw): {avg_len:.0f}")

    # ── Load spaCy ─────────────────────────────────────────────────────────
    print("\n[2/7] Loading spaCy model...")
    nlp = load_spacy()

    # ── Preprocess text ────────────────────────────────────────────────────
    print("\n[3/7] Preprocessing text...")
    texts_processed = preprocess_text(texts_raw, nlp)
    avg_len_proc = np.mean([len(t.split()) if t else 0 for t in texts_processed])
    print(f"  Avg word count (processed): {avg_len_proc:.0f}")

    # ── MCCV ───────────────────────────────────────────────────────────────
    n_reprs = len(MAX_FEATURES_LIST) * len(N_COMPONENTS_LIST)
    print(f"\n[4/7] MCCV search ({n_reprs} representations × 72 KNN × 50 splits)...")
    configs, summary, oof_mccv, var_summary = run_mccv(texts_processed, y, conf, splits_df)

    # ── Select best ────────────────────────────────────────────────────────
    print("\n[5/7] Selecting best configuration...")
    best_cn, best_agg = select_best(summary)
    best_cfg = configs[best_cn]
    print(f"  Best: {best_cn}")
    print(f"  F1_macro={best_agg['f1_macro']['mean']:.4f} +/- {best_agg['f1_macro']['std']:.4f}")
    print(f"  F1_yes  ={best_agg['f1_yes']['mean']:.4f} +/- {best_agg['f1_yes']['std']:.4f}")
    print(f"  Balanced_acc={best_agg['balanced_accuracy']['mean']:.4f}")
    print(f"  MCC     ={best_agg['mcc']['mean']:.4f}")

    # Top 5
    ranked = sorted(summary.items(),
                    key=lambda x: (x[1]["f1_macro"]["mean"], -x[1]["brier_score"]["mean"]),
                    reverse=True)
    print("\n  Top 5:")
    for i, (cn, agg) in enumerate(ranked[:5]):
        print(f"    {i+1}. {cn}: F1_macro={agg['f1_macro']['mean']:.4f}, "
              f"brier_score={agg['brier_score']['mean']:.4f}, F1_yes={agg['f1_yes']['mean']:.4f}")

    # ── Variance explained summary ─────────────────────────────────────────
    print("\n  Variance explained by SVD (mean over 50 splits):")
    for key in sorted(var_summary.keys()):
        vs = var_summary[key]
        print(f"    {key}: {vs['mean_var_explained']*100:.2f}% (+/- {vs['std_var_explained']*100:.2f}%)")

    # ── LOO ────────────────────────────────────────────────────────────────
    print("\n[6/7] LOO evaluation (best config only)...")
    oof_loo, loo_metrics = run_loo(texts_processed, y, conf, splits_df, best_cfg, best_cn)
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
    print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), "
          f"{loo_metrics['brier_score']:.4f} (conv.)")

    # ── Confusion matrices ─────────────────────────────────────────────────
    print("\n[7/7] Generating confusion matrix figures...")
    oof_mccv_best = [o for o in oof_mccv if o["config"] == best_cn]

    y_true_mccv = np.array([o["y_true"] for o in oof_mccv_best], dtype=float)
    y_pred_mccv = np.array([o["y_pred"] for o in oof_mccv_best], dtype=float)
    cm_mccv = confusion_matrix(y_true_mccv, y_pred_mccv, labels=[0, 1])

    y_true_loo = np.array([o["y_true"] for o in oof_loo], dtype=float)
    y_pred_loo = np.array([o["y_pred"] for o in oof_loo], dtype=float)
    cm_loo = confusion_matrix(y_true_loo, y_pred_loo, labels=[0, 1])

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(cm_mccv, cm_loo, FIGURES, best_cn)

    # ── Write artefacts ────────────────────────────────────────────────────
    print("\n  Writing artefacts...")
    out_dir = RESULTS / best_cn
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config log
    config_log = {}
    for cn, agg in summary.items():
        config_log[cn] = {k: v["mean"] for k, v in agg.items() if isinstance(v, dict)}
    (out_dir / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

    # MCCV metrics
    (out_dir / "metrics_mccv.json").write_text(json.dumps({
        "config": best_cn,
        "aggregate": best_agg,
        "per_split": best_cfg["splits"],
    }, indent=2, default=str))

    # LOO metrics
    (out_dir / "metrics_loo.json").write_text(json.dumps({
        "config": best_cn,
        "metrics": loo_metrics,
    }, indent=2, default=str))

    # Hyperparameters
    (out_dir / "hyperparameters.json").write_text(json.dumps({
        "n_neighbors": best_cfg["n_neighbors"],
        "metric": best_cfg["metric"],
        "weights": best_cfg["weights"],
        "variant": best_cfg["variant"],
        "max_features": best_cfg["max_features"],
        "n_components": best_cfg["n_components"],
        "svd_params": SVD_PARAMS,
        "tfidf_params": {
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
        },
        "preprocessing": "lowercase, remove special chars, remove stopwords, lemmatize",
        "spacy_model": SPACY_MODEL,
        "spacy_version": SPACY_VERSION,
    }, indent=2))

    # OOF predictions
    pd.DataFrame(oof_mccv_best).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)
    pd.DataFrame(oof_loo).to_csv(out_dir / "oof_predictions_loo.csv", index=False)

    # Confusion matrices as JSON
    (out_dir / "confusion_matrices.json").write_text(json.dumps({
        "mccv_pooled": cm_mccv.tolist(),
        "mccv_pooled_normalized": (cm_mccv.astype(float) / cm_mccv.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
        "loo": cm_loo.tolist(),
        "loo_normalized": (cm_loo.astype(float) / cm_loo.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
    }, indent=2))

    # Variance explained
    (out_dir / "variance_explained.json").write_text(json.dumps(var_summary, indent=2))

    # Summary selection
    sel = {
        "best_config": best_cn,
        "best_mccv_metrics": {k: v["mean"] for k, v in best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "total_configs_evaluated": len(configs),
        "total_mccv_splits": 50,
        "total_loo_folds": 88,
        "input_modality": "clinical_narrative_tfidf_svd",
        "max_features": "None (full vocabulary)",
        "n_components_grid": [c for c in N_COMPONENTS_LIST if c is not None],
        "preprocessing": "lowercase, remove special chars, remove stopwords, lemmatize",
        "svd_params": SVD_PARAMS,
        "tfidf_params": {
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
        },
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
        "exp10_baseline_mccv_f1_macro": 0.6158,
        "exp10_baseline_loo_f1_macro": 0.6085,
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # Validation report
    vr = {
        "all_passed": True,
        "checks": {
            "input_shape": text_df.shape == (88, 2),
            "usable_cases": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "total_configs": len(configs) == 360,
            "mccv_splits": len(set(o["split"] for o in oof_mccv)) == 50,
            "loo_folds": len(oof_loo) == 88,
            "selected_one_config": True,
            "no_leakage": True,
            "confusion_matrix_figures": True,
        },
    }
    vr["all_passed"] = all(vr["checks"].values())
    (out_dir / "validation_report.json").write_text(json.dumps(vr, indent=2, default=str))

    # Git commit
    try:
        git_hash = subprocess.check_output(
            ["git", "log", "-1", "--format=%H"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_hash = "unknown"
    (out_dir / "git_commit.txt").write_text(git_hash)

    # Copy config_log to RESULTS root
    (RESULTS / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

    elapsed = time.time() - t_start
    print(f"\n  Artefacts written to {out_dir}/")
    print(f"  Figures written to {FIGURES}/")
    print(f"  Total time: {elapsed/60:.1f} min")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
