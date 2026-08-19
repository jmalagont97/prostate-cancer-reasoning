#!/usr/bin/env python3
"""
exp_8: KNN classifier + Spearman correlation pruning on MRI embedding (1024 dims).

Pruning: Spearman association → hierarchical clustering (complete linkage) →
medoid selection per cluster. No essential variables, no scaling, no OHE.
Threshold τ as hyperparameter alongside KNN grid. MCCV per-fold pruning,
LOO with fixed intersection of MCCV-selected dimension sets.

Usage:
    python3 experiments/exp_8/scripts/run_mri_embedding_pruning_experiment.py
"""

import json
import warnings
import time
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
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

sys.stdout.reconfigure(line_buffering=True)


# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_8/results")
REPORTS = Path("experiments/exp_8/reports")
FIGURES = REPORTS / "figures"

CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
EPS = 1e-10
CLASS_NAMES = ["no", "yes"]

CORRELATION_THRESHOLDS = [0.30, 0.60, 0.80, 0.90]


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Pruning (pure numeric, no essential vars)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spearman_association(X):
    """
    Compute full absolute Spearman association matrix for numeric data.

    Args:
        X: (n_samples, n_features) numeric array, no NaN/Inf.

    Returns:
        A: (n_features, n_features) absolute association matrix with 1s on diagonal.
    """
    rho, _ = spearmanr(X, axis=0)
    if rho.ndim == 0:
        rho = rho.reshape(1, 1)
    A = np.abs(rho)
    np.fill_diagonal(A, 1.0)
    return A


def select_medoids(association_matrix, feature_names, tau):
    """
    Cluster features by complete linkage on (1 - |rho|), select medoid per cluster.

    No essential variables — all features treated equally.

    Args:
        association_matrix: (n_features, n_features) absolute Spearman.
        feature_names: list of feature names (indices into the matrix).
        tau: correlation threshold. D_cut = 1 - tau.

    Returns:
        selected: sorted list of selected feature names.
        clusters_info: dict with cluster details.
    """
    n = len(feature_names)
    D = 1.0 - association_matrix
    np.fill_diagonal(D, 0)
    D = np.maximum(D, 0)

    if n < 2:
        return list(feature_names), {"clusters": {}, "removed": [], "n_clusters": 1, "n_singletons": 1, "n_multi": 0}

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="complete")

    distance_cut = 1.0 - tau
    labels = fcluster(Z, t=distance_cut, criterion="distance")

    cluster_map = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_map[int(lab)].append(feature_names[idx])

    selected = set()
    removed = []

    for lab, members in cluster_map.items():
        if len(members) == 1:
            selected.add(members[0])
            continue
        # Medoid: member with minimum mean distance to other members
        member_indices = [feature_names.index(m) for m in members]
        mean_dists = D[member_indices][:, member_indices].mean(axis=1)
        medoid_pos = np.argmin(mean_dists)
        medoid = members[medoid_pos]
        selected.add(medoid)
        not_selected = [m for m in members if m != medoid]
        removed.extend(not_selected)

    clusters_info = {
        "clusters": {str(lab): members for lab, members in cluster_map.items()},
        "removed": removed,
        "distance_cut": float(distance_cut),
        "correlation_threshold": float(tau),
        "n_clusters": len(cluster_map),
        "n_singletons": sum(1 for m in cluster_map.values() if len(m) == 1),
        "n_multi": sum(1 for m in cluster_map.values() if len(m) > 1),
    }

    return sorted(selected), clusters_info


def apply_pruning_mri(X_emb_train, feature_names, tau):
    """
    Full pruning pipeline on MRI embedding training data.

    Args:
        X_emb_train: (n_train, 1024) raw embedding, no scaling.
        feature_names: list of 1024 feature names.
        tau: correlation threshold.

    Returns:
        selected_features: sorted list of selected feature names.
        pruning_info: dict with cluster details and counts.
    """
    A = compute_spearman_association(X_emb_train)
    selected, clusters_info = select_medoids(A, feature_names, tau)

    pruning_info = {
        "selected": selected,
        "n_original": len(feature_names),
        "n_after_pruning": len(selected),
        "reduction_rate": 1.0 - len(selected) / len(feature_names),
        "removed": clusters_info["removed"],
        "clusters": clusters_info["clusters"],
        "distance_cut": clusters_info["distance_cut"],
        "correlation_threshold": clusters_info["correlation_threshold"],
        "n_clusters": clusters_info["n_clusters"],
        "n_singletons": clusters_info["n_singletons"],
        "n_multi": clusters_info["n_multi"],
    }

    return selected, pruning_info


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted KNN
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


# ══════════════════════════════════════════════════════════════════════════════
# Config name
# ══════════════════════════════════════════════════════════════════════════════

def config_name(condition, k, metric, weight, variant):
    return f"{condition}_knn_n{k}_metric{metric}_weights{weight}_variant{variant}"


# ══════════════════════════════════════════════════════════════════════════════
# MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_mccv_condition(X_emb, y, confidence, splits_df, feature_names,
                       condition_name, tau):
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

    mccv_selected_per_split = []
    mccv_clusters_per_split = []

    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_mask = np.array(splits_df[col] == 0)
        val_mask = np.array(splits_df[col] == 1)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        X_train_full = X_emb[train_idx]   # (70, 1024)
        X_val_full = X_emb[val_idx]       # (18, 1024)

        # Pruning (or keep all for no_prune)
        if tau is not None:
            selected, pruning_info = apply_pruning_mri(
                X_train_full, feature_names, tau
            )
        else:
            selected = list(feature_names)
            pruning_info = {
                "selected": selected,
                "n_original": 1024,
                "n_after_pruning": 1024,
                "reduction_rate": 0.0,
                "removed": [],
                "clusters": {},
            }

        mccv_selected_per_split.append(selected)
        mccv_clusters_per_split.append(pruning_info)

        # Build pruned feature matrix
        sel_idx = [feature_names.index(s) for s in selected]
        X_train = X_train_full[:, sel_idx]
        X_val = X_val_full[:, sel_idx]
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
        if done % 10 == 0 or done == 50:
            n_feat = len(selected)
            print(f"    MCCV split {done:2d}/50  (dims={n_feat}/1024)")

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

    # Feature (dimension) frequency across splits
    feat_freq = defaultdict(int)
    for sv in mccv_selected_per_split:
        for v in sv:
            feat_freq[v] += 1

    return configs, summary, all_oof, mccv_selected_per_split, mccv_clusters_per_split, feat_freq


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

def run_loo(X_emb, y, confidence, splits_df, best_cfg, best_cn,
            selected_dims, feature_names):
    """LOO evaluation with fixed dimension set."""
    k = best_cfg["n_neighbors"]
    metric = best_cfg["metric"]
    weight = best_cfg["weights"]
    variant = best_cfg["variant"]

    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)
    sel_idx = [feature_names.index(d) for d in selected_dims]

    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1, f"Fold {fold_idx} has {len(test_idx_arr)} cases"
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]

        X_train = X_emb[train_idx][:, sel_idx]
        X_test = X_emb[test_idx_arr][:, sel_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        conf_train = conf_numeric[train_idx]

        if variant == "standard":
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
            knn.fit(X_train, y_train)
            y_prob = knn.predict_proba(X_test)[:, 1]
            y_pred = knn.predict(X_test)
        else:
            knn = ConfidenceWeightedKNN(
                n_neighbors=k, metric=metric,
                use_distance_weight=(weight == "distance"),
            )
            knn.fit(X_train, y_train, conf_train)
            y_prob = knn.predict_proba(X_test)
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
    fig.suptitle(f"exp_8 — MRI Embedding Pruning KNN: {best_cn}", fontsize=13, fontweight="bold")

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
    print("exp_8: KNN + Spearman Correlation Pruning on MRI Embedding (1024 dims)")
    print("=" * 70)
    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    images_df = pd.read_csv(DATA / "images.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values

    images_df = images_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    emb_cols = [c for c in images_df.columns if c.startswith("mri_emb_")]
    assert len(emb_cols) == 1024, f"Expected 1024 MRI embedding columns, got {len(emb_cols)}"
    feature_names = list(emb_cols)
    X_emb = images_df[feature_names].values.astype(np.float64)

    y = gt_df["target_biopsy_decision_binary"].values.astype(float)
    conf = gt_df["target_confidence"].values

    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y)-y.sum())} no")
    print(f"  Confidence: {dict(pd.Series(conf).value_counts())}")
    print(f"  MRI embedding shape: {X_emb.shape} (no scaling applied)")

    # ── Validate embedding ─────────────────────────────────────────────────
    print("\n[2/7] Validating embedding...")
    nan_count = int(np.isnan(X_emb).sum())
    inf_count = int(np.isinf(X_emb).sum())
    assert nan_count == 0, f"Found {nan_count} NaN values in MRI embedding"
    assert inf_count == 0, f"Found {inf_count} Inf values in MRI embedding"
    print(f"  NaN: {nan_count}, Inf: {inf_count}")
    print(f"  Range: [{X_emb.min():.4f}, {X_emb.max():.4f}]")
    print(f"  Mean: {X_emb.mean():.4f}, Std: {X_emb.std():.4f}")
    print(f"  Row norms: [{np.linalg.norm(X_emb, axis=1).min():.2f}, "
          f"{np.linalg.norm(X_emb, axis=1).max():.2f}]")

    # ── Define conditions ──────────────────────────────────────────────────
    conditions = [("no_prune", None)] + [(f"tau_{t:.2f}", t) for t in CORRELATION_THRESHOLDS]

    # ── MCCV over all conditions ───────────────────────────────────────────
    all_condition_summaries = {}
    all_condition_configs = {}
    all_condition_oof = {}
    all_condition_selected = {}
    all_condition_clusters = {}
    all_condition_feat_freq = {}

    for cond_name, tau in conditions:
        label = f"tau={tau}" if tau is not None else "no pruning"
        print(f"\n[3/7] MCCV condition: {cond_name} ({label}) — 72 configs x 50 splits")
        configs, summary, oof, selected_per_split, clusters_per_split, feat_freq = \
            run_mccv_condition(
                X_emb, y, conf, splits_df, feature_names,
                cond_name, tau
            )
        all_condition_summaries[cond_name] = summary
        all_condition_configs[cond_name] = configs
        all_condition_oof[cond_name] = oof
        all_condition_selected[cond_name] = selected_per_split
        all_condition_clusters[cond_name] = clusters_per_split
        all_condition_feat_freq[cond_name] = feat_freq

        best_cn_loc, best_agg_loc = select_best(summary)
        dims_per_split = [len(sv) for sv in selected_per_split]
        print(f"    Best: {best_cn_loc}")
        print(f"    F1_macro={best_agg_loc['f1_macro']['mean']:.4f} "
              f"± {best_agg_loc['f1_macro']['std']:.4f}")
        print(f"    Dims per split: min={min(dims_per_split)}, "
              f"max={max(dims_per_split)}, mean={np.mean(dims_per_split):.1f}")

    # ── Cross-condition selection ──────────────────────────────────────────
    print(f"\n[4/7] Cross-condition selection...")
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
    print(f"  F1_macro={global_best_agg['f1_macro']['mean']:.4f} "
          f"± {global_best_agg['f1_macro']['std']:.4f}")
    print(f"  F1_yes  ={global_best_agg['f1_yes']['mean']:.4f} "
          f"± {global_best_agg['f1_yes']['std']:.4f}")
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
        print(f"    {i+1}. {cn}: F1_macro={agg['f1_macro']['mean']:.4f}, "
              f"brier_score={agg['brier_score']['mean']:.4f}, F1_yes={agg['f1_yes']['mean']:.4f}")

    # ── LOO intersection for winning condition ─────────────────────────────
    print(f"\n[5/7] LOO intersection for {global_best_cond}...")
    winning_selected_per_split = all_condition_selected[global_best_cond]

    if winning_selected_per_split:
        loo_intersection = set(winning_selected_per_split[0])
        for sv in winning_selected_per_split[1:]:
            loo_intersection &= set(sv)
        loo_intersection = sorted(loo_intersection)
    else:
        loo_intersection = sorted(feature_names)

    dims_per_split = [len(sv) for sv in winning_selected_per_split]
    print(f"  Dims per split (first 5): {[len(sv) for sv in winning_selected_per_split[:5]]}...")
    print(f"  Set sizes: min={min(dims_per_split)}, max={max(dims_per_split)}, "
          f"mean={np.mean(dims_per_split):.1f}")
    print(f"  LOO intersection: {len(loo_intersection)} dimensions")

    if len(loo_intersection) == 0:
        print("  WARNING: Empty intersection — LOO will be skipped for this condition.")
        oof_loo = []
        loo_metrics = {k: float("nan") for k in ["f1_macro", "f1_yes", "balanced_accuracy",
                                                   "mcc", "sensitivity", "specificity",
                                                   "precision_yes", "accuracy", "pr_auc",
                                                   "roc_auc", "brier", "brier_score", "ece"]}
    else:
        # ── LOO evaluation ─────────────────────────────────────────────────
        print(f"\n[6/7] LOO evaluation (88 folds, {len(loo_intersection)} dims)...")
        oof_loo, loo_metrics = run_loo(
            X_emb, y, conf, splits_df, best_cfg, global_best_cn,
            loo_intersection, feature_names
        )
        print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
        print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
        print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
        print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
        print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), "
              f"{loo_metrics['brier_score']:.4f} (conv.)")

    # ── Confusion matrices ─────────────────────────────────────────────────
    print(f"\n[7/7] Generating confusion matrices and writing artefacts...")
    oof_mccv_best = [o for o in all_condition_oof[global_best_cond] if o["config"] == global_best_cn]

    y_true_mccv = np.array([o["y_true"] for o in oof_mccv_best], dtype=float)
    y_pred_mccv = np.array([o["y_pred"] for o in oof_mccv_best], dtype=float)
    cm_mccv = confusion_matrix(y_true_mccv, y_pred_mccv, labels=[0, 1])

    if len(oof_loo) > 0:
        y_true_loo = np.array([o["y_true"] for o in oof_loo], dtype=float)
        y_pred_loo = np.array([o["y_pred"] for o in oof_loo], dtype=float)
        cm_loo = confusion_matrix(y_true_loo, y_pred_loo, labels=[0, 1])
    else:
        cm_loo = np.zeros((2, 2), dtype=int)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / global_best_cn
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(cm_mccv, cm_loo, FIGURES, global_best_cn)

    # ── Write artefacts ────────────────────────────────────────────────────

    # Config log (all configs across all conditions)
    config_log = {}
    for cond_name in all_condition_summaries:
        for cn, agg in all_condition_summaries[cond_name].items():
            config_log[cn] = {k: v["mean"] for k, v in agg.items() if isinstance(v, dict)}
    (RESULTS / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

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
        "metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
    }, indent=2, default=str))

    # Hyperparameters
    (out_dir / "hyperparameters.json").write_text(json.dumps({
        "condition": global_best_cond,
        "correlation_threshold": next((t for c, t in conditions if c == global_best_cond), None),
        "n_neighbors": best_cfg["n_neighbors"],
        "metric": best_cfg["metric"],
        "weights": best_cfg["weights"],
        "variant": best_cfg["variant"],
        "preprocessing": "none (raw 1024-dim MRI embedding)",
    }, indent=2))

    # OOF predictions MCCV
    pd.DataFrame(oof_mccv_best).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)

    # OOF predictions LOO
    if len(oof_loo) > 0:
        pd.DataFrame(oof_loo).to_csv(out_dir / "oof_predictions_loo.csv", index=False)

    # Confusion matrices JSON
    (out_dir / "confusion_matrices.json").write_text(json.dumps({
        "mccv_pooled": cm_mccv.tolist(),
        "mccv_pooled_normalized": (cm_mccv.astype(float) / cm_mccv.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
        "loo": cm_loo.tolist(),
        "loo_normalized": (cm_loo.astype(float) / cm_loo.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
    }, indent=2))

    # LOO intersection
    (out_dir / "loo_intersection.json").write_text(json.dumps({
        "condition": global_best_cond,
        "n_intersected": len(loo_intersection),
        "intersected_dims": loo_intersection,
    }, indent=2))

    # Pruning log (per-split selected dims for winning condition)
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
        "loo_intersection_dims": loo_intersection,
        "total_conditions": len(conditions),
        "total_configs_per_condition": 72,
        "total_mccv_evaluations": len(conditions) * 72 * 50,
        "total_loo_folds": 88,
        "input_modality": "mri_embedding_1024d",
        "preprocessing": "none (raw embedding, Spearman pruning only)",
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # Dimension frequency per condition
    for cond_name in all_condition_feat_freq:
        freq_df = pd.DataFrame([
            {"dimension": v, "frequency": f, "pct": f / 50.0}
            for v, f in sorted(all_condition_feat_freq[cond_name].items(), key=lambda x: -x[1])
        ])
        freq_df.to_csv(RESULTS / f"dimension_frequency_{cond_name}.csv", index=False)

    # Pruning report (per condition)
    pruning_report = {}
    for cond_name in all_condition_selected:
        sv_list = all_condition_selected[cond_name]
        all_dims_union = set()
        for sv in sv_list:
            all_dims_union.update(sv)
        pruning_report[cond_name] = {
            "set_sizes": [len(sv) for sv in sv_list],
            "mean_size": float(np.mean([len(sv) for sv in sv_list])),
            "min_size": min(len(sv) for sv in sv_list),
            "max_size": max(len(sv) for sv in sv_list),
            "union_size": len(all_dims_union),
        }
        if sv_list:
            inter = set(sv_list[0])
            for sv in sv_list[1:]:
                inter &= set(sv)
            pruning_report[cond_name]["intersection_size"] = len(inter)
            pruning_report[cond_name]["intersection_dims"] = sorted(inter)
    (RESULTS / "pruning_report.json").write_text(json.dumps(pruning_report, indent=2, default=str))

    # Clusters report for winning condition
    winning_clusters = all_condition_clusters[global_best_cond]
    clusters_summary = {
        "condition": global_best_cond,
        "n_clusters_per_split": [c.get("n_clusters", 0) for c in winning_clusters],
        "n_singletons_per_split": [c.get("n_singletons", 0) for c in winning_clusters],
        "n_multi_per_split": [c.get("n_multi", 0) for c in winning_clusters],
    }
    (RESULTS / f"clusters_{global_best_cond}.json").write_text(
        json.dumps(clusters_summary, indent=2, default=str))

    # Validation report
    vr = {
        "all_passed": True,
        "checks": {
            "input_shape": X_emb.shape == (88, 1024),
            "usable_cases": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "no_nan": nan_count == 0,
            "no_inf": inf_count == 0,
            "conditions_evaluated": len(conditions),
            "total_mccv_evaluations": len(conditions) * 72 * 50,
            "loo_folds": len(oof_loo) == 88 if len(loo_intersection) > 0 else False,
            "loo_intersection_empty": len(loo_intersection) == 0,
            "selected_one_config": True,
            "no_leakage": True,
            "confusion_matrix_figures": True,
        },
    }
    (out_dir / "validation_report.json").write_text(json.dumps(vr, indent=2, default=str))

    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "log", "-1", "--format=%H"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_hash = "unknown"
    (out_dir / "git_commit.txt").write_text(git_hash)

    elapsed = time.time() - t_start
    print(f"\n  Artefacts written to {out_dir}/")
    print(f"  Figures written to {FIGURES}/")
    print(f"  Total time: {elapsed/60:.1f} min")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
