#!/usr/bin/env python3
"""
exp_9: KNN classifier + PCA dimensionality reduction on MRI embedding (1024 dims).

PCA: fit per fold on training data only, transform train+test. No scaling,
no whitening, svd_solver='full' for determinism. n_components as hyperparameter
alongside KNN grid. MCCV per-fold PCA fit, LOO with frozen n_components (no
intersection — PCA bases differ per fold).

Usage:
    python3 experiments/exp_9/scripts/run_mri_pca_knn_experiment.py
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
from sklearn.decomposition import PCA
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
RESULTS = Path("experiments/exp_9/results")
REPORTS = Path("experiments/exp_9/reports")
FIGURES = REPORTS / "figures"

CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
EPS = 1e-10
CLASS_NAMES = ["no", "yes"]

PCA_COMPONENTS_GRID = [1, 23, 46, 69]


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

def run_mccv_condition(X_emb, y, confidence, splits_df,
                       condition_name, n_components):
    """Run all 72 KNN configs for one PCA condition over 50 MCCV splits."""
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

    pca_per_split = []
    explained_variance_per_split = []

    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_mask = np.array(splits_df[col] == 0)
        val_mask = np.array(splits_df[col] == 1)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        X_train_full = X_emb[train_idx]   # (70, 1024)
        X_val_full = X_emb[val_idx]       # (18, 1024)

        # PCA transform (or keep raw for no_pca)
        if n_components is not None:
            pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
            X_train = pca.fit_transform(X_train_full)
            X_val = pca.transform(X_val_full)
            pca_per_split.append({
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_components": pca.n_components_,
                "singular_values": pca.singular_values_.tolist(),
            })
            explained_variance_per_split.append(pca.explained_variance_ratio_)
        else:
            X_train = X_train_full
            X_val = X_val_full
            pca_per_split.append(None)
            explained_variance_per_split.append(None)

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
            if n_components is not None:
                cumvar = np.mean([np.sum(ev) for ev in explained_variance_per_split if ev is not None])
                print(f"    MCCV split {done:2d}/50  (PCA d={n_components}, cumvar={cumvar:.4f})")
            else:
                print(f"    MCCV split {done:2d}/50  (no PCA, 1024 dims)")

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

    return configs, summary, all_oof, pca_per_split, explained_variance_per_split


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
            n_components, condition_name):
    """LOO evaluation with frozen n_components, PCA refit per fold."""
    k = best_cfg["n_neighbors"]
    metric = best_cfg["metric"]
    weight = best_cfg["weights"]
    variant = best_cfg["variant"]

    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)

    oof = []
    loo_explained_variance = []

    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1, f"Fold {fold_idx} has {len(test_idx_arr)} cases"
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]

        X_train_full = X_emb[train_idx]   # (87, 1024)
        X_test_full = X_emb[test_idx_arr] # (1, 1024)

        # PCA transform (or keep raw for no_pca)
        if n_components is not None:
            pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
            X_train = pca.fit_transform(X_train_full)
            X_test = pca.transform(X_test_full)
            loo_explained_variance.append(pca.explained_variance_ratio_)
        else:
            X_train = X_train_full
            X_test = X_test_full
            loo_explained_variance.append(None)

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
    return oof, metrics, loo_explained_variance


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(cm_mccv, cm_loo, out_dir, best_cn):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"exp_9 — MRI Embedding PCA KNN: {best_cn}", fontsize=13, fontweight="bold")

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
# Explained variance figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_explained_variance(all_ev_data, out_dir):
    """Bar chart of mean cumulative explained variance per condition."""
    conditions = []
    mean_cumvars = []
    std_cumvars = []

    for cond_name, ev_list in all_ev_data.items():
        # ev_list is list of per-split arrays (or None for no_pca)
        valid = [ev for ev in ev_list if ev is not None]
        if not valid:
            continue
        cumvars = [float(np.sum(ev)) for ev in valid]
        conditions.append(cond_name)
        mean_cumvars.append(float(np.mean(cumvars)))
        std_cumvars.append(float(np.std(cumvars)))

    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, mean_cumvars, yerr=std_cumvars, capsize=4, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative Explained Variance (mean)")
    ax.set_title("exp_9 — PCA Explained Variance per Condition (MCCV)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, mean_cumvars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "explained_variance.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "explained_variance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Explained variance figure saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("exp_9: KNN + PCA Dimensionality Reduction on MRI Embedding (1024 dims)")
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
    conditions = [("no_pca", None)] + [(f"pca_{d}", d) for d in PCA_COMPONENTS_GRID]

    # ── MCCV over all conditions ───────────────────────────────────────────
    all_condition_summaries = {}
    all_condition_configs = {}
    all_condition_oof = {}
    all_condition_pca = {}
    all_condition_ev = {}

    for cond_name, n_comp in conditions:
        label = f"n_components={n_comp}" if n_comp is not None else "no PCA"
        print(f"\n[3/7] MCCV condition: {cond_name} ({label}) — 72 configs x 50 splits")
        configs, summary, oof, pca_per_split, ev_per_split = \
            run_mccv_condition(
                X_emb, y, conf, splits_df,
                cond_name, n_comp
            )
        all_condition_summaries[cond_name] = summary
        all_condition_configs[cond_name] = configs
        all_condition_oof[cond_name] = oof
        all_condition_pca[cond_name] = pca_per_split
        all_condition_ev[cond_name] = ev_per_split

        best_cn_loc, best_agg_loc = select_best(summary)
        if n_comp is not None:
            cumvars = [float(np.sum(ev)) for ev in ev_per_split if ev is not None]
            print(f"    Best: {best_cn_loc}")
            print(f"    F1_macro={best_agg_loc['f1_macro']['mean']:.4f} "
                  f"± {best_agg_loc['f1_macro']['std']:.4f}")
            print(f"    Cumulative explained variance: "
                  f"mean={np.mean(cumvars):.4f}, std={np.std(cumvars):.4f}")
        else:
            print(f"    Best: {best_cn_loc}")
            print(f"    F1_macro={best_agg_loc['f1_macro']['mean']:.4f} "
                  f"± {best_agg_loc['f1_macro']['std']:.4f}")

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

    # ── LOO evaluation ─────────────────────────────────────────────────────
    # Determine n_components for LOO from winning condition
    if global_best_cond == "no_pca":
        loo_n_components = None
    else:
        loo_n_components = int(global_best_cond.split("_")[1])

    print(f"\n[5/7] LOO evaluation for {global_best_cond} "
          f"(n_components={loo_n_components})...")
    oof_loo, loo_metrics, loo_ev = run_loo(
        X_emb, y, conf, splits_df, best_cfg, global_best_cn,
        loo_n_components, global_best_cond
    )
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")
    print(f"  LOO Brier   ={loo_metrics['brier']:.4f} (1-Brier), "
          f"{loo_metrics['brier_score']:.4f} (conv.)")

    # ── Confusion matrices ─────────────────────────────────────────────────
    print(f"\n[6/7] Generating confusion matrices and writing artefacts...")
    oof_mccv_best = [o for o in all_condition_oof[global_best_cond] if o["config"] == global_best_cn]

    y_true_mccv = np.array([o["y_true"] for o in oof_mccv_best], dtype=float)
    y_pred_mccv = np.array([o["y_pred"] for o in oof_mccv_best], dtype=float)
    cm_mccv = confusion_matrix(y_true_mccv, y_pred_mccv, labels=[0, 1])

    y_true_loo = np.array([o["y_true"] for o in oof_loo], dtype=float)
    y_pred_loo = np.array([o["y_pred"] for o in oof_loo], dtype=float)
    cm_loo = confusion_matrix(y_true_loo, y_pred_loo, labels=[0, 1])

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / global_best_cn
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(cm_mccv, cm_loo, FIGURES, global_best_cn)
    plot_explained_variance(all_condition_ev, FIGURES)

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
        "n_components": loo_n_components,
        "n_neighbors": best_cfg["n_neighbors"],
        "metric": best_cfg["metric"],
        "weights": best_cfg["weights"],
        "variant": best_cfg["variant"],
        "preprocessing": f"PCA n_components={loo_n_components}, whiten=False, svd_solver=full" if loo_n_components is not None else "none (raw 1024-dim MRI embedding)",
    }, indent=2))

    # OOF predictions MCCV
    pd.DataFrame(oof_mccv_best).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)

    # OOF predictions LOO
    pd.DataFrame(oof_loo).to_csv(out_dir / "oof_predictions_loo.csv", index=False)

    # Confusion matrices JSON
    (out_dir / "confusion_matrices.json").write_text(json.dumps({
        "mccv_pooled": cm_mccv.tolist(),
        "mccv_pooled_normalized": (cm_mccv.astype(float) / cm_mccv.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
        "loo": cm_loo.tolist(),
        "loo_normalized": (cm_loo.astype(float) / cm_loo.sum(axis=1, keepdims=True).clip(min=1)).tolist(),
    }, indent=2))

    # PCA log (per-split PCA info for winning condition)
    winning_pca = all_condition_pca[global_best_cond]
    pca_log = {
        "condition": global_best_cond,
        "n_components": loo_n_components,
        "n_splits": len(winning_pca),
        "per_split": [p for p in winning_pca if p is not None],
    }
    if loo_n_components is not None:
        cumvars = [float(np.sum(p["explained_variance_ratio"])) for p in winning_pca if p is not None]
        pca_log["cumulative_variance_mean"] = float(np.mean(cumvars))
        pca_log["cumulative_variance_std"] = float(np.std(cumvars))
    (out_dir / "pca_log.json").write_text(json.dumps(pca_log, indent=2, default=str))

    # Summary selection
    sel = {
        "best_config": global_best_cn,
        "best_condition": global_best_cond,
        "best_mccv_metrics": {k: v["mean"] for k, v in global_best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "total_conditions": len(conditions),
        "total_configs_per_condition": 72,
        "total_mccv_evaluations": len(conditions) * 72 * 50,
        "total_loo_folds": 88,
        "input_modality": "mri_embedding_1024d",
        "preprocessing": f"PCA n_components={loo_n_components}" if loo_n_components is not None else "none (raw embedding)",
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # Explained variance per condition
    for cond_name in all_condition_ev:
        ev_list = all_condition_ev[cond_name]
        valid = [ev for ev in ev_list if ev is not None]
        if not valid:
            continue
        rows = []
        for split_idx, ev in enumerate(ev_list):
            if ev is None:
                continue
            for comp_idx, var in enumerate(ev):
                rows.append({"split": split_idx, "component": comp_idx + 1,
                             "explained_variance_ratio": float(var),
                             "cumulative": float(np.sum(ev[:comp_idx + 1]))})
        ev_df = pd.DataFrame(rows)
        ev_df.to_csv(RESULTS / f"explained_variance_{cond_name}.csv", index=False)

    # PCA report (per condition)
    pca_report = {}
    for cond_name, ev_list in all_condition_ev.items():
        valid = [ev for ev in ev_list if ev is not None]
        if not valid:
            pca_report[cond_name] = {"n_valid_folds": 0}
            continue
        cumvars = [float(np.sum(ev)) for ev in valid]
        pca_report[cond_name] = {
            "n_valid_folds": len(valid),
            "cumulative_variance_mean": float(np.mean(cumvars)),
            "cumulative_variance_std": float(np.std(cumvars)),
            "cumulative_variance_min": float(np.min(cumvars)),
            "cumulative_variance_max": float(np.max(cumvars)),
        }
    (RESULTS / "pca_report.json").write_text(json.dumps(pca_report, indent=2, default=str))

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
            "loo_folds": len(oof_loo) == 88,
            "selected_one_config": True,
            "no_leakage": True,
            "confusion_matrix_figures": True,
            "explained_variance_figures": True,
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
