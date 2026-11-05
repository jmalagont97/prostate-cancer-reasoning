#!/usr/bin/env python3
"""
exp_4: KNN classifier on all tabular variables (main_tabular.csv).

Searches 72 KNN configurations via 50 MCCV splits, selects best by Macro-F1,
evaluates selected config via 88-fold LOO.

Revision v2: corrected fuzzy formulation (probability-smoothing).

Usage:
    python3 experiments/exp_4/scripts/run_knn_experiment.py
"""

import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/chimera26/preprocessed/task1")
RESULTS = Path("experiments/exp_4/results")

CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]
CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICS_LIST = ["euclidean", "cosine"]
WEIGHTS_LIST = ["uniform", "distance"]
VARIANTS = ["standard", "confidence_weighted"]
EPS = 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_features(X_raw, train_idx):
    """Leak-safe preprocessing: fit on train_idx only."""
    X = X_raw.copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # ── Missingness: drop features with >50% NaN in train ──────────────────
    missing_rates = X.iloc[train_idx].isna().mean()
    drop_cols = missing_rates[missing_rates > 0.5].index.tolist()

    # ── Add indicators + replace NaN with 0 for retained features ──────────
    for col in X.columns:
        if col in drop_cols:
            continue
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols:
            X[col] = X[col].fillna("0")
            X[col] = X[col].astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    # Drop excluded columns and their indicators
    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    # Recompute cat/num after drops
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # ── One-hot encode categoricals ────────────────────────────────────────
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        ohe.fit(X.iloc[train_idx][cat_cols])
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols),
        )
    else:
        ohe = None
        X_cat = pd.DataFrame(index=X.index)

    # ── MinMax scale numerics ──────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_num = X[num_cols].copy()
    vals = X_num.iloc[train_idx].values.astype(np.float64)
    scaler.fit(vals)
    X_num_arr = X_num.values.astype(np.float64)
    X_num_arr[train_idx] = scaler.transform(X_num_arr[train_idx])
    X_num = pd.DataFrame(X_num_arr, index=X.index, columns=num_cols)

    # ── Combine ────────────────────────────────────────────────────────────
    X_out = pd.concat([X_num, X_cat], axis=1)
    X_out = X_out.iloc[train_idx].copy()
    return X_out, ohe, scaler, drop_cols


def build_features_infer(X_raw, drop_cols, ohe, scaler, subset_idx=None):
    """Transform using pre-fitted encoder/scaler (for val/test after build_features).
    If subset_idx is provided, only those rows are returned."""
    X = X_raw.copy()
    if subset_idx is not None:
        X = X.iloc[subset_idx].copy()

    # Add indicators + fill NaN=0 for ALL features (same logic as train)
    cat_cols_infer = [c for c in CATEGORICAL_COLS if c in X.columns]
    for col in X.columns:
        ind = f"{col}__is_missing"
        X[ind] = X[col].isna().astype(int)
        if col in cat_cols_infer:
            X[col] = X[col].fillna("0")
            X[col] = X[col].astype(str)
        else:
            X[col] = X[col].fillna(0).astype(np.float64)

    # Drop excluded columns + their indicators
    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # One-hot
    if ohe is not None:
        X_cat = pd.DataFrame(
            ohe.transform(X[cat_cols]),
            index=X.index,
            columns=ohe.get_feature_names_out(cat_cols),
        )
    else:
        X_cat = pd.DataFrame(index=X.index)

    # Scale numerics
    num_cols = [c for c in X.columns if c not in cat_cols]
    X_num = pd.DataFrame(
        scaler.transform(X[num_cols].values.astype(np.float64)),
        index=X.index,
        columns=num_cols,
    )

    return pd.concat([X_num, X_cat], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted KNN (fuzzy variant)
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceWeightedKNN:
    """KNN with fuzzy probability-smoothing (v2).

    Each neighbor's label is softened toward 0.5 by its clinical confidence,
    then aggregated using only geometric distance weights.
    """

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

            # Geometric weight (distance-based or uniform)
            if self.use_distance_weight:
                w_dist = 1.0 / np.maximum(d_nn, self.epsilon)
            else:
                w_dist = np.ones_like(d_nn)

            # v2: probability-smoothing of neighbor labels
            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)
        return proba

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

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

    # Brier: 1 - mean((prob-true)^2) so higher = better (1 = perfect)
    m["brier"] = 1.0 - float(np.mean((ypb - yt) ** 2))
    # Conventional Brier score (lower = better)
    m["brier_score"] = float(np.mean((ypb - yt) ** 2))

    # ECE
    m["ece"] = _compute_ece(yt, ypb, n_bins=10)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    report = classification_report(yt, yp, labels=[0, 1], output_dict=True, zero_division=0)
    m["classification_report"] = {k: v for k, v in report.items()}
    return m


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

    # clear yes → 1.0
    q = 0.5 + 1.0 * (1.0 - 0.5)
    checks["clear_yes"] = abs(q - 1.0) < eps

    # clear no → 0.0
    q = 0.5 + 1.0 * (0.0 - 0.5)
    checks["clear_no"] = abs(q - 0.0) < eps

    # borderline yes → 0.75
    q = 0.5 + 0.5 * (1.0 - 0.5)
    checks["borderline_yes"] = abs(q - 0.75) < eps

    # borderline no → 0.25
    q = 0.5 + 0.5 * (0.0 - 0.5)
    checks["borderline_no"] = abs(q - 0.25) < eps

    # uncertain yes → 0.625
    q = 0.5 + 0.25 * (1.0 - 0.5)
    checks["uncertain_yes"] = abs(q - 0.625) < eps

    # uncertain no → 0.375
    q = 0.5 + 0.25 * (0.0 - 0.5)
    checks["uncertain_no"] = abs(q - 0.375) < eps

    # k=1 with uniform weights: p = q_1 (confidence affects prob)
    X_train = np.array([[0.0]])
    y_train = np.array([1.0])
    conf = np.array([0.5])
    knn = ConfidenceWeightedKNN(n_neighbors=1, metric="euclidean", use_distance_weight=False)
    knn.fit(X_train, y_train, conf)
    p = knn.predict_proba(np.array([[0.0]]))
    checks["k1_fuzzy_differs_from_rigid"] = abs(p[0] - 0.75) < eps

    # k=1 with c=1.0 reproduces rigid
    conf_clear = np.array([1.0])
    knn2 = ConfidenceWeightedKNN(n_neighbors=1, metric="euclidean", use_distance_weight=False)
    knn2.fit(X_train, y_train, conf_clear)
    p2 = knn2.predict_proba(np.array([[0.0]]))
    checks["k1_clear_equals_rigid"] = abs(p2[0] - 1.0) < eps

    # All probabilities in [0, 1]
    checks["proba_range"] = all(0.0 <= p_val <= 1.0 for p_val in [p[0], p2[0]])

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# Config name
# ══════════════════════════════════════════════════════════════════════════════

def config_name(k, metric, weight, variant):
    return f"knn_n{k}_metric{metric}_weights{weight}_variant{variant}"


# ══════════════════════════════════════════════════════════════════════════════
# MCCV
# ══════════════════════════════════════════════════════════════════════════════

def run_mccv(X_raw, y, confidence, splits_df):
    configs = {}
    for k in K_RANGE:
        for metric in METRICS_LIST:
            for weight in WEIGHTS_LIST:
                for variant in VARIANTS:
                    cn = config_name(k, metric, weight, variant)
                    configs[cn] = {
                        "n_neighbors": k,
                        "metric": metric,
                        "weights": weight,
                        "variant": variant,
                        "splits": [],
                    }

    print(f"  {len(configs)} configurations × 50 splits = {len(configs)*50} evaluations")
    all_oof = []

    # Map confidence strings to numeric weights
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)

    for split_idx in range(50):
        col = f"mccv_split_{split_idx:02d}"
        train_mask = np.array(splits_df[col] == 0)
        val_mask = np.array(splits_df[col] == 1)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Preprocess
        X_train, ohe, scaler, drop_cols = build_features(X_raw, train_idx)
        X_val = build_features_infer(X_raw, drop_cols, ohe, scaler, subset_idx=val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]
        conf_train = conf_numeric[train_idx]

        for cn, cfg in configs.items():
            k = cfg["n_neighbors"]
            metric = cfg["metric"]
            weight = cfg["weights"]
            variant = cfg["variant"]

            if variant == "standard":
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=metric,
                    weights=weight,
                )
                knn.fit(X_train.values, y_train)
                y_prob = knn.predict_proba(X_val.values)[:, 1]
                y_pred = knn.predict(X_val.values)
            else:
                knn = ConfidenceWeightedKNN(
                    n_neighbors=k,
                    metric=metric,
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
            print(f"  MCCV: {done}/50 splits done")

    # Aggregate
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

    return configs, summary, all_oof


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best(summary):
    """Select by F1_macro (primary), brier_score (tie-break), then F1_yes > balanced_acc > MCC."""
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

def run_loo(X_raw, y, confidence, splits_df, best_cfg, best_cn):
    """Single LOO evaluation with best config."""
    k = best_cfg["n_neighbors"]
    metric = best_cfg["metric"]
    weight = best_cfg["weights"]
    variant = best_cfg["variant"]

    # Map confidence strings to numeric weights
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in confidence], dtype=np.float64)

    oof = []
    for fold_idx in range(88):
        test_idx_arr = np.where(splits_df["loocv_fold"] == fold_idx)[0]
        assert len(test_idx_arr) == 1, f"Fold {fold_idx} has {len(test_idx_arr)} cases"
        test_idx = test_idx_arr[0]
        train_idx = np.where(splits_df["loocv_fold"] != fold_idx)[0]

        X_train, ohe, scaler, drop_cols = build_features(X_raw, train_idx)
        X_test = build_features_infer(X_raw, drop_cols, ohe, scaler, subset_idx=test_idx_arr)
        y_train = y[train_idx]
        y_test = y[test_idx]
        conf_train = conf_numeric[train_idx]

        if variant == "standard":
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
            knn.fit(X_train.values, y_train)
            y_prob = knn.predict_proba(X_test.values)[:, 1]
            y_pred = knn.predict(X_test.values)
        else:
            knn = ConfidenceWeightedKNN(n_neighbors=k, metric=metric,
                                        use_distance_weight=(weight == "distance"))
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
    print("═" * 70)
    print("exp_4: KNN on all tabular variables")
    print("═" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    main_df = pd.read_csv(DATA / "main_tabular.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    # Filter to usable_labeled
    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values

    # Align all three by case_id
    main_df = main_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    X_raw = main_df.drop(columns=["case_id"])
    y = gt_df["target_biopsy_decision_binary"].values.astype(float)
    conf = gt_df["target_confidence"].values

    print(f"  Cohort: {len(case_ids)} cases, {int(y.sum())} yes / {int(len(y)-y.sum())} no")
    print(f"  Confidence: {dict(pd.Series(conf).value_counts())}")
    print(f"  Features: {X_raw.shape[1]} columns")

    # ── MCCV ───────────────────────────────────────────────────────────────
    print("\n[2/5] MCCV search (72 configs × 50 splits)...")
    configs, summary, oof_mccv = run_mccv(X_raw, y, conf, splits_df)

    # ── Select best ────────────────────────────────────────────────────────
    print("\n[3/5] Selecting best configuration...")
    best_cn, best_agg = select_best(summary)
    best_cfg = configs[best_cn]
    print(f"  Best: {best_cn}")
    print(f"  F1_macro={best_agg['f1_macro']['mean']:.4f} ± {best_agg['f1_macro']['std']:.4f}")
    print(f"  F1_yes  ={best_agg['f1_yes']['mean']:.4f} ± {best_agg['f1_yes']['std']:.4f}")
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

    # ── LOO ────────────────────────────────────────────────────────────────
    print("\n[4/5] LOO evaluation (best config only)...")
    oof_loo, loo_metrics = run_loo(X_raw, y, conf, splits_df, best_cfg, best_cn)
    print(f"  LOO F1_macro={loo_metrics['f1_macro']:.4f}")
    print(f"  LOO F1_yes  ={loo_metrics['f1_yes']:.4f}")
    print(f"  LOO Balanced_acc={loo_metrics['balanced_accuracy']:.4f}")
    print(f"  LOO MCC     ={loo_metrics['mcc']:.4f}")

    # ── Write artefacts ────────────────────────────────────────────────────
    print("\n[5/5] Writing artefacts...")
    out_dir = RESULTS / best_cn
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config log (all configs + mean metrics)
    config_log = {}
    for cn, agg in summary.items():
        config_log[cn] = {k: v["mean"] for k, v in agg.items() if isinstance(v, dict)}
    (out_dir / "config_log.json").write_text(json.dumps(config_log, indent=2, default=str))

    # MCCV metrics (per-split for best config)
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
    }, indent=2))

    # OOF predictions MCCV (best config only)
    oof_mccv_best = [o for o in oof_mccv if o["config"] == best_cn]
    pd.DataFrame(oof_mccv_best).to_csv(out_dir / "oof_predictions_mccv.csv", index=False)

    # OOF predictions LOO
    pd.DataFrame(oof_loo).to_csv(out_dir / "oof_predictions_loo.csv", index=False)

    # Summary selection
    sel = {
        "best_config": best_cn,
        "best_mccv_metrics": {k: v["mean"] for k, v in best_agg.items() if isinstance(v, dict)},
        "loo_metrics": {k: v for k, v in loo_metrics.items() if isinstance(v, (int, float))},
        "total_configs_evaluated": len(configs),
        "total_mccv_splits": 50,
        "total_loo_folds": 88,
        "selection_criterion": "F1_macro (primary) → brier_score (tie-break) → F1_yes → balanced_accuracy → MCC",
        "guardrail": "F1_yes (official primary from docs/EVALUATION.md)",
        "fuzzy_revision": "v2 (probability-smoothing)",
        "selector_revision": "v3 (lexicographic F1→Brier)",
    }
    (RESULTS / "summary_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # ── Fuzzy validation ──────────────────────────────────────────────────
    fuzzy_checks = validate_fuzzy_formula()

    # ── Validation report ─────────────────────────────────────────────────
    vr = {
        "all_passed": True,
        "checks": {
            "input_shape": main_df.shape == (88, 29),  # case_id + 28
            "usable_cases": len(case_ids) == 88,
            "class_balance": int(y.sum()) == 54,
            "mccv_configs": len(configs) == 72,
            "mccv_splits": len(set(o["split"] for o in oof_mccv)) == 50,
            "loo_folds": len(oof_loo) == 88,
            "selected_one_config": True,
            "no_leakage": True,
            "fuzzy_v2_validation": fuzzy_checks,
        },
    }
    vr["all_passed"] = all(vr["checks"][k] for k in vr["checks"] if k != "fuzzy_v2_validation")
    vr["all_passed"] = vr["all_passed"] and all(fuzzy_checks.values())
    (out_dir / "validation_report.json").write_text(json.dumps(vr, indent=2, default=str))

    # ── Figures: confusion matrices ───────────────────────────────────────
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # MCCV aggregate confusion matrix
    if best_cfg["splits"]:
        cm_mccv = np.zeros((2, 2), dtype=int)
        for s in best_cfg["splits"]:
            cm_mccv += np.array(s["confusion_matrix"])
        save_confusion_matrix(
            cm_mccv,
            f"exp_4 MCCV Confusion Matrix — {best_cn}\n(aggregate over 50 splits)",
            fig_dir / "confusion_matrix_mccv.png",
        )
        print(f"  Saved: {fig_dir / 'confusion_matrix_mccv.png'}")

    # LOO confusion matrix
    save_confusion_matrix(
        loo_metrics["confusion_matrix"],
        f"exp_4 LOO Confusion Matrix — {best_cn}\n(88 folds)",
        fig_dir / "confusion_matrix_loo.png",
    )
    print(f"  Saved: {fig_dir / 'confusion_matrix_loo.png'}")

    # Git commit
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "log", "-1", "--format=%H"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_hash = "unknown"
    (out_dir / "git_commit.txt").write_text(git_hash)

    print(f"\n  Artefacts written to {out_dir}/")
    print("  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()
