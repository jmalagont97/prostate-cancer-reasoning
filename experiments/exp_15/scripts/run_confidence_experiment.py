#!/usr/bin/env python3
"""
exp_15 (corrected v2): 4D Vector [ICI, p_T, p_M, p_X] — Canonical Evaluation

Evaluates 3 feature families for predicting target_confidence:
  - ici_only: [ICI]
  - p_only: [p_T, p_M, p_X]
  - ici_plus_p: [ICI, p_T, p_M, p_X]

CORRECTIONS from v1:
  - Base models use ConfidenceWeightedKNN (NOT StandardKNN).
  - KNNs are trained on target_biopsy_decision_binary (y_binary).
  - Clinical confidence is used ONLY as neighbor weights.
"""

import os
import re
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
import spacy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"
RESULTS_DIR = ROOT / "experiments" / "exp_15" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

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
TOTAL_CONFIGS = 6
TOTAL_MCCV_FOLDS = 50
TOTAL_LOO_FOLDS = 88
EPS = 1e-10


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
        X_cat = pd.DataFrame(ohe.transform(X[cat_cols_kept]), index=X.index, columns=ohe.get_feature_names_out(cat_cols_kept))
    else:
        ohe = None
        X_cat = pd.DataFrame(index=X.index)
    scaler = MinMaxScaler()
    X_num = X[num_cols].copy()
    scaler.fit(X_num.values.astype(np.float64))
    X_num_arr = scaler.transform(X_num.values.astype(np.float64))
    X_num = pd.DataFrame(X_num_arr, index=X.index, columns=num_cols)
    X_out = pd.concat([X_num, X_cat], axis=1)
    return X_out.values.astype(np.float64), ohe, scaler, drop_cols


def build_features_infer(X_raw, cat_cols, drop_cols, ohe, scaler, subset_idx):
    X = X_raw.iloc[subset_idx].copy()
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
    for v in FROZEN_21_VARS:
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
        X_cat = pd.DataFrame(ohe.transform(X[cat_cols]), index=X.index, columns=ohe.get_feature_names_out(cat_cols))
    else:
        X_cat = pd.DataFrame(index=X.index)
    num_cols = [c for c in X.columns if c not in cat_cols]
    X_num = pd.DataFrame(scaler.transform(X[num_cols].values.astype(np.float64)), index=X.index, columns=num_cols)
    X_out = pd.concat([X_num, X_cat], axis=1)
    return X_out.values.astype(np.float64)


def compute_ici(p_t, p_m, p_x):
    p_bar = (p_t + p_m + p_x) / 3.0
    sigma = np.std(np.stack([p_t, p_m, p_x], axis=1), axis=1, ddof=0)
    return 2.0 * np.abs(p_bar - 0.5) * (1.0 - 2.0 * sigma)


def build_feature_vector(ici_val, p_t_val, p_m_val, p_x_val, family):
    if family == "ici_only":
        return ici_val.reshape(-1, 1)
    elif family == "p_only":
        return np.stack([p_t_val, p_m_val, p_x_val], axis=1)
    elif family == "ici_plus_p":
        return np.stack([ici_val, p_t_val, p_m_val, p_x_val], axis=1)
    else:
        raise ValueError(f"Unknown family: {family}")


def compute_moe_abs(y_true_ord, y_pred_ord):
    y_true_ord = np.array(y_true_ord, dtype=int)
    y_pred_ord = np.array(y_pred_ord, dtype=int)
    errs = []
    for c in range(3):
        mask = y_true_ord == c
        if mask.sum() == 0:
            continue
        errs.append(np.mean(np.abs(y_pred_ord[mask] - y_true_ord[mask]) / 2.0))
    return float(np.mean(errs)) if errs else np.nan


def compute_confusion_matrices(y_true_ord, y_pred_ord):
    y_true_ord = np.array(y_true_ord, dtype=int)
    y_pred_ord = np.array(y_pred_ord, dtype=int)
    cm_abs = confusion_matrix(y_true_ord, y_pred_ord, labels=[0, 1, 2])
    cm_norm = cm_abs.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    return cm_abs, cm_norm


def check_zero_recall(cm_abs):
    for c in range(3):
        row_sum = cm_abs[c, :].sum()
        if row_sum > 0 and cm_abs[c, c] == 0:
            return True
    return False


def compute_metrics(y_true_ord, y_pred_ord):
    y_true_ord = np.array(y_true_ord, dtype=int)
    y_pred_ord = np.array(y_pred_ord, dtype=int)
    moe_abs = compute_moe_abs(y_true_ord, y_pred_ord)
    f1_macro = float(f1_score(y_true_ord, y_pred_ord, average="macro", zero_division=0))
    cm_abs, cm_norm = compute_confusion_matrices(y_true_ord, y_pred_ord)
    has_zero_recall = check_zero_recall(cm_abs)
    return {
        "moe_abs": moe_abs, "f1_macro": f1_macro,
        "cm_abs": cm_abs.tolist(), "cm_norm": cm_norm.tolist(),
        "has_zero_recall": has_zero_recall, "n": len(y_true_ord),
    }


def _plot_cm(ax, cm, title, cmap="YlOrRd"):
    cm_arr = np.array(cm, dtype=float)
    total = cm_arr.sum()
    annot = np.array([
        f"{int(cm_arr[i, j])}\n({cm_arr[i, j] / total * 100:.1f}%)"
        if total > 0 else f"{int(cm_arr[i, j])}"
        for i in range(cm_arr.shape[0]) for j in range(cm_arr.shape[1])
    ]).reshape(cm_arr.shape)
    sns.heatmap(cm_arr, annot=annot, fmt="", cmap=cmap, cbar=True,
                xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, ax=ax,
                annot_kws={"size": 9}, vmin=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_yticklabels(ORD_NAMES, rotation=0)


def plot_cm_family(cm_data_dict, config_names, out_path, n_predictions, family):
    n = len(config_names)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 10), squeeze=False)
    fig.suptitle(f"CM — {family} — MCCV Pooled (N={n_predictions})", fontsize=14, fontweight="bold", y=0.98)
    for col, cfg_name in enumerate(config_names):
        cm_abs = np.array(cm_data_dict[cfg_name]["cm_abs"], dtype=float)
        cm_norm = np.array(cm_data_dict[cfg_name]["cm_norm"], dtype=float)
        _plot_cm(axes[0, col], cm_abs, f"{cfg_name} (counts)")
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True,
                    xticklabels=ORD_NAMES, yticklabels=ORD_NAMES,
                    ax=axes[1, col], annot_kws={"size": 10}, vmin=0, vmax=1)
        axes[1, col].set_title(f"{cfg_name} (row-norm)", fontsize=11, fontweight="bold")
        axes[1, col].set_xlabel("Predicted")
        axes[1, col].set_ylabel("True")
        axes[1, col].set_yticklabels(ORD_NAMES, rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cm_loo(cm_abs, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    _plot_cm(ax, cm_abs, title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def select_best_per_family(mccv_results, baseline_moe, families):
    winners = {}
    for family in families:
        family_cfgs = [c for c in mccv_results if c.startswith(family)]
        candidates = []
        for cfg_name in family_cfgs:
            cfg = mccv_results[cfg_name]
            pm = cfg["pooled_metrics"]
            if cfg["valid_rate"] < 1.0:
                print(f"  {cfg_name}: EXCLUDED (valid={cfg['valid_rate']:.0%})")
                continue
            if pm["moe_abs"] >= baseline_moe:
                print(f"  {cfg_name}: EXCLUDED (MOE_abs={pm['moe_abs']:.4f} >= baseline={baseline_moe:.4f})")
                continue
            if pm["has_zero_recall"]:
                print(f"  {cfg_name}: EXCLUDED (zero recall)")
                continue
            candidates.append({"name": cfg_name, "moe_abs": pm["moe_abs"], "f1_macro": pm["f1_macro"]})
        if candidates:
            candidates.sort(key=lambda c: (c["moe_abs"], -c["f1_macro"]))
            w = candidates[0]
            winners[family] = w["name"]
            print(f"  {family} SELECTED: {w['name']} (MOE_abs={w['moe_abs']:.4f})")
        else:
            winners[family] = None
            print(f"  {family}: NO eligible config")
    return winners


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[SEARCH] exp_15 | base_pipelines=3 | configs={TOTAL_CONFIGS} | ConfidenceWeightedKNN")
    print("=" * 70)
    t_start = time.time()

    print("\n[1/8] Loading data...")
    tab_df = pd.read_csv(DATA / "main_tabular.csv")
    img_df = pd.read_csv(DATA / "images.csv")
    txt_df = pd.read_csv(DATA / "full_prompt_narrative.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable = splits_df["cohort_status"] == "usable_labeled"
    case_ids = splits_df.loc[usable, "case_id"].values
    N = len(case_ids)

    tab_df = tab_df.set_index("case_id").loc[case_ids].reset_index()
    img_df = img_df.set_index("case_id").loc[case_ids].reset_index()
    txt_df = txt_df.set_index("case_id").loc[case_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[case_ids].reset_index()
    splits_df = splits_df.set_index("case_id").loc[case_ids].reset_index()

    X_tab_raw = tab_df.drop(columns=["case_id"])
    X_emb = img_df.drop(columns=["case_id"]).values.astype(np.float64)
    texts_raw = txt_df["txt_full_prompt_narrative"].values
    conf = gt_df["target_confidence"].values
    conf_ord = np.array([ORD_MAP[c] for c in conf], dtype=int)
    conf_numeric = np.array([CONFIDENCE_MAP.get(c, 0.5) for c in conf], dtype=np.float64)
    y_binary = gt_df["target_biopsy_decision_binary"].values.astype(np.float64)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_tab_raw.columns]

    conf_counts = gt_df["target_confidence"].value_counts()
    print(f"  Cohort: {N} usable_labeled cases")
    print(f"  Confidence: clear={conf_counts.get('clear',0)}, borderline={conf_counts.get('borderline',0)}, uncertain={conf_counts.get('uncertain',0)}")
    print(f"  Biopsy binary: yes={int(y_binary.sum())}, no={int(N - y_binary.sum())}")

    print("\n[2/8] Loading spaCy...")
    nlp = load_spacy()

    print("\n[3/8] Preprocessing text...")
    texts_processed = preprocess_text(texts_raw, nlp)

    FAMILIES = ["ici_only", "p_only", "ici_plus_p"]
    TREE_CONFIGS = {
        "ici_only_none": {"family": "ici_only", "class_weight": None},
        "ici_only_balanced": {"family": "ici_only", "class_weight": "balanced"},
        "p_only_none": {"family": "p_only", "class_weight": None},
        "p_only_balanced": {"family": "p_only", "class_weight": "balanced"},
        "ici_plus_p_none": {"family": "ici_plus_p", "class_weight": None},
        "ici_plus_p_balanced": {"family": "ici_plus_p", "class_weight": "balanced"},
    }

    print("\n" + "=" * 70)
    print(f"[SEARCH] exp_15 | base_pipelines=3 | configs={TOTAL_CONFIGS} | "
          f"config=0/{TOTAL_CONFIGS} | fold=0/{TOTAL_MCCV_FOLDS} | completed=0/{TOTAL_CONFIGS * TOTAL_MCCV_FOLDS}")
    print("=" * 70)

    mccv_results = {}
    mccv_all_predictions = {}
    fold_counter = 0

    for cfg_idx, (config_name, config_params) in enumerate(TREE_CONFIGS.items(), 1):
        family = config_params["family"]
        cw = config_params["class_weight"]
        print(f"\n  === Config {cfg_idx}/{TOTAL_CONFIGS}: {config_name} (family={family}, cw={cw}) ===")
        all_preds = []
        t_config = time.time()

        for fold in range(TOTAL_MCCV_FOLDS):
            t_fold = time.time()
            col = f"mccv_split_{fold:02d}"
            train_idx = np.where(splits_df[col].values == 0)[0]
            val_idx = np.where(splits_df[col].values == 1)[0]
            n_inner = len(train_idx) // INNER_N_SPLITS

            p_t_oof = np.zeros(len(train_idx), dtype=np.float64)
            p_m_oof = np.zeros(len(train_idx), dtype=np.float64)
            p_x_oof = np.zeros(len(train_idx), dtype=np.float64)

            for k in range(INNER_N_SPLITS):
                ivs = k * n_inner
                ive = (k + 1) * n_inner if k < INNER_N_SPLITS - 1 else len(train_idx)
                inner_val_idx = train_idx[ivs:ive]
                inner_train_idx = np.concatenate([train_idx[:ivs], train_idx[ive:]])

                X_it, ohe_it, sc_it, dc_it = build_features_train(X_tab_raw, inner_train_idx, cat_cols)
                X_iv = build_features_infer(X_tab_raw, cat_cols, dc_it, ohe_it, sc_it, inner_val_idx)

                knn_t = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
                knn_t.fit(X_it, y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_t_oof[ivs:ive] = knn_t.predict_proba(X_iv)

                pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
                knn_m = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, (MRI_WEIGHT == "distance"))
                knn_m.fit(pca.fit_transform(X_emb[inner_train_idx]),
                          y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_m_oof[ivs:ive] = knn_m.predict_proba(pca.transform(X_emb[inner_val_idx]))

                texts_arr = np.array(texts_processed)
                vec = TfidfVectorizer(**TEXT_TFIDF_PARAMS, max_features=TEXT_MAX_FEATURES)
                X_txt_tr = vec.fit_transform(texts_arr[inner_train_idx]).toarray().astype(np.float64)
                X_txt_iv = vec.transform(texts_arr[inner_val_idx]).toarray().astype(np.float64)
                knn_x = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, (TEXT_WEIGHT == "distance"))
                knn_x.fit(X_txt_tr, y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_x_oof[ivs:ive] = knn_x.predict_proba(X_txt_iv)

            ici_train = compute_ici(p_t_oof, p_m_oof, p_x_oof)

            X_tr, ohe_tr, sc_tr, dc_tr = build_features_train(X_tab_raw, train_idx, cat_cols)
            X_v = build_features_infer(X_tab_raw, cat_cols, dc_tr, ohe_tr, sc_tr, val_idx)

            knn_t = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
            knn_t.fit(X_tr, y_binary[train_idx], conf_numeric[train_idx])
            p_t_val = knn_t.predict_proba(X_v)

            pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
            knn_m = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, (MRI_WEIGHT == "distance"))
            knn_m.fit(pca.fit_transform(X_emb[train_idx]),
                      y_binary[train_idx], conf_numeric[train_idx])
            p_m_val = knn_m.predict_proba(pca.transform(X_emb[val_idx]))

            texts_arr = np.array(texts_processed)
            vec = TfidfVectorizer(**TEXT_TFIDF_PARAMS, max_features=TEXT_MAX_FEATURES)
            X_txt_tr = vec.fit_transform(texts_arr[train_idx]).toarray().astype(np.float64)
            X_txt_v = vec.transform(texts_arr[val_idx]).toarray().astype(np.float64)
            knn_x = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, (TEXT_WEIGHT == "distance"))
            knn_x.fit(X_txt_tr, y_binary[train_idx], conf_numeric[train_idx])
            p_x_val = knn_x.predict_proba(X_txt_v)

            ici_val = compute_ici(p_t_val, p_m_val, p_x_val)
            X_feat_train = build_feature_vector(ici_train, p_t_oof, p_m_oof, p_x_oof, family)
            X_feat_val = build_feature_vector(ici_val, p_t_val, p_m_val, p_x_val, family)

            cw_param = {k: v for k, v in {"class_weight": cw}.items() if v is not None}
            dt = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=5, random_state=42, **cw_param)
            dt.fit(X_feat_train, conf_ord[train_idx])

            y_pred_proba = dt.predict_proba(X_feat_val)
            y_pred_ord = np.argmax(y_pred_proba, axis=1)
            y_true_ord = conf_ord[val_idx]

            metrics = compute_metrics(y_true_ord, y_pred_ord)

            for i, vi in enumerate(val_idx):
                all_preds.append({
                    "fold": fold, "case_id": case_ids[vi],
                    "y_true": int(y_true_ord[i]), "y_true_name": ORD_NAMES[int(y_true_ord[i])],
                    "y_pred": int(y_pred_ord[i]), "y_pred_name": ORD_NAMES[int(y_pred_ord[i])],
                })

            dt_time = time.time() - t_fold
            fold_counter += 1
            print(f"  [SEARCH] exp_15 | configs={TOTAL_CONFIGS} | "
                  f"config={cfg_idx}/{TOTAL_CONFIGS} | fold={fold+1}/{TOTAL_MCCV_FOLDS} | "
                  f"completed={fold_counter}/{TOTAL_CONFIGS * TOTAL_MCCV_FOLDS} | "
                  f"moe={metrics['moe_abs']:.4f} f1m={metrics['f1_macro']:.3f} ({dt_time:.1f}s)")

        y_true_pooled = np.array([p["y_true"] for p in all_preds], dtype=int)
        y_pred_pooled = np.array([p["y_pred"] for p in all_preds], dtype=int)
        pooled_metrics = compute_metrics(y_true_pooled, y_pred_pooled)

        config_time = time.time() - t_config
        mccv_results[config_name] = {
            "params": config_params, "pooled_metrics": pooled_metrics,
            "valid_folds": TOTAL_MCCV_FOLDS, "valid_rate": 1.0,
        }
        mccv_all_predictions[config_name] = all_preds

        print(f"\n  {config_name}: MOE_abs={pooled_metrics['moe_abs']:.4f}  "
              f"F1m={pooled_metrics['f1_macro']:.4f}  time={config_time:.1f}s")

    print("\n" + "=" * 70)
    print("[5/8] Baseline + Selection")
    print("=" * 70)

    first_cfg = list(mccv_all_predictions.keys())[0]
    y_true_mccv_arr = np.array([p["y_true"] for p in mccv_all_predictions[first_cfg]], dtype=int)
    maj_pred_mccv = np.full(len(y_true_mccv_arr), ORD_MAP["clear"], dtype=int)
    baseline_metrics = compute_metrics(y_true_mccv_arr, maj_pred_mccv)
    baseline_moe = baseline_metrics["moe_abs"]
    print(f"  Baseline (always clear, N={len(y_true_mccv_arr)}):")
    print(f"    MOE_abs:  {baseline_moe:.4f}")
    print(f"    F1_macro: {baseline_metrics['f1_macro']:.4f}")

    print("\n  --- Selection (per family) ---")
    family_winners = select_best_per_family(mccv_results, baseline_moe, FAMILIES)

    print("\n" + "=" * 70)
    print("[LOO] exp_15 — per-family winners")
    print("=" * 70)

    loo_results = {}
    for family in FAMILIES:
        winner = family_winners[family]
        if winner is None:
            print(f"\n  {family}: skipped (no eligible config)")
            continue

        print(f"\n  === LOO: {winner} (family={family}) ===")
        cfg_params = TREE_CONFIGS[winner]
        cw = cfg_params["class_weight"]
        loo_preds = []
        loo_y_true = []
        loo_y_pred = []
        t_loo = time.time()

        for fold in range(TOTAL_LOO_FOLDS):
            t_fold = time.time()
            test_idx_arr = np.where(splits_df["loocv_fold"].values == fold)[0]
            test_idx = test_idx_arr[0]
            train_idx = np.where(splits_df["loocv_fold"].values != fold)[0]
            n_inner = len(train_idx) // INNER_N_SPLITS

            p_t_oof = np.zeros(len(train_idx), dtype=np.float64)
            p_m_oof = np.zeros(len(train_idx), dtype=np.float64)
            p_x_oof = np.zeros(len(train_idx), dtype=np.float64)

            for k in range(INNER_N_SPLITS):
                ivs = k * n_inner
                ive = (k + 1) * n_inner if k < INNER_N_SPLITS - 1 else len(train_idx)
                inner_val_idx = train_idx[ivs:ive]
                inner_train_idx = np.concatenate([train_idx[:ivs], train_idx[ive:]])

                X_it, ohe_it, sc_it, dc_it = build_features_train(X_tab_raw, inner_train_idx, cat_cols)
                X_iv = build_features_infer(X_tab_raw, cat_cols, dc_it, ohe_it, sc_it, inner_val_idx)

                knn_t = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
                knn_t.fit(X_it, y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_t_oof[ivs:ive] = knn_t.predict_proba(X_iv)

                pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
                knn_m = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, (MRI_WEIGHT == "distance"))
                knn_m.fit(pca.fit_transform(X_emb[inner_train_idx]),
                          y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_m_oof[ivs:ive] = knn_m.predict_proba(pca.transform(X_emb[inner_val_idx]))

                texts_arr = np.array(texts_processed)
                vec = TfidfVectorizer(**TEXT_TFIDF_PARAMS, max_features=TEXT_MAX_FEATURES)
                X_txt_tr = vec.fit_transform(texts_arr[inner_train_idx]).toarray().astype(np.float64)
                X_txt_iv = vec.transform(texts_arr[inner_val_idx]).toarray().astype(np.float64)
                knn_x = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, (TEXT_WEIGHT == "distance"))
                knn_x.fit(X_txt_tr, y_binary[inner_train_idx], conf_numeric[inner_train_idx])
                p_x_oof[ivs:ive] = knn_x.predict_proba(X_txt_iv)

            ici_train = compute_ici(p_t_oof, p_m_oof, p_x_oof)

            X_tr, ohe_tr, sc_tr, dc_tr = build_features_train(X_tab_raw, train_idx, cat_cols)
            X_te = build_features_infer(X_tab_raw, cat_cols, dc_tr, ohe_tr, sc_tr, test_idx_arr)

            knn_t = ConfidenceWeightedKNN(TABULAR_K, TABULAR_METRIC, False)
            knn_t.fit(X_tr, y_binary[train_idx], conf_numeric[train_idx])
            p_t_te = knn_t.predict_proba(X_te)

            pca = PCA(n_components=MRI_N_COMPONENTS, svd_solver="full", whiten=False)
            knn_m = ConfidenceWeightedKNN(MRI_K, MRI_METRIC, (MRI_WEIGHT == "distance"))
            knn_m.fit(pca.fit_transform(X_emb[train_idx]),
                      y_binary[train_idx], conf_numeric[train_idx])
            p_m_te = knn_m.predict_proba(pca.transform(X_emb[test_idx_arr]))

            texts_arr = np.array(texts_processed)
            vec = TfidfVectorizer(**TEXT_TFIDF_PARAMS, max_features=TEXT_MAX_FEATURES)
            X_txt_tr = vec.fit_transform(texts_arr[train_idx]).toarray().astype(np.float64)
            X_txt_te = vec.transform(texts_arr[test_idx_arr]).toarray().astype(np.float64)
            knn_x = ConfidenceWeightedKNN(TEXT_K, TEXT_METRIC, (TEXT_WEIGHT == "distance"))
            knn_x.fit(X_txt_tr, y_binary[train_idx], conf_numeric[train_idx])
            p_x_te = knn_x.predict_proba(X_txt_te)

            ici_test = compute_ici(p_t_te, p_m_te, p_x_te)
            X_feat_train = build_feature_vector(ici_train, p_t_oof, p_m_oof, p_x_oof, family)
            X_feat_test = build_feature_vector(ici_test, p_t_te, p_m_te, p_x_te, family)

            cw_param = {k: v for k, v in {"class_weight": cw}.items() if v is not None}
            dt = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=5, random_state=42, **cw_param)
            dt.fit(X_feat_train, conf_ord[train_idx])

            pred_proba = dt.predict_proba(X_feat_test)
            pred_ord = int(np.argmax(pred_proba))
            true_ord = int(conf_ord[test_idx])
            loo_y_true.append(true_ord)
            loo_y_pred.append(pred_ord)
            loo_preds.append({"fold": fold, "case_id": case_ids[test_idx],
                              "y_true": true_ord, "y_pred": pred_ord})

            dt_time = time.time() - t_fold
            correct = "OK" if pred_ord == true_ord else "ERR"
            print(f"  [LOO] exp_15 | config={winner} | fold={fold+1}/{TOTAL_LOO_FOLDS} | "
                  f"completed={fold+1}/{TOTAL_LOO_FOLDS} | "
                  f"true={ORD_NAMES[true_ord]:10s} pred={ORD_NAMES[pred_ord]:10s} {correct} ({dt_time:.1f}s)")

        loo_time = time.time() - t_loo
        loo_metrics = compute_metrics(loo_y_true, loo_y_pred)
        loo_results[winner] = {
            "family": family, "metrics": loo_metrics,
            "predictions": loo_preds, "time": loo_time,
        }
        print(f"\n  LOO {winner}: MOE_abs={loo_metrics['moe_abs']:.4f}  F1m={loo_metrics['f1_macro']:.4f}  ({loo_time:.1f}s)")

    # Confusion matrices
    print("\n" + "=" * 70)
    print("[6/8] Confusion Matrices + Figures")
    print("=" * 70)

    for family in FAMILIES:
        family_cfgs = [c for c in TREE_CONFIGS if TREE_CONFIGS[c]["family"] == family]
        cm_data = {}
        for cfg_name in family_cfgs:
            preds = mccv_all_predictions[cfg_name]
            y_t = np.array([p["y_true"] for p in preds], dtype=int)
            y_p = np.array([p["y_pred"] for p in preds], dtype=int)
            cm_abs, cm_norm = compute_confusion_matrices(y_t, y_p)
            cm_data[cfg_name] = {"cm_abs": cm_abs.tolist(), "cm_norm": cm_norm.tolist()}
        cm_bl_abs, cm_bl_norm = compute_confusion_matrices(y_true_mccv_arr, maj_pred_mccv)
        cm_data["baseline_clear"] = {"cm_abs": cm_bl_abs.tolist(), "cm_norm": cm_bl_norm.tolist()}
        plot_cm_family(cm_data, family_cfgs + ["baseline_clear"],
                       FIGURES_DIR / f"cm_mccv_{family}.png", len(y_true_mccv_arr), family)

    for winner_name, loo_data in loo_results.items():
        loo_y_t = np.array([p["y_true"] for p in loo_data["predictions"]], dtype=int)
        loo_y_p = np.array([p["y_pred"] for p in loo_data["predictions"]], dtype=int)
        cm_abs, cm_norm = compute_confusion_matrices(loo_y_t, loo_y_p)
        plot_cm_loo(cm_abs.tolist(), f"LOO CM — {winner_name} (N=88)",
                    FIGURES_DIR / f"cm_loo_{winner_name}.png")
        plot_cm_loo(cm_norm.tolist(), f"LOO Norm CM — {winner_name} (N=88)",
                    FIGURES_DIR / f"cm_loo_{winner_name}_norm.png")

    # Save outputs
    print("\n" + "=" * 70)
    print("[7/8] Save Outputs")
    print("=" * 70)

    summary = {
        "experiment": "exp_15_feature_families_canonical_v2",
        "version": "v2_confidence_weighted",
        "base_model": "ConfidenceWeightedKNN (trained on target_biopsy_decision_binary)",
        "cohort_size": N,
        "confidence_distribution": {c: int(conf_counts.get(c, 0)) for c in ORD_NAMES},
        "configurations": {},
        "family_winners": family_winners,
        "baseline_mccv": {"moe_abs": baseline_moe, "f1_macro": baseline_metrics["f1_macro"], "n_predictions": len(y_true_mccv_arr)},
        "loo_results": {},
    }
    for cfg_name, cfg_data in mccv_results.items():
        summary["configurations"][cfg_name] = {
            "family": cfg_data["params"]["family"],
            "class_weight": cfg_data["params"]["class_weight"],
            "moe_abs": cfg_data["pooled_metrics"]["moe_abs"],
            "f1_macro": cfg_data["pooled_metrics"]["f1_macro"],
        }
    for winner_name, loo_data in loo_results.items():
        summary["loo_results"][winner_name] = {
            "family": loo_data["family"],
            "moe_abs": loo_data["metrics"]["moe_abs"],
            "f1_macro": loo_data["metrics"]["f1_macro"],
        }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: summary.json")

    mccv_rows = []
    for cfg_name, preds in mccv_all_predictions.items():
        for p in preds:
            mccv_rows.append({"config": cfg_name, **p})
    pd.DataFrame(mccv_rows).to_csv(RESULTS_DIR / "predictions_mccv.csv", index=False)
    print(f"  Saved: predictions_mccv.csv")

    for winner_name, loo_data in loo_results.items():
        pd.DataFrame(loo_data["predictions"]).to_csv(
            RESULTS_DIR / f"predictions_loo_{winner_name}.csv", index=False)
        print(f"  Saved: predictions_loo_{winner_name}.csv")

    scorecard_rows = []
    for cfg_name, cfg_data in mccv_results.items():
        pm = cfg_data["pooled_metrics"]
        scorecard_rows.append({
            "config": cfg_name, "scope": "MCCV_pooled",
            "moe_abs": pm["moe_abs"], "f1_macro": pm["f1_macro"],
            "n_predictions": pm["n"], "valid_structure_rate": cfg_data["valid_rate"],
            "eligible": (pm["moe_abs"] < baseline_moe and not pm["has_zero_recall"]),
            "selection_status": "selected" if cfg_name in family_winners.values() else "",
        })
    for winner_name, loo_data in loo_results.items():
        scorecard_rows.append({
            "config": winner_name, "scope": "LOO",
            "moe_abs": loo_data["metrics"]["moe_abs"],
            "f1_macro": loo_data["metrics"]["f1_macro"],
            "n_predictions": loo_data["metrics"]["n"],
            "valid_structure_rate": 1.0, "eligible": "", "selection_status": "",
        })
    scorecard_rows.append({
        "config": "baseline_clear", "scope": "baseline_MCCV",
        "moe_abs": baseline_moe, "f1_macro": baseline_metrics["f1_macro"],
        "n_predictions": len(y_true_mccv_arr), "valid_structure_rate": 1.0,
        "eligible": "", "selection_status": "",
    })

    pd.DataFrame(scorecard_rows).to_csv(RESULTS_DIR / "evaluation_scorecard.csv", index=False)
    print(f"  Saved: evaluation_scorecard.csv")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("[SUMMARY] exp_15 (v2: ConfidenceWeightedKNN)")
    print("=" * 70)
    for family in FAMILIES:
        winner = family_winners[family]
        if winner:
            pm = mccv_results[winner]["pooled_metrics"]
            loo_m = loo_results[winner]["metrics"]
            print(f"  {family}: MCCV MOE_abs={pm['moe_abs']:.4f}  LOO MOE_abs={loo_m['moe_abs']:.4f}  winner={winner}")
        else:
            print(f"  {family}: NO eligible config")
    print(f"  Baseline: MOE_abs={baseline_moe:.4f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
