#!/usr/bin/env python3
"""
Canonical MCCV-30 Benchmark for CHIMERA Task 2 Treatment Recommendation (exp_16).

Models evaluated:
  - H1: Trimodal Hard-SVC Late Fusion
  - H2: Trimodal Fuzzy-SVR Late Fusion (Champion configuration from jtarquino_task2)

Protocol:
  - MCCV-30 multi-subtask stratification from data/processed/task2/mccv_splits_30.csv
  - Strictly evaluated through chimera_metrics.task2.decision.compute_treatment_decision_gate
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHIMERA_PKG = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHIMERA_PKG))

from chimera_metrics.task2.decision import compute_treatment_decision_gate
from chimera_metrics.core.base import norm_treatment_decision

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_16" / "results"
MODELS_DIR = PROJECT_ROOT / "experiments" / "exp_16" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 30
N_CLASSES = 4
CLASS_ORDER = [
    "watchful_waiting",
    "continued_surveillance",
    "active_surveillance",
    "active_treatment",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_ORDER)}
CONFIDENCE_C = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}

T2_NUMERIC_COLS = [
    "age", "psa", "months", "pirads", "psad", "psav", "psap", "vol", "cspca",
    "bx_isup", "bx_gl_prim", "bx_gl_sec",
    "vitals_weight_kg", "vitals_height_cm", "vitals_bmi",
    "vitals_bp_sys", "vitals_bp_dia", "vitals_hr_bpm", "smoking_pack_years",
    "ipss_score", "allergies_count", "comorbidity_count",
    "comorb_diabetes", "comorb_hypertension", "comorb_cardiovascular",
    "comorb_respiratory", "comorb_renal", "comorb_surgery", "comorb_cancer_other",
]
T2_CATEGORICAL_COLS = ["dre", "bx", "ct"]


class Task2TabularPreprocessor:
    def __init__(self) -> None:
        self.impute_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_columns_: Dict[str, List[str]] = {}
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                med = df[col].median()
                self.impute_medians[col] = 0.0 if pd.isna(med) else float(med)
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                mode = df[col].mode()
                self.cat_modes[col] = str(mode.iloc[0]) if not mode.empty else "unknown"

        X_train = self._build_features(df, fit_cats=True)
        self.scaler = StandardScaler().fit(X_train)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df, fit_cats=False)
        return self.scaler.transform(X)

    def _build_features(self, df: pd.DataFrame, fit_cats: bool) -> np.ndarray:
        num_parts: List[np.ndarray] = []
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                vals = df[col].fillna(self.impute_medians.get(col, 0.0)).to_numpy(dtype=float)
            else:
                vals = np.zeros(len(df), dtype=float)
            num_parts.append(vals.reshape(-1, 1))

        cat_parts: List[np.ndarray] = []
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                series = df[col].fillna(self.cat_modes.get(col, "unknown")).astype(str)
            else:
                series = pd.Series(["unknown"] * len(df))

            if fit_cats:
                dummies = pd.get_dummies(series, prefix=col)
                self.cat_columns_[col] = list(dummies.columns)
                cat_parts.append(dummies.to_numpy(dtype=float))
            else:
                dummies = pd.get_dummies(series, prefix=col)
                aligned = dummies.reindex(columns=self.cat_columns_[col], fill_value=0)
                cat_parts.append(aligned.to_numpy(dtype=float))

        all_parts = num_parts + cat_parts
        return np.hstack(all_parts)


def compute_soft_targets(df_gt_train: pd.DataFrame) -> np.ndarray:
    n = len(df_gt_train)
    Y_soft = np.zeros((n, N_CLASSES), dtype=float)
    for i, (_, row) in enumerate(df_gt_train.iterrows()):
        dec = str(row["treatment_decision"]).strip()
        conf = str(row.get("confidence", "clear")).strip()
        k = CLASS_TO_IDX.get(dec, -1)
        c = CONFIDENCE_C.get(conf, 1.0)
        if k >= 0:
            y_k = 0.25 + 0.75 * c
            y_other = (1.0 - y_k) / (N_CLASSES - 1)
            Y_soft[i, :] = y_other
            Y_soft[i, k] = y_k
        else:
            Y_soft[i, :] = 1.0 / N_CLASSES
    return Y_soft


def generate_simplex_grid(step: float = 0.05) -> List[Tuple[float, float, float]]:
    grid = []
    ticks = int(round(1.0 / step))
    for i in range(ticks + 1):
        for j in range(ticks + 1 - i):
            k = ticks - i - j
            grid.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
    return grid


def compute_metrics(
    y_true_str: List[str],
    y_pred_str: List[str],
    probs: np.ndarray,
) -> Dict[str, float]:
    y_true_idx = np.array([CLASS_TO_IDX[s] for s in y_true_str])
    y_pred_idx = np.array([CLASS_TO_IDX[s] for s in y_pred_str])

    gate_scores = [
        compute_treatment_decision_gate(
            {"treatment_recommendation": {"primary": yt}},
            {"treatment_recommendation": {"primary": yp}},
        )[0]
        for yt, yp in zip(y_true_str, y_pred_str)
    ]
    gate_acc = float(np.mean(gate_scores))

    macro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0))

    # Multiclass Brier score
    Y_true_onehot = label_binarize(y_true_idx, classes=list(range(N_CLASSES)))
    if Y_true_onehot.shape[1] == 1:
        temp = np.zeros((len(y_true_idx), N_CLASSES))
        temp[:, y_true_idx] = 1.0
        Y_true_onehot = temp
    brier = float(np.mean(np.sum((probs - Y_true_onehot) ** 2, axis=1)))

    # AUROC OVR
    try:
        if len(np.unique(y_true_idx)) > 1:
            auroc = float(roc_auc_score(Y_true_onehot, probs, multi_class="ovr", average="macro"))
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5

    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(range(N_CLASSES)), zero_division=0
    )

    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "brier": brier,
        "auroc": auroc,
        "gate_acc": gate_acc,
    }
    for idx, cls_name in enumerate(CLASS_ORDER):
        metrics[f"f1_{cls_name}"] = float(f1_per_class[idx])

    return metrics


def main():
    print("=" * 80)
    print("RUNNING CHIMERA TASK 2 (SUBTASK 2.1) MCCV-30 BENCHMARK (exp_16)")
    print("=" * 80)

    # Load data
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_txt = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_splits = pd.read_csv(DATA_DIR / "mccv_splits_30.csv")

    pids = df_gt["pid"].tolist()
    mri_feature_cols = [c for c in df_mri.columns if c != "pid"]

    # Champion hyperparams from jtarquino_task2 exp_11
    # H1: hard SVC linear
    # H2: fuzzy LinearSVR
    # Weights for H2: (0.55 tab, 0.20 mri, 0.25 txt)
    h2_fixed_weights = (0.55, 0.20, 0.25)
    h1_fixed_weights = (0.85, 0.15, 0.00)

    h1_split_metrics: List[Dict[str, float]] = []
    h2_split_metrics: List[Dict[str, float]] = []

    for split_idx in range(N_SPLITS):
        split_col = f"split_{split_idx}"
        train_pids = set(df_splits[df_splits[split_col] == "train"]["pid"])
        val_pids = set(df_splits[df_splits[split_col] == "val"]["pid"])

        # Train / Val masks
        train_mask = df_gt["pid"].isin(train_pids)
        val_mask = df_gt["pid"].isin(val_pids)

        gt_train = df_gt[train_mask].copy()
        gt_val = df_gt[val_mask].copy()

        # 1. Tabular features
        tab_prep = Task2TabularPreprocessor()
        X_tab_train = tab_prep.fit(df_tab[train_mask]).transform(df_tab[train_mask])
        X_tab_val = tab_prep.transform(df_tab[val_mask])

        # 2. MRI features
        mri_scaler = StandardScaler()
        X_mri_train = mri_scaler.fit_transform(df_mri[train_mask][mri_feature_cols].to_numpy(dtype=float))
        X_mri_val = mri_scaler.transform(df_mri[val_mask][mri_feature_cols].to_numpy(dtype=float))

        # 3. Text features
        tfidf = TfidfVectorizer(max_features=None, lowercase=True)
        txt_train_series: pd.Series = pd.Series(df_txt[train_mask]["full_encounter_narrative"])
        txt_val_series: pd.Series = pd.Series(df_txt[val_mask]["full_encounter_narrative"])
        raw_txt_train = txt_train_series.fillna("").tolist()
        raw_txt_val = txt_val_series.fillna("").tolist()
        X_txt_train = tfidf.fit_transform(raw_txt_train).toarray()
        X_txt_val = tfidf.transform(raw_txt_val).toarray()

        # Targets
        y_train_str = [norm_treatment_decision(d) for d in gt_train["treatment_decision"]]
        y_val_str = [norm_treatment_decision(d) for d in gt_val["treatment_decision"]]
        y_train_idx = np.array([CLASS_TO_IDX[s] for s in y_train_str])

        # Soft targets for H2
        Y_soft_train = compute_soft_targets(gt_train)

        # -------------------------------------------------------------
        # Condition H1: Hard-SVC Trimodal Late Fusion
        # -------------------------------------------------------------
        svc_tab = SVC(kernel="linear", C=0.1, probability=True, random_state=42).fit(X_tab_train, y_train_idx)
        svc_mri = SVC(kernel="linear", C=0.1, probability=True, random_state=42).fit(X_mri_train, y_train_idx)
        svc_txt = SVC(kernel="linear", C=10.0, probability=True, random_state=42).fit(X_txt_train, y_train_idx)

        p_tab_val_h1 = svc_tab.predict_proba(X_tab_val)
        p_mri_val_h1 = svc_mri.predict_proba(X_mri_val)
        p_txt_val_h1 = svc_txt.predict_proba(X_txt_val)

        # Fix column alignment for predict_proba if classes were missed in fold
        def align_probs(prob_arr: np.ndarray, classes_present: np.ndarray) -> np.ndarray:
            full_prob = np.zeros((len(prob_arr), N_CLASSES))
            for local_i, global_c in enumerate(classes_present):
                full_prob[:, int(global_c)] = prob_arr[:, local_i]
            return full_prob

        p_tab_val_h1 = align_probs(p_tab_val_h1, svc_tab.classes_)
        p_mri_val_h1 = align_probs(p_mri_val_h1, svc_mri.classes_)
        p_txt_val_h1 = align_probs(p_txt_val_h1, svc_txt.classes_)

        w_tab_h1, w_mri_h1, w_txt_h1 = h1_fixed_weights
        p_fused_h1 = w_tab_h1 * p_tab_val_h1 + w_mri_h1 * p_mri_val_h1 + w_txt_h1 * p_txt_val_h1
        # Normalize
        row_sums = p_fused_h1.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        p_fused_h1 = p_fused_h1 / row_sums

        preds_idx_h1 = np.argmax(p_fused_h1, axis=1)
        preds_str_h1 = [CLASS_ORDER[i] for i in preds_idx_h1]

        m_h1 = compute_metrics(y_val_str, preds_str_h1, p_fused_h1)
        h1_split_metrics.append(m_h1)

        # -------------------------------------------------------------
        # Condition H2: Fuzzy-SVR Trimodal Late Fusion
        # -------------------------------------------------------------
        svr_tab = MultiOutputRegressor(LinearSVR(C=0.1, epsilon=0.05, random_state=42)).fit(X_tab_train, Y_soft_train)
        svr_mri = MultiOutputRegressor(LinearSVR(C=0.1, epsilon=0.1, random_state=42)).fit(X_mri_train, Y_soft_train)
        svr_txt = MultiOutputRegressor(LinearSVR(C=10.0, epsilon=0.05, random_state=42)).fit(X_txt_train, Y_soft_train)

        raw_tab_val = np.clip(svr_tab.predict(X_tab_val), 0, None)
        raw_mri_val = np.clip(svr_mri.predict(X_mri_val), 0, None)
        raw_txt_val = np.clip(svr_txt.predict(X_txt_val), 0, None)

        def l1_norm(arr):
            s = arr.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return arr / s

        p_tab_val_h2 = l1_norm(raw_tab_val)
        p_mri_val_h2 = l1_norm(raw_mri_val)
        p_txt_val_h2 = l1_norm(raw_txt_val)

        w_tab_h2, w_mri_h2, w_txt_h2 = h2_fixed_weights
        p_fused_h2 = w_tab_h2 * p_tab_val_h2 + w_mri_h2 * p_mri_val_h2 + w_txt_h2 * p_txt_val_h2
        p_fused_h2 = l1_norm(p_fused_h2)

        preds_idx_h2 = np.argmax(p_fused_h2, axis=1)
        preds_str_h2 = [CLASS_ORDER[i] for i in preds_idx_h2]

        m_h2 = compute_metrics(y_val_str, preds_str_h2, p_fused_h2)
        h2_split_metrics.append(m_h2)

    # Compute aggregate stats
    def aggregate(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        df = pd.DataFrame(metrics_list)
        agg = {}
        for col in df.columns:
            agg[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=1)),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        return agg

    agg_h1 = aggregate(h1_split_metrics)
    agg_h2 = aggregate(h2_split_metrics)

    results = {
        "H1_Hard_SVC": agg_h1,
        "H2_Fuzzy_SVR": agg_h2,
    }

    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "split_metrics.json", "w") as f:
        json.dump({"H1": h1_split_metrics, "H2": h2_split_metrics}, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (30 Splits MCCV):")
    print("=" * 80)
    print(f"H1 Hard-SVC : Macro-F1 = {agg_h1['macro_f1']['mean']:.4f} ± {agg_h1['macro_f1']['std']:.4f} | Brier = {agg_h1['brier']['mean']:.4f} | Gate Acc = {agg_h1['gate_acc']['mean']:.4f}")
    print(f"H2 Fuzzy-SVR: Macro-F1 = {agg_h2['macro_f1']['mean']:.4f} ± {agg_h2['macro_f1']['std']:.4f} | Brier = {agg_h2['brier']['mean']:.4f} | Gate Acc = {agg_h2['gate_acc']['mean']:.4f}")
    print("=" * 80)
    print("Per-class F1 for H2 Fuzzy-SVR:")
    for cls_name in CLASS_ORDER:
        print(f"  - {cls_name:24s}: {agg_h2[f'f1_{cls_name}']['mean']:.4f} ± {agg_h2[f'f1_{cls_name}']['std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
