#!/usr/bin/env python3
"""
Canonical MCCV-100 + LOOCV Benchmark for CHIMERA Task 2 Treatment Recommendation (exp_10).

Families:
  H1 = hard-KNN trimodal late fusion (one-hot 4-class targets)
  H2 = fuzzy-KNN trimodal late fusion (confidence-guided soft 4-class targets)

Protocol: DESIGN.md / IMPLEMENTATION.md (exp_10)
Governance: chimera_metrics.task2.decision.compute_treatment_decision_gate
            + sklearn.metrics homologation for Macro/Micro/Weighted-F1, AUROC, Brier.
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
from scipy.stats import wilcoxon
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    pairwise_distances,
)
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHIMERA_PKG = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHIMERA_PKG))

from chimera_metrics.task2.decision import compute_treatment_decision_gate  # noqa: E402
from experiments.exp_4.scripts.models import MRIPreprocessor, TextPreprocessor  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_10" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"

N_SPLITS = 100
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


def log(msg: str) -> None:
    """Compact real-time logging to stdout and file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Task 2 Tabular Preprocessor (fold-isolated, leak-safe)
# ---------------------------------------------------------------------------

class Task2TabularPreprocessor:
    """Fold-isolated MinMax scaler + one-hot encoding for the Task 2 tabular frame."""

    def __init__(self) -> None:
        self.impute_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_columns_: Dict[str, List[str]] = {}  # per-categorical-column dummy columns
        self.scaler: Optional[MinMaxScaler] = None

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
        self.scaler = MinMaxScaler().fit(X_train)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df, fit_cats=False)
        return np.clip(self.scaler.transform(X), 0.0, 1.0)

    def _build_features(self, df: pd.DataFrame, fit_cats: bool) -> np.ndarray:
        num_parts: List[np.ndarray] = []
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                num_parts.append(s.fillna(self.impute_medians.get(col, 0.0)).values)
            else:
                num_parts.append(np.zeros(len(df)))

        cat_parts: List[pd.DataFrame] = []
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                s = df[col].fillna(self.cat_modes.get(col, "unknown")).astype(str)
            else:
                s = pd.Series(["unknown"] * len(df))
            dummies = pd.get_dummies(s, prefix=col, drop_first=False)
            if fit_cats:
                self.cat_columns_[col] = dummies.columns.tolist()
            else:
                dummies = dummies.reindex(columns=self.cat_columns_.get(col, []), fill_value=0.0)
            cat_parts.append(dummies.astype(float))

        X = np.column_stack(num_parts)
        if cat_parts:
            X_cat = pd.concat(cat_parts, axis=1).values
            X = np.hstack([X, X_cat])
        return X


# ---------------------------------------------------------------------------
# Multiclass KNN (hard / fuzzy)
# ---------------------------------------------------------------------------

class MulticlassKNN:
    """4-class KNN classifier (hard) or fuzzy regressor (soft 4-d targets)."""

    def __init__(self, k: int, metric: str = "euclidean", weights: str = "uniform", fuzzy: bool = False):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.fuzzy = fuzzy
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None  # (n_train, 4) hard one-hot or soft

    def _build_targets(
        self,
        y_multiclass: List[str],
        y_confidence: Optional[List[str]] = None,
    ) -> np.ndarray:
        n = len(y_multiclass)
        Y = np.zeros((n, N_CLASSES))
        for i, cls in enumerate(y_multiclass):
            idx = CLASS_TO_IDX.get(cls)
            if idx is None:
                raise ValueError(f"Unknown class label: {cls!r}")
            Y[i, idx] = 1.0

        if self.fuzzy:
            conf = y_confidence or ["clear"] * n
            for i, cls in enumerate(y_multiclass):
                c = CONFIDENCE_C.get(str(conf[i]).lower() if i < len(conf) else "clear", 1.0)
                idx = CLASS_TO_IDX[cls]
                y_k = 0.25 + 0.75 * c
                Y[i, :] = (1.0 - y_k) / (N_CLASSES - 1)
                Y[i, idx] = y_k
        return Y

    def fit(
        self,
        X_train: np.ndarray,
        y_multiclass: List[str],
        y_confidence: Optional[List[str]] = None,
    ) -> "MulticlassKNN":
        self.X_train = np.asarray(X_train, dtype=float)
        self.Y_train = self._build_targets(y_multiclass, y_confidence)
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        X_v = np.asarray(X_val, dtype=float)
        dists = pairwise_distances(X_v, self.X_train, metric=self.metric)
        dists = np.nan_to_num(dists, nan=1e6)

        k_eff = min(self.k, self.X_train.shape[0])
        idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
        probs = np.zeros((len(X_v), N_CLASSES))
        for i in range(len(X_v)):
            row_idx = idx[i]
            row_dist = dists[i, row_idx]
            order = np.argsort(row_dist)
            sorted_idx = row_idx[order]
            sorted_dist = row_dist[order]
            w = 1.0 / (sorted_dist + 1e-5) if self.weights == "distance" else np.ones_like(sorted_dist)
            probs[i] = np.sum(w[:, None] * self.Y_train[sorted_idx], axis=0) / (np.sum(w) + 1e-9)

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        return probs / row_sums

    def predict(self, X_val: np.ndarray) -> List[str]:
        probs = self.predict_proba(X_val)
        return [CLASS_ORDER[int(np.argmax(p))] for p in probs]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_split_metrics(
    y_true: List[str],
    y_pred: List[str],
    probs: Optional[np.ndarray] = None,
    soft_targets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    per_class_f1 = {cls: float(f) for cls, f in zip(CLASS_ORDER, per_class[2])}

    auroc = None
    if probs is not None and len(np.unique([CLASS_TO_IDX[c] for c in y_true])) >= 2:
        try:
            y_bin = np.array([CLASS_TO_IDX[c] for c in y_true])
            auroc = float(roc_auc_score(
                y_bin, probs, multi_class="ovr", average="macro", labels=list(range(N_CLASSES))
            ))
        except ValueError:
            auroc = None

    gate_scores = []
    for gt, pred in zip(y_true, y_pred):
        score, _, _, _ = compute_treatment_decision_gate(gt, pred)
        gate_scores.append(score)
    gate_acc = float(np.mean(gate_scores)) if gate_scores else 0.0

    brier = None
    if probs is not None:
        if soft_targets is not None:
            Y_ref = soft_targets[: len(y_true)]
        else:
            Y_ref = np.zeros((len(y_true), N_CLASSES))
            for i, cls in enumerate(y_true):
                Y_ref[i, CLASS_TO_IDX[cls]] = 1.0
        brier = float(np.mean(np.sum((probs - Y_ref) ** 2, axis=1)))

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "auroc": auroc,
        "gate_acc": gate_acc,
        "brier": brier,
        "per_class_f1": per_class_f1,
    }


def load_task2_splits() -> Tuple[pd.DataFrame, List[Dict[str, List[str]]]]:
    """Load 100 MCCV splits (custom loader, bypassing 30-split validator)."""
    df_splits = pd.read_csv(DATA_DIR / "mccv_splits_100.csv")
    split_cols = [c for c in df_splits.columns if c.startswith("split_")]
    if len(split_cols) != N_SPLITS:
        raise ValueError(f"Expected {N_SPLITS} split columns, found {len(split_cols)}.")
    split_configs = []
    for col in split_cols:
        train_pids = df_splits.loc[df_splits[col] == "train", "pid"].astype(str).tolist()
        val_pids = df_splits.loc[df_splits[col] == "val", "pid"].astype(str).tolist()
        split_configs.append({"train_pids": train_pids, "val_pids": val_pids})
    return df_splits, split_configs


# ---------------------------------------------------------------------------
# Phase A: unimodal grids + probability extraction
# ---------------------------------------------------------------------------

def prepare_fold_cache(
    df_tab: pd.DataFrame,
    df_mri: pd.DataFrame,
    df_text: pd.DataFrame,
    df_gt: pd.DataFrame,
    split_configs: List[Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    fold_cache = []
    for split_cfg in split_configs:
        tr_pids, val_pids = split_cfg["train_pids"], split_cfg["val_pids"]

        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[val_pids].reset_index()
        prep_tab = Task2TabularPreprocessor().fit(tr_tab)
        X_tab_tr, X_tab_val = prep_tab.transform(tr_tab), prep_tab.transform(val_tab)

        X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_val = df_mri.set_index("pid").loc[val_pids][emb_cols].values.astype(float)
        prep_mri = MRIPreprocessor(n_components=None).fit(X_mri_tr)
        X_mri_tr_p, X_mri_val_p = prep_mri.transform(X_mri_tr), prep_mri.transform(X_mri_val)

        tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        val_txt = df_text.set_index("pid").loc[val_pids]["full_encounter_narrative"].tolist()
        text_cache = {}
        for max_feat in (500, None):
            prep_txt = TextPreprocessor(max_features=max_feat, n_components=None).fit(tr_txt)
            text_cache[max_feat] = (prep_txt.transform(tr_txt), prep_txt.transform(val_txt))

        gt_tr = df_gt.set_index("pid").loc[tr_pids]
        gt_val = df_gt.set_index("pid").loc[val_pids]
        fold_cache.append({
            "val_pids": val_pids,
            "y_tr": gt_tr["treatment_decision"].tolist(),
            "y_tr_conf": gt_tr["confidence"].fillna("clear").tolist(),
            "y_val": gt_val["treatment_decision"].tolist(),
            "X_tab_tr": X_tab_tr, "X_tab_val": X_tab_val,
            "X_mri_tr": X_mri_tr_p, "X_mri_val": X_mri_val_p,
            "text_cache": text_cache,
        })
    return fold_cache


def run_unimodal_phase_a(
    fold_cache: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
    modality: str,
    family: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Grid search over one modality for one family.

    Returns:
        (best_config, config_scores, fold_probs)
    """
    best_cfg, best_f1 = None, -1.0
    cfg_scores = []
    fold_probs_best = None

    for cfg in grid:
        split_f1s = []
        split_probs = []
        for fc in fold_cache:
            if modality == "tab":
                X_tr, X_val = fc["X_tab_tr"], fc["X_tab_val"]
            elif modality == "mri":
                X_tr, X_val = fc["X_mri_tr"], fc["X_mri_val"]
            else:
                mf = cfg.get("max_features", 500)
                X_tr, X_val = fc["text_cache"][mf]

            knn = MulticlassKNN(
                k=cfg["k"], metric=cfg["metric"], weights=cfg["weights"], fuzzy=cfg["fuzzy"]
            ).fit(X_tr, fc["y_tr"], fc["y_tr_conf"])
            probs_val = knn.predict_proba(X_val)
            y_pred = [CLASS_ORDER[int(np.argmax(p))] for p in probs_val]

            m = compute_split_metrics(fc["y_val"], y_pred, probs_val)
            split_f1s.append(m["macro_f1"])
            split_probs.append({"val_pids": fc["val_pids"], "probs": probs_val, "y_val": fc["y_val"]})

        mean_f1 = float(np.mean(split_f1s))
        cfg_scores.append({
            "config": cfg,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": float(np.std(split_f1s)),
        })
        if mean_f1 > best_f1:
            best_f1, best_cfg, fold_probs_best = mean_f1, cfg, split_probs

    log(f"  Best {family}/{modality}: k={best_cfg['k']}, weights={best_cfg['weights']}, "
        f"metric={best_cfg['metric']}, fuzzy={best_cfg['fuzzy']} | Macro-F1={best_f1:.4f}")
    return best_cfg, cfg_scores, fold_probs_best


# ---------------------------------------------------------------------------
# Late fusion (weights grid over simplex Δ²)
# ---------------------------------------------------------------------------

def run_fusion_search(
    probs_tab: List[Dict[str, Any]],
    probs_mri: List[Dict[str, Any]],
    probs_txt: List[Dict[str, Any]],
    family: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Search fusion weights w ∈ Δ² (step 0.05) maximizing mean val Macro-F1."""
    step = 0.05
    weight_candidates = []
    for w1_i in range(int(1.0 / step) + 1):
        for w2_i in range(int(1.0 / step) + 1 - w1_i):
            w1 = round(w1_i * step, 2)
            w2 = round(w2_i * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= -1e-6:
                weight_candidates.append((w1, w2, max(0.0, w3)))

    best_cfg, best_f1 = None, -1.0
    fusion_results = []

    for (w_tab, w_mri, w_txt) in weight_candidates:
        split_f1s, split_briers = [], []
        for p_tab, p_mri, p_txt in zip(probs_tab, probs_mri, probs_txt):
            p_fused = w_tab * p_tab["probs"] + w_mri * p_mri["probs"] + w_txt * p_txt["probs"]
            y_pred = [CLASS_ORDER[int(np.argmax(p))] for p in p_fused]
            m = compute_split_metrics(p_tab["y_val"], y_pred, p_fused)
            split_f1s.append(m["macro_f1"])
            if m["brier"] is not None:
                split_briers.append(m["brier"])

        mean_f1 = float(np.mean(split_f1s))
        cfg_entry = {
            "family": family,
            "w_tab": w_tab,
            "w_mri": w_mri,
            "w_txt": w_txt,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": float(np.std(split_f1s)),
            "mean_brier": float(np.mean(split_briers)) if split_briers else None,
        }
        fusion_results.append(cfg_entry)
        if mean_f1 > best_f1:
            best_f1, best_cfg = mean_f1, cfg_entry

    log(f"  Best fusion {family}: w=({best_cfg['w_tab']}, {best_cfg['w_mri']}, {best_cfg['w_txt']}) | "
        f"Macro-F1={best_cfg['mean_macro_f1']:.4f} | Brier={best_cfg['mean_brier']:.4f}")
    return best_cfg, fusion_results


# ---------------------------------------------------------------------------
# Phase B: Frozen LOOCV
# ---------------------------------------------------------------------------

def run_loocv(
    df_tab: pd.DataFrame,
    df_mri: pd.DataFrame,
    df_text: pd.DataFrame,
    df_gt: pd.DataFrame,
    best_cfg_hard: Dict[str, Any],
    best_cfg_fuzzy: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase B: Leave-One-Out CV with hyperparameters frozen from Phase A.

    Evaluates both families (H1 hard fusion, H2 fuzzy fusion).
    """
    pids = df_gt["pid"].tolist()
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    n = len(pids)

    conditions = {"H1_hard_fusion": "hard", "H2_fuzzy_fusion": "fuzzy"}
    results = {}

    for cond_name, family in conditions.items():
        y_true_all, y_pred_all = [], []
        proba_all = []
        soft_all = []

        for hold_out in pids:
            tr_pids = [p for p in pids if p != hold_out]

            tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
            val_tab = df_tab.set_index("pid").loc[[hold_out]].reset_index()
            prep_tab = Task2TabularPreprocessor().fit(tr_tab)
            X_tab_tr, X_tab_val = prep_tab.transform(tr_tab), prep_tab.transform(val_tab)

            X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
            X_mri_val = df_mri.set_index("pid").loc[[hold_out]][emb_cols].values.astype(float)
            prep_mri = MRIPreprocessor(n_components=None).fit(X_mri_tr)
            X_mri_tr_p, X_mri_val_p = prep_mri.transform(X_mri_tr), prep_mri.transform(X_mri_val)

            tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
            val_txt = df_text.set_index("pid").loc[[hold_out]]["full_encounter_narrative"].tolist()

            gt_tr = df_gt.set_index("pid").loc[tr_pids]
            y_tr = gt_tr["treatment_decision"].tolist()
            y_tr_conf = gt_tr["confidence"].fillna("clear").tolist()

            # Build per-modality frozen KNNs
            cfgs = best_cfg_hard if family == "hard" else best_cfg_fuzzy

            knn_tab = MulticlassKNN(
                k=cfgs["tab"]["k"], metric=cfgs["tab"]["metric"],
                weights=cfgs["tab"]["weights"], fuzzy=cfgs["tab"]["fuzzy"],
            ).fit(X_tab_tr, y_tr, y_tr_conf)
            p_tab = knn_tab.predict_proba(X_tab_val)[0]

            max_feat = cfgs["text"].get("max_features", 500)
            prep_txt = TextPreprocessor(max_features=max_feat, n_components=None).fit(tr_txt)
            X_txt_tr_p, X_txt_val_p = prep_txt.transform(tr_txt), prep_txt.transform(val_txt)
            knn_txt = MulticlassKNN(
                k=cfgs["text"]["k"], metric=cfgs["text"]["metric"],
                weights=cfgs["text"]["weights"], fuzzy=cfgs["text"]["fuzzy"],
            ).fit(X_txt_tr_p, y_tr, y_tr_conf)
            p_txt = knn_txt.predict_proba(X_txt_val_p)[0]

            knn_mri = MulticlassKNN(
                k=cfgs["mri"]["k"], metric=cfgs["mri"]["metric"],
                weights=cfgs["mri"]["weights"], fuzzy=cfgs["mri"]["fuzzy"],
            ).fit(X_mri_tr_p, y_tr, y_tr_conf)
            p_mri = knn_mri.predict_proba(X_mri_val_p)[0]

            w = cfgs["fusion"]
            p_fused = w["w_tab"] * p_tab + w["w_mri"] * p_mri + w["w_txt"] * p_txt
            y_pred = CLASS_ORDER[int(np.argmax(p_fused))]

            y_true_all.append(df_gt.set_index("pid").loc[hold_out, "treatment_decision"])
            y_pred_all.append(y_pred)
            proba_all.append(p_fused)

            # Soft target reference for Brier
            y_k = df_gt.set_index("pid").loc[hold_out, "treatment_decision"]
            conf_hold = df_gt.set_index("pid").loc[hold_out, "confidence"]
            if family == "fuzzy":
                c = CONFIDENCE_C.get(str(conf_hold).lower(), 1.0)
                k_idx = CLASS_TO_IDX[y_k]
                soft = np.full(N_CLASSES, (1.0 - (0.25 + 0.75 * c)) / (N_CLASSES - 1))
                soft[k_idx] = 0.25 + 0.75 * c
            else:
                soft = np.zeros(N_CLASSES)
                soft[CLASS_TO_IDX[y_k]] = 1.0
            soft_all.append(soft)

        probs_arr = np.array(proba_all)
        soft_arr = np.array(soft_all)
        m = compute_split_metrics(y_true_all, y_pred_all, probs_arr, soft_arr)
        results[cond_name] = m
        log(f"  LOOCV {cond_name}: Macro-F1={m['macro_f1']:.4f} | Micro-F1={m['micro_f1']:.4f} | "
            f"Weighted-F1={m['weighted_f1']:.4f} | Brier={m['brier']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 90)
    log("CHIMERA Task 2 -- Trimodal Hard/Fuzzy KNN Late Fusion (exp_10)")
    log("=" * 90)

    # 1. Ingest
    log("[1/5] Ingesting canonical data + 100 MCCV splits...")
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    _, split_configs = load_task2_splits()

    # Leakage audit
    leaked = [c for c in df_tab.columns if "active_treatment_flag" in c]
    if leaked:
        raise RuntimeError(f"Leakage detected: {leaked}")

    log(f"  Loaded: N={len(df_gt)} | Splits={N_SPLITS} | Features: tab={df_tab.shape[1]-1}, "
        f"mri={df_mri.shape[1]-1}, text=1")

    # 2. Phase A: prepare caches + unimodal grids
    log("[2/5] Phase A: preparing per-split caches and running unimodal grids...")
    t0 = time.time()
    fold_cache = prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs)

    tab_grid_hard = [{"k": k, "metric": "euclidean", "weights": w, "fuzzy": False}
                     for k in (1, 3, 5, 7) for w in ("uniform", "distance")]
    tab_grid_fuzzy = [{"k": k, "metric": "euclidean", "weights": w, "fuzzy": True}
                      for k in (1, 3, 5) for w in ("uniform", "distance")]
    mri_grid_hard = [{"k": k, "metric": "euclidean", "weights": w, "fuzzy": False}
                     for k in (1, 3, 5) for w in ("uniform", "distance")]
    mri_grid_fuzzy = [{"k": k, "metric": "euclidean", "weights": w, "fuzzy": True}
                      for k in (1, 3, 5) for w in ("uniform", "distance")]
    text_grid_hard = [{"k": k, "metric": "cosine", "weights": "uniform", "fuzzy": False, "max_features": 500}
                      for k in (1, 3)]
    text_grid_fuzzy = [{"k": k, "metric": "cosine", "weights": "uniform", "fuzzy": True, "max_features": None}
                       for k in (1, 3, 5)]

    best_cfgs = {"hard": {}, "fuzzy": {}}
    probs_cache = {"hard": {}, "fuzzy": {}}

    for family, prefix in (("hard", "H1"), ("fuzzy", "H2")):
        grids = {
            "tab": tab_grid_hard if family == "hard" else tab_grid_fuzzy,
            "mri": mri_grid_hard if family == "hard" else mri_grid_fuzzy,
            "text": text_grid_hard if family == "hard" else text_grid_fuzzy,
        }
        for modality in ("tab", "mri", "text"):
            best_cfg, _, probs = run_unimodal_phase_a(fold_cache, grids[modality], modality, family)
            best_cfgs[family][modality] = best_cfg
            probs_cache[family][modality] = probs

        # 3. Fusion search
        log(f"[3/5] Fusion search ({prefix}) over Δ²...")
        best_fusion, fusion_results = run_fusion_search(
            probs_cache[family]["tab"],
            probs_cache[family]["mri"],
            probs_cache[family]["text"],
            family,
        )
        best_cfgs[family]["fusion"] = best_fusion
        best_cfgs[family]["fusion_results"] = fusion_results

    elapsed_a = time.time() - t0
    log(f"Phase A completed in {elapsed_a:.1f}s")

    # Persist best hyperparameters (excluding bulky fusion_results)
    persist_cfgs = {
        family: {k: v for k, v in cfgs.items() if k != "fusion_results"}
        for family, cfgs in best_cfgs.items()
    }
    with open(RESULTS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(persist_cfgs, f, indent=2)

    # 4. Phase B: frozen LOOCV
    log("[4/5] Phase B: frozen LOOCV...")
    loocv_results = run_loocv(
        df_tab, df_mri, df_text, df_gt,
        best_cfgs["hard"], best_cfgs["fuzzy"],
    )

    # 5. Aggregate + persist
    log("[5/5] Aggregating metrics + statistical tests...")

    # Per-split metrics for both families using frozen best configs
    split_metrics_all = {"H1_hard_fusion": [], "H2_fuzzy_fusion": []}

    for family, cond_name in (("hard", "H1_hard_fusion"), ("fuzzy", "H2_fuzzy_fusion")):
        w = best_cfgs[family]["fusion"]
        for i, fc in enumerate(fold_cache):
            p_tab = probs_cache[family]["tab"][i]["probs"]
            p_mri = probs_cache[family]["mri"][i]["probs"]
            p_txt = probs_cache[family]["text"][i]["probs"]
            p_fused = w["w_tab"] * p_tab + w["w_mri"] * p_mri + w["w_txt"] * p_txt
            y_pred = [CLASS_ORDER[int(np.argmax(p))] for p in p_fused]
            m = compute_split_metrics(fc["y_val"], y_pred, p_fused)
            m["split_idx"] = i
            split_metrics_all[cond_name].append(m)

    # Aggregate
    aggregate_metrics = {}
    for cond_name, metrics_list in split_metrics_all.items():
        df_m = pd.DataFrame(metrics_list)
        agg = {}
        for col in ["macro_f1", "micro_f1", "weighted_f1", "gate_acc"]:
            agg[col] = {"mean": float(df_m[col].mean()), "std": float(df_m[col].std())}
        for col in ["auroc", "brier"]:
            valid = df_m[col].dropna()
            agg[col] = {"mean": float(valid.mean()) if len(valid) else None,
                        "std": float(valid.std()) if len(valid) else None}
        agg["per_class_f1"] = {
            cls: {"mean": float(df_m["per_class_f1"].apply(lambda x: x.get(cls)).mean()),
                  "std": float(df_m["per_class_f1"].apply(lambda x: x.get(cls)).std())}
            for cls in CLASS_ORDER
        }
        agg["loocv_"] = loocv_results.get(cond_name, {})
        aggregate_metrics[cond_name] = agg

    # Wilcoxon tests H1 vs H2 (paired across 100 splits)
    f1_h1 = [m["macro_f1"] for m in split_metrics_all["H1_hard_fusion"]]
    f1_h2 = [m["macro_f1"] for m in split_metrics_all["H2_fuzzy_fusion"]]
    brier_h1 = [m["brier"] for m in split_metrics_all["H1_hard_fusion"] if m["brier"] is not None]
    brier_h2 = [m["brier"] for m in split_metrics_all["H2_fuzzy_fusion"] if m["brier"] is not None]

    wilcoxon_tests = {}
    diffs_f1 = np.array(f1_h1) - np.array(f1_h2)
    if np.all(diffs_f1 == 0):
        wilcoxon_tests["H1_vs_H2_macro_f1"] = {"statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0}
    else:
        res = wilcoxon(f1_h1, f1_h2, zero_method="wilcox", alternative="two-sided")
        wilcoxon_tests["H1_vs_H2_macro_f1"] = {
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "mean_difference": float(np.mean(diffs_f1)),
            "is_significant_alpha_0_05": bool(res.pvalue < 0.05),
        }
    if len(brier_h1) == len(brier_h2) and len(brier_h1) > 0:
        diffs_b = np.array(brier_h1) - np.array(brier_h2)
        if np.all(diffs_b == 0):
            wilcoxon_tests["H1_vs_H2_brier"] = {"statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0}
        else:
            res_b = wilcoxon(brier_h1, brier_h2, zero_method="wilcox", alternative="two-sided")
            wilcoxon_tests["H1_vs_H2_brier"] = {
                "statistic": float(res_b.statistic), "p_value": float(res_b.pvalue),
                "mean_difference": float(np.mean(diffs_b)),
                "is_significant_alpha_0_05": bool(res_b.pvalue < 0.05),
            }

    # Persist
    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)
    with open(RESULTS_DIR / "split_metrics.json", "w", encoding="utf-8") as f:
        json.dump(split_metrics_all, f, indent=2)
    with open(RESULTS_DIR / "wilcoxon_tests.json", "w", encoding="utf-8") as f:
        json.dump(wilcoxon_tests, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_10 execution timestamp: {datetime.now().isoformat()}\n")
        f.write(f"git SHA: (commit hash unavailable from runner)\n")

    # Summary table
    log("\n" + "=" * 100)
    log(f"{'Condition':<20} | {'Macro-F1':<20} | {'Micro-F1':<20} | {'Weighted-F1':<20} | Brier")
    log("-" * 100)
    for cond in ["H1_hard_fusion", "H2_fuzzy_fusion"]:
        a = aggregate_metrics[cond]
        log(f"{cond:<20} | {a['macro_f1']['mean']:.4f}±{a['macro_f1']['std']:.4f} | "
            f"{a['micro_f1']['mean']:.4f} | {a['weighted_f1']['mean']:.4f} | "
            f"{a['brier']['mean']:.4f}")
    log("=" * 100)

    for test, res in wilcoxon_tests.items():
        sig = "SIG" if res.get("is_significant_alpha_0_05") else "NS"
        log(f"  {test}: diff={res['mean_difference']:+.4f} | p={res['p_value']:.4e} [{sig}]")

    log("\nexp_10 completed successfully.")


if __name__ == "__main__":
    main()
