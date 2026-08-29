#!/usr/bin/env python3
"""
Canonical MCCV-100 + LOOCV benchmark for the three Task-2 subtasks (exp_12).

Subtask 2.1: Treatment recommendation (4-class) -- frozen exp_10 configs.
Subtask 2.2: Clinical confidence (3-class ordinal) -- fuzzy-KNN grid + fusion.
Subtask 2.3: Variable weights (11 vars x 4 ordinal levels) -- RF/ExtraTrees meta-models.

Governance: chimera_metrics.task2 / task1.reasoning gates + sklearn homologation.
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
from scipy.stats import wilcoxon, spearmanr
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHIMERA_PKG = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHIMERA_PKG))

from chimera_metrics.task2.decision import compute_treatment_decision_gate  # noqa: E402
from chimera_metrics.task1.reasoning import (  # noqa: E402
    compute_confidence_score,
    compute_variable_weights_score,
    compute_important_decisive_f1,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_12" / "results"
EXP10_RESULTS = PROJECT_ROOT / "experiments" / "exp_10" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"

N_SPLITS = 100
CLASS_ORDER_4 = ["watchful_waiting", "continued_surveillance", "active_surveillance", "active_treatment"]
CLASS_ORDER_3 = ["uncertain", "borderline", "clear"]
WEIGHT_ORDER = ["not_used", "noted", "important", "decisive"]
VARIABLES_11 = ["age", "bx_gl_prim", "bx_gl_sec", "bx_isup", "comorbidity", "cspca", "ct", "fh", "pirads", "psa", "psad"]

IDX_4 = {c: i for i, c in enumerate(CLASS_ORDER_4)}
IDX_3 = {c: i for i, c in enumerate(CLASS_ORDER_3)}
IDX_W = {c: i for i, c in enumerate(WEIGHT_ORDER)}
WEIGHT_IDX = IDX_W
CONF_ORDINAL = {"uncertain": 0.25, "borderline": 0.5, "clear": 1.0}
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
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Task2TabularPreprocessor:
    def __init__(self) -> None:
        self.impute_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_columns_: Dict[str, List[str]] = {}
        self.scaler: Optional[MinMaxScaler] = None

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                med = df[col].median()
                self.impute_medians[col] = 0.0 if pd.isna(med) else float(med)
        for col in T2_CATEGORICAL_COLS:
            st = df[col].mode()
            self.cat_modes[col] = str(st.iloc[0]) if not st.empty else "unknown"
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
            X = np.hstack([X, pd.concat(cat_parts, axis=1).values])
        return X


class MRIPreprocessor:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MRIPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.mean_vector = np.mean(X_arr, axis=0)
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0):
                X_arr[i] = self.mean_vector
        if self.n_components is not None:
            max_comp = min(X_arr.shape[0] - 1, X_arr.shape[1], self.n_components)
            self.pca = PCA(n_components=max(1, max_comp), random_state=42).fit(X_arr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0) and self.mean_vector is not None:
                X_arr[i] = self.mean_vector
        if self.pca is not None:
            return self.pca.transform(X_arr)
        return X_arr


class TextPreprocessor:
    def __init__(self, max_features: Optional[int] = 1000, n_components: Optional[int] = None):
        self.max_features = max_features
        self.n_components = n_components
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: List[str]) -> "TextPreprocessor":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b\w+(?:-\w+)*\b",
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


class MulticlassKNN:
    def __init__(self, k: int, metric: str = "euclidean", weights: str = "uniform", fuzzy: bool = False, n_classes: int = 4):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.fuzzy = fuzzy
        self.n_classes = n_classes
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None

    def _build_targets(self, y_multiclass: List[str], y_confidence: Optional[List[str]] = None, conf_type: str = "treatment") -> np.ndarray:
        n = len(y_multiclass)
        Y = np.zeros((n, self.n_classes))
        if conf_type == "treatment":
            for i, cls in enumerate(y_multiclass):
                if cls not in IDX_4:
                    raise ValueError(f"Unknown treatment label {cls!r}")
                Y[i, IDX_4[cls]] = 1.0
            if self.fuzzy:
                conf = y_confidence or ["clear"] * n
                for i, cls in enumerate(y_multiclass):
                    c = CONFIDENCE_C.get(str(conf[i]).lower() if i < len(conf) else "clear", 1.0)
                    y_k = 0.25 + 0.75 * c
                    Y[i, :] = (1.0 - y_k) / (self.n_classes - 1)
                    Y[i, IDX_4[cls]] = y_k
        else:  # confidence 3-class ordinal
            for i, cls in enumerate(y_multiclass):
                if cls not in IDX_3:
                    raise ValueError(f"Unknown confidence label {cls!r}")
                Y[i, IDX_3[cls]] = 1.0
                if self.fuzzy:
                    Y[i, IDX_3[cls]] = CONF_ORDINAL.get(cls, 0.5)
        return Y

    def fit(self, X_train: np.ndarray, y_multiclass: List[str], y_confidence: Optional[List[str]] = None, conf_type: str = "treatment") -> "MulticlassKNN":
        self.X_train = np.asarray(X_train, dtype=float)
        self.Y_train = self._build_targets(y_multiclass, y_confidence, conf_type)
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        X_v = np.asarray(X_val, dtype=float)
        dists = pairwise_distances(X_v, self.X_train, metric=self.metric)
        dists = np.nan_to_num(dists, nan=1e6)
        k_eff = min(self.k, self.X_train.shape[0])
        idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
        probs = np.zeros((len(X_v), self.n_classes))
        for i in range(len(X_v)):
            row_idx = idx[i]
            row_dist = dists[i, row_idx]
            order = np.argsort(row_dist)
            si = row_idx[order]
            sd = row_dist[order]
            w = 1.0 / (sd + 1e-5) if self.weights == "distance" else np.ones_like(sd)
            probs[i] = np.sum(w[:, None] * self.Y_train[si], axis=0) / (np.sum(w) + 1e-9)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        return probs / row_sums

    def predict(self, X_val: np.ndarray, class_order: List[str]) -> List[str]:
        probs = self.predict_proba(X_val)
        return [class_order[int(np.argmax(p))] for p in probs]


def compute_split_metrics(y_true: List[str], y_pred: List[str], probs: Optional[np.ndarray] = None,
                          soft_targets: Optional[np.ndarray] = None, class_order: Optional[List[str]] = None,
                          n_classes: int = 4) -> Dict[str, Any]:
    class_order = class_order or CLASS_ORDER_4
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0))
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=class_order, zero_division=0)
    per_class_f1 = {c: float(f) for c, f in zip(class_order, per_class[2])}

    auroc = None
    if probs is not None and len(np.unique([class_order.index(c) for c in y_true])) >= 2:
        try:
            y_bin = np.array([class_order.index(c) for c in y_true])
            auroc = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro", labels=list(range(n_classes))))
        except ValueError:
            auroc = None

    brier = None
    if probs is not None:
        if soft_targets is not None:
            Y_ref = soft_targets[: len(y_true)]
        else:
            Y_ref = np.zeros((len(y_true), n_classes))
            for i, c in enumerate(y_true):
                Y_ref[i, class_order.index(c)] = 1.0
        brier = float(np.mean(np.sum((probs - Y_ref) ** 2, axis=1)))

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1,
            "auroc": auroc, "brier": brier, "per_class_f1": per_class_f1}


def load_task2_splits() -> Tuple[pd.DataFrame, List[Dict[str, List[str]]]]:
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


def load_var_weights_map(pids: List[str]) -> Dict[str, Dict[str, str]]:
    """Load variable_weights dict per pid; missing variables are left absent (implicit not_used)."""
    vw_map: Dict[str, Dict[str, str]] = {}
    for pid in pids:
        rp = RAW_DIR / pid / "prostate-treatment-decision-reasoning.json"
        with open(rp, "r", encoding="utf-8") as f:
            rr = json.load(f)
        vw_map[pid] = rr.get("variable_weights", {})
    return vw_map


def prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map) -> List[Dict[str, Any]]:
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
        prep_mri = MRIPreprocessor().fit(X_mri_tr)
        X_mri_tr_p, X_mri_val_p = prep_mri.transform(X_mri_tr), prep_mri.transform(X_mri_val)

        tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        val_txt = df_text.set_index("pid").loc[val_pids]["full_encounter_narrative"].tolist()
        text_cache = {}
        for mf in (500, None, 50):
            prep_txt = TextPreprocessor(max_features=mf).fit(tr_txt)
            text_cache[mf] = (prep_txt.transform(tr_txt), prep_txt.transform(val_txt))

        gt_tr = df_gt.set_index("pid").loc[tr_pids]
        gt_val = df_gt.set_index("pid").loc[val_pids]
        fold_cache.append({
            "val_pids": val_pids,
            "y_tr_treat": gt_tr["treatment_decision"].tolist(),
            "y_tr_conf": gt_tr["confidence"].fillna("clear").tolist(),
            "y_val_treat": gt_val["treatment_decision"].tolist(),
            "y_val_conf": gt_val["confidence"].fillna("clear").tolist(),
            "X_tab_tr": X_tab_tr, "X_tab_val": X_tab_val,
            "X_mri_tr": X_mri_tr_p, "X_mri_val": X_mri_val_p,
            "text_cache": text_cache,
            "vw_tr": {p: vw_map[p] for p in tr_pids},
            "vw_val": {p: vw_map[p] for p in val_pids},
        })
    return fold_cache


def run_reeval_21(fold_cache, best_exp10) -> Dict[str, Any]:
    results = {}
    for cond_name, family in (("C1_fuzzy", "fuzzy"), ("C2_hard", "hard")):
        cfg = best_exp10[family]
        split_metrics = []
        for fc in fold_cache:
            pt = MulticlassKNN(k=cfg["tab"]["k"], metric=cfg["tab"]["metric"], weights=cfg["tab"]["weights"],
                               fuzzy=cfg["tab"]["fuzzy"], n_classes=4).fit(fc["X_tab_tr"], fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(fc["X_tab_val"])
            pm = MulticlassKNN(k=cfg["mri"]["k"], metric=cfg["mri"]["metric"], weights=cfg["mri"]["weights"],
                               fuzzy=cfg["mri"]["fuzzy"], n_classes=4).fit(fc["X_mri_tr"], fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(fc["X_mri_val"])
            mf = cfg["text"].get("max_features", 500) if cfg["text"].get("max_features") is not None else None
            key = mf if mf in fc["text_cache"] else (500 if mf is None else mf)
            X_ttr, X_tva = fc["text_cache"].get(key, fc["text_cache"][500])
            px = MulticlassKNN(k=cfg["text"]["k"], metric=cfg["text"]["metric"], weights=cfg["text"]["weights"],
                               fuzzy=cfg["text"]["fuzzy"], n_classes=4).fit(X_ttr, fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(X_tva)
            w = cfg["fusion"]
            p_fused = w["w_tab"] * pt + w["w_mri"] * pm + w["w_txt"] * px
            y_pred = [CLASS_ORDER_4[int(np.argmax(p))] for p in p_fused]
            m = compute_split_metrics(fc["y_val_treat"], y_pred, p_fused, class_order=CLASS_ORDER_4, n_classes=4)
            split_metrics.append(m)
        results[cond_name] = split_metrics
        log(f"  [2.1] {cond_name}: Macro-F1={np.mean([m['macro_f1'] for m in split_metrics]):.4f}")
    return results


def run_grid_22(fold_cache) -> Tuple[Dict[str, Any], Dict[str, List[np.ndarray]]]:
    grids = {
        "tab": [{"k": k, "w": w} for k in (1, 3, 5, 7) for w in ("uniform", "distance")],
        "mri": [{"k": k, "w": w} for k in (1, 3, 5) for w in ("uniform", "distance")],
        "text": [{"k": k, "mf": mf} for k in (1, 3, 5) for mf in (500, None)],
    }
    best_cfg: Dict[str, Any] = {}
    probs_cache: Dict[str, List[np.ndarray]] = {}
    all_scores: Dict[str, List[Dict[str, Any]]] = {}
    for modality in ("tab", "mri", "text"):
        best_f1 = -1.0
        best_c: Optional[Dict[str, Any]] = None
        best_probs: Optional[List[np.ndarray]] = None
        best_scores: List[Dict[str, Any]] = []
        for cand in grids[modality]:
            split_f1s, split_probs = [], []
            for fc in fold_cache:
                if modality == "tab":
                    X_tr, X_val = fc["X_tab_tr"], fc["X_tab_val"]
                    knn = MulticlassKNN(k=cand["k"], metric="euclidean", weights=cand["w"], fuzzy=True, n_classes=3)
                elif modality == "mri":
                    X_tr, X_val = fc["X_mri_tr"], fc["X_mri_val"]
                    knn = MulticlassKNN(k=cand["k"], metric="euclidean", weights=cand["w"], fuzzy=True, n_classes=3)
                else:
                    key = cand["mf"] if cand["mf"] in fc["text_cache"] else 500
                    X_tr, X_val = fc["text_cache"][key]
                    knn = MulticlassKNN(k=cand["k"], metric="cosine", weights="uniform", fuzzy=True, n_classes=3)
                knn.fit(X_tr, fc["y_tr_conf"], None, "confidence")
                probs = knn.predict_proba(X_val)
                y_pred = [CLASS_ORDER_3[int(np.argmax(p))] for p in probs]
                m = compute_split_metrics(fc["y_val_conf"], y_pred, probs, class_order=CLASS_ORDER_3, n_classes=3)
                split_f1s.append(m["macro_f1"])
                split_probs.append(probs)
            mean_f1 = float(np.mean(split_f1s))
            if mean_f1 > best_f1:
                best_f1, best_c, best_probs = mean_f1, cand, split_probs
                best_scores = []  # recomputed later
        best_cfg[modality] = best_c
        probs_cache[modality] = best_probs
        log(f"  [2.2] Best {modality}: {best_c} | Macro-F1={best_f1:.4f}")

    # Fusion search over Delta^2
    step = 0.05
    w_cands = []
    for w1_i in range(int(1.0 / step) + 1):
        for w2_i in range(int(1.0 / step) + 1 - w1_i):
            w1 = round(w1_i * step, 2)
            w2 = round(w2_i * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= -1e-6:
                w_cands.append((w1, w2, max(0.0, w3)))
    best_w, best_f1w = None, -1.0
    for (wt, wm, wx) in w_cands:
        fs = []
        for i, fc in enumerate(fold_cache):
            fused = wt * probs_cache["tab"][i] + wm * probs_cache["mri"][i] + wx * probs_cache["text"][i]
            y_pred = [CLASS_ORDER_3[int(np.argmax(p))] for p in fused]
            m = compute_split_metrics(fc["y_val_conf"], y_pred, fused, class_order=CLASS_ORDER_3, n_classes=3)
            fs.append(m["macro_f1"])
        mean_f = float(np.mean(fs))
        if mean_f > best_f1w:
            best_f1w, best_w = mean_f, (wt, wm, wx)
    best_cfg["fusion"] = {"w_tab": best_w[0], "w_mri": best_w[1], "w_txt": best_w[2], "mean_macro_f1": best_f1w}
    log(f"  [2.2] Best fusion: w={best_w} | Macro-F1={best_f1w:.4f}")
    return best_cfg, probs_cache


def run_grid_23(fold_cache) -> Dict[str, Any]:
    """Subtask 2.3: per-variable RF vs ExtraTrees over joint 32+16+50 features."""
    joint_per_fold = []
    for fc in fold_cache:
        X_txt_tr_50, X_txt_val_50 = fc["text_cache"][50]
        prep_mri50 = MRIPreprocessor(n_components=16).fit(fc["X_mri_tr"])
        mri_tr16 = prep_mri50.transform(fc["X_mri_tr"])
        mri_val16 = prep_mri50.transform(fc["X_mri_val"])
        X_tr = np.hstack([fc["X_tab_tr"], mri_tr16, X_txt_tr_50])
        X_val = np.hstack([fc["X_tab_val"], mri_val16, X_txt_val_50])
        joint_per_fold.append((X_tr, X_val))

    algorithms = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}
    per_var_best: Dict[str, Dict[str, Any]] = {}
    per_var_scores: Dict[str, Dict[str, List[float]]] = {}
    for var in VARIABLES_11:
        per_var_scores[var] = {"rf": [], "et": []}
        for alg_name, alg in algorithms.items():
            sf1 = []
            for i, fc in enumerate(fold_cache):
                X_tr, X_val = joint_per_fold[i]
                y_tr = [WEIGHT_ORDER.index(fc["vw_tr"][p].get(var, "not_used")) for p in fc["vw_tr"]]
                y_val = [fc["vw_val"][p].get(var, "not_used") for p in fc["vw_val"]]
                model = alg(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
                model.fit(X_tr, y_tr)
                pred_idx = model.predict(X_val)
                pred = [WEIGHT_ORDER[i] for i in pred_idx]
                m = f1_score(y_val, pred, labels=WEIGHT_ORDER, average="macro", zero_division=0)
                sf1.append(m)
            per_var_scores[var][alg_name] = sf1
        best_alg = "rf" if np.mean(per_var_scores[var]["rf"]) >= np.mean(per_var_scores[var]["et"]) else "et"
        per_var_best[var] = {"algorithm": best_alg, "mean_macro_f1": float(np.mean(per_var_scores[var][best_alg]))}
        log(f"  [2.3] {var:<14} -> {best_alg} | Macro-F1={per_var_best[var]['mean_macro_f1']:.4f}")
    return {"per_var_best": per_var_best, "per_var_scores": per_var_scores}


def main() -> None:
    log("=" * 90)
    log("CHIMERA Task-2 -- Three-Subtask Benchmark (exp_12)")
    log("=" * 90)
    t0 = time.time()

    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    _, split_configs = load_task2_splits()
    leaked = [c for c in df_tab.columns if "active_treatment_flag" in c]
    if leaked:
        raise RuntimeError(f"Leakage detected: {leaked}")
    vw_map = load_var_weights_map(df_gt["pid"].tolist())

    with open(EXP10_RESULTS / "best_hyperparams.json", "r", encoding="utf-8") as f:
        best_exp10 = json.load(f)

    log("Preparing shared fold cache (features + confidence + variable-weights targets)...")
    fold_cache = prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    log("[1/3] Subtask 2.1: frozen exp_10 re-evaluation...")
    res_21 = run_reeval_21(fold_cache, best_exp10)

    log("[2/3] Subtask 2.2: fuzzy-KNN 3-class ordinal grid + fusion...")
    best_22, probs_22 = run_grid_22(fold_cache)

    log("[3/3] Subtask 2.3: RF/ExtraTrees per-variable meta-models...")
    res_23 = run_grid_23(fold_cache)

    elapsed = time.time() - t0
    log(f"All three subtasks completed in {elapsed:.1f}s")

    # Persist compact JSONs
    agg = {
        "subtask2_1": {c: {"mean_macro_f1": float(np.mean([m["macro_f1"] for m in res_21[c]])),
                           "std_macro_f1": float(np.std([m["macro_f1"] for m in res_21[c]]))} for c in res_21},
        "subtask2_2_best_config": best_22,
        "subtask2_3_per_var_best": res_23["per_var_best"],
    }
    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(RESULTS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump({"subtask2_2": best_22, "subtask2_3": res_23["per_var_best"]}, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_12 execution timestamp: {datetime.now().isoformat()}\n")
        f.write("Sub-skeleton complete: 2.1 reeval + 2.2 grid/fusion + 2.3 per-variable grid.\n")
    log("exp_12 benchmark completed and persisted.")


if __name__ == "__main__":
    main()
