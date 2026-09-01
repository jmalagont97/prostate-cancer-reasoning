#!/usr/bin/env python3
"""
Canonical MCCV-100 + LOOCV benchmark for the three Task-2 subtasks with linear-SVM (exp_13).

Subtask 2.1: Treatment recommendation (4-class) -- frozen exp_11 SVC/SVR configs.
Subtask 2.2: Clinical confidence (3-class ordinal) -- SVC/SVR grids + fusion + binary C6.
Subtask 2.3: Variable weights (11 vars x 4 ordinal) -- SVC / single-output SVR meta-models.

Phase B LOOCV is included for all three subtasks (closes the exp_12 protocol gap).
Governance: chimera_metrics.task2 / task1.reasoning gates + sklearn homologation.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

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
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_13" / "results"
EXP11_RESULTS = PROJECT_ROOT / "experiments" / "exp_11" / "results"
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
CONFIDENCE_C = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
CONF_ORDINAL = {"uncertain": 0.25, "borderline": 0.5, "clear": 1.0}
WEIGHT_ORDINAL = {"not_used": 0.0, "noted": 1.0 / 3.0, "important": 2.0 / 3.0, "decisive": 1.0}
WEIGHT_THRESHOLDS = [1.0 / 6.0, 0.5, 5.0 / 6.0]

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
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                med = df[col].median()
                self.impute_medians[col] = 0.0 if pd.isna(med) else float(med)
        for col in T2_CATEGORICAL_COLS:
            st = df[col].mode()
            self.cat_modes[col] = str(st.iloc[0]) if not st.empty else "unknown"
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


class MRIStandardPreprocessor:
    def __init__(self, use_pca_fallback: bool = False):
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.use_pca_fallback = use_pca_fallback

    def fit(self, X: np.ndarray) -> "MRIStandardPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.scaler.fit(X_arr)
        if self.use_pca_fallback:
            max_comp = min(X_arr.shape[0] - 1, X_arr.shape[1], 64)
            self.pca = PCA(n_components=max(1, max_comp), random_state=42).fit(self.scaler.transform(X_arr))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        X_scaled = self.scaler.transform(X_arr)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled


class TextPreprocessor:
    def __init__(self, max_features: Optional[int] = 1000):
        self.max_features = max_features
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


class HardLinearSVM:
    def __init__(self, C: float = 1.0, class_weight: Optional[str] = None, n_classes: Optional[int] = None):
        self.C = C
        self.class_weight = class_weight
        self.n_classes = n_classes
        self.model: Optional[SVC] = None
        self.fallback_model: Optional[LinearSVC] = None
        self._used_platt = True

    def fit(self, X_train: np.ndarray, y_idx: np.ndarray) -> "HardLinearSVM":
        self.classes_ = np.unique(np.asarray(y_idx))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.model = SVC(
                    kernel="linear", C=self.C, probability=True,
                    class_weight=self.class_weight, random_state=42,
                )
                self.model.fit(X_train, y_idx)
            self._used_platt = True
        except (Warning, ValueError, RuntimeError):
            self.fallback_model = LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=5000, random_state=42)
            self.fallback_model.fit(X_train, y_idx)
            self._used_platt = False
        return self

    def _pad_to_n_classes(self, raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
        if self.n_classes is None or raw.shape[1] >= self.n_classes:
            return raw
        full = np.zeros((raw.shape[0], self.n_classes))
        for j, c in enumerate(classes):
            full[:, int(c)] = raw[:, j]
        return full

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        if self._used_platt and self.model is not None:
            raw = self.model.predict_proba(X_val)
            return self._pad_to_n_classes(raw, self.classes_)
        if self.fallback_model is not None:
            scores = self.fallback_model.decision_function(X_val)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            raw = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self._pad_to_n_classes(raw, self.classes_)
        raise RuntimeError("No fitted model available.")


class SoftLinearSVR:
    def __init__(self, C: float = 1.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self.model: Optional[MultiOutputRegressor] = None

    def fit(self, X_train: np.ndarray, Y: np.ndarray) -> "SoftLinearSVR":
        base = LinearSVR(C=self.C, epsilon=self.epsilon, max_iter=5000, random_state=42)
        self.model = MultiOutputRegressor(base, n_jobs=2)
        self.model.fit(X_train, Y)
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X_val)
        clipped = np.clip(raw, 0.0, 1.0)
        row_sums = clipped.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        return clipped / row_sums


def build_treatment_soft(y_treat: List[str], y_conf: List[str], n_classes: int = 4) -> np.ndarray:
    Y = np.zeros((len(y_treat), n_classes))
    for i, cls in enumerate(y_treat):
        Y[i, IDX_4[cls]] = 1.0
    for i, cls in enumerate(y_treat):
        c = CONFIDENCE_C.get(str(y_conf[i]).lower() if i < len(y_conf) else "clear", 1.0)
        y_k = 0.25 + 0.75 * c
        Y[i, :] = (1.0 - y_k) / (n_classes - 1)
        Y[i, IDX_4[cls]] = y_k
    return Y


def build_confidence_soft(y_conf: List[str]) -> np.ndarray:
    Y = np.zeros((len(y_conf), 3))
    for i, cls in enumerate(y_conf):
        Y[i, IDX_3[cls]] = CONF_ORDINAL.get(cls, 0.5)
    return Y


def compute_split_metrics(y_true, y_pred, probs=None, soft_targets=None, class_order=None, n_classes=None):
    class_order = class_order or CLASS_ORDER_4
    n_classes = n_classes or len(class_order)
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0))
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=class_order, zero_division=0)
    per_class_f1 = {c: float(f) for c, f in zip(class_order, per_class[2])}

    auroc = None
    if probs is not None and len(np.unique([class_order.index(c) for c in y_true if c in class_order])) >= 2:
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
                if c in class_order:
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
        prep_mri = MRIStandardPreprocessor().fit(X_mri_tr)
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


def run_reeval_21(fold_cache, best_exp11) -> Dict[str, List[Dict[str, Any]]]:
    results = {}
    for cond_name, family in (("C1_fuzzy", "fuzzy"), ("C2_hard", "hard")):
        cfg = best_exp11[family]
        split_metrics = []
        for fc in fold_cache:
            if family == "fuzzy":
                Y_tr = build_treatment_soft(fc["y_tr_treat"], fc["y_tr_conf"], 4)
                m_tab = SoftLinearSVR(C=cfg["tab"]["C"], epsilon=cfg["tab"]["epsilon"]).fit(fc["X_tab_tr"], Y_tr)
                m_mri = SoftLinearSVR(C=cfg["mri"]["C"], epsilon=cfg["mri"]["epsilon"]).fit(fc["X_mri_tr"], Y_tr)
            else:
                y_idx = np.array([IDX_4[c] for c in fc["y_tr_treat"]])
                m_tab = HardLinearSVM(C=cfg["tab"]["C"], class_weight=cfg["tab"]["class_weight"]).fit(fc["X_tab_tr"], y_idx)
                m_mri = HardLinearSVM(C=cfg["mri"]["C"], class_weight=cfg["mri"]["class_weight"]).fit(fc["X_mri_tr"], y_idx)
            mf = cfg["text"].get("max_features")
            key = mf if mf in fc["text_cache"] else (500 if mf is None else mf)
            X_ttr, X_tva = fc["text_cache"][key]
            if family == "fuzzy":
                m_txt = SoftLinearSVR(C=cfg["text"]["C"], epsilon=cfg["text"]["epsilon"]).fit(X_ttr, Y_tr)
            else:
                m_txt = HardLinearSVM(C=cfg["text"]["C"], class_weight=cfg["text"]["class_weight"]).fit(X_ttr, y_idx)
            pt = m_tab.predict_proba(fc["X_tab_val"])
            pm = m_mri.predict_proba(fc["X_mri_val"])
            px = m_txt.predict_proba(X_tva)
            w = cfg["fusion"]
            p_fused = w["w_tab"] * pt + w["w_mri"] * pm + w["w_txt"] * px
            y_pred = [CLASS_ORDER_4[int(np.argmax(p))] for p in p_fused]
            m = compute_split_metrics(fc["y_val_treat"], y_pred, p_fused, class_order=CLASS_ORDER_4, n_classes=4)
            split_metrics.append(m)
        results[cond_name] = split_metrics
        log(f"  [2.1] {cond_name}: Macro-F1={np.mean([m['macro_f1'] for m in split_metrics]):.4f} | Brier={np.mean([m['brier'] for m in split_metrics if m['brier'] is not None]):.4f}")
    return results


def run_grid_22(fold_cache) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    grids = {
        "tab": {"hard": [{"C": C, "cw": cw} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced")],
                 "soft": [{"C": C, "eps": e} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1)]},
        "mri": {"hard": [{"C": C, "cw": cw} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced")],
                 "soft": [{"C": C, "eps": e} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1)]},
        "text": {"hard": [{"C": C, "cw": cw, "mf": mf} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced") for mf in (500, None)],
                  "soft": [{"C": C, "eps": e, "mf": mf} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1) for mf in (500, None)]},
    }
    best_cfg: Dict[str, Any] = {}
    probs_cache: Dict[str, Dict[str, List[np.ndarray]]] = {"hard": {}, "soft": {}}

    for family in ("hard", "soft"):
        for modality in ("tab", "mri", "text"):
            best_f1 = -1.0
            best_c = None
            best_probs = None
            for cand in grids[modality][family]:
                split_f1s, split_probs = [], []
                for fc in fold_cache:
                    if modality == "tab":
                        X_tr, X_val = fc["X_tab_tr"], fc["X_tab_val"]
                    elif modality == "mri":
                        X_tr, X_val = fc["X_mri_tr"], fc["X_mri_val"]
                    else:
                        key = cand.get("mf", 500) if cand.get("mf") in fc["text_cache"] else 500
                        X_tr, X_val = fc["text_cache"][key]
                    if family == "hard":
                        y_idx = np.array([IDX_3[c] for c in fc["y_tr_conf"]])
                        model = HardLinearSVM(C=cand["C"], class_weight=cand["cw"], n_classes=3).fit(X_tr, y_idx)
                    else:
                        Y_tr = build_confidence_soft(fc["y_tr_conf"])
                        model = SoftLinearSVR(C=cand["C"], epsilon=cand["eps"]).fit(X_tr, Y_tr)
                    probs = model.predict_proba(X_val)
                    y_pred = [CLASS_ORDER_3[int(np.argmax(p))] for p in probs]
                    m = compute_split_metrics(fc["y_val_conf"], y_pred, probs, class_order=CLASS_ORDER_3, n_classes=3)
                    split_f1s.append(m["macro_f1"])
                    split_probs.append(probs)
                mean_f1 = float(np.mean(split_f1s))
                if mean_f1 > best_f1:
                    best_f1, best_c, best_probs = mean_f1, cand, split_probs
            best_cfg[f"{family}_{modality}"] = best_c
            probs_cache[family][modality] = best_probs
            log(f"  [2.2] Best {family}/{modality}: {best_c} | Macro-F1={best_f1:.4f}")

    step = 0.05
    w_cands = []
    for w1_i in range(int(1.0 / step) + 1):
        for w2_i in range(int(1.0 / step) + 1 - w1_i):
            w1 = round(w1_i * step, 2)
            w2 = round(w2_i * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= -1e-6:
                w_cands.append((w1, w2, max(0.0, w3)))
    for family in ("hard", "soft"):
        best_w = None
        best_f1w = -1.0
        for (wt, wm, wx) in w_cands:
            fs = []
            for i, fc in enumerate(fold_cache):
                fused = wt * probs_cache[family]["tab"][i] + wm * probs_cache[family]["mri"][i] + wx * probs_cache[family]["text"][i]
                y_pred = [CLASS_ORDER_3[int(np.argmax(p))] for p in fused]
                m = compute_split_metrics(fc["y_val_conf"], y_pred, fused, class_order=CLASS_ORDER_3, n_classes=3)
                fs.append(m["macro_f1"])
            mean_f = float(np.mean(fs))
            if mean_f > best_f1w:
                best_f1w, best_w = mean_f, (wt, wm, wx)
        best_cfg[f"{family}_fusion"] = {"w_tab": best_w[0], "w_mri": best_w[1], "w_txt": best_w[2], "mean_macro_f1": best_f1w}
        log(f"  [2.2] {family} fusion: w={best_w} | Macro-F1={best_f1w:.4f}")

    # Binary clear-vs-borderline (C6)
    binary_res = {}
    class_order_bin = ["borderline", "clear"]
    for family in ("hard", "soft"):
        bins = []
        for fc in fold_cache:
            tr_mask = [c in ("clear", "borderline") for c in fc["y_tr_conf"]]
            val_mask = [c in ("clear", "borderline") for c in fc["y_val_conf"]]
            y_tr_b = [fc["y_tr_conf"][j] for j in range(len(tr_mask)) if tr_mask[j]]
            y_val_b = [fc["y_val_conf"][j] for j in range(len(val_mask)) if val_mask[j]]
            if len(set(y_tr_b)) < 2 or len(set(y_val_b)) < 2:
                continue
            X_tr = fc["X_tab_tr"][np.array(tr_mask)]
            X_val = fc["X_tab_val"][np.array(val_mask)]
            if family == "hard":
                y_idx = np.array([0 if c == "borderline" else 1 for c in y_tr_b])
                model = HardLinearSVM(C=1.0, class_weight="balanced").fit(X_tr, y_idx)
            else:
                Y_tr = np.zeros((len(y_tr_b), 2))
                for j, c in enumerate(y_tr_b):
                    Y_tr[j, 0 if c == "borderline" else 1] = CONF_ORDINAL.get(c, 0.5)
                model = SoftLinearSVR(C=1.0, epsilon=0.1).fit(X_tr, Y_tr)
            probs = model.predict_proba(X_val)
            y_pred = ["clear" if np.argmax(p) == 1 else "borderline" for p in probs]
            f = f1_score(y_val_b, y_pred, labels=class_order_bin, average="macro", zero_division=0)
            bins.append(f)
        binary_res[family] = float(np.mean(bins)) if bins else None
        if binary_res[family] is not None:
            log(f"  [2.2] binary C6 {family}: Macro-F1={binary_res[family]:.4f}")
        else:
            log(f"  [2.2] binary C6 {family}: no valid splits")
    best_cfg["binary_C6"] = binary_res
    return best_cfg, probs_cache


def run_grid_23(fold_cache) -> Dict[str, Dict[str, Any]]:
    joint_per_fold = []
    for fc in fold_cache:
        X_txt_tr_50, X_txt_val_50 = fc["text_cache"][50]
        pca16 = PCA(n_components=min(16, fc["X_mri_tr"].shape[0] - 1, fc["X_mri_tr"].shape[1]), random_state=42).fit(fc["X_mri_tr"])
        mri_tr16 = pca16.transform(fc["X_mri_tr"])
        mri_val16 = pca16.transform(fc["X_mri_val"])
        X_tr = np.hstack([fc["X_tab_tr"], mri_tr16, X_txt_tr_50])
        X_val = np.hstack([fc["X_tab_val"], mri_val16, X_txt_val_50])
        joint_per_fold.append((X_tr, X_val))

    per_var_best: Dict[str, Dict[str, Any]] = {}
    for var in VARIABLES_11:
        svc_f1, svr_f1 = [], []
        for i, fc in enumerate(fold_cache):
            X_tr, X_val = joint_per_fold[i]
            y_tr = np.array([IDX_W[fc["vw_tr"][p].get(var, "not_used")] for p in fc["vw_tr"]])
            y_val_str = [fc["vw_val"][p].get(var, "not_used") for p in fc["vw_val"]]
            svc = HardLinearSVM(C=1.0, class_weight="balanced", n_classes=4).fit(X_tr, y_tr)
            pred_idx = svc.predict_proba(X_val).argmax(axis=1)
            pred = [WEIGHT_ORDER[i] for i in pred_idx]
            svc_f1.append(f1_score(y_val_str, pred, labels=WEIGHT_ORDER, average="macro", zero_division=0))
            y_tr_val = np.array([WEIGHT_ORDINAL[fc["vw_tr"][p].get(var, "not_used")] for p in fc["vw_tr"]])
            svr = LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, random_state=42).fit(X_tr, y_tr_val)
            pred_val = svr.predict(X_val)
            pred_idx = np.digitize(pred_val, WEIGHT_THRESHOLDS)
            pred = [WEIGHT_ORDER[i] for i in pred_idx]
            svr_f1.append(f1_score(y_val_str, pred, labels=WEIGHT_ORDER, average="macro", zero_division=0))
        ms = float(np.mean(svc_f1))
        mr = float(np.mean(svr_f1))
        alg = "svc" if ms >= mr else "svr"
        per_var_best[var] = {"algorithm": alg, "svc_macro_f1": ms, "svr_macro_f1": mr, "best_macro_f1": max(ms, mr)}
        log(f"  [2.3] {var:<14} -> {alg.upper()} | SVC={ms:.4f} SVR={mr:.4f}")
    return per_var_best


def run_loocv_all(df_gt, df_tab, df_mri, df_text, vw_map, best_exp11, best_22, best_23) -> Dict[str, Any]:
    """Unified LOOCV for the three subtasks with frozen hyperparameters."""
    pids = df_gt["pid"].tolist()
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    loocv = {"21": {"C1_fuzzy": [], "C2_hard": []}, "22": {"hard": [], "soft": []}, "23": {v: [] for v in VARIABLES_11}}

    for hold_out in pids:
        tr_pids = [p for p in pids if p != hold_out]
        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[[hold_out]].reset_index()
        prep_tab = Task2TabularPreprocessor().fit(tr_tab)
        X_tab_tr = prep_tab.transform(tr_tab)
        X_tab_val = prep_tab.transform(val_tab)
        X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_val = df_mri.set_index("pid").loc[[hold_out]][emb_cols].values.astype(float)
        prep_mri = MRIStandardPreprocessor().fit(X_mri_tr)
        X_mri_tr_p = prep_mri.transform(X_mri_tr)
        X_mri_val_p = prep_mri.transform(X_mri_val)
        tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        val_txt = df_text.set_index("pid").loc[[hold_out]]["full_encounter_narrative"].tolist()
        gt_tr = df_gt.set_index("pid").loc[tr_pids]
        gt_val = df_gt.set_index("pid").loc[[hold_out]]
        y_tr_treat = gt_tr["treatment_decision"].tolist()
        y_tr_conf = gt_tr["confidence"].fillna("clear").tolist()
        y_hold_treat = gt_val["treatment_decision"].iloc[0]
        y_hold_conf = gt_val["confidence"].iloc[0]

        # Subtask 2.1
        for family, cond_name in (("fuzzy", "C1_fuzzy"), ("hard", "C2_hard")):
            cfg = best_exp11[family]
            mf = cfg["text"].get("max_features")
            key = mf if mf is not None else 500
            if key not in (500, None):
                key = 500
            prep_txt = TextPreprocessor(max_features=key).fit(tr_txt)
            X_ttr = prep_txt.transform(tr_txt)
            X_tva = prep_txt.transform(val_txt)
            if family == "fuzzy":
                Y_tr = build_treatment_soft(y_tr_treat, y_tr_conf, 4)
                m_tab = SoftLinearSVR(C=cfg["tab"]["C"], epsilon=cfg["tab"]["epsilon"]).fit(X_tab_tr, Y_tr)
                m_mri = SoftLinearSVR(C=cfg["mri"]["C"], epsilon=cfg["mri"]["epsilon"]).fit(X_mri_tr_p, Y_tr)
                m_txt = SoftLinearSVR(C=cfg["text"]["C"], epsilon=cfg["text"]["epsilon"]).fit(X_ttr, Y_tr)
            else:
                y_idx = np.array([IDX_4[c] for c in y_tr_treat])
                m_tab = HardLinearSVM(C=cfg["tab"]["C"], class_weight=cfg["tab"]["class_weight"]).fit(X_tab_tr, y_idx)
                m_mri = HardLinearSVM(C=cfg["mri"]["C"], class_weight=cfg["mri"]["class_weight"]).fit(X_mri_tr_p, y_idx)
                m_txt = HardLinearSVM(C=cfg["text"]["C"], class_weight=cfg["text"]["class_weight"]).fit(X_ttr, y_idx)
            pt = m_tab.predict_proba(X_tab_val)
            pm = m_mri.predict_proba(X_mri_val_p)
            px = m_txt.predict_proba(X_tva)
            w = cfg["fusion"]
            p_fused = w["w_tab"] * pt + w["w_mri"] * pm + w["w_txt"] * px
            pred = CLASS_ORDER_4[int(np.argmax(p_fused[0]))]
            loocv["21"][cond_name].append(1.0 if pred == y_hold_treat else 0.0)

        # Subtask 2.2 (frozen best per-modality tab config from Phase A)
        for family in ("hard", "soft"):
            bc = best_22[f"{family}_tab"]
            if family == "hard":
                y_idx = np.array([IDX_3[c] for c in y_tr_conf])
                model = HardLinearSVM(C=bc["C"], class_weight=bc.get("cw"), n_classes=3).fit(X_tab_tr, y_idx)
            else:
                Y_tr = build_confidence_soft(y_tr_conf)
                model = SoftLinearSVR(C=bc["C"], epsilon=bc.get("eps", 0.1)).fit(X_tab_tr, Y_tr)
            probs = model.predict_proba(X_tab_val)[0]
            pred = CLASS_ORDER_3[int(np.argmax(probs))]
            loocv["22"][family].append(1.0 if pred == y_hold_conf else 0.0)

        # Subtask 2.3 (best per-variable algorithm, tab features only for LOOCV)
        for var in VARIABLES_11:
            alg = best_23[var]["algorithm"]
            y_tr = np.array([IDX_W[vw_map[p].get(var, "not_used")] for p in tr_pids])
            hold_w = vw_map[hold_out].get(var, "not_used")
            if alg == "svc":
                model = HardLinearSVM(C=1.0, class_weight="balanced", n_classes=4).fit(X_tab_tr, y_tr)
                pred = WEIGHT_ORDER[int(model.predict_proba(X_tab_val)[0].argmax())]
            else:
                y_tr_val = np.array([WEIGHT_ORDINAL[vw_map[p].get(var, "not_used")] for p in tr_pids])
                model = LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, random_state=42).fit(X_tab_tr, y_tr_val)
                pred = WEIGHT_ORDER[int(np.digitize(model.predict(X_tab_val)[0], WEIGHT_THRESHOLDS))]
            loocv["23"][var].append(1.0 if pred == hold_w else 0.0)

    loocv_agg = {
        "21": {c: float(np.mean(v)) for c, v in loocv["21"].items()},
        "22": {c: float(np.mean(v)) for c, v in loocv["22"].items()},
        "23": {v: float(np.mean(lst)) for v, lst in loocv["23"].items()},
        "23_avg": float(np.mean([np.mean(lst) for lst in loocv["23"].values()])),
    }
    log(f"  LOOCV 2.1: {loocv_agg['21']}")
    log(f"  LOOCV 2.2: {loocv_agg['22']}")
    log(f"  LOOCV 2.3 avg: {loocv_agg['23_avg']:.4f}")
    return loocv_agg


def main() -> None:
    log("=" * 90)
    log("CHIMERA Task-2 -- Three-Subtask Linear-SVM Benchmark (exp_13)")
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

    with open(EXP11_RESULTS / "best_hyperparams.json", "r", encoding="utf-8") as f:
        best_exp11 = json.load(f)

    log("Preparing shared fold cache (StandardScaler + confidence + variable-weight targets)...")
    fold_cache = prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    log("[1/4] Subtask 2.1: frozen exp_11 SVC/SVR re-evaluation...")
    res_21 = run_reeval_21(fold_cache, best_exp11)

    log("[2/4] Subtask 2.2: SVM 3-class ordinal grids + fusion + binary C6...")
    best_22, probs_22 = run_grid_22(fold_cache)

    log("[3/4] Subtask 2.3: SVC/SVR per-variable meta-models...")
    res_23 = run_grid_23(fold_cache)

    log("[4/4] Phase B LOOCV (all three subtasks)...")
    loocv_agg = run_loocv_all(df_gt, df_tab, df_mri, df_text, vw_map, best_exp11, best_22, res_23)

    elapsed = time.time() - t0
    log(f"All phases completed in {elapsed:.1f}s")

    agg = {
        "subtask2_1": {c: {"mean_macro_f1": float(np.mean([m["macro_f1"] for m in res_21[c]])),
                           "mean_brier": float(np.mean([m["brier"] for m in res_21[c] if m["brier"] is not None]))} for c in res_21},
        "subtask2_2_best": best_22,
        "subtask2_3_per_var": res_23,
        "loocv": loocv_agg,
    }
    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(RESULTS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump({"subtask2_2": best_22, "subtask2_3": res_23}, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_13 execution timestamp: {datetime.now().isoformat()}\n")
        f.write("3 subtasks + Phase B LOOCV completed.\n")
    log("exp_13 benchmark completed and persisted.")


if __name__ == "__main__":
    main()
