#!/usr/bin/env python3
"""
Persist best exp_12 models as pickle bundles for inference (Task-2 three subtasks).

Subtask 2.1: fuzzy-KNN 4-class (frozen exp_10 H2 configs) -> subtask1_treatment_fuzzy_knn.pkl
Subtask 2.2: fuzzy-KNN 3-class ordinal (exp_12 Phase-A winner) -> subtask2_confidence_fuzzy_knn.pkl
Subtask 2.3: per-variable RF/ExtraTrees (exp_12 Phase-A selection) -> subtask3_weights_ensemble.pkl

All models retrained on ALL N=70 labeled cases with Phase-A frozen hyperparameters.
Bundle format: {'models': ..., 'preprocessors': ..., ...}.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"
EXP10_DIR = PROJECT_ROOT / "experiments" / "exp_10"
EXP12_DIR = PROJECT_ROOT / "experiments" / "exp_12"
MODELS_DIR = EXP12_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER_4 = ["watchful_waiting", "continued_surveillance", "active_surveillance", "active_treatment"]
CLASS_ORDER_3 = ["uncertain", "borderline", "clear"]
WEIGHT_ORDER = ["not_used", "noted", "important", "decisive"]
VARIABLES_11 = ["age", "bx_gl_prim", "bx_gl_sec", "bx_isup", "comorbidity", "cspca", "ct", "fh", "pirads", "psa", "psad"]

IDX_4 = {c: i for i, c in enumerate(CLASS_ORDER_4)}
IDX_3 = {c: i for i, c in enumerate(CLASS_ORDER_3)}
IDX_W = {c: i for i, c in enumerate(WEIGHT_ORDER)}
CONFIDENCE_C = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
CONF_ORDINAL = {"uncertain": 0.25, "borderline": 0.5, "clear": 1.0}

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
                Y[i, IDX_4[cls]] = 1.0
            if self.fuzzy:
                conf = y_confidence or ["clear"] * n
                for i, cls in enumerate(y_multiclass):
                    c = CONFIDENCE_C.get(str(conf[i]).lower() if i < len(conf) else "clear", 1.0)
                    y_k = 0.25 + 0.75 * c
                    Y[i, :] = (1.0 - y_k) / (self.n_classes - 1)
                    Y[i, IDX_4[cls]] = y_k
        else:
            for i, cls in enumerate(y_multiclass):
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


def load_var_weights_map(pids: List[str]) -> Dict[str, Dict[str, str]]:
    vw_map: Dict[str, Dict[str, str]] = {}
    for pid in pids:
        rp = RAW_DIR / pid / "prostate-treatment-decision-reasoning.json"
        with open(rp, "r", encoding="utf-8") as f:
            rr = json.load(f)
        vw_map[pid] = rr.get("variable_weights", {})
    return vw_map


def build_23_joint(X_tab, X_mri_raw, X_txt_50) -> np.ndarray:
    prep_mri = MRIPreprocessor(n_components=16).fit(X_mri_raw)
    mri16 = prep_mri.transform(X_mri_raw)
    return np.hstack([X_tab, mri16, X_txt_50]), prep_mri


def main() -> None:
    print("=" * 80)
    print("Saving best exp_12 models as pickle bundles (retrained on all N=70)")
    print("=" * 80)

    # --- Load canonical data ---
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    assert not any("active_treatment_flag" in c for c in df_tab.columns), "Leakage detected!"

    y_treat = df_gt["treatment_decision"].tolist()
    y_conf = df_gt["confidence"].fillna("clear").tolist()
    vw_map = load_var_weights_map(df_gt["pid"].tolist())

    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    mri_raw = df_mri[emb_cols].values.astype(float)
    texts = df_text["full_encounter_narrative"].tolist()

    # --- Subtask 2.1 (frozen exp_10 H2 fuzzy) ---
    with open(EXP10_DIR / "results" / "best_hyperparams.json", "r", encoding="utf-8") as f:
        exp10_best = json.load(f)
    cfg21 = exp10_best["fuzzy"]
    prep_tab = Task2TabularPreprocessor().fit(df_tab)
    X_tab_all = prep_tab.transform(df_tab)
    prep_mri_full = MRIPreprocessor().fit(mri_raw)
    X_mri_all = prep_mri_full.transform(mri_raw)
    mf21 = cfg21["text"].get("max_features")
    prep_txt_full = TextPreprocessor(max_features=mf21).fit(texts)
    X_txt_all = prep_txt_full.transform(texts)
    knn_tab21 = MulticlassKNN(k=cfg21["tab"]["k"], metric=cfg21["tab"]["metric"], weights=cfg21["tab"]["weights"], fuzzy=True, n_classes=4).fit(X_tab_all, y_treat, y_conf, "treatment")
    knn_mri21 = MulticlassKNN(k=cfg21["mri"]["k"], metric=cfg21["mri"]["metric"], weights=cfg21["mri"]["weights"], fuzzy=True, n_classes=4).fit(X_mri_all, y_treat, y_conf, "treatment")
    knn_txt21 = MulticlassKNN(k=cfg21["text"]["k"], metric=cfg21["text"]["metric"], weights=cfg21["text"]["weights"], fuzzy=True, n_classes=4).fit(X_txt_all, y_treat, y_conf, "treatment")
    bundle21 = {
        "models": {"tab": knn_tab21, "mri": knn_mri21, "text": knn_txt21},
        "preprocessors": {"tab": prep_tab, "mri": prep_mri_full, "text": prep_txt_full},
        "fusion_weights": cfg21["fusion"],
        "class_order": CLASS_ORDER_4,
    }
    with open(MODELS_DIR / "subtask1_treatment_fuzzy_knn.pkl", "wb") as f:
        pickle.dump(bundle21, f)
    with open(MODELS_DIR / "subtask1_fusion_weights.json", "w", encoding="utf-8") as f:
        json.dump(cfg21["fusion"], f, indent=2)
    print(f"  subtask1: {MODELS_DIR / 'subtask1_treatment_fuzzy_knn.pkl'} saved")

    # --- Subtask 2.2 (exp_12 Phase-A winner) ---
    with open(EXP12_DIR / "results" / "best_hyperparams.json", "r", encoding="utf-8") as f:
        exp12_best = json.load(f)
    cfg22 = exp12_best["subtask2_2"]
    knn_tab22 = MulticlassKNN(k=cfg22["tab"]["k"], metric="euclidean", weights=cfg22["tab"]["w"], fuzzy=True, n_classes=3).fit(X_tab_all, y_conf, None, "confidence")
    knn_mri22 = MulticlassKNN(k=cfg22["mri"]["k"], metric="euclidean", weights=cfg22["mri"]["w"], fuzzy=True, n_classes=3).fit(X_mri_all, y_conf, None, "confidence")
    mf22 = cfg22["text"]["mf"]
    prep_txt22 = TextPreprocessor(max_features=mf22).fit(texts)
    X_txt_22 = prep_txt22.transform(texts)
    knn_txt22 = MulticlassKNN(k=cfg22["text"]["k"], metric="cosine", weights="uniform", fuzzy=True, n_classes=3).fit(X_txt_22, y_conf, None, "confidence")
    bundle22 = {
        "models": {"tab": knn_tab22, "mri": knn_mri22, "text": knn_txt22},
        "preprocessors": {"tab": prep_tab, "mri": prep_mri_full, "text": prep_txt22},
        "fusion_weights": {"w_tab": cfg22["fusion"]["w_tab"], "w_mri": cfg22["fusion"]["w_mri"], "w_txt": cfg22["fusion"]["w_txt"]},
        "class_order": CLASS_ORDER_3,
    }
    with open(MODELS_DIR / "subtask2_confidence_fuzzy_knn.pkl", "wb") as f:
        pickle.dump(bundle22, f)
    with open(MODELS_DIR / "subtask2_fusion_weights.json", "w", encoding="utf-8") as f:
        json.dump({"w_tab": cfg22["fusion"]["w_tab"], "w_mri": cfg22["fusion"]["w_mri"], "w_txt": cfg22["fusion"]["w_txt"]}, f, indent=2)
    print(f"  subtask2: {MODELS_DIR / 'subtask2_confidence_fuzzy_knn.pkl'} saved")

    # --- Subtask 2.3 (per-variable RF/ET over 32+16+50D joint) ---
    prep_txt50 = TextPreprocessor(max_features=50).fit(texts)
    X_txt_50 = prep_txt50.transform(texts)
    X_joint, prep_mri16 = build_23_joint(X_tab_all, mri_raw, X_txt_50)
    per_var_models: Dict[str, Any] = {}
    per_var_meta: Dict[str, str] = {}
    for var in VARIABLES_11:
        alg_name = exp12_best["subtask2_3"][var]["algorithm"]
        per_var_meta[var] = alg_name
        y_var = np.array([WEIGHT_ORDER.index(vw_map[pid].get(var, "not_used")) for pid in df_gt["pid"]])
        if alg_name == "rf":
            est = RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        else:
            est = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        est.fit(X_joint, y_var)
        per_var_models[var] = est
    bundle23 = {
        "models": per_var_models,
        "per_var_algorithm": per_var_meta,
        "preprocessors": {"tab": prep_tab, "mri": prep_mri16, "text": prep_txt50},
        "variables": VARIABLES_11,
        "weight_order": WEIGHT_ORDER,
    }
    with open(MODELS_DIR / "subtask3_weights_ensemble.pkl", "wb") as f:
        pickle.dump(bundle23, f)
    with open(MODELS_DIR / "variable_weights_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"per_var_algorithm": per_var_meta, "variables": VARIABLES_11, "weight_order": WEIGHT_ORDER}, f, indent=2)
    print(f"  subtask3: {MODELS_DIR / 'subtask3_weights_ensemble.pkl'} saved")

    # Metadata
    metadata = {
        "experiment": "exp_12",
        "created_at": datetime.now().isoformat(),
        "cohort": {"n": len(df_gt), "classes_4": CLASS_ORDER_4, "classes_3": CLASS_ORDER_3},
        "subtask1_fusion": cfg21["fusion"],
        "subtask2_fusion": {"w_tab": cfg22["fusion"]["w_tab"], "w_mri": cfg22["fusion"]["w_mri"], "w_txt": cfg22["fusion"]["w_txt"]},
        "subtask3_algorithms": per_var_meta,
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("\nSaved artifacts:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")
    print("\nAll best exp_12 models persisted successfully.")


if __name__ == "__main__":
    main()
