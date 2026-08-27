#!/usr/bin/env python3
"""
Persist best exp_10 models as pickle files for inference.

Re-trains the winning hard/fuzzy KNN per modality on ALL N=70 labeled cases
using the best hyperparameters from best_hyperparams.json, then saves pickles
with the fitted preprocessors so new cases can be transformed and predicted.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
EXP10_DIR = PROJECT_ROOT / "experiments" / "exp_10"
MODELS_DIR = EXP10_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
        self.pca = None
        self.mean_vector: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MRIPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.mean_vector = np.mean(X_arr, axis=0)
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0):
                X_arr[i] = self.mean_vector
        if self.n_components is not None:
            from sklearn.decomposition import PCA
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
        self.svd: Optional[TruncatedSVD] = None

    def fit(self, texts: List[str]) -> "TextPreprocessor":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b\w+(?:-\w+)*\b",
        )
        X_tfidf = self.vectorizer.fit_transform(texts)
        if self.n_components is not None:
            max_comp = min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1, self.n_components)
            if max_comp > 0:
                self.svd = TruncatedSVD(n_components=max_comp, random_state=42).fit(X_tfidf)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(texts)
        if self.svd is not None:
            return self.svd.transform(X_tfidf)
        return X_tfidf.toarray()


class MulticlassKNN:
    def __init__(self, k: int, metric: str = "euclidean", weights: str = "uniform", fuzzy: bool = False):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.fuzzy = fuzzy
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None

    def _build_targets(self, y_multiclass: List[str], y_confidence: Optional[List[str]] = None) -> np.ndarray:
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

    def fit(self, X_train: np.ndarray, y_multiclass: List[str], y_confidence: Optional[List[str]] = None) -> "MulticlassKNN":
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
            si = row_idx[order]
            sd = row_dist[order]
            w = 1.0 / (sd + 1e-5) if self.weights == "distance" else np.ones_like(sd)
            probs[i] = np.sum(w[:, None] * self.Y_train[si], axis=0) / (np.sum(w) + 1e-9)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        return probs / row_sums

    def predict(self, X_val: np.ndarray) -> List[str]:
        probs = self.predict_proba(X_val)
        return [CLASS_ORDER[int(np.argmax(p))] for p in probs]


def main() -> None:
    print("=" * 80)
    print("Saving best exp_10 models as pickle files (retrained on all N=70)")
    print("=" * 80)

    # Load canonical data
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")

    # Leakage audit
    assert not any("active_treatment_flag" in c for c in df_tab.columns), "Leakage detected!"

    y_multiclass = df_gt["treatment_decision"].tolist()
    y_confidence = df_gt["confidence"].fillna("clear").tolist()

    # Load best hyperparams
    with open(EXP10_DIR / "results" / "best_hyperparams.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    saved = {}

    for family in ("hard", "fuzzy"):
        cfg_family = best[family]

        # --- Tabular ---
        cfg_tab = cfg_family["tab"]
        prep_tab = Task2TabularPreprocessor().fit(df_tab)
        X_tab = prep_tab.transform(df_tab)
        knn_tab = MulticlassKNN(
            k=cfg_tab["k"], metric=cfg_tab["metric"], weights=cfg_tab["weights"], fuzzy=cfg_tab["fuzzy"]
        ).fit(X_tab, y_multiclass, y_confidence)
        tab_path = MODELS_DIR / f"{family}_tabular_knn.pkl"
        with open(tab_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_tab, "model": knn_tab}, fh)
        saved[str(tab_path)] = cfg_tab

        # --- MRI ---
        cfg_mri = cfg_family["mri"]
        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        X_mri_raw = df_mri[emb_cols].values.astype(float)
        prep_mri = MRIPreprocessor(n_components=cfg_mri.get("pca_components")).fit(X_mri_raw)
        X_mri = prep_mri.transform(X_mri_raw)
        knn_mri = MulticlassKNN(
            k=cfg_mri["k"], metric=cfg_mri["metric"], weights=cfg_mri["weights"], fuzzy=cfg_mri["fuzzy"]
        ).fit(X_mri, y_multiclass, y_confidence)
        mri_path = MODELS_DIR / f"{family}_mri_knn.pkl"
        with open(mri_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_mri, "model": knn_mri}, fh)
        saved[str(mri_path)] = cfg_mri

        # --- Text ---
        cfg_txt = cfg_family["text"]
        texts = df_text["full_encounter_narrative"].tolist()
        prep_txt = TextPreprocessor(
            max_features=cfg_txt.get("max_features"), n_components=cfg_txt.get("svd_components")
        ).fit(texts)
        X_txt = prep_txt.transform(texts)
        knn_txt = MulticlassKNN(
            k=cfg_txt["k"], metric=cfg_txt["metric"], weights=cfg_txt["weights"], fuzzy=cfg_txt["fuzzy"]
        ).fit(X_txt, y_multiclass, y_confidence)
        txt_path = MODELS_DIR / f"{family}_text_knn.pkl"
        with open(txt_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_txt, "model": knn_txt}, fh)
        saved[str(txt_path)] = cfg_txt

        # --- Fusion weights ---
        fus_path = MODELS_DIR / f"{family}_fusion_weights.json"
        with open(fus_path, "w", encoding="utf-8") as fh:
            json.dump(cfg_family["fusion"], fh, indent=2)
        saved[str(fus_path)] = cfg_family["fusion"]

    # Metadata
    metadata = {
        "experiment": "exp_10",
        "created_at": datetime.now().isoformat(),
        "cohort": {"n": len(df_gt), "classes": CLASS_ORDER},
        "soft_target_formula": "y_k = 0.25 + 0.75*c; y_j!=k = (1-y_k)/3",
        "confidence_map": CONFIDENCE_C,
        "class_order": CLASS_ORDER,
        "notes": "Models retrained on ALL N=70 labeled cases. For inference on new cases, apply the corresponding preprocessor before predict_proba.",
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Copy of best hyperparams for reference
    with open(MODELS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    print("\nSaved artifacts:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")
    print("\nAll best exp_10 models persisted successfully.")


if __name__ == "__main__":
    main()
