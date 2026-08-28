#!/usr/bin/env python3
"""
Persist best exp_11 models as pickle files for inference.

Re-trains the winning hard-SVC / fuzzy-SVR per modality on ALL N=70 labeled cases
using the best hyperparameters from best_hyperparams.json, then saves pickles
with the fitted preprocessors so new cases can be transformed and predicted.
"""

from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
EXP11_DIR = PROJECT_ROOT / "experiments" / "exp_11"
MODELS_DIR = EXP11_DIR / "models"
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
    """Fold-isolated StandardScaler preprocessing for the exp_11 tabular frame."""

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
    """StandardScaler preprocessor for MRI embeddings (exp_11), PCA fallback."""

    def __init__(self, use_pca_fallback: bool = False):
        self.scaler = StandardScaler()
        self.pca: Optional[Any] = None
        self.use_pca_fallback = use_pca_fallback

    def fit(self, X: np.ndarray) -> "MRIStandardPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.scaler.fit(X_arr)
        if self.use_pca_fallback:
            from sklearn.decomposition import PCA
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
    """TF-IDF preprocessor reused from exp_4."""

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


class HardLinearSVM:
    """SVC(kernel='linear', probability=True) wrapper for 4-class classification."""

    def __init__(self, C: float = 1.0, class_weight: Optional[str] = None):
        self.C = C
        self.class_weight = class_weight
        self.model: Optional[SVC] = None
        self.fallback_model: Optional[LinearSVC] = None
        self._used_platt = True

    def fit(self, X_train: np.ndarray, y_multiclass: List[str], y_confidence: Optional[List[str]] = None) -> "HardLinearSVM":
        y_idx = np.array([CLASS_TO_IDX[c] for c in y_multiclass])
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
            self.fallback_model = LinearSVC(
                C=self.C, class_weight=self.class_weight, max_iter=5000, random_state=42
            )
            self.fallback_model.fit(X_train, y_idx)
            self._used_platt = False
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        if self._used_platt and self.model is not None:
            return self.model.predict_proba(X_val)
        if self.fallback_model is not None:
            scores = self.fallback_model.decision_function(X_val)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        raise RuntimeError("No fitted model available.")

    def predict(self, X_val: np.ndarray) -> List[str]:
        probs = self.predict_proba(X_val)
        return [CLASS_ORDER[int(np.argmax(p))] for p in probs]


class SoftLinearSVR:
    """MultiOutputRegressor(LinearSVR) on 4-d soft confidence-guided targets."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self.model: Optional[MultiOutputRegressor] = None

    def fit(self, X_train: np.ndarray, y_multiclass: List[str], y_confidence: Optional[List[str]] = None) -> "SoftLinearSVR":
        n = len(y_multiclass)
        Y = np.zeros((n, N_CLASSES))
        for i, cls in enumerate(y_multiclass):
            Y[i, CLASS_TO_IDX[cls]] = 1.0
        conf = y_confidence or ["clear"] * n
        for i, cls in enumerate(y_multiclass):
            c = CONFIDENCE_C.get(str(conf[i]).lower() if i < len(conf) else "clear", 1.0)
            idx = CLASS_TO_IDX[cls]
            y_k = 0.25 + 0.75 * c
            Y[i, :] = (1.0 - y_k) / (N_CLASSES - 1)
            Y[i, idx] = y_k
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

    def predict(self, X_val: np.ndarray) -> List[str]:
        probs = self.predict_proba(X_val)
        return [CLASS_ORDER[int(np.argmax(p))] for p in probs]


def main() -> None:
    print("=" * 80)
    print("Saving best exp_11 models as pickle files (retrained on all N=70)")
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
    with open(EXP11_DIR / "results" / "best_hyperparams.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    saved = {}

    for family in ("hard", "fuzzy"):
        cfg_family = best[family]

        # --- Tabular ---
        cfg_tab = cfg_family["tab"]
        prep_tab = Task2TabularPreprocessor().fit(df_tab)
        X_tab = prep_tab.transform(df_tab)
        if family == "hard":
            model_tab = HardLinearSVM(C=cfg_tab["C"], class_weight=cfg_tab["class_weight"])
        else:
            model_tab = SoftLinearSVR(C=cfg_tab["C"], epsilon=cfg_tab["epsilon"])
        model_tab.fit(X_tab, y_multiclass, y_confidence)
        tab_path = MODELS_DIR / f"{family}_tabular_{'svc' if family=='hard' else 'svr'}.pkl"
        with open(tab_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_tab, "model": model_tab}, fh)
        saved[str(tab_path)] = cfg_tab

        # --- MRI ---
        cfg_mri = cfg_family["mri"]
        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        X_mri_raw = df_mri[emb_cols].values.astype(float)
        prep_mri = MRIStandardPreprocessor(use_pca_fallback=False).fit(X_mri_raw)
        X_mri = prep_mri.transform(X_mri_raw)
        if family == "hard":
            model_mri = HardLinearSVM(C=cfg_mri["C"], class_weight=cfg_mri["class_weight"])
        else:
            model_mri = SoftLinearSVR(C=cfg_mri["C"], epsilon=cfg_mri["epsilon"])
        model_mri.fit(X_mri, y_multiclass, y_confidence)
        mri_path = MODELS_DIR / f"{family}_mri_{'svc' if family=='hard' else 'svr'}.pkl"
        with open(mri_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_mri, "model": model_mri}, fh)
        saved[str(mri_path)] = cfg_mri

        # --- Text ---
        cfg_txt = cfg_family["text"]
        texts = df_text["full_encounter_narrative"].tolist()
        prep_txt = TextPreprocessor(max_features=cfg_txt.get("max_features")).fit(texts)
        X_txt = prep_txt.transform(texts)
        if family == "hard":
            model_txt = HardLinearSVM(C=cfg_txt["C"], class_weight=cfg_txt["class_weight"])
        else:
            model_txt = SoftLinearSVR(C=cfg_txt["C"], epsilon=cfg_txt["epsilon"])
        model_txt.fit(X_txt, y_multiclass, y_confidence)
        txt_path = MODELS_DIR / f"{family}_text_{'svc' if family=='hard' else 'svr'}.pkl"
        with open(txt_path, "wb") as fh:
            pickle.dump({"preprocessor": prep_txt, "model": model_txt}, fh)
        saved[str(txt_path)] = cfg_txt

        # --- Fusion weights ---
        fus_path = MODELS_DIR / f"{family}_fusion_weights.json"
        with open(fus_path, "w", encoding="utf-8") as fh:
            json.dump(cfg_family["fusion"], fh, indent=2)
        saved[str(fus_path)] = cfg_family["fusion"]

    # Metadata
    metadata = {
        "experiment": "exp_11",
        "created_at": datetime.now().isoformat(),
        "cohort": {"n": len(df_gt), "classes": CLASS_ORDER},
        "soft_target_formula": "y_k = 0.25 + 0.75*c; y_j!=k = (1-y_k)/3 | fuzzy only",
        "confidence_map": CONFIDENCE_C,
        "class_order": CLASS_ORDER,
        "note": "Models retrained on ALL N=70 cases. Preprocessors use StandardScaler (tabular, MRI) and TF-IDF (text).",
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    with open(MODELS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    print("\nSaved artifacts:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")
    print("\nAll best exp_11 models persisted successfully.")


if __name__ == "__main__":
    main()
