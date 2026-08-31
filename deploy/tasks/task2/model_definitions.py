"""
Canonical Production Model Classes and Feature Pipeline for Task 2 (CHIMERA-Agent Challenge).

Self-contained, deployable models:
  1. Subtask1FuzzySVREnsemble (Fuzzy-SVR Tuned Decision Classifier + Multi-modal kNN Neighbor Retriever with Free-Text Reasoning)
  2. Subtask2FuzzyKNNClassifier (Multimodal Fuzzy-KNN Confidence Predictor)
  3. Subtask3MultiVariableEnsemble (11 ExtraTrees/RandomForest Meta-Classifiers on Joint 98D Multimodal Space)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from deploy.tasks.task2.preprocessing import (
    T2_CATEGORICAL_COLS,
    T2_NUMERICAL_COLS,
    extract_raw_patient_payload_task2,
)

TREATMENT_CLASSES_4 = [
    "active_surveillance",
    "continued_surveillance",
    "active_treatment",
    "watchful_waiting",
]

CONFIDENCE_CLASSES_3 = ["uncertain", "borderline", "clear"]

WEIGHT_ORDER_4 = ["not_used", "noted", "important", "decisive"]
VARIABLES_11 = [
    "age",
    "bx_gl_prim",
    "bx_gl_sec",
    "bx_isup",
    "comorbidity",
    "cspca",
    "ct",
    "fh",
    "pirads",
    "psa",
    "psad",
]


class Task2FeatureProcessor:
    """Encapsulates feature preprocessing and transformation for Tabular, MRI, and Text modalities."""

    def __init__(self):
        self.num_cols = list(T2_NUMERICAL_COLS)
        self.cat_cols = list(T2_CATEGORICAL_COLS)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.cat_columns_: List[str] = []
        self.num_defaults_: Dict[str, float] = {}
        self.cat_modes_: Dict[str, str] = {}
        self.is_fitted = False

    def fit(self, df_tab: pd.DataFrame) -> "Task2FeatureProcessor":
        df = df_tab.copy()
        for col in self.num_cols:
            if col in df.columns:
                self.num_defaults_[col] = float(df[col].median(skipna=True))
            else:
                self.num_defaults_[col] = 0.0

        for col in self.cat_cols:
            if col in df.columns:
                mode = df[col].dropna().mode()
                self.cat_modes_[col] = str(mode.iloc[0]) if len(mode) > 0 else "unknown"
            else:
                self.cat_modes_[col] = "unknown"

        X_num = df[[c for c in self.num_cols if c in df.columns]].copy()
        X_num_imp = self.imputer.fit_transform(X_num)
        X_num_scaled = self.scaler.fit_transform(X_num_imp)

        cat_subset = df[[c for c in self.cat_cols if c in df.columns]]
        X_cat = pd.get_dummies(cat_subset, drop_first=False) if len(cat_subset.columns) > 0 else pd.DataFrame(index=df.index)
        self.cat_columns_ = list(X_cat.columns)

        self.is_fitted = True
        return self

    def transform_df(self, df_tab: pd.DataFrame) -> np.ndarray:
        df = df_tab.copy()
        for col in self.num_cols:
            if col not in df.columns:
                df[col] = self.num_defaults_.get(col, 0.0)
        X_num = df[self.num_cols].copy()
        X_num_imp = self.imputer.transform(X_num)
        X_num_scaled = self.scaler.transform(X_num_imp)

        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = self.cat_modes_.get(col, "unknown")
        if self.cat_columns_:
            X_cat = pd.get_dummies(df[self.cat_cols], drop_first=False)
            for c in self.cat_columns_:
                if c not in X_cat.columns:
                    X_cat[c] = 0
            X_cat = X_cat[self.cat_columns_]
        else:
            X_cat = pd.DataFrame(index=df.index)

        return np.hstack([X_num_scaled, X_cat.values.astype(float)])

    def transform_single(self, features_dict: Dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([features_dict])
        return self.transform_df(df)[0]


class Subtask1FuzzySVREnsemble:
    """
    Subtask 2.1 Champion: Fuzzy-SVR Tuned Decision Classifier with Late Fusion and Multi-modal kNN Neighbor Retrieval.
    """

    def __init__(
        self,
        weight_tab: float = 0.50,
        weight_mri: float = 0.05,
        weight_txt: float = 0.45,
        k_neighbors: int = 3,
        c_tab: float = 0.1,
        eps_tab: float = 0.05,
        c_mri: float = 0.1,
        eps_mri: float = 0.1,
        c_txt: float = 10.0,
        eps_txt: float = 0.1,
    ):
        self.weight_tab = weight_tab
        self.weight_mri = weight_mri
        self.weight_txt = weight_txt
        self.k_neighbors = k_neighbors
        self.c_tab = c_tab
        self.eps_tab = eps_tab
        self.c_mri = c_mri
        self.eps_mri = eps_mri
        self.c_txt = c_txt
        self.eps_txt = eps_txt
        self.classes = list(TREATMENT_CLASSES_4)

        self.processor = Task2FeatureProcessor()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words="english")

        self.svr_tab = None
        self.svr_mri = None
        self.svr_txt = None

        self.nn_tab = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean")
        self.nn_mri = NearestNeighbors(n_neighbors=self.k_neighbors, metric="cosine")
        self.nn_txt = NearestNeighbors(n_neighbors=self.k_neighbors, metric="cosine")

        self.train_pids: List[str] = []
        self.reasonings_map: Dict[str, str] = {}
        self.is_fitted = False

    def fit(
        self,
        df_tab: pd.DataFrame,
        df_txt: pd.DataFrame,
        df_mri: pd.DataFrame,
        df_gt: pd.DataFrame,
        reasonings_map: Optional[Dict[str, str]] = None,
    ) -> "Subtask1FuzzySVREnsemble":
        df_merged = df_gt.merge(df_tab, on="pid").merge(df_txt, on="pid").merge(df_mri, on="pid")
        df_merged = df_merged.dropna(subset=["treatment_decision"]).copy()
        self.train_pids = df_merged["pid"].tolist()

        if reasonings_map:
            self.reasonings_map = dict(reasonings_map)
        else:
            self.reasonings_map = {pid: "Reasoning not provided in ground truth." for pid in self.train_pids}

        # 1. Fit Processors & Representations
        self.processor.fit(df_merged)
        X_tab = self.processor.transform_df(df_merged)

        txt_col = "full_encounter_narrative" if "full_encounter_narrative" in df_merged.columns else "clinical_text"
        X_txt = self.tfidf.fit_transform(df_merged[str(txt_col)].fillna("").astype(str)).toarray()

        mri_cols = [f"emb_{i}" for i in range(1024) if f"emb_{i}" in df_merged.columns]
        if len(mri_cols) == 1024:
            X_mri = df_merged[mri_cols].values.astype(float)
        else:
            mri_cols = [c for c in df_merged.columns if c.startswith("emb_") or c.startswith("mri_")]
            X_mri = df_merged[mri_cols].iloc[:, :1024].values.astype(float)

        # 2. Build soft targets for Fuzzy-SVR
        conf_map = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.0}
        n_samples = len(df_merged)
        Y_soft = np.zeros((n_samples, len(self.classes)), dtype=float)
        for i, (_, row) in enumerate(df_merged.iterrows()):
            dec = row["treatment_decision"]
            k = self.classes.index(dec) if dec in self.classes else 0
            c_val = conf_map.get(str(row.get("confidence", "clear")).lower(), 1.0)
            p_k = 0.25 + 0.75 * c_val
            p_rest = (1.0 - p_k) / (len(self.classes) - 1)
            Y_soft[i, :] = p_rest
            Y_soft[i, k] = p_k

        # 3. Fit MultiOutput LinearSVR per modality on soft targets
        self.svr_tab = MultiOutputRegressor(LinearSVR(C=self.c_tab, epsilon=self.eps_tab, random_state=42))
        self.svr_tab.fit(X_tab, Y_soft)

        self.svr_mri = MultiOutputRegressor(LinearSVR(C=self.c_mri, epsilon=self.eps_mri, random_state=42))
        self.svr_mri.fit(X_mri, Y_soft)

        self.svr_txt = MultiOutputRegressor(LinearSVR(C=self.c_txt, epsilon=self.eps_txt, random_state=42))
        self.svr_txt.fit(X_txt, Y_soft)

        # 4. Fit NearestNeighbors indexers
        self.nn_tab.fit(X_tab)
        self.nn_mri.fit(X_mri)
        self.nn_txt.fit(X_txt)

        self.is_fitted = True
        return self

    def predict_components(
        self, X_tab_1d: np.ndarray, X_mri_1d: np.ndarray, text_str: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if not self.is_fitted:
            raise RuntimeError("Subtask1FuzzySVREnsemble is not fitted.")

        # Modality predictions
        raw_tab = np.clip(self.svr_tab.predict([X_tab_1d])[0], 0.0, None)
        sum_tab = np.sum(raw_tab)
        p_tab = raw_tab / sum_tab if sum_tab > 1e-9 else np.ones(len(self.classes)) / len(self.classes)

        raw_mri = np.clip(self.svr_mri.predict([X_mri_1d])[0], 0.0, None)
        sum_mri = np.sum(raw_mri)
        p_mri = raw_mri / sum_mri if sum_mri > 1e-9 else np.ones(len(self.classes)) / len(self.classes)

        X_txt_1d = self.tfidf.transform([text_str]).toarray()[0]
        raw_txt = np.clip(self.svr_txt.predict([X_txt_1d])[0], 0.0, None)
        sum_txt = np.sum(raw_txt)
        p_txt = raw_txt / sum_txt if sum_txt > 1e-9 else np.ones(len(self.classes)) / len(self.classes)

        fused = self.weight_tab * p_tab + self.weight_mri * p_mri + self.weight_txt * p_txt
        fused = fused / (np.sum(fused) + 1e-9)

        # Retrieve Neighbors (Union of sets)
        _, idx_tab = self.nn_tab.kneighbors([X_tab_1d])
        _, idx_mri = self.nn_mri.kneighbors([X_mri_1d])
        _, idx_txt = self.nn_txt.kneighbors([X_txt_1d])

        neighbor_indices = list(dict.fromkeys(list(idx_tab[0]) + list(idx_mri[0]) + list(idx_txt[0])))
        neighbor_pids = [self.train_pids[i] for i in neighbor_indices if i < len(self.train_pids)]

        return fused, p_tab, p_mri, p_txt, neighbor_pids

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        case_id, feat_dict, narrative, mri_vec = extract_raw_patient_payload_task2(raw_input)
        X_tab_1d = self.processor.transform_single(feat_dict)
        fused, p_tab, p_mri, p_txt, neighbor_pids = self.predict_components(X_tab_1d, mri_vec, narrative)

        best_idx = int(np.argmax(fused))
        decision = self.classes[best_idx]

        neighbor_reasonings = [
            self.reasonings_map.get(pid, "Reasoning unavailable.") for pid in neighbor_pids
        ]

        return {
            "case_id": case_id,
            "decision": decision,
            "treatment_decision": decision,
            "treatment_probabilities": {self.classes[i]: float(fused[i]) for i in range(len(self.classes))},
            "modality_probabilities": {
                "tabular": {self.classes[i]: float(p_tab[i]) for i in range(len(self.classes))},
                "mri": {self.classes[i]: float(p_mri[i]) for i in range(len(self.classes))},
                "text": {self.classes[i]: float(p_txt[i]) for i in range(len(self.classes))},
            },
            "neighbors_union": {
                "total_unique_neighbors": len(neighbor_pids),
                "neighbor_pids": neighbor_pids,
                "reasoning_free_texts": neighbor_reasonings,
            },
        }


class Subtask2FuzzyKNNClassifier:
    """
    Subtask 2.2 Champion: Multimodal Fuzzy-KNN Confidence Classifier (exp_12 baseline).
    """

    def __init__(
        self,
        weight_tab: float = 0.0,
        weight_mri: float = 0.55,
        weight_txt: float = 0.45,
    ):
        self.weight_tab = weight_tab
        self.weight_mri = weight_mri
        self.weight_txt = weight_txt
        self.classes = list(CONFIDENCE_CLASSES_3)
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.processor = Task2FeatureProcessor()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words="english")

        self.train_labels: np.ndarray = np.array([])
        self.X_tab_train: np.ndarray = np.array([])
        self.X_mri_train: np.ndarray = np.array([])
        self.X_txt_train: np.ndarray = np.array([])
        self.is_fitted = False

    def fit(
        self,
        df_tab: pd.DataFrame,
        df_txt: pd.DataFrame,
        df_mri: pd.DataFrame,
        df_gt: pd.DataFrame,
    ) -> "Subtask2FuzzyKNNClassifier":
        df_merged = df_gt.merge(df_tab, on="pid").merge(df_txt, on="pid").merge(df_mri, on="pid")
        df_merged = df_merged.dropna(subset=["confidence"]).copy()

        self.processor.fit(df_merged)
        self.X_tab_train = self.processor.transform_df(df_merged)

        txt_col = "full_encounter_narrative" if "full_encounter_narrative" in df_merged.columns else "clinical_text"
        self.X_txt_train = self.tfidf.fit_transform(df_merged[str(txt_col)].fillna("").astype(str)).toarray()

        mri_cols = [f"emb_{i}" for i in range(1024) if f"emb_{i}" in df_merged.columns]
        if len(mri_cols) == 1024:
            self.X_mri_train = df_merged[mri_cols].values.astype(float)
        else:
            mri_cols = [c for c in df_merged.columns if c.startswith("emb_") or c.startswith("mri_")]
            self.X_mri_train = df_merged[mri_cols].iloc[:, :1024].values.astype(float)

        self.train_labels = np.array([self.classes_to_idx.get(c, 2) for c in df_merged["confidence"]])
        self.is_fitted = True
        return self

    def _knn_probs(self, X_train: np.ndarray, x_val: np.ndarray, k: int = 1) -> np.ndarray:
        dists = np.linalg.norm(X_train - x_val, axis=1)
        idx = np.argsort(dists)[:k]
        counts = np.bincount(self.train_labels[idx], minlength=len(self.classes))
        return counts / np.sum(counts)

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Subtask2FuzzyKNNClassifier is not fitted.")

        case_id, feat_dict, narrative, mri_vec = extract_raw_patient_payload_task2(raw_input)
        x_tab = self.processor.transform_single(feat_dict)
        x_txt = self.tfidf.transform([narrative]).toarray()[0]

        p_tab = self._knn_probs(self.X_tab_train, x_tab, k=1)
        p_mri = self._knn_probs(self.X_mri_train, mri_vec, k=1)
        p_txt = self._knn_probs(self.X_txt_train, x_txt, k=3)

        fused = self.weight_tab * p_tab + self.weight_mri * p_mri + self.weight_txt * p_txt
        fused = fused / (np.sum(fused) + 1e-9)

        pred_idx = int(np.argmax(fused))
        conf_label = self.classes[pred_idx]

        # Normalized ordinal confidence score (0 to 1)
        score = float(pred_idx) / 2.0

        return {
            "case_id": case_id,
            "confidence": conf_label,
            "confidence_score": score,
            "class_probabilities": {self.classes[i]: float(fused[i]) for i in range(len(self.classes))},
        }


class Subtask3MultiVariableEnsemble:
    """
    Subtask 2.3 Champion: ExtraTrees/RandomForest Meta-Classifiers on Joint 98D Space (exp_12 baseline).
    """

    def __init__(self):
        self.variables = list(VARIABLES_11)
        self.weight_order = list(WEIGHT_ORDER_4)
        self.weight_to_idx = {w: i for i, w in enumerate(self.weight_order)}

        self.processor = Task2FeatureProcessor()
        self.pca_mri = PCA(n_components=16, random_state=42)
        self.tfidf = TfidfVectorizer(max_features=50, stop_words="english")

        self.models: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        df_tab: pd.DataFrame,
        df_txt: pd.DataFrame,
        df_mri: pd.DataFrame,
        var_weights_dict: Dict[str, Dict[str, str]],
    ) -> "Subtask3MultiVariableEnsemble":
        df_merged = df_tab.merge(df_txt, on="pid").merge(df_mri, on="pid")
        pids = df_merged["pid"].tolist()

        # 1. Fit Representations
        self.processor.fit(df_merged)
        X_tab = self.processor.transform_df(df_merged)  # 32D

        mri_cols = [f"emb_{i}" for i in range(1024) if f"emb_{i}" in df_merged.columns]
        if len(mri_cols) == 1024:
            mri_raw = df_merged[mri_cols].values.astype(float)
        else:
            mri_cols = [c for c in df_merged.columns if c.startswith("emb_") or c.startswith("mri_")]
            mri_raw = df_merged[mri_cols].iloc[:, :1024].values.astype(float)
        X_mri = self.pca_mri.fit_transform(mri_raw)  # 16D

        txt_col = "full_encounter_narrative" if "full_encounter_narrative" in df_merged.columns else "clinical_text"
        X_txt = self.tfidf.fit_transform(df_merged[str(txt_col)].fillna("").astype(str)).toarray()  # 50D

        X_joint = np.hstack([X_tab, X_mri, X_txt])  # 98D

        # 2. Fit per-variable models
        for var in self.variables:
            y_var = [self.weight_to_idx.get(var_weights_dict.get(pid, {}).get(var, "not_used"), 0) for pid in pids]

            # Use RF for bx_isup and ct; ExtraTrees for the rest (exp_12 champion architecture)
            if var in ["bx_isup", "ct"]:
                clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
            else:
                clf = ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)

            clf.fit(X_joint, y_var)
            self.models[var] = clf

        self.is_fitted = True
        return self

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Subtask3MultiVariableEnsemble is not fitted.")

        case_id, feat_dict, narrative, mri_vec = extract_raw_patient_payload_task2(raw_input)
        x_tab = self.processor.transform_single(feat_dict)
        x_mri = self.pca_mri.transform([mri_vec])[0]
        x_txt = self.tfidf.transform([narrative]).toarray()[0]

        x_joint = np.hstack([x_tab, x_mri, x_txt]).reshape(1, -1)

        weights_res: Dict[str, str] = {}
        decisive_factors: List[str] = []
        important_factors: List[str] = []

        for var in self.variables:
            pred_idx = int(self.models[var].predict(x_joint)[0])
            w_str = self.weight_order[pred_idx]
            weights_res[var] = w_str
            if w_str == "decisive":
                decisive_factors.append(var)
            elif w_str == "important":
                important_factors.append(var)

        return {
            "case_id": case_id,
            "variable_weights": weights_res,
            "decisive_factors": decisive_factors,
            "important_factors": important_factors,
        }
