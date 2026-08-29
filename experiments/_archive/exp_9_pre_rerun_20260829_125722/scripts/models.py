"""
Models, feature processors, SHAP extractors, and joint 23D meta-classifiers for exp_9.
Task 1: Subtask 1.3 Clinical Relevance Weighting with Joint Feature-Attribution Fusion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import shap

CLINICAL_VARIABLES = [
    "age",
    "psa",
    "vol",
    "psad",
    "dre",
    "pirads",
    "fh",
    "comorbidity",
    "bx",
    "cspca",
]

ORDINAL_WEIGHT_MAP = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
ORDINAL_WEIGHT_INV = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}


class TabularPreprocessor:
    """Preprocesses clinical tabular data for C1* Fuzzy-KNN model and raw 10D feature vector."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.raw_scaler = MinMaxScaler()
        self.is_fitted = False
        self.medians: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        # Missing value medians
        num_cols = ["age", "psa", "vol", "psad", "pirads", "bx", "cspca", "psav", "psap", "months", "vitals_bmi", "vitals_bp_sys", "ipss_score"]
        for col in num_cols:
            if col in df.columns:
                self.medians[col] = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0

        # Fit model scaler
        X_mat = self._extract_knn_matrix(df, is_train=True)
        self.scaler.fit(X_mat)

        # Fit raw 10D scaler
        X_raw_mat = self._extract_raw_10d(df)
        self.raw_scaler.fit(X_raw_mat)

        self.is_fitted = True
        return self

    def transform_knn(self, df: pd.DataFrame) -> np.ndarray:
        X_mat = self._extract_knn_matrix(df, is_train=False)
        return self.scaler.transform(X_mat)

    def transform_raw_10d(self, df: pd.DataFrame) -> np.ndarray:
        X_raw_mat = self._extract_raw_10d(df)
        return self.raw_scaler.transform(X_raw_mat)

    def _extract_knn_matrix(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        # Standard clean leak-free feature matrix for C1* Fuzzy KNN
        feats = []
        for _, row in df.iterrows():
            row_feats = [
                row.get("age", self.medians.get("age", 65.0)) if pd.notna(row.get("age")) else self.medians.get("age", 65.0),
                row.get("psa", self.medians.get("psa", 6.0)) if pd.notna(row.get("psa")) else self.medians.get("psa", 6.0),
                row.get("vol", self.medians.get("vol", 40.0)) if pd.notna(row.get("vol")) else self.medians.get("vol", 40.0),
                row.get("psad", self.medians.get("psad", 0.15)) if pd.notna(row.get("psad")) else self.medians.get("psad", 0.15),
                row.get("pirads", self.medians.get("pirads", 3.0)) if pd.notna(row.get("pirads")) else self.medians.get("pirads", 3.0),
                row.get("cspca", self.medians.get("cspca", 0.3)) if pd.notna(row.get("cspca")) else self.medians.get("cspca", 0.3),
                row.get("psav", self.medians.get("psav", 0.0)) if pd.notna(row.get("psav")) else self.medians.get("psav", 0.0),
                row.get("psap", self.medians.get("psap", 0.0)) if pd.notna(row.get("psap")) else self.medians.get("psap", 0.0),
                row.get("months", self.medians.get("months", 12.0)) if pd.notna(row.get("months")) else self.medians.get("months", 12.0),
                row.get("ipss_score", self.medians.get("ipss_score", 10.0)) if pd.notna(row.get("ipss_score")) else self.medians.get("ipss_score", 10.0),
                1.0 if str(row.get("dre", "")).lower() in ["nodus", "abnormal", "suspicious"] else 0.0,
                row.get("bx", self.medians.get("bx", 0.0)) if pd.notna(row.get("bx")) else self.medians.get("bx", 0.0),
            ]
            feats.append(row_feats)
        return np.array(feats, dtype=np.float32)

    def _extract_raw_10d(self, df: pd.DataFrame) -> np.ndarray:
        raw_feats = []
        for _, row in df.iterrows():
            age = float(row.get("age", 65.0)) if pd.notna(row.get("age")) else self.medians.get("age", 65.0)
            psa = float(row.get("psa", 6.0)) if pd.notna(row.get("psa")) else self.medians.get("psa", 6.0)
            vol = float(row.get("vol", 40.0)) if pd.notna(row.get("vol")) else self.medians.get("vol", 40.0)
            psad = float(row.get("psad", 0.15)) if pd.notna(row.get("psad")) else (psa / max(vol, 1.0))
            
            dre_str = str(row.get("dre", "")).strip().lower()
            dre_val = 2.0 if dre_str in ["nodus", "abnormal", "suspicious"] else (1.0 if dre_str == "normal" else 0.0)
            
            pirads = float(row.get("pirads", 3.0)) if pd.notna(row.get("pirads")) else self.medians.get("pirads", 3.0)
            
            fh_str = str(row.get("free_text", "")).lower() + " " + str(row.get("notes", "")).lower()
            fh_val = 1.0 if ("family history" in fh_str or "father" in fh_str or "brother" in fh_str) else 0.0
            
            comorb = float(row.get("ipss_score", 0.0)) if pd.notna(row.get("ipss_score")) else 0.0
            
            enc_type_str = str(row.get("enc_type", "")).lower()
            bx_value = float(row.get("bx", self.medians.get("bx", 0.0))) if pd.notna(row.get("bx")) else self.medians.get("bx", 0.0)
            
            cspca = float(row.get("cspca", 0.3)) if pd.notna(row.get("cspca")) else self.medians.get("cspca", 0.3)

            raw_feats.append([age, psa, vol, psad, dre_val, pirads, fh_val, comorb, bx_value, cspca])
        return np.array(raw_feats, dtype=np.float32)


class FuzzyKNNRegressor:
    """Optimal C1* Tabular Model from exp_4 (k=5, Euclidean, distance-weighted)."""

    def __init__(self, k: int = 5, eps: float = 1e-5):
        self.k = k
        self.eps = eps
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyKNNRegressor":
        self.X_train = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train = np.ascontiguousarray(y, dtype=np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :], axis=2)
        k = min(self.k, len(self.X_train))
        topk_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

        probs = np.zeros(len(X), dtype=np.float32)
        for i in range(len(X)):
            idx_i = topk_idx[i]
            dists_i = dists[i, idx_i]
            weights = 1.0 / (dists_i + self.eps)
            weights /= weights.sum()
            probs[i] = np.sum(weights * self.y_train[idx_i])
        return np.clip(probs, 0.0, 1.0)


class ShapAttributionExtractor:
    """Extracts local Shapley feature attribution vectors from the trained FuzzyKNN model."""

    def __init__(self, model: FuzzyKNNRegressor, X_train: np.ndarray, n_background: int = 40):
        self.model = model
        bg_idx = np.linspace(0, len(X_train) - 1, min(len(X_train), n_background), dtype=int)
        self.background = X_train[bg_idx]
        self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background)

    def extract_shap_10d(self, X: np.ndarray) -> np.ndarray:
        shap_values = self.explainer.shap_values(X, nsamples=100, silent=True)
        if isinstance(shap_values, list):
            shap_mat = np.array(shap_values[0])
        else:
            shap_mat = np.array(shap_values)

        if shap_mat.ndim == 3:
            shap_mat = shap_mat[:, :, 1] if shap_mat.shape[2] > 1 else shap_mat[:, :, 0]

        # Map the 12 features of C1* to the 10 canonical clinical variables
        # C1* features: [0:age, 1:psa, 2:vol, 3:psad, 4:pirads, 5:cspca, 6:psav, 7:psap, 8:months, 9:ipss, 10:dre, 11:bx]
        # Canonical 10: [age, psa, vol, psad, dre, pirads, fh, comorbidity, bx, cspca]
        N = len(X)
        shap_10d = np.zeros((N, 10), dtype=np.float32)
        shap_10d[:, 0] = np.abs(shap_mat[:, 0])  # age
        shap_10d[:, 1] = np.abs(shap_mat[:, 1]) + 0.5 * np.abs(shap_mat[:, 6]) + 0.5 * np.abs(shap_mat[:, 7])  # psa suite
        shap_10d[:, 2] = np.abs(shap_mat[:, 2])  # vol
        shap_10d[:, 3] = np.abs(shap_mat[:, 3])  # psad
        shap_10d[:, 4] = np.abs(shap_mat[:, 10])  # dre
        shap_10d[:, 5] = np.abs(shap_mat[:, 4])  # pirads
        shap_10d[:, 6] = 0.5 * np.abs(shap_mat[:, 8])  # fh surrogate
        shap_10d[:, 7] = np.abs(shap_mat[:, 9])  # comorbidity (ipss)
        shap_10d[:, 8] = np.abs(shap_mat[:, 11])  # bx
        shap_10d[:, 9] = np.abs(shap_mat[:, 5])  # cspca

        return shap_10d


class JointFeatureAugmenter:
    """Constructs the 13D coalesced SHAP space and the 23D joint representation."""

    @staticmethod
    def augment_coalesced_13d(shap_10d: np.ndarray) -> np.ndarray:
        N = len(shap_10d)
        aug_13d = np.zeros((N, 13), dtype=np.float32)
        aug_13d[:, :10] = shap_10d

        # Coalitions:
        # 10: {PSA, PSAD}
        aug_13d[:, 10] = shap_10d[:, 1] + shap_10d[:, 3]
        # 11: {PIRADS, csPCa}
        aug_13d[:, 11] = shap_10d[:, 5] + shap_10d[:, 9]
        # 12: {Vol, PSAD}
        aug_13d[:, 12] = shap_10d[:, 2] + shap_10d[:, 3]

        return aug_13d

    @staticmethod
    def build_joint_23d(raw_10d: np.ndarray, aug_13d: np.ndarray) -> np.ndarray:
        return np.hstack([raw_10d, aug_13d])


class JointRelevanceMetaPredictor:
    """Meta-Predictor supporting SHAP-Only, Raw-Only, Joint-Forest 23D, ExtraTrees 23D, and Targeted Hybrid."""

    def __init__(self, condition_name: str, random_state: int = 42):
        self.condition_name = condition_name
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train_df: pd.DataFrame) -> "JointRelevanceMetaPredictor":
        for var_idx, var_name in enumerate(CLINICAL_VARIABLES):
            col_target = f"weight_{var_name}"
            y_raw = y_train_df[col_target].map(lambda v: ORDINAL_WEIGHT_MAP.get(str(v).lower(), 1)).values

            if len(np.unique(y_raw)) <= 1:
                self.models[var_name] = ("constant", y_raw[0])
                continue

            if self.condition_name == "C1_SHAP_Only_13D":
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=self.random_state + var_idx,
                )
                clf.fit(X_train, y_raw)
                self.models[var_name] = ("model", clf)

            elif self.condition_name == "C2_Raw_Only_10D":
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=self.random_state + var_idx,
                )
                clf.fit(X_train, y_raw)
                self.models[var_name] = ("model", clf)

            elif self.condition_name == "C3_Joint_Forest_23D":
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state + var_idx,
                )
                clf.fit(X_train, y_raw)
                self.models[var_name] = ("model", clf)

            elif self.condition_name == "C4_Joint_ExtraTrees_23D":
                clf = ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=3,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state + var_idx,
                )
                clf.fit(X_train, y_raw)
                self.models[var_name] = ("model", clf)

            elif self.condition_name == "C5_Targeted_Hybrid_23D":
                # Targeted architecture: deterministic categories use decision trees focused on raw values + SHAP
                if var_name in ["dre", "bx", "fh"]:
                    clf = DecisionTreeClassifier(
                        max_depth=2,
                        class_weight="balanced",
                        random_state=self.random_state + var_idx,
                    )
                else:
                    clf = ExtraTreesClassifier(
                        n_estimators=50,
                        max_depth=3,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=self.random_state + var_idx,
                    )
                clf.fit(X_train, y_raw)
                self.models[var_name] = ("model", clf)

        return self

    def predict(self, X_val: np.ndarray) -> Dict[str, List[str]]:
        preds_dict: Dict[str, List[str]] = {}
        for var_name in CLINICAL_VARIABLES:
            model_info = self.models.get(var_name)
            if model_info is None:
                preds_dict[var_name] = ["noted"] * len(X_val)
                continue

            m_type, estimator = model_info
            if m_type == "constant":
                preds_int = np.full(len(X_val), estimator, dtype=int)
            else:
                preds_int = estimator.predict(X_val)

            preds_dict[var_name] = [ORDINAL_WEIGHT_INV.get(int(p), "noted") for p in preds_int]

        return preds_dict
