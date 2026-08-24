"""
Models, feature processors, SHAP attribution extractors, and meta-classifiers for exp_8.
Task 1: Pre-Biopsy Decision & Subtask 1.3: Variable Relevance Weighting.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import shap

warnings.filterwarnings("ignore")

# 10 clinical variables to report relevance on (Task 1.3 target weights)
CLINICAL_VARIABLES_10 = [
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

# Mapping between target weights and numerical codes
WEIGHT_STR_TO_INT = {
    "not_used": 0,
    "noted": 1,
    "important": 2,
    "decisive": 3,
}
WEIGHT_INT_TO_STR = {v: k for k, v in WEIGHT_STR_TO_INT.items()}

# Mapping from Task 1 target decision + confidence to continuous fuzzy risk target
FUZZY_TARGET_MAP = {
    ("yes", "clear"): 1.00,
    ("yes", "borderline"): 0.75,
    ("yes", "uncertain"): 0.50,
    ("no", "uncertain"): 0.50,
    ("no", "borderline"): 0.25,
    ("no", "clear"): 0.00,
}


class TabularPreprocessor:
    """
    Leak-free clinical tabular preprocessor for CHIMERA Task 1.
    Strictly excludes all 14 biopsy/pathology-derived forbidden fields.
    """

    def __init__(self):
        self.num_cols = ["age", "psa", "vol", "psad", "pirads", "cspca", "psav", "psap", "months", "ipss_score"]
        self.cat_cols = ["dre", "enc_type"]
        self.imputer_num: Optional[SimpleImputer] = None
        self.imputer_cat: Optional[SimpleImputer] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.encoded_feature_names: List[str] = []
        self.cat_dummies_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        # 1. Copiar y verificar columnas
        df_sub = df.copy()
        for col in self.num_cols:
            if col not in df_sub.columns:
                df_sub[col] = np.nan
        for col in self.cat_cols:
            if col not in df_sub.columns:
                df_sub[col] = "Missing"

        # 2. Imputación numérica y categórica
        self.imputer_num = SimpleImputer(strategy="median")
        num_imputed = self.imputer_num.fit_transform(df_sub[self.num_cols])

        self.imputer_cat = SimpleImputer(strategy="most_frequent")
        cat_imputed = self.imputer_cat.fit_transform(df_sub[self.cat_cols].astype(str))

        # 3. One-hot encoding de categóricas
        cat_df = pd.DataFrame(cat_imputed, columns=self.cat_cols, index=df_sub.index)
        cat_encoded = pd.get_dummies(cat_df, drop_first=False, dtype=float)
        self.cat_dummies_columns = list(cat_encoded.columns)

        # 4. Concatenación
        num_df = pd.DataFrame(num_imputed, columns=self.num_cols, index=df_sub.index)
        combined = pd.concat([num_df, cat_encoded], axis=1)

        # 5. Escalamiento MinMaxScaler
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(combined)
        self.encoded_feature_names = list(combined.columns)

        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.imputer_num is None or self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        df_sub = df.copy()
        for col in self.num_cols:
            if col not in df_sub.columns:
                df_sub[col] = np.nan
        for col in self.cat_cols:
            if col not in df_sub.columns:
                df_sub[col] = "Missing"

        num_imputed = self.imputer_num.transform(df_sub[self.num_cols])
        cat_imputed = self.imputer_cat.transform(df_sub[self.cat_cols].astype(str))

        cat_df = pd.DataFrame(cat_imputed, columns=self.cat_cols, index=df_sub.index)
        cat_encoded = pd.get_dummies(cat_df, drop_first=False, dtype=float)

        # Alinear columnas One-Hot con las de fit
        cat_encoded = cat_encoded.reindex(columns=self.cat_dummies_columns, fill_value=0.0)

        num_df = pd.DataFrame(num_imputed, columns=self.num_cols, index=df_sub.index)
        combined = pd.concat([num_df, cat_encoded], axis=1)

        return self.scaler.transform(combined)


class FuzzyKNNRegressor(BaseEstimator, RegressorMixin):
    """
    Fuzzy k-Nearest Neighbors Regressor matching the optimal C1* from exp_4.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "distance"):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.nn: Optional[NearestNeighbors] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyKNNRegressor":
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        actual_k = min(self.k, len(self.X_train_))
        self.nn = NearestNeighbors(n_neighbors=actual_k, metric=self.metric)
        self.nn.fit(self.X_train_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.nn is None or self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = np.asarray(X, dtype=float)
        dists, indices = self.nn.kneighbors(X_arr)
        preds = np.zeros(len(X_arr), dtype=float)

        for i in range(len(X_arr)):
            d = dists[i]
            idx = indices[i]
            y_neigh = self.y_train_[idx]

            if np.any(d == 0):
                preds[i] = np.mean(y_neigh[d == 0])
            else:
                if self.weights == "distance":
                    w = 1.0 / (d + 1e-6)
                    preds[i] = np.sum(w * y_neigh) / np.sum(w)
                else:
                    preds[i] = np.mean(y_neigh)

        return np.clip(preds, 0.0, 1.0)


class ShapAttributionExtractor:
    """
    Computes local SHAP attributions from the base Tabular model (C1*) strictly on the fold.
    Maps encoded features back to the 10 target clinical variables.
    """

    def __init__(self, model: FuzzyKNNRegressor, preprocessor: TabularPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer: Optional[shap.KernelExplainer] = None
        self.background_summary: Optional[np.ndarray] = None

    def fit_explainer(self, X_train: np.ndarray):
        # Usar k-means summary para eficiencia y estabilidad matemática
        if len(X_train) > 25:
            self.background_summary = shap.kmeans(X_train, 15)
        else:
            self.background_summary = X_train

        # Wrapper de predicción continua
        def model_predict(x_eval):
            return self.model.predict(x_eval)

        self.explainer = shap.KernelExplainer(model_predict, self.background_summary)

    def extract_10d_shap(self, X: np.ndarray, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Calculates the 10D Shapley attribution vector:
        [ |phi_age|, |phi_psa|, |phi_vol|, |phi_psad|, |phi_dre|,
          |phi_pirads|, |phi_fh|, |phi_comorb|, |phi_bx|, |phi_cspca| ]
        """
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted before extracting attributions.")

        # KernelSHAP local values
        shap_raw = self.explainer.shap_values(X, nsamples=100, silent=True)
        if isinstance(shap_raw, list):
            shap_raw = shap_raw[0]

        feature_names = self.preprocessor.encoded_feature_names
        n_samples = len(X)
        phi_10d = np.zeros((n_samples, 10), dtype=float)

        for i in range(n_samples):
            feat_abs_shap = {feat: abs(shap_raw[i, j]) for j, feat in enumerate(feature_names)}

            # 1. age
            phi_10d[i, 0] = feat_abs_shap.get("age", 0.0)
            # 2. psa
            phi_10d[i, 1] = feat_abs_shap.get("psa", 0.0)
            # 3. vol
            phi_10d[i, 2] = feat_abs_shap.get("vol", 0.0)
            # 4. psad
            phi_10d[i, 3] = feat_abs_shap.get("psad", 0.0)
            # 5. dre (agrupa one-hot features de DRE)
            dre_val = sum(v for k, v in feat_abs_shap.items() if k.startswith("dre_"))
            phi_10d[i, 4] = dre_val
            # 6. pirads
            phi_10d[i, 5] = feat_abs_shap.get("pirads", 0.0)
            # 7. fh (antecedentes familiares - proxy clínico pre-biopsia)
            phi_10d[i, 6] = feat_abs_shap.get("ipss_score", 0.0) * 0.5
            # 8. comorbidity (proxy comorbilidad)
            phi_10d[i, 7] = feat_abs_shap.get("months", 0.0) * 0.5 + feat_abs_shap.get("ipss_score", 0.0) * 0.5
            # 9. bx (motivo consulta: antecedente de biopsia o sospecha previa)
            enc_val = sum(v for k, v in feat_abs_shap.items() if "biopsy" in k.lower() or "pca" in k.lower())
            phi_10d[i, 8] = enc_val
            # 10. cspca
            phi_10d[i, 9] = feat_abs_shap.get("cspca", 0.0)

        return phi_10d


class CoalescedAugmenter:
    """
    Augments the 10D attribution vector with coalesced coalition energy terms:
    1. Phi_{PSA, PSAD}    = |phi_psa| + |phi_psad|
    2. Phi_{PIRADS, csPCa} = |phi_pirads| + |phi_cspca|
    3. Phi_{Vol, PSAD}    = |phi_vol| + |phi_psad|
    Resulting in a 13D space: Phi_aug in R^13.
    """

    @staticmethod
    def augment(phi_10d: np.ndarray) -> np.ndarray:
        psa = phi_10d[:, 1]
        vol = phi_10d[:, 2]
        psad = phi_10d[:, 3]
        pirads = phi_10d[:, 5]
        cspca = phi_10d[:, 9]

        coal_psa_psad = (psa + psad)[:, np.newaxis]
        coal_pirads_cspca = (pirads + cspca)[:, np.newaxis]
        coal_vol_psad = (vol + psad)[:, np.newaxis]

        phi_13d = np.hstack([phi_10d, coal_psa_psad, coal_pirads_cspca, coal_vol_psad])
        return phi_13d


class RelevanceMetaPredictor:
    """
    Predicts the 10 discrete clinical variable weights y_j in {0, 1, 2, 3}
    using multivariate balanced decision trees or random forests.
    """

    def __init__(
        self,
        condition: str = "C3_Coalesced_Balanced_Tree",
        max_depth: int = 3,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.condition = condition
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimators: Dict[str, Union[DecisionTreeClassifier, RandomForestClassifier]] = {}

    def fit(self, X_train: np.ndarray, y_train_weights: pd.DataFrame, raw_train_df: Optional[pd.DataFrame] = None):
        """
        X_train: (N, 10) or (N, 13) attribution vector.
        y_train_weights: DataFrame with columns weight_age, weight_psa, ..., weight_cspca.
        """
        self.estimators = {}

        for var in CLINICAL_VARIABLES_10:
            target_col = f"weight_{var}"
            if target_col not in y_train_weights.columns:
                continue

            y_raw = y_train_weights[target_col].values
            y_int = np.array([WEIGHT_STR_TO_INT.get(str(v).lower(), 1) for v in y_raw], dtype=int)

            # Instanciar según condición
            if "Forest" in self.condition:
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=self.max_depth,
                    class_weight=self.class_weight,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                )
            else:
                clf = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    class_weight=self.class_weight,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                )

            # Ajuste supervisado
            clf.fit(X_train, y_int)
            self.estimators[var] = clf

    def predict(
        self,
        X_test: np.ndarray,
        raw_test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary: { 'age': array_of_strings, 'psa': array_of_strings, ... }
        """
        preds_dict = {}

        for var in CLINICAL_VARIABLES_10:
            if var not in self.estimators:
                preds_dict[var] = np.array(["noted"] * len(X_test))
                continue

            clf = self.estimators[var]
            y_pred_int = clf.predict(X_test)

            # Condición C5: Regularización híbrida / Monotone Prior para DRE y Bx si aplica
            if self.condition == "C5_Hybrid_Regularized" and raw_test_df is not None:
                if var == "dre" and "dre" in raw_test_df.columns:
                    dre_vals = raw_test_df["dre"].astype(str).str.lower().values
                    for idx, dv in enumerate(dre_vals):
                        if "nodus" in dv or "abnormal" in dv or "suspicious" in dv:
                            y_pred_int[idx] = max(y_pred_int[idx], 2)  # important or decisive
                elif var == "bx" and "enc_type" in raw_test_df.columns:
                    enc_vals = raw_test_df["enc_type"].astype(str).str.lower().values
                    for idx, ev in enumerate(enc_vals):
                        if "prior pca" in ev or "prior negative biopsy" in ev:
                            y_pred_int[idx] = max(y_pred_int[idx], 2)

            preds_dict[var] = np.array([WEIGHT_INT_TO_STR.get(val, "noted") for val in y_pred_int])

        return preds_dict
