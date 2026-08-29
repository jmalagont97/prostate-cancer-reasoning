"""
Multimodal Model Implementations, Preprocessors, and KNN/Fuzzy-KNN Estimators
for CHIMERA Subtask 1.1 Biopsy Decision (exp_4).
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import complete, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

CONFIDENCE_FUZZY_WEIGHTS = {
    ("yes", "clear"): 1.00,
    ("yes", "borderline"): 0.75,
    ("yes", "uncertain"): 0.50,
    ("no", "uncertain"): 0.50,
    ("no", "borderline"): 0.25,
    ("no", "clear"): 0.00,
}

NUMERIC_COLS = [
    "age", "psa", "vol", "psad", "pirads", "bx", "psav", "psap", "months",
    "ipss_score", "allergies_count", "comorbidity_count",
    "vitals_weight_kg", "vitals_height_cm", "vitals_bmi",
    "vitals_bp_sys", "vitals_bp_dia", "vitals_hr_bpm", "smoking_pack_years"
]

CATEGORICAL_COLS = [
    "cli_dre", "cli_fh_binary", "vit_smoking_status"
]


class TabularPreprocessor:
    def __init__(self, spearman_tau: Optional[float] = None, spearman_threshold: Optional[float] = None):
        self.spearman_threshold = spearman_tau if spearman_tau is not None else spearman_threshold
        self.impute_values: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.scaler = MinMaxScaler()
        self.kept_cols: List[str] = []
        self.feature_names_: List[str] = []


    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        df_work = df.copy()
        for col in NUMERIC_COLS:
            if col in df_work.columns:
                val = df_work[col].median()
                self.impute_values[col] = float(val) if not pd.isna(val) else 0.0

        for col in CATEGORICAL_COLS:
            if col in df_work.columns:
                m = df_work[col].mode()
                self.cat_modes[col] = str(m.iloc[0]) if not m.empty else "unknown"

        X_num, X_cat, X_ind = self._extract_raw_features(df_work, is_fit=True)

        if self.spearman_threshold is not None and X_num.shape[1] > 1:
            stds = np.std(X_num, axis=0)
            non_const_indices = np.where(stds > 1e-7)[0]
            if len(non_const_indices) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(X_num[:, non_const_indices])
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                dist_matrix = np.clip(1.0 - np.abs(corr), 0.0, 1.0)
                np.fill_diagonal(dist_matrix, 0.0)
                condensed = squareform(dist_matrix, checks=False)
                clusters = fcluster(complete(condensed), t=1.0 - self.spearman_threshold, criterion="distance")
                selected_rel = []
                for cl in np.unique(clusters):
                    members = np.where(clusters == cl)[0]
                    selected_rel.append(members[0])
                kept_num_indices = non_const_indices[selected_rel]
            else:
                kept_num_indices = non_const_indices
            X_num = X_num[:, kept_num_indices]
            self.kept_num_indices = kept_num_indices
        else:
            self.kept_num_indices = np.arange(X_num.shape[1])

        X_full = np.hstack([X_num, X_cat, X_ind]) if X_ind.shape[1] > 0 else np.hstack([X_num, X_cat])
        self.scaler.fit(X_full)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_work = df.copy()
        X_num, X_cat, X_ind = self._extract_raw_features(df_work, is_fit=False)
        X_num = X_num[:, self.kept_num_indices]
        X_full = np.hstack([X_num, X_cat, X_ind]) if X_ind.shape[1] > 0 else np.hstack([X_num, X_cat])
        return np.clip(self.scaler.transform(X_full), 0.0, 1.0)

    def _extract_raw_features(self, df: pd.DataFrame, is_fit: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_features = []
        missing_indicators = []
        for col in NUMERIC_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                missing_indicators.append(series.isna().astype(float).values)
                fill_val = self.impute_values.get(col, 0.0)
                num_features.append(series.fillna(fill_val).values)
            else:
                num_features.append(np.zeros(len(df)))
                missing_indicators.append(np.ones(len(df)))

        X_num = np.column_stack(num_features)
        X_ind = np.column_stack(missing_indicators)

        cat_encoded_dfs = []
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                s = df[col].fillna(self.cat_modes.get(col, "unknown")).astype(str)
            else:
                s = pd.Series(["unknown"] * len(df))
            dummies = pd.get_dummies(s, prefix=col, drop_first=False)
            cat_encoded_dfs.append(dummies)

        if cat_encoded_dfs:
            cat_df = pd.concat(cat_encoded_dfs, axis=1).astype(float)
            if is_fit:
                self.cat_columns_ = cat_df.columns.tolist()
            else:
                cat_df = cat_df.reindex(columns=self.cat_columns_, fill_value=0.0)
            X_cat = cat_df.values
        else:
            X_cat = np.empty((len(df), 0))

        return X_num, X_cat, X_ind


class MRIPreprocessor:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MRIPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.mean_vector = np.mean(X_arr, axis=0)

        # Replace any all-zero rows with mean_vector
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0):
                X_arr[i] = self.mean_vector

        if self.n_components is not None:
            max_comp = min(X_arr.shape[0] - 1, X_arr.shape[1], self.n_components)
            self.pca = PCA(n_components=max(1, max_comp), random_state=42)
            self.pca.fit(X_arr)
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
                self.svd = TruncatedSVD(n_components=max_comp, random_state=42)
                self.svd.fit(X_tfidf)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(texts)
        if self.svd is not None:
            return self.svd.transform(X_tfidf)
        return X_tfidf.toarray()


class CustomKNNPredictor:
    def __init__(
        self,
        k: int = 5,
        metric: str = "cosine",
        weights: str = "uniform",
        is_fuzzy: bool = False,
        model_type: Optional[str] = None,
    ):
        self.k = k
        self.metric = metric
        self.weights = weights
        if model_type is not None:
            self.is_fuzzy = ("fuzzy" in model_type.lower())
        else:
            self.is_fuzzy = is_fuzzy
        self.X_train: Optional[np.ndarray] = None
        self.y_binary: Optional[np.ndarray] = None
        self.y_continuous: Optional[np.ndarray] = None


    def fit(
        self,
        X_train: np.ndarray,
        y_decision: List[str],
        y_confidence: Optional[List[str]] = None,
    ) -> "CustomKNNPredictor":
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_binary = np.array([1 if d == "yes" else 0 for d in y_decision], dtype=float)

        if self.is_fuzzy:
            if y_confidence is None:
                y_confidence = ["clear"] * len(y_decision)
            self.y_continuous = np.array(
                [
                    CONFIDENCE_FUZZY_WEIGHTS.get((d.lower(), str(c).lower()), 0.5)
                    for d, c in zip(y_decision, y_confidence)
                ],
                dtype=float,
            )
        else:
            self.y_continuous = self.y_binary

        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        X_val_arr = np.asarray(X_val, dtype=float)
        distances = pairwise_distances(X_val_arr, self.X_train, metric=self.metric)
        distances = np.nan_to_num(distances, nan=1e6)

        k_eff = min(self.k, self.X_train.shape[0])
        idx = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]

        probs = []
        for i in range(len(X_val_arr)):
            row_idx = idx[i]
            row_dist = distances[i, row_idx]
            order = np.argsort(row_dist)
            sorted_idx = row_idx[order]
            sorted_dist = row_dist[order]

            y_targets = self.y_continuous[sorted_idx]

            if self.weights == "distance":
                w = 1.0 / (sorted_dist + 1e-5)
            else:
                w = np.ones_like(sorted_dist)

            p_i = float(np.sum(w * y_targets) / (np.sum(w) + 1e-9))
            probs.append(np.clip(p_i, 0.0, 1.0))

        return np.array(probs, dtype=float)

    def predict(self, X_val: np.ndarray, threshold: float = 0.50) -> List[str]:
        probs = self.predict_proba(X_val)
        return ["yes" if p >= threshold else "no" for p in probs]


CustomKNN = CustomKNNPredictor

