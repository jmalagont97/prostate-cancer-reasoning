"""
Preprocessors, BrentMemKDM Kernel Density Memory, and Multimodal Late Fusion
for CHIMERA Subtask 1.1 Biopsy Decision (exp_5).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import complete, fcluster
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
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
    """Tabular preprocessor preserving the exact representation of exp_4."""

    def __init__(self, spearman_tau: Optional[float] = None):
        self.spearman_threshold = spearman_tau
        self.impute_values: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.scaler = MinMaxScaler()
        self.kept_num_indices: np.ndarray = np.array([])
        self.cat_columns_: List[str] = []

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
                clusters = fcluster(
                    complete(condensed), t=1.0 - self.spearman_threshold, criterion="distance"
                )
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

    def _extract_raw_features(
        self, df: pd.DataFrame, is_fit: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """MRI embedding preprocessor preserving exact PCA projection from exp_4."""

    def __init__(self, n_components: Optional[int] = 16):
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
    """Text preprocessor preserving TF-IDF representation from exp_4."""

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


class BrentMemKDM:
    """
    Brent-Optimized Memory Kernel Density Matrix (BrentMemKDM).
    Implements continuous RBF bandwidth alignment via 1-D Brent search on MCCV.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        is_fuzzy: bool = False,
        eps: float = 1e-12,
        metric: str = "euclidean",
    ):
        self.sigma = sigma
        self.is_fuzzy = is_fuzzy
        self.eps = eps
        self.metric = metric
        self.X_train: Optional[np.ndarray] = None
        self.y_targets: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_decision: List[str],
        y_confidence: Optional[List[str]] = None,
    ) -> "BrentMemKDM":
        self.X_train = np.asarray(X_train, dtype=float)
        y_bin = np.array([1.0 if str(d).lower() == "yes" else 0.0 for d in y_decision], dtype=float)

        if self.is_fuzzy:
            if y_confidence is None:
                y_confidence = ["clear"] * len(y_decision)
            self.y_targets = np.array(
                [
                    CONFIDENCE_FUZZY_WEIGHTS.get((str(d).lower(), str(c).lower()), 0.5)
                    for d, c in zip(y_decision, y_confidence)
                ],
                dtype=float,
            )
        else:
            self.y_targets = y_bin

        return self

    @staticmethod
    def estimate_reference_sigma(X: np.ndarray, k_neighbor: int = 3, metric: str = "euclidean") -> float:
        """Calculate unsupervised reference bandwidth (mean distance to k-th nearest neighbor)."""
        X_arr = np.asarray(X, dtype=float)
        n_samples = len(X_arr)
        if n_samples <= 1:
            return 1.0
        k_eff = min(k_neighbor + 1, n_samples)
        nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
        nn.fit(X_arr)
        distances, _ = nn.kneighbors(X_arr)
        kth_dists = distances[:, -1]
        sigma_ref = float(np.mean(kth_dists))
        return max(sigma_ref, 1e-4)

    def predict_proba(self, X_val: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Compute conditional probability p(y=1|x) via density matrix associative memory.
        k_j(x) = exp( - ||x - c_j||^2 / (2 * sigma^2) )
        w_j(x) = clamped_j / sum_l clamped_l
        """
        if self.X_train is None or self.y_targets is None:
            raise ValueError("Model must be fitted before calling predict_proba.")

        s = sigma if sigma is not None else self.sigma
        if s is None or s <= 1e-7:
            s = 1.0

        X_val_arr = np.asarray(X_val, dtype=float)
        sq_dist = pairwise_distances(X_val_arr, self.X_train, metric="sqeuclidean")

        # RBF Kernel Affinities
        gamma = 1.0 / (2.0 * (s ** 2))
        k_aff = np.exp(-gamma * sq_dist)  # [N_val, N_train]

        # Clamped mixture component densities
        N_tr = self.X_train.shape[0]
        clamped = np.maximum((1.0 / N_tr) * (k_aff ** 2), self.eps)

        # Normalized associative weights
        sum_clamped = np.sum(clamped, axis=1, keepdims=True)
        weights = clamped / (sum_clamped + 1e-12)

        # Expected decision target
        probs = np.dot(weights, self.y_targets)
        return np.clip(probs, 0.0, 1.0)

    def predict(self, X_val: np.ndarray, threshold: float = 0.50, sigma: Optional[float] = None) -> List[str]:
        probs = self.predict_proba(X_val, sigma=sigma)
        return ["yes" if p >= threshold else "no" for p in probs]
