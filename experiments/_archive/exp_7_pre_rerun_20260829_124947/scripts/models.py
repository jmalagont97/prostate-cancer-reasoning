"""
Multimodal Model Implementations, Preprocessors, and Composite Fuzzy ICI Estimators
for CHIMERA Subtask 1.2 Clinical Confidence (exp_7).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

CONFIDENCE_FUZZY_WEIGHTS = {
    ("yes", "clear"): 1.00,
    ("yes", "borderline"): 0.75,
    ("yes", "uncertain"): 0.50,
    ("no", "uncertain"): 0.50,
    ("no", "borderline"): 0.25,
    ("no", "clear"): 0.00,
}

CONF_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}
ORD_NAMES = ["uncertain", "borderline", "clear"]

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
    def __init__(self, spearman_tau: Optional[float] = None):
        self.spearman_tau = spearman_tau
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
    ):
        self.k = k
        self.metric = metric
        self.weights = weights
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


def compute_composite_ici(
    p_t: np.ndarray,
    p_m: np.ndarray,
    p_x: np.ndarray,
    weights: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Computes continuous Composite Inconsistency/Confidence Index (ICI) in [0, 1].
    
    Equation:
        p_bar = sum(w_m * p_m)
        sigma = sqrt( (1/3) * sum (p_m - p_mean_unweighted)^2 )
        ICI = 2.0 * |p_bar - 0.5| * (1.0 - 2.0 * sigma)
    """
    p_t = np.asarray(p_t, dtype=float)
    p_m = np.asarray(p_m, dtype=float)
    p_x = np.asarray(p_x, dtype=float)

    if weights is not None:
        w_t, w_m, w_x = weights
        p_bar = w_t * p_t + w_m * p_m + w_x * p_x
    else:
        p_bar = (p_t + p_m + p_x) / 3.0

    p_stack = np.stack([p_t, p_m, p_x], axis=1)
    sigma = np.std(p_stack, axis=1, ddof=0)
    ici = 2.0 * np.abs(p_bar - 0.5) * (1.0 - 2.0 * sigma)
    return np.clip(ici, 0.0, 1.0)


class MetaThresholdTreeFitter:
    """
    Fits 1D Decision Tree on continuous ICI to predict ordinal confidence classes {0, 1, 2}.
    Supports balanced class weighting or uniform weighting.
    """
    def __init__(self, mode: str = "regressor", criterion: str = "absolute_error", use_balanced_weights: bool = True):
        self.mode = mode
        self.criterion = criterion
        self.use_balanced_weights = use_balanced_weights
        self.model: Optional[Union[DecisionTreeRegressor, DecisionTreeClassifier]] = None

    def fit(self, ici_train: np.ndarray, y_train_ord: np.ndarray) -> "MetaThresholdTreeFitter":
        ici_train_2d = np.asarray(ici_train, dtype=float).reshape(-1, 1)
        y_train_ord = np.asarray(y_train_ord, dtype=int)
        N = len(y_train_ord)

        sample_weight = None
        if self.use_balanced_weights:
            class_counts = np.bincount(y_train_ord, minlength=3).astype(float)
            class_counts[class_counts == 0] = 1.0
            w_map = N / (3.0 * class_counts)
            sample_weight = w_map[y_train_ord]

        if self.mode == "regressor":
            self.model = DecisionTreeRegressor(
                max_depth=2,
                max_leaf_nodes=3,
                min_samples_leaf=5,
                criterion=self.criterion,
                random_state=42,
            )
            self.model.fit(ici_train_2d, y_train_ord.astype(float), sample_weight=sample_weight)
        else:
            self.model = DecisionTreeClassifier(
                max_depth=2,
                max_leaf_nodes=3,
                min_samples_leaf=5,
                criterion="gini",
                class_weight="balanced" if self.use_balanced_weights else None,
                random_state=42,
            )
            self.model.fit(ici_train_2d, y_train_ord)

        return self

    def predict(self, ici_val: np.ndarray) -> np.ndarray:
        ici_val_2d = np.asarray(ici_val, dtype=float).reshape(-1, 1)
        if self.mode == "regressor":
            score_hat = self.model.predict(ici_val_2d)
            y_pred_ord = np.clip(np.floor(score_hat + 0.5).astype(int), 0, 2)
            return y_pred_ord
        else:
            return self.model.predict(ici_val_2d).astype(int)
