"""Self-contained production Cox-Ridge model for Task 3."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from deploy.tasks.task3.preprocessing import FEATURE_COLUMNS

PENALIZER = 0.3
PCA_COMPONENTS = 5
EVENT_HORIZON_MONTHS = 60.0
SURVIVAL_THRESHOLD = 0.7
MAX_MONTHS = 330.0


class Task3CoxPCA5:
    """Cox Ridge model using canonical features and pooled prostatectomy PCA-5."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=0)
        self.model = CoxPHFitter(penalizer=PENALIZER)
        self.feature_columns = list(FEATURE_COLUMNS)
        self.fitted = False

    @staticmethod
    def _design(features: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
        base = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        pooled = np.asarray([np.nanmean(embeddings[str(cid)], axis=0) for cid in features.case_id], dtype=float)
        if not np.isfinite(pooled).all():
            raise ValueError("Non-finite prostatectomy embedding")
        return pd.concat([base, pd.DataFrame(pooled)], axis=1)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> "Task3CoxPCA5":
        design = self._design(features, embeddings)
        transformed = self.pca.fit_transform(design.iloc[:, len(FEATURE_COLUMNS):])
        design = pd.concat([design.iloc[:, :len(FEATURE_COLUMNS)], pd.DataFrame(transformed, columns=[f"pca_{i}" for i in range(PCA_COMPONENTS)])], axis=1)
        design = pd.DataFrame(self.imputer.fit_transform(design), columns=design.columns)
        cox = design.copy()
        cox["months_to_recurrence"] = labels.months_to_recurrence.to_numpy()
        cox["event"] = labels.event.astype(int).to_numpy()
        self.model.fit(cox, duration_col="months_to_recurrence", event_col="event", show_progress=False)
        self.fitted = True
        return self

    def _transform(self, features: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Task 3 model is not fitted")
        design = self._design(features, embeddings)
        transformed = self.pca.transform(design.iloc[:, len(FEATURE_COLUMNS):])
        design = pd.concat([design.iloc[:, :len(FEATURE_COLUMNS)], pd.DataFrame(transformed, columns=[f"pca_{i}" for i in range(PCA_COMPONENTS)])], axis=1)
        return pd.DataFrame(self.imputer.transform(design), columns=design.columns)

    def predict(self, features: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
        design = self._transform(features, embeddings)
        months = np.asarray(self.model.predict_percentile(design, p=0.90), dtype=float).reshape(-1)
        months = np.clip(np.where(np.isfinite(months), months, MAX_MONTHS), 0.0, MAX_MONTHS)
        survival = self.model.predict_survival_function(design, times=[EVENT_HORIZON_MONTHS])
        event = (np.asarray(survival.values, dtype=float).reshape(-1) < SURVIVAL_THRESHOLD).astype(int)
        return {"months_to_recurrence": float(months[0]), "event": int(event[0])}

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Task3CoxPCA5":
        return joblib.load(path)
