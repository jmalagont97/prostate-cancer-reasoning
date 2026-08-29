"""
Exact Replication of exp_12 Scalar Direct-KDM for Subtask 1.2 Clinical Confidence.
Zero Data Leakage: Excludes all biopsy/pathology-derived fields (bx_positive, bx_missing, isup).
Input Feature Frames:
  - 23col Clean: PSA suite (psa, psap, psav, psad), vol, age, cspca, vit_bmi,
                 pirads, dre (ordinal + not_done), 6 comorbidity flags, MRI-PCA(2) + mri_missing.
  - 19col Clean: Drops psap, psav, vit_bmi (exp_3 frame baseline).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIDENCE_CLASSES = ["uncertain", "borderline", "clear"]
CONFIDENCE_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}
INV_CONFIDENCE_MAP = {0: "uncertain", 1: "borderline", 2: "clear"}

COMORBIDITY_GROUPS = {
    "comorb_cardiometabolic": [
        "Hypertension", "Hypercholesterolaemia", "Coronary artery disease", "Type 2 diabetes", "Obesity"
    ],
    "comorb_renal": ["Chronic kidney disease"],
    "comorb_bleeding_risk": ["Atrial fibrillation"],
    "comorb_respiratory": ["COPD"],
    "comorb_bph": ["Benign prostatic hyperplasia"],
}
_KNOWN_CONDITIONS = [c for conditions in COMORBIDITY_GROUPS.values() for c in conditions] + ["Osteoarthritis"]
_KNOWN_CONDITIONS_LOWER = {c.lower() for c in _KNOWN_CONDITIONS}


def extract_comorbidity_flags(medhx_series: pd.Series) -> pd.DataFrame:
    """Extract 6 standardized comorbidity binary flags from clinical history text."""
    def _split(raw):
        if pd.isna(raw):
            return []
        parts = []
        for entry in str(raw).split(","):
            entry = re.sub(r"\s*\(.*?\)", "", entry).strip()
            if entry:
                parts.append(entry)
        return parts

    parsed = medhx_series.apply(_split)
    out = {}
    for group, conditions in COMORBIDITY_GROUPS.items():
        cond_set = {c.lower() for c in conditions}
        out[group] = parsed.apply(lambda conds, cs=cond_set: int(any(c.lower() in cs for c in conds)))

    out["comorb_other_unmatched"] = parsed.apply(
        lambda conds: int(any(c.lower() not in _KNOWN_CONDITIONS_LOWER for c in conds))
    )
    return pd.DataFrame(out, index=medhx_series.index)


class Exp12FeatureBuilder:
    """
    Builds the clean, zero-leakage feature matrices corresponding to exp_12 (23col and 19col).
    Fitted strictly on train_pids per fold to prevent PCA and imputation leakage.
    """

    def __init__(self, frame_tag: str = "23col", n_mri_pca: int = 2):
        assert frame_tag in ("23col", "19col")
        self.frame_tag = frame_tag
        self.n_mri_pca = n_mri_pca
        self.pca: Optional[PCA] = None
        self.mri_mean_vec: Optional[np.ndarray] = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def _extract_raw_frame(self, df_tab: pd.DataFrame, df_txt: pd.DataFrame) -> pd.DataFrame:
        if self.frame_tag == "23col":
            cont_cols = ["psa", "psap", "psav", "psad", "vol", "age", "cspca", "vitals_bmi"]
        else:
            cont_cols = ["psa", "psad", "vol", "age", "cspca"]

        frame = df_tab[["pid"] + cont_cols].copy()
        frame["bx"] = pd.to_numeric(df_tab["bx"], errors="coerce")
        frame["bx_missing"] = df_tab["bx"].isna().astype(float)
        frame["pirads"] = df_tab["pirads"].values

        dre_map = {"Normal": 0, "Abnormal": 1, "Nodus": 2, "Suspicious": 3}
        dre_raw = df_tab["dre"]
        frame["dre_not_done"] = (dre_raw == "Not done").astype(float)
        frame["dre_ordinal"] = dre_raw.map(dre_map).astype(float)

        flags = extract_comorbidity_flags(df_txt.set_index("pid").loc[df_tab["pid"]]["medhx"])
        for col in flags.columns:
            frame[col] = flags[col].values

        return frame

    def fit(self, df_tab: pd.DataFrame, df_txt: pd.DataFrame, df_mri: pd.DataFrame, train_pids: List[str]) -> "Exp12FeatureBuilder":
        tab_train = df_tab[df_tab["pid"].isin(train_pids)]
        raw_frame_tr = self._extract_raw_frame(tab_train, df_txt)
        feature_cols = [c for c in raw_frame_tr.columns if c != "pid"]

        # 1. Fit tabular median imputer
        self.imputer.fit(raw_frame_tr[feature_cols].values)

        # 2. Fit MRI PCA strictly on training embeddings
        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        mri_tr_df = df_mri.set_index("pid").loc[train_pids][emb_cols]
        mri_tr_arr = np.nan_to_num(mri_tr_df.values, nan=0.0)

        non_zero_tr = mri_tr_arr[~np.all(mri_tr_arr == 0.0, axis=1)]
        if len(non_zero_tr) > 0:
            self.mri_mean_vec = np.mean(non_zero_tr, axis=0)
        else:
            self.mri_mean_vec = np.zeros(1024, dtype=float)

        for i in range(len(mri_tr_arr)):
            if np.all(mri_tr_arr[i] == 0.0):
                mri_tr_arr[i] = self.mri_mean_vec

        self.pca = PCA(n_components=self.n_mri_pca, random_state=42)
        self.pca.fit(mri_tr_arr)

        # 3. Fit standard scaler on imputed tabular + MRI PCA features
        X_tab_tr = self.imputer.transform(raw_frame_tr[feature_cols].values)
        X_mri_pca_tr = self.pca.transform(mri_tr_arr)
        mri_missing_tr = np.all(df_mri.set_index("pid").loc[train_pids][emb_cols].isna().values, axis=1).astype(float)[:, None]

        X_full_tr = np.hstack([X_tab_tr, X_mri_pca_tr, mri_missing_tr])
        self.scaler.fit(X_full_tr)

        self.feature_names = feature_cols + [f"mri_pca_{i}" for i in range(self.n_mri_pca)] + ["mri_missing"]
        return self

    def transform(self, df_tab: pd.DataFrame, df_txt: pd.DataFrame, df_mri: pd.DataFrame, pids: List[str]) -> np.ndarray:
        tab_sub = df_tab.set_index("pid").loc[pids].reset_index()
        raw_frame = self._extract_raw_frame(tab_sub, df_txt)
        feature_cols = [c for c in raw_frame.columns if c != "pid"]

        X_tab = self.imputer.transform(raw_frame[feature_cols].values)

        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        mri_df = df_mri.set_index("pid").loc[pids][emb_cols]
        mri_arr = np.nan_to_num(mri_df.values, nan=0.0)

        for i in range(len(mri_arr)):
            if np.all(mri_arr[i] == 0.0) and self.mri_mean_vec is not None:
                mri_arr[i] = self.mri_mean_vec

        X_mri_pca = self.pca.transform(mri_arr)
        mri_missing = np.all(df_mri.set_index("pid").loc[pids][emb_cols].isna().values, axis=1).astype(float)[:, None]

        X_full = np.hstack([X_tab, X_mri_pca, mri_missing])
        return self.scaler.transform(X_full)


class ScalarDirectKDM:
    """
    Quantum-inspired Kernel Density Matrix classifier with frozen prototypes and scalar sigma.
    Direct 3-class training exactly matching exp_12/kdm_backbone.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_epochs: int = 300,
        lr: float = 0.01,
        random_state: int = 0,
        supervision_type: str = "hard",  # "hard" or "ordinal_weighted"
    ):
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.supervision_type = supervision_type
        self.c_x: Optional[torch.Tensor] = None
        self.c_y: Optional[torch.Tensor] = None
        self.sigma: Optional[torch.Tensor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScalarDirectKDM":
        torch.manual_seed(self.random_state)
        n, dim_x = X.shape

        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.long)

        # Build target matrix
        if self.supervision_type == "ordinal_weighted":
            # Soft targets: Y_ik proportional to 1.0 - |k - y_i|/2.0
            Y_soft = torch.zeros((n, self.n_classes), dtype=torch.float32)
            for k in range(self.n_classes):
                Y_soft[:, k] = 1.0 - (yt - k).abs().float() / 2.0
            Y_soft = Y_soft / Y_soft.sum(dim=1, keepdim=True)
            self.c_y = Y_soft
        else:
            self.c_y = F.one_hot(yt, self.n_classes).float()

        self.c_x = Xt.clone()

        # Initial sigma heuristic: median pairwise distance
        with torch.no_grad():
            diff = Xt.unsqueeze(1) - Xt.unsqueeze(0)
            dists = torch.norm(diff, dim=2)
            mask = ~torch.eye(n, dtype=torch.bool)
            med_dist = torch.median(dists[mask]).item()
            init_sigma = max(0.1, med_dist / np.sqrt(2.0))

        log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma), dtype=torch.float32, requires_grad=True))
        optimizer = torch.optim.Adam([log_sigma], lr=self.lr)

        for _ in range(self.n_epochs):
            sigma_val = torch.exp(log_sigma)
            # RBF kernel distance: (n, n)
            diff = Xt.unsqueeze(1) - self.c_x.unsqueeze(0)  # (n, n, dim_x)
            sq_dists = (diff ** 2).sum(dim=2)  # (n, n)
            weights = F.softmax(-sq_dists / (2.0 * (sigma_val ** 2) + 1e-7), dim=1)  # (n, n)

            probs = torch.matmul(weights, self.c_y)  # (n, n_classes)
            probs = probs.clamp_min(1e-7)

            if self.supervision_type == "ordinal_weighted":
                loss = -(self.c_y * torch.log(probs)).sum(dim=1).mean()
            else:
                loss = F.nll_loss(torch.log(probs), yt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.sigma = torch.exp(log_sigma).detach()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            diff = Xt.unsqueeze(1) - self.c_x.unsqueeze(0)  # (m, n, dim_x)
            sq_dists = (diff ** 2).sum(dim=2)
            weights = F.softmax(-sq_dists / (2.0 * (self.sigma ** 2) + 1e-7), dim=1)
            probs = torch.matmul(weights, self.c_y)
        return probs.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
