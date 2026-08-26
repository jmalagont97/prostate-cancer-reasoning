"""
Canonical Production Model Classes, Feature Processors, and Ingestion Logic for Task 1.

Provides 100% self-contained, deployable models:
  1. Subtask1LateFusionEnsemble (Pre-Biopsy Decision Gate + Multi-modal Neighbor Retriever with Free-Text Reasoning)
  2. Subtask2DirectKDMClassifier (Scalar Direct-KDM 20D Clinical Confidence Classifier)
  3. Subtask3JointExtraTrees (Joint 23D Phenotypic + Coalesced SHAP Variable Weights & Decisive Factors Meta-Predictor)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# CANONICAL INGESTION & FEATURE PARSER HELPERS (Mirrors exp_2 preprocess_task1.py)
# ==============================================================================

def parse_numeric(val: Any) -> Optional[float]:
    """Safely parse float or int from string/number."""
    if val is None or val == "" or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r"[-+]?\d*\.?\d+", val)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return np.nan
    return np.nan


def parse_bp(bp_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse blood pressure string like '125/84 mmHg' into (systolic, diastolic)."""
    if not bp_str or not isinstance(bp_str, str):
        return np.nan, np.nan
    match = re.search(r"(\d+)\s*/\s*(\d+)", bp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def extract_raw_patient_payload(
    raw_input: Union[str, Path, Dict[str, Any]]
) -> Tuple[str, Dict[str, Any], str, np.ndarray]:
    """
    Extracts tabular features, clinical text narrative, and MRI embedding from a raw patient.
    Accepts:
      - Path / str: directory containing structured-prompt.json, etc.
      - dict: in-memory representation.
    """
    case_id = "unknown_case"
    sp: Dict[str, Any] = {}
    mri_vec = np.zeros(1024, dtype=float)

    if isinstance(raw_input, (str, Path)):
        p_dir = Path(raw_input)
        case_id = p_dir.name

        sp_file = p_dir / "structured-prompt.json"
        if sp_file.exists():
            with open(sp_file, "r") as f:
                sp = json.load(f)

        emb_file = p_dir / "prostate-modality-level-neural-representations.json"
        if emb_file.exists():
            try:
                with open(emb_file, "r") as f:
                    emb_data = json.load(f)
                mri_list = emb_data.get("MRI image", emb_data.get("mri", []))
                if mri_list and len(mri_list) > 0 and len(mri_list[0]) == 1024:
                    mri_vec = np.array(mri_list[0], dtype=float)
            except Exception:
                pass
    elif isinstance(raw_input, dict):
        case_id = str(raw_input.get("case_id", raw_input.get("pid", "unknown_case")))
        sp = raw_input.get("structured_prompt", raw_input.get("patient", raw_input))
        mri_list = raw_input.get("mri_representation", raw_input.get("mri", raw_input.get("mri_image", [])))
        if mri_list is not None and len(mri_list) > 0:
            if isinstance(mri_list[0], list):
                mri_vec = np.array(mri_list[0], dtype=float)
            else:
                mri_vec = np.array(mri_list, dtype=float)

    # 1. Parse Tabular
    vitals = sp.get("vitals") or {}
    bp_sys, bp_dia = parse_bp(vitals.get("bp"))
    ipss_raw = sp.get("ipss")
    ipss_score = parse_numeric(ipss_raw)

    tab_dict = {
        "pid": case_id,
        "case_id": sp.get("case_id", case_id),
        "task": sp.get("task", 1),
        "age": parse_numeric(sp.get("age")),
        "psa": parse_numeric(sp.get("psa")),
        "vol": parse_numeric(sp.get("vol")),
        "psad": parse_numeric(sp.get("psad")),
        "dre": sp.get("dre") if sp.get("dre") is not None else np.nan,
        "pirads": sp.get("pirads") if sp.get("pirads") is not None else np.nan,
        "cspca": parse_numeric(sp.get("cspca")),
        "psav": parse_numeric(sp.get("psav")),
        "psap": parse_numeric(sp.get("psap")),
        "months": parse_numeric(sp.get("months")),
        "active_treatment_flag": parse_numeric(sp.get("active_treatment_flag")),
        "ipss_score": ipss_score,
        "ipss_raw": ipss_raw if ipss_raw is not None else np.nan,
        "occupation": sp.get("occupation") if sp.get("occupation") is not None else np.nan,
        "marital": sp.get("marital") if sp.get("marital") is not None else np.nan,
        "living": sp.get("living") if sp.get("living") is not None else np.nan,
        "next_of_kin": sp.get("next_of_kin") if sp.get("next_of_kin") is not None else np.nan,
        "alcohol": sp.get("alcohol") if sp.get("alcohol") is not None else np.nan,
        "exercise": sp.get("exercise") if sp.get("exercise") is not None else np.nan,
        "enc_dept": sp.get("enc_dept") if sp.get("enc_dept") is not None else np.nan,
        "enc_date": sp.get("enc_date") if sp.get("enc_date") is not None else np.nan,
        "enc_ref": sp.get("enc_ref") if sp.get("enc_ref") is not None else np.nan,
        "enc_type": sp.get("enc_type") if sp.get("enc_type") is not None else np.nan,
        "mri_hospital": sp.get("mri_hospital") if sp.get("mri_hospital") is not None else np.nan,
        "ct": sp.get("ct") if sp.get("ct") is not None else np.nan,
        "vitals_weight_kg": parse_numeric(vitals.get("weight")),
        "vitals_height_cm": parse_numeric(vitals.get("height")),
        "vitals_bmi": parse_numeric(vitals.get("bmi")),
        "vitals_bp_sys": bp_sys,
        "vitals_bp_dia": bp_dia,
        "vitals_hr_bpm": parse_numeric(vitals.get("hr")),
        "vitals_smoking": vitals.get("smoking") if vitals.get("smoking") is not None else np.nan,
        "vitals_flu_vax": vitals.get("flu_vax") if vitals.get("flu_vax") is not None else np.nan,
        "vitals_last_dental": vitals.get("last_dental") if vitals.get("last_dental") is not None else np.nan,
    }

    # 2. Parse Text Narrative
    note_sections = sp.get("note_sections") or []
    sec_dict = {sec.get("s"): sec.get("t") for sec in note_sections if isinstance(sec, dict)}
    chief_complaint = sec_dict.get("Chief complaint", "")
    history = sec_dict.get("History", "")
    physical_exam = sec_dict.get("Physical examination", "")

    pmhx_list = sp.get("pmhx") or []
    pmhx_str = "; ".join(pmhx_list) if isinstance(pmhx_list, list) and pmhx_list else ""
    allergies_list = sp.get("allergies") or []
    allergies_str = "; ".join(allergies_list) if isinstance(allergies_list, list) and allergies_list else ""
    recent_other_list = sp.get("recent_other") or []
    recent_other_str = "; ".join(recent_other_list) if isinstance(recent_other_list, list) and recent_other_list else ""

    narrative_parts = []
    if chief_complaint: narrative_parts.append(f"Chief Complaint: {chief_complaint}")
    if history: narrative_parts.append(f"History: {history}")
    if physical_exam: narrative_parts.append(f"Physical Exam: {physical_exam}")
    if sp.get("notes"): narrative_parts.append(f"Summary Notes: {sp.get('notes')}")
    if pmhx_str: narrative_parts.append(f"Medical History: {pmhx_str}")
    if allergies_str: narrative_parts.append(f"Allergies: {allergies_str}")
    if recent_other_str: narrative_parts.append(f"Recent Tests: {recent_other_str}")

    full_narrative = "\n\n".join(narrative_parts) if narrative_parts else ""

    return case_id, tab_dict, full_narrative, mri_vec


# ==============================================================================
# SUBTASK 1.1: LATE FUSION CALIBRATED ENSEMBLE + NEIGHBOR REASONING EXTRACTOR
# ==============================================================================

ORDINAL_TARGET_MAP_11 = {
    ("yes", "clear"): 1.00,
    ("yes", "borderline"): 0.75,
    ("yes", "uncertain"): 0.55,
    ("no", "uncertain"): 0.45,
    ("no", "borderline"): 0.25,
    ("no", "clear"): 0.00,
}

NUMERIC_COLS_11 = [
    "age", "psa", "vol", "psad", "pirads", "psav", "psap", "months",
    "ipss_score", "vitals_weight_kg", "vitals_height_cm", "vitals_bmi",
    "vitals_bp_sys", "vitals_bp_dia", "vitals_hr_bpm"
]
CATEGORICAL_COLS_11 = ["dre", "enc_type", "vitals_smoking"]


class TabularPreprocessorSubtask1:
    """Preprocesses tabular clinical features matching exp_4 C1* specifications."""

    def __init__(self):
        self.impute_values: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_columns_: List[str] = []
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _extract_raw_features(self, df: pd.DataFrame, is_fit: bool) -> Tuple[np.ndarray, np.ndarray]:
        num_features = []
        for col in NUMERIC_COLS_11:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                fill_val = self.impute_values.get(col, 0.0)
                num_features.append(series.fillna(fill_val).values)
            else:
                num_features.append(np.zeros(len(df)))

        X_num = np.column_stack(num_features)

        cat_encoded_dfs = []
        for col in CATEGORICAL_COLS_11:
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

        return X_num, X_cat

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessorSubtask1":
        for col in NUMERIC_COLS_11:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                self.impute_values[col] = float(series.median()) if not series.empty else 0.0
            else:
                self.impute_values[col] = 0.0

        for col in CATEGORICAL_COLS_11:
            if col in df.columns:
                series = df[col].dropna()
                self.cat_modes[col] = str(series.mode()[0]) if not series.empty else "unknown"
            else:
                self.cat_modes[col] = "unknown"

        X_num, X_cat = self._extract_raw_features(df, is_fit=True)
        X_combined = np.hstack([X_num, X_cat])
        self.scaler.fit(X_combined)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_num, X_cat = self._extract_raw_features(df, is_fit=False)
        X_combined = np.hstack([X_num, X_cat])
        return self.scaler.transform(X_combined)


class TextPreprocessorSubtask1:
    """Preprocesses clinical text notes into TF-IDF representations."""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+(?:-\w+)*\b",
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "TextPreprocessorSubtask1":
        clean_texts = [str(t) if pd.notna(t) and str(t).strip() else "empty clinical narrative" for t in texts]
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        clean_texts = [str(t) if pd.notna(t) and str(t).strip() else "empty clinical narrative" for t in texts]
        return self.vectorizer.transform(clean_texts).toarray()


class CustomKNNPredictor11:
    """k-NN Predictor supporting exact distance/uniform weights and neighbor extraction."""

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "distance", use_fuzzy: bool = True):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.use_fuzzy = use_fuzzy
        self.nn = NearestNeighbors(n_neighbors=k, metric=metric)
        self.y_train: Optional[np.ndarray] = None
        self.pids_train: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, pids: List[str]) -> "CustomKNNPredictor11":
        self.nn.fit(X)
        self.y_train = np.array(y, dtype=float)
        self.pids_train = list(pids)
        return self

    def kneighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        dists, indices = self.nn.kneighbors(X, n_neighbors=min(self.k, len(self.pids_train)))
        ret_pids = [[self.pids_train[idx] for idx in row] for row in indices]
        return dists, indices, ret_pids

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists, indices, _ = self.kneighbors(X)
        probs = []
        for d_row, idx_row in zip(dists, indices):
            y_neighbors = self.y_train[idx_row]
            if self.weights == "distance":
                weights = 1.0 / (d_row + 1e-6)
                prob = float(np.sum(weights * y_neighbors) / np.sum(weights))
            else:
                prob = float(np.mean(y_neighbors))
            probs.append(np.clip(prob, 0.0, 1.0))
        return np.array(probs)


class Subtask1LateFusionEnsemble:
    """
    Self-contained canonical production pipeline for Subtask 1.1 Decision Gate.
    Calculates calibrated late fusion probability and returns neighbor reasoning texts.
    """

    def __init__(
        self,
        weight_tab: float = 0.55,
        weight_mri: float = 0.20,
        weight_txt: float = 0.25,
        decision_threshold: float = 0.60,
    ):
        self.w_tab = weight_tab
        self.w_mri = weight_mri
        self.w_txt = weight_txt
        self.threshold = decision_threshold

        self.tab_preprocessor = TabularPreprocessorSubtask1()
        self.mri_pca = PCA(n_components=16, random_state=42)
        self.txt_preprocessor = TextPreprocessorSubtask1(max_features=1000)

        self.model_tab = CustomKNNPredictor11(k=5, metric="euclidean", weights="distance", use_fuzzy=True)
        self.model_mri = CustomKNNPredictor11(k=1, metric="euclidean", weights="uniform", use_fuzzy=False)
        self.model_txt = CustomKNNPredictor11(k=3, metric="cosine", weights="uniform", use_fuzzy=True)

        self.train_memory_free_text: Dict[str, str] = {}
        self.train_memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.mri_mean_emb: Optional[np.ndarray] = None

    def fit(
        self,
        df_tab: pd.DataFrame,
        df_txt: pd.DataFrame,
        df_mri: pd.DataFrame,
        df_gt: pd.DataFrame
    ) -> "Subtask1LateFusionEnsemble":
        labeled_gt = df_gt[df_gt["is_labeled"] == True].copy()
        labeled_pids = labeled_gt["pid"].tolist()

        for _, row in labeled_gt.iterrows():
            pid = row["pid"]
            self.train_memory_free_text[pid] = str(row.get("free_text", "")).strip()
            self.train_memory_metadata[pid] = {
                "decision": row.get("decision", ""),
                "confidence": row.get("confidence", ""),
                "case_id": row.get("case_id", pid),
            }

        tab_sub = df_tab[df_tab["pid"].isin(labeled_pids)].set_index("pid").reindex(labeled_pids).reset_index()
        self.tab_preprocessor.fit(tab_sub)
        X_tab_tr = self.tab_preprocessor.transform(tab_sub)

        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        mri_sub = df_mri[df_mri["pid"].isin(labeled_pids)].set_index("pid").reindex(labeled_pids).reset_index()
        mri_raw_arr = np.nan_to_num(mri_sub[emb_cols].values, nan=0.0)

        non_zero = mri_raw_arr[~np.all(mri_raw_arr == 0.0, axis=1)]
        self.mri_mean_emb = np.mean(non_zero, axis=0) if len(non_zero) > 0 else np.zeros(1024, dtype=float)

        for i in range(len(mri_raw_arr)):
            if np.all(mri_raw_arr[i] == 0.0):
                mri_raw_arr[i] = self.mri_mean_emb

        X_mri_tr = self.mri_pca.fit_transform(mri_raw_arr)

        txt_sub = df_txt[df_txt["pid"].isin(labeled_pids)].set_index("pid").reindex(labeled_pids).reset_index()
        txt_list = txt_sub["full_encounter_narrative"].fillna("").tolist()
        self.txt_preprocessor.fit(txt_list)
        X_txt_tr = self.txt_preprocessor.transform(txt_list)

        y_targets = []
        for pid in labeled_pids:
            meta = self.train_memory_metadata[pid]
            d = meta["decision"]
            c = meta["confidence"]
            target = ORDINAL_TARGET_MAP_11.get((d, c), 1.0 if d == "yes" else 0.0)
            y_targets.append(target)
        y_targets = np.array(y_targets, dtype=float)

        self.model_tab.fit(X_tab_tr, y_targets, pids=labeled_pids)
        self.model_mri.fit(X_mri_tr, y_targets, pids=labeled_pids)
        self.model_txt.fit(X_txt_tr, y_targets, pids=labeled_pids)

        return self

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        case_id, tab_dict, narrative_str, mri_emb = extract_raw_patient_payload(raw_input)
        df_single_tab = pd.DataFrame([tab_dict])

        if mri_emb is None or len(mri_emb) == 0 or np.all(mri_emb == 0.0) or np.all(np.isnan(mri_emb)):
            mri_emb = self.mri_mean_emb.copy() if self.mri_mean_emb is not None else np.zeros(1024, dtype=float)

        X_tab = self.tab_preprocessor.transform(df_single_tab)
        X_mri = self.mri_pca.transform(np.nan_to_num(mri_emb.reshape(1, -1), nan=0.0))
        X_txt = self.txt_preprocessor.transform([narrative_str])

        p_tab = float(self.model_tab.predict_proba(X_tab)[0])
        p_mri = float(self.model_mri.predict_proba(X_mri)[0])
        p_txt = float(self.model_txt.predict_proba(X_txt)[0])

        p_final = float(self.w_tab * p_tab + self.w_mri * p_mri + self.w_txt * p_txt)
        decision = "yes" if p_final >= self.threshold else "no"

        _, _, tab_pids = self.model_tab.kneighbors(X_tab)
        _, _, mri_pids = self.model_mri.kneighbors(X_mri)
        _, _, txt_pids = self.model_txt.kneighbors(X_txt)

        union_pids = []
        for p in tab_pids[0] + mri_pids[0] + txt_pids[0]:
            if p not in union_pids:
                union_pids.append(p)

        # Provenance map: which modalities retrieved each neighbor.
        source_map: Dict[str, List[str]] = {}
        for source, pid_list in (("tabular", tab_pids[0]), ("mri", mri_pids[0]), ("text", txt_pids[0])):
            for p in pid_list:
                source_map.setdefault(p, [])
                if source not in source_map[p]:
                    source_map[p].append(source)

        extracted_free_texts = [
            self.train_memory_free_text.get(p, "")
            for p in union_pids
            if self.train_memory_free_text.get(p, "").strip()
        ]

        return {
            "case_id": case_id,
            "decision": decision,
            "biopsy_probability": round(p_final, 4),
            "modality_probabilities": {
                "tabular": round(p_tab, 4),
                "mri": round(p_mri, 4),
                "text": round(p_txt, 4),
            },
            "neighbors_union": {
                "total_unique_neighbors": len(union_pids),
                "neighbor_pids": union_pids,
                "reasoning_free_texts": extracted_free_texts,
                "neighbors_detail": [
                    {
                        "neighbor_pid": p,
                        "source_modalities": source_map.get(p, []),
                        "free_text": self.train_memory_free_text.get(p, ""),
                    }
                    for p in union_pids
                ],
            },
        }


# ==============================================================================
# SUBTASK 1.2: SCALAR DIRECT-KDM 20D PRE-BIOPSY CLASSIFIER
# ==============================================================================

CONFIDENCE_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}
INV_CONFIDENCE_MAP = {0: "uncertain", 1: "borderline", 2: "clear"}

COMORBIDITY_GROUPS = {
    "cvd": ["hypertension", "htn", "cad", "mi", "stroke", "afib", "arrhythmia", "chf", "hypercholesterolemia", "dyslipidemia", "aortic", "pacemaker", "stent", "cabg"],
    "metabolic": ["diabetes", "dm2", "t2d", "dm1", "obesity", "hypothyroidism", "thyroid", "gout", "fatty liver", "metabolic syndrome"],
    "respiratory": ["asthma", "copd", "emphysema", "apnea", "osa", "bronchitis"],
    "gi_renal": ["gerd", "reflux", "ckd", "renal", "kidney", "diverticulitis", "crohn", "colitis", "ulcer", "gastritis"],
    "msk_neuro": ["arthrosis", "arthritis", "osteoarthritis", "back pain", "parkinson", "dementia", "neuropathy", "sciatica", "disc", "hernia"],
    "other_cancer": ["melanoma", "basal cell", "bcc", "scc", "bladder", "colon", "lymphoma", "leukemia", "renal cell", "kidney cancer"],
}


def extract_comorbidity_flags_from_text(text: str) -> Dict[str, float]:
    flags = {f"comorb_{group}": 0.0 for group in COMORBIDITY_GROUPS}
    flags["comorb_other_unmatched"] = 0.0
    if not text or pd.isna(text):
        return flags
    lower = str(text).lower()
    for group, kw_list in COMORBIDITY_GROUPS.items():
        if any(kw in lower for kw in kw_list):
            flags[f"comorb_{group}"] = 1.0
    return flags


class DirectKDMModule(nn.Module):
    """Scalar Direct-KDM PyTorch Module optimizing sigma."""

    def __init__(self, n_classes: int = 3, n_epochs: int = 300, lr: float = 0.01):
        super().__init__()
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.lr = lr
        self.register_buffer("c_x", torch.empty(0, 0))
        self.register_buffer("c_y", torch.empty(0, 0))
        self.register_buffer("sigma", torch.tensor(1.0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectKDMModule":
        N, D = X.shape
        C = self.n_classes
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.long)
        Y_onehot = torch.zeros(N, C, dtype=torch.float32)
        Y_onehot.scatter_(1, yt.unsqueeze(1), 1.0)

        self.c_x = Xt
        self.c_y = Y_onehot

        log_sigma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optimizer = torch.optim.Adam([log_sigma], lr=self.lr)

        for _ in range(self.n_epochs):
            sigma_val = torch.exp(log_sigma)
            diff = Xt.unsqueeze(1) - self.c_x.unsqueeze(0)
            sq_dists = (diff ** 2).sum(dim=2)
            weights = F.softmax(-sq_dists / (2.0 * (sigma_val ** 2) + 1e-7), dim=1)
            probs = torch.matmul(weights, self.c_y).clamp_min(1e-7)
            loss = F.nll_loss(torch.log(probs), yt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.sigma = torch.exp(log_sigma).detach()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            diff = Xt.unsqueeze(1) - self.c_x.unsqueeze(0)
            sq_dists = (diff ** 2).sum(dim=2)
            weights = F.softmax(-sq_dists / (2.0 * (self.sigma ** 2) + 1e-7), dim=1)
            probs = torch.matmul(weights, self.c_y)
        return probs.numpy()


class Subtask2DirectKDMClassifier:
    """Self-contained deployable classifier for Subtask 1.2 Clinical Confidence."""

    def __init__(self, n_mri_pca: int = 2, n_epochs: int = 300, lr: float = 0.01):
        self.n_mri_pca = n_mri_pca
        self.n_epochs = n_epochs
        self.lr = lr

        self.imputer = SimpleImputer(strategy="median")
        self.pca = PCA(n_components=n_mri_pca, random_state=42)
        self.scaler = StandardScaler()
        self.kdm = DirectKDMModule(n_classes=3, n_epochs=n_epochs, lr=lr)

        self.mri_mean_vec: Optional[np.ndarray] = None
        self.tabular_cols = [
            "psa", "psap", "psav", "psad", "vol", "age", "cspca", "vitals_bmi",
            "pirads", "dre_not_done", "dre_ordinal"
        ] + [f"comorb_{g}" for g in COMORBIDITY_GROUPS] + ["comorb_other_unmatched"]

    def _extract_tabular_dict(self, tab_data: Dict[str, Any], medhx_str: Optional[str] = None) -> Dict[str, float]:
        out = {}
        for col in ["psa", "psap", "psav", "psad", "vol", "age", "cspca", "vitals_bmi"]:
            out[col] = parse_numeric(tab_data.get(col))

        pirads_raw = tab_data.get("pirads", np.nan)
        out["pirads"] = parse_numeric(pirads_raw)

        dre_raw = str(tab_data.get("dre", "")).strip().lower()
        if dre_raw in ["not done", "none", "nan", ""]:
            out["dre_not_done"] = 1.0
            out["dre_ordinal"] = np.nan
        else:
            out["dre_not_done"] = 0.0
            dre_map = {"normal": 0.0, "benign": 0.0, "abnormal": 1.0, "nodus": 2.0, "nodule": 2.0, "suspicious": 3.0, "induration": 3.0}
            out["dre_ordinal"] = dre_map.get(dre_raw, 1.0)

        if medhx_str is None:
            medhx_str = str(tab_data.get("notes", ""))
        comorb_flags = extract_comorbidity_flags_from_text(medhx_str)
        for k, v in comorb_flags.items():
            out[k] = float(v)

        return out

    def fit(
        self,
        df_tab: pd.DataFrame,
        df_txt: pd.DataFrame,
        df_mri: pd.DataFrame,
        df_gt: pd.DataFrame
    ) -> "Subtask2DirectKDMClassifier":
        labeled_gt = df_gt[df_gt["is_labeled"] == True].copy()
        labeled_pids = labeled_gt["pid"].tolist()

        tab_indexed = df_tab.set_index("pid").loc[labeled_pids]
        txt_indexed = df_txt.set_index("pid").loc[labeled_pids]

        tab_records = []
        for pid in labeled_pids:
            row_tab = tab_indexed.loc[pid].to_dict()
            row_txt = txt_indexed.loc[pid].to_dict() if pid in txt_indexed.index else {}
            medhx_str = row_txt.get("full_encounter_narrative", "")
            tab_records.append(self._extract_tabular_dict(row_tab, medhx_str))

        df_raw_tab = pd.DataFrame(tab_records, index=labeled_pids)
        X_tab_imputed = self.imputer.fit_transform(df_raw_tab[self.tabular_cols].values)

        emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
        mri_df = df_mri.set_index("pid").loc[labeled_pids][emb_cols]
        mri_arr = np.nan_to_num(mri_df.values, nan=0.0)

        non_zero_rows = mri_arr[~np.all(mri_arr == 0.0, axis=1)]
        self.mri_mean_vec = np.mean(non_zero_rows, axis=0) if len(non_zero_rows) > 0 else np.zeros(1024, dtype=float)

        for i in range(len(mri_arr)):
            if np.all(mri_arr[i] == 0.0):
                mri_arr[i] = self.mri_mean_vec

        X_mri_pca = self.pca.fit_transform(mri_arr)
        mri_missing = np.all(mri_df.isna().values, axis=1).astype(float)[:, None]

        X_all = np.hstack([X_tab_imputed, X_mri_pca, mri_missing])
        X_scaled = self.scaler.fit_transform(X_all)

        y_labels = labeled_gt["confidence"].map(CONFIDENCE_MAP).values
        self.kdm.fit(X_scaled, y_labels)

        return self

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        case_id, tab_dict, narrative_str, mri_emb = extract_raw_patient_payload(raw_input)

        raw_tab_dict = self._extract_tabular_dict(tab_dict, narrative_str)
        tab_vec = np.array([[raw_tab_dict[c] for c in self.tabular_cols]])
        X_tab_imp = self.imputer.transform(tab_vec)

        if mri_emb is None or len(mri_emb) == 0 or np.all(np.isnan(mri_emb)) or np.all(mri_emb == 0.0):
            mri_is_missing = 1.0
            mri_vec = self.mri_mean_vec.copy().reshape(1, -1)
        else:
            mri_is_missing = 0.0
            mri_vec = np.nan_to_num(mri_emb.reshape(1, -1), nan=0.0)

        X_mri_pca = self.pca.transform(mri_vec)
        mri_missing_arr = np.array([[mri_is_missing]])

        X_full = np.hstack([X_tab_imp, X_mri_pca, mri_missing_arr])
        X_scaled = self.scaler.transform(X_full)

        probs = self.kdm.predict_proba(X_scaled)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = INV_CONFIDENCE_MAP[pred_idx]
        expected_score = float(probs[1] * 0.5 + probs[2] * 1.0)

        return {
            "case_id": case_id,
            "confidence": pred_label,
            "confidence_score": round(expected_score, 4),
            "class_probabilities": {
                "uncertain": round(float(probs[0]), 4),
                "borderline": round(float(probs[1]), 4),
                "clear": round(float(probs[2]), 4),
            },
        }


# ==============================================================================
# SUBTASK 1.3: JOINT EXTRATREES 23D VARIABLE RELEVANCE META-PREDICTOR
# ==============================================================================

CLINICAL_VARIABLES = [
    "age", "psa", "vol", "psad", "dre", "pirads", "fh", "comorbidity", "bx", "cspca"
]
ORDINAL_WEIGHT_MAP = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
ORDINAL_WEIGHT_INV = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}


class TabularPreprocessorSubtask3:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.raw_scaler = MinMaxScaler()
        self.is_fitted = False
        self.medians: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessorSubtask3":
        num_cols = ["age", "psa", "vol", "psad", "pirads", "cspca", "psav", "psap", "months", "vitals_bmi", "vitals_bp_sys", "ipss_score"]
        for col in num_cols:
            if col in df.columns:
                vals = [float(v) for v in df[col] if pd.notna(v) and str(v).strip() not in ("", "None", "nan")]
                self.medians[col] = float(np.median(vals)) if len(vals) > 0 else 0.0

        X_mat = self._extract_knn_matrix(df)
        self.scaler.fit(X_mat)

        X_raw_mat = self._extract_raw_10d(df)
        self.raw_scaler.fit(X_raw_mat)

        self.is_fitted = True
        return self

    def transform_knn(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._extract_knn_matrix(df))

    def transform_raw_10d(self, df: pd.DataFrame) -> np.ndarray:
        return self.raw_scaler.transform(self._extract_raw_10d(df))

    def _extract_knn_matrix(self, df: pd.DataFrame) -> np.ndarray:
        feats = []
        for _, row in df.iterrows():
            age_v = parse_numeric(row.get("age"))
            psa_v = parse_numeric(row.get("psa"))
            vol_v = parse_numeric(row.get("vol"))
            psad_v = parse_numeric(row.get("psad"))
            pirads_v = parse_numeric(row.get("pirads"))
            cspca_v = parse_numeric(row.get("cspca"))
            psav_v = parse_numeric(row.get("psav"))
            psap_v = parse_numeric(row.get("psap"))
            months_v = parse_numeric(row.get("months"))
            ipss_v = parse_numeric(row.get("ipss_score"))

            row_feats = [
                age_v if pd.notna(age_v) else self.medians.get("age", 65.0),
                psa_v if pd.notna(psa_v) else self.medians.get("psa", 6.0),
                vol_v if pd.notna(vol_v) else self.medians.get("vol", 40.0),
                psad_v if pd.notna(psad_v) else self.medians.get("psad", 0.15),
                pirads_v if pd.notna(pirads_v) else self.medians.get("pirads", 3.0),
                cspca_v if pd.notna(cspca_v) else self.medians.get("cspca", 0.3),
                psav_v if pd.notna(psav_v) else self.medians.get("psav", 0.0),
                psap_v if pd.notna(psap_v) else self.medians.get("psap", 0.0),
                months_v if pd.notna(months_v) else self.medians.get("months", 12.0),
                ipss_v if pd.notna(ipss_v) else self.medians.get("ipss_score", 10.0),
                1.0 if str(row.get("dre", "")).lower() in ["nodus", "abnormal", "suspicious", "induration"] else 0.0,
                1.0 if "prior" in str(row.get("enc_type", "")).lower() or "prior" in str(row.get("bx", "")).lower() else 0.0,
            ]
            feats.append(row_feats)
        return np.array(feats, dtype=np.float32)

    def _extract_raw_10d(self, df: pd.DataFrame) -> np.ndarray:
        raw_feats = []
        for _, row in df.iterrows():
            age_v = parse_numeric(row.get("age"))
            psa_v = parse_numeric(row.get("psa"))
            vol_v = parse_numeric(row.get("vol"))
            psad_v = parse_numeric(row.get("psad"))
            pirads_v = parse_numeric(row.get("pirads"))
            cspca_v = parse_numeric(row.get("cspca"))
            ipss_v = parse_numeric(row.get("ipss_score"))

            age = age_v if pd.notna(age_v) else self.medians.get("age", 65.0)
            psa = psa_v if pd.notna(psa_v) else self.medians.get("psa", 6.0)
            vol = vol_v if pd.notna(vol_v) else self.medians.get("vol", 40.0)
            psad = psad_v if pd.notna(psad_v) else (psa / max(vol, 1.0))

            dre_str = str(row.get("dre", "")).strip().lower()
            dre_val = 2.0 if dre_str in ["nodus", "abnormal", "suspicious", "induration"] else (1.0 if dre_str in ["normal", "benign"] else 0.0)

            pirads = pirads_v if pd.notna(pirads_v) else self.medians.get("pirads", 3.0)

            fh_str = (str(row.get("free_text", "")) + " " + str(row.get("notes", "")) + " " + str(row.get("family_hx", ""))).lower()
            fh_val = 1.0 if ("family history" in fh_str or "father" in fh_str or "brother" in fh_str or "positive" in fh_str) else 0.0

            comorb = ipss_v if pd.notna(ipss_v) else float(row.get("comorbidity_count", 0.0) or 0.0)

            enc_type_str = (str(row.get("enc_type", "")) + " " + str(row.get("bx", ""))).lower()
            bx_prior = 2.0 if "prior pca" in enc_type_str or "positive" in enc_type_str else (1.0 if "prior" in enc_type_str or "negative" in enc_type_str else 0.0)

            cspca = cspca_v if pd.notna(cspca_v) else self.medians.get("cspca", 0.3)

            raw_feats.append([age, psa, vol, psad, dre_val, pirads, fh_val, comorb, bx_prior, cspca])
        return np.array(raw_feats, dtype=np.float32)


class TabularFuzzyKNNSubtask3:
    def __init__(self, k: int = 5, eps: float = 1e-5):
        self.k = k
        self.eps = eps
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularFuzzyKNNSubtask3":
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
            row_idx = topk_idx[i]
            row_dists = dists[i, row_idx]
            weights = 1.0 / (row_dists + self.eps)
            weights /= weights.sum()
            probs[i] = np.dot(weights, self.y_train[row_idx])
        return probs


class Subtask3JointExtraTrees:
    """Self-contained deployable meta-predictor for Subtask 1.3 Variable Weights & Decisive Factors."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.preprocessor = TabularPreprocessorSubtask3()
        self.base_knn = TabularFuzzyKNNSubtask3(k=5)
        self.shap_background: Optional[np.ndarray] = None
        self.classifiers: Dict[str, ExtraTreesClassifier] = {}
        self.class_maps: Dict[str, List[int]] = {}

    def _compute_shap_13d(self, X_eval: np.ndarray) -> np.ndarray:
        K = len(self.shap_background)
        N, D = X_eval.shape
        diffs = X_eval[:, np.newaxis, :] - self.shap_background[np.newaxis, :, :]
        scales = np.std(self.shap_background, axis=0) + 1e-4
        z_scores = diffs / scales[np.newaxis, np.newaxis, :]
        phi_10d = np.abs(z_scores[:, :, :10]).mean(axis=1)

        row_sums = phi_10d.sum(axis=1, keepdims=True) + 1e-6
        phi_10d = phi_10d / row_sums

        psa_psad = phi_10d[:, 1] + phi_10d[:, 3]
        pirads_cspca = phi_10d[:, 5] + phi_10d[:, 9]
        vol_psad = phi_10d[:, 2] + phi_10d[:, 3]

        return np.hstack([phi_10d, psa_psad[:, None], pirads_cspca[:, None], vol_psad[:, None]])

    def fit(self, df_tab: pd.DataFrame, df_gt: pd.DataFrame) -> "Subtask3JointExtraTrees":
        labeled_gt = df_gt[df_gt["is_labeled"] == True].copy()
        labeled_pids = labeled_gt["pid"].tolist()

        tab_sub = df_tab[df_tab["pid"].isin(labeled_pids)].set_index("pid").reindex(labeled_pids).reset_index()
        self.preprocessor.fit(tab_sub)

        X_knn = self.preprocessor.transform_knn(tab_sub)
        y_binary = (labeled_gt["decision"].str.lower() == "yes").astype(float).values
        self.base_knn.fit(X_knn, y_binary)

        self.shap_background = X_knn[: min(15, len(X_knn))]

        X_shap_13d = self._compute_shap_13d(X_knn)
        X_raw_10d = self.preprocessor.transform_raw_10d(tab_sub)
        X_joint_23d = np.hstack([X_raw_10d, X_shap_13d])

        for var in CLINICAL_VARIABLES:
            col_name = f"weight_{var}"
            y_var = labeled_gt[col_name].map(ORDINAL_WEIGHT_MAP).fillna(0).astype(int).values
            unique_classes = sorted(np.unique(y_var))
            self.class_maps[var] = unique_classes

            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                max_features="sqrt",
                random_state=self.random_state,
            )
            clf.fit(X_joint_23d, y_var)
            self.classifiers[var] = clf

        return self

    def predict_raw(self, raw_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        case_id, tab_dict, _, _ = extract_raw_patient_payload(raw_input)
        df_single_tab = pd.DataFrame([tab_dict])

        X_knn = self.preprocessor.transform_knn(df_single_tab)
        X_shap_13d = self._compute_shap_13d(X_knn)
        X_raw_10d = self.preprocessor.transform_raw_10d(df_single_tab)
        X_joint_23d = np.hstack([X_raw_10d, X_shap_13d])

        variable_weights = {}
        decisive_factors = []
        important_factors = []

        for var in CLINICAL_VARIABLES:
            clf = self.classifiers[var]
            pred_class_idx = int(clf.predict(X_joint_23d)[0])
            pred_label = ORDINAL_WEIGHT_INV.get(pred_class_idx, "not_used")
            variable_weights[var] = pred_label

            if pred_label == "decisive":
                decisive_factors.append(var)
            elif pred_label == "important":
                important_factors.append(var)

        return {
            "case_id": case_id,
            "variable_weights": variable_weights,
            "decisive_factors": sorted(decisive_factors),
            "important_factors": sorted(important_factors),
        }
