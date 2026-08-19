#!/usr/bin/env python3
"""
deploy/subtask_1.1/preprocess_input.py

Data transformation module for Agent Tools in Subtask 1.1.
Transforms a single raw patient case (dict/JSON/DataFrame) into preprocessed
numerical feature vectors for Tabular (T), MRI (M), and Text (X) modalities.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATION_STOPWORDS = {"no", "not", "without", "never", "neither", "nor", "none"}

RAW_TO_MODEL_KEY_MAP = {
    "age": "cli_age",
    "psa": "cli_psa",
    "pirads": "cli_pirads",
    "psad": "cli_psad",
    "psav": "cli_psav",
    "vol": "cli_vol",
    "dre": "cli_dre",
    "bx": "cli_bx",
    "cspca": "cli_cspca",
    "months": "cli_months",
    "fh": "cli_fh_binary",
    "family_history": "cli_fh_binary",
    "comorbidity": "cli_comorbidity_count",
    "comorbidities": "cli_comorbidity_count",
    "allergies": "cli_allergies_count",
    "ipss": "cli_ipss_score",
}


def clean_text_narrative(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t in NEGATION_STOPWORDS]
    return " ".join(tokens)


def _safe_float(val):
    if val is None or val == "" or val == "NONE" or val == "Not reported":
        return np.nan
    if isinstance(val, (list, tuple, dict)):
        return float(len(val))
    try:
        return float(val)
    except (ValueError, TypeError):
        m = re.search(r"(\d+\.?\d*)", str(val))
        return float(m.group(1)) if m else np.nan


def standardize_tabular_dict(tab_dict):
    """
    Standardizes a raw tabular dictionary mapping raw JSON keys (e.g. age, psa, pirads)
    to model feature column names (e.g. cli_age, cli_psa, cli_pirads).
    Also parses vitals sub-dictionary and collection counts safely.
    """
    std = {}

    # Alias raw keys to cli_ prefixed keys
    for raw_k, model_k in RAW_TO_MODEL_KEY_MAP.items():
        if raw_k in tab_dict:
            val = tab_dict[raw_k]
            if raw_k in ["allergies", "pmhx", "comorbidity", "comorbidities"] and isinstance(val, (list, tuple)):
                std[model_k] = float(len(val))
            elif raw_k == "ipss" and isinstance(val, str):
                m = re.search(r"(\d+)", val)
                std[model_k] = float(m.group(1)) if m else np.nan
            else:
                std[model_k] = val

    # Copy all other keys
    for k, v in tab_dict.items():
        if k not in std:
            std[k] = v

    # Handle vitals sub-dictionary
    vitals = tab_dict.get("vitals", {})
    if isinstance(vitals, dict):
        if "height" in vitals and "vit_height_cm" not in std:
            std["vit_height_cm"] = _safe_float(vitals["height"])

        if "weight" in vitals and "vit_weight_kg" not in std:
            std["vit_weight_kg"] = _safe_float(vitals["weight"])

        if "hr" in vitals and "vit_heart_rate_bpm" not in std:
            std["vit_heart_rate_bpm"] = _safe_float(vitals["hr"])

        if "bp" in vitals and ("vit_bp_systolic" not in std or "vit_bp_diastolic" not in std):
            parts = str(vitals["bp"]).split("/")
            if len(parts) >= 2:
                m_sys = re.search(r"(\d+)", parts[0])
                m_dia = re.search(r"(\d+)", parts[1])
                if m_sys:
                    std["vit_bp_systolic"] = float(m_sys.group(1))
                if m_dia:
                    std["vit_bp_diastolic"] = float(m_dia.group(1))

        if "smoking" in vitals and "vit_smoking_status" not in std:
            std["vit_smoking_status"] = str(vitals["smoking"])
            m_pk = re.search(r"(\d+)\s*pack", str(vitals["smoking"]))
            if m_pk:
                std["vit_smoking_pack_years"] = float(m_pk.group(1))

    return std


def preprocess_case(raw_case, model_bundle):
    """
    Transforms a single patient case into preprocessed feature arrays.

    Parameters:
      raw_case (dict):
        {
          "tabular": dict of 21 tabular features (or subset),
          "text": str narrative text,
          "mri_emb": 1024-dim list or np.ndarray
        }
      model_bundle (dict): Loaded pipeline dictionary from model_subtask_1.1.pkl

    Returns:
      tuple: (X_tab_proc, X_mri_pca, X_text_vec)
    """
    frozen_vars = model_bundle["frozen_tabular_vars"]
    cat_cols = model_bundle["categorical_cols"]
    num_cols = model_bundle["num_cols"]
    transformers = model_bundle["transformers"]

    ohe = transformers["ohe"]
    scaler = transformers["scaler"]
    pca = transformers["pca"]
    tfidf = transformers["tfidf"]

    # 1. Process Tabular Modality
    tab_data = raw_case.get("tabular", {})
    if isinstance(tab_data, pd.Series):
        tab_dict = tab_data.to_dict()
    elif isinstance(tab_data, dict):
        tab_dict = tab_data
    else:
        tab_dict = {}

    tab_dict = standardize_tabular_dict(tab_dict)
    X_raw_df = pd.DataFrame([tab_dict])

    # Ensure all 21 frozen variables exist
    for col in frozen_vars:
        if col not in X_raw_df.columns:
            X_raw_df[col] = np.nan

    X_tab_proc = pd.DataFrame(index=[0])
    for col in frozen_vars:
        ind = f"{col}__is_missing"
        X_tab_proc[ind] = X_raw_df[col].isna().astype(int)
        if col in cat_cols:
            X_raw_df[col] = X_raw_df[col].fillna("0").astype(str)
        else:
            # Safely cast numerical columns
            vals = X_raw_df[col].apply(_safe_float)
            X_raw_df[col] = vals.fillna(0.0).astype(np.float64)

    X_cat_scaled = pd.DataFrame(
        ohe.transform(X_raw_df[cat_cols]),
        index=[0],
        columns=ohe.get_feature_names_out(cat_cols)
    )

    X_num_scaled = pd.DataFrame(
        scaler.transform(X_raw_df[num_cols].values.astype(np.float64)),
        index=[0],
        columns=num_cols
    )

    X_tab_final = pd.concat([X_num_scaled, X_cat_scaled], axis=1)
    for c in X_tab_proc.columns:
        X_tab_final[c] = X_tab_proc[c]

    # 2. Process MRI Embedding Modality
    mri_raw = raw_case.get("mri_emb", np.zeros(1024))
    if isinstance(mri_raw, list):
        mri_arr = np.array(mri_raw, dtype=np.float64).reshape(1, -1)
    elif isinstance(mri_raw, np.ndarray):
        mri_arr = mri_raw.reshape(1, -1)
    else:
        mri_arr = np.zeros((1, 1024), dtype=np.float64)

    if mri_arr.shape[1] != 1024:
        mri_arr = np.zeros((1, 1024), dtype=np.float64)

    X_mri_pca = pca.transform(mri_arr)

    # 3. Process Text Narrative Modality
    raw_text = raw_case.get("text", "")
    cleaned_text = clean_text_narrative(raw_text)
    X_text_vec = tfidf.transform([cleaned_text]).toarray()

    return X_tab_final.values, X_mri_pca, X_text_vec
