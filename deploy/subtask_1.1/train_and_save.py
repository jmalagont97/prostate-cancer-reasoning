#!/usr/bin/env python3
"""
deploy/subtask_1.1/train_and_save.py

Trains the canonical Subtask 1.1 Multimodal Fusion model (exp_12) on ALL N=88 usable labeled cases.
Serializes the fitted pipeline, feature transformers, and KNN support sets into model_subtask_1.1.pkl.
"""

import os
import sys
import re
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR))

from knn_model import ConfidenceWeightedKNN

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"

CATEGORICAL_COLS = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]

FROZEN_21_VARS = [
    "cli_age", "cli_allergies_count", "cli_bx", "cli_comorbidity_count",
    "cli_cspca", "cli_dre", "cli_fh_binary", "cli_ipss_score", "cli_months",
    "cli_pirads", "cli_psa", "cli_psad", "cli_psav", "cli_vol",
    "vit_bp_diastolic", "vit_bp_systolic", "vit_heart_rate_bpm",
    "vit_height_cm", "vit_smoking_pack_years", "vit_smoking_status",
    "vit_weight_kg",
]

CONFIDENCE_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.25}
NEGATION_STOPWORDS = {"no", "not", "without", "never", "neither", "nor", "none"}
CUSTOM_STOPWORDS = list(set(ENGLISH_STOP_WORDS) - NEGATION_STOPWORDS)


def clean_text_narrative(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t in NEGATION_STOPWORDS]
    return " ".join(tokens)


def main():
    print("=" * 80)
    print("Subtask 1.1: Training & Exporting Model Pipeline (exp_12)")
    print("=" * 80)

    # 1. Load data
    inputs_df = pd.read_csv(DATA / "inputs.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")
    text_df = pd.read_csv(DATA / "full_prompt_narrative.csv")
    images_df = pd.read_csv(DATA / "images.csv")

    usable_df = splits_df[splits_df["cohort_status"] == "usable_labeled"].sort_values("case_id").reset_index(drop=True)
    usable_ids = usable_df["case_id"].tolist()
    N = len(usable_ids)
    print(f"  Training on ALL {N} usable labeled cohort cases...")

    inputs_df = inputs_df.set_index("case_id").loc[usable_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[usable_ids].reset_index()
    text_df = text_df.set_index("case_id").loc[usable_ids].reset_index()
    images_df = images_df.set_index("case_id").loc[usable_ids].reset_index()

    y_binary = gt_df["target_biopsy_decision_binary"].values.astype(int)
    y_conf_labels = gt_df["target_confidence"].tolist()
    conf_weights = np.array([CONFIDENCE_MAP[c] for c in y_conf_labels], dtype=np.float64)

    # 2. Train Tabular Modality (T)
    print("  Fitting Tabular Modality (T)...")
    X_tab_raw = inputs_df[FROZEN_21_VARS].copy()
    num_cols = [c for c in FROZEN_21_VARS if c not in CATEGORICAL_COLS]

    X_tab_proc = pd.DataFrame(index=X_tab_raw.index)
    for col in FROZEN_21_VARS:
        ind = f"{col}__is_missing"
        X_tab_proc[ind] = X_tab_raw[col].isna().astype(int)
        if col in CATEGORICAL_COLS:
            X_tab_raw[col] = X_tab_raw[col].fillna("0").astype(str)
        else:
            X_tab_raw[col] = X_tab_raw[col].fillna(0).astype(np.float64)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    ohe.fit(X_tab_raw[CATEGORICAL_COLS])
    X_cat_scaled = pd.DataFrame(ohe.transform(X_tab_raw[CATEGORICAL_COLS]), index=X_tab_raw.index, columns=ohe.get_feature_names_out(CATEGORICAL_COLS))

    scaler = MinMaxScaler()
    scaler.fit(X_tab_raw[num_cols].values.astype(np.float64))
    X_num_scaled = pd.DataFrame(scaler.transform(X_tab_raw[num_cols].values.astype(np.float64)), index=X_tab_raw.index, columns=num_cols)

    X_tab_final = pd.concat([X_num_scaled, X_cat_scaled], axis=1)
    for c in X_tab_proc.columns:
        X_tab_final[c] = X_tab_proc[c]

    knn_tabular = ConfidenceWeightedKNN(n_neighbors=1, metric="cosine", use_distance_weight=False)
    knn_tabular.fit(X_tab_final.values, y_binary, conf_weights)

    # 3. Train MRI Embedding Modality (M)
    print("  Fitting MRI Embedding Modality (M)...")
    emb_cols = [c for c in images_df.columns if c.startswith("mri_emb_")]
    X_mri_raw = images_df[emb_cols].values.astype(np.float64)

    pca = PCA(n_components=1, random_state=42)
    X_mri_pca = pca.fit_transform(X_mri_raw)

    knn_mri = ConfidenceWeightedKNN(n_neighbors=1, metric="euclidean", use_distance_weight=True)
    knn_mri.fit(X_mri_pca, y_binary, conf_weights)

    # 4. Train Text Modality (X)
    print("  Fitting Clinical Narrative Text Modality (X)...")
    raw_texts = text_df["txt_full_prompt_narrative"].tolist()
    cleaned_texts = [clean_text_narrative(t) for t in raw_texts]

    tfidf = TfidfVectorizer(max_features=2000, stop_words=CUSTOM_STOPWORDS, dtype=np.float64)
    X_text_vec = tfidf.fit_transform(cleaned_texts).toarray()

    knn_text = ConfidenceWeightedKNN(n_neighbors=3, metric="cosine", use_distance_weight=True)
    knn_text.fit(X_text_vec, y_binary, conf_weights)

    # 5. Pack and Serialize Pipeline Bundle
    pipeline_bundle = {
        "subtask": "1.1_biopsy_decision",
        "model_name": "exp_12_late_multimodal_fusion",
        "frozen_tabular_vars": FROZEN_21_VARS,
        "categorical_cols": CATEGORICAL_COLS,
        "num_cols": num_cols,
        "transformers": {
            "ohe": ohe,
            "scaler": scaler,
            "pca": pca,
            "tfidf": tfidf,
        },
        "knn_models": {
            "tabular": knn_tabular,
            "mri": knn_mri,
            "text": knn_text,
        },
        "fusion_weights": {"tabular": 1/3, "mri": 1/3, "text": 1/3},
        "decision_threshold": 0.5,
    }

    pkl_path = DEPLOY_DIR / "model_subtask_1.1.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    print(f"\n✓ Successfully trained and serialized pipeline model to: {pkl_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
