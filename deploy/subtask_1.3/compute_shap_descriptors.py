#!/usr/bin/env python3
"""
deploy/subtask_1.3/compute_shap_descriptors.py

Extracts SHAP feature attributions for the 10 target clinical variables from
the Subtask 1.1 tabular decision model, and max-normalizes them against training population maxes.
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

sys.path.insert(0, str(SUBTASK_1_1_DIR))
from knn_model import ConfidenceWeightedKNN
from preprocess_input import preprocess_case

TARGET_VARS = ["age", "fh", "cspca", "pirads", "vol", "psa", "comorbidity", "psad", "dre", "bx"]
VAR_TO_TABULAR_NAME = {
    "age": "cli_age", "fh": "cli_fh_binary", "cspca": "cli_cspca",
    "pirads": "cli_pirads", "vol": "cli_vol", "psa": "cli_psa",
    "comorbidity": "cli_comorbidity_count", "psad": "cli_psad",
    "dre": "cli_dre", "bx": "cli_bx"
}
EPS = 1e-10


def compute_shap_attributions(raw_case, model_bundle_1_1, population_max_shap=None):
    """
    Computes max-normalized SHAP attributions psi_{i, k} for the 10 target variables.

    Parameters:
      raw_case (dict): Raw patient case containing tabular data.
      model_bundle_1_1 (dict): Loaded pipeline bundle from model_subtask_1.1.pkl.
      population_max_shap (dict, optional): Population max SHAP values per target var.

    Returns:
      tuple: (psi_shap_norm_dict, raw_shap_dict)
    """
    # 1. Preprocess tabular input
    X_tab_proc, _, _ = preprocess_case(raw_case, model_bundle_1_1)

    knn_tabular = model_bundle_1_1["knn_models"]["tabular"]
    frozen_vars = model_bundle_1_1["frozen_tabular_vars"]
    cat_cols = model_bundle_1_1["categorical_cols"]
    num_cols = model_bundle_1_1["num_cols"]
    ohe = model_bundle_1_1["transformers"]["ohe"]

    # Reconstruct feature column names of X_tab_proc
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    proc_col_names = num_cols + cat_feature_names + [f"{c}__is_missing" for c in frozen_vars]
    X_tab_df = pd.DataFrame(X_tab_proc, columns=proc_col_names)

    # 2. Sample background for Kernel Explainer
    background = shap.sample(knn_tabular.X_train, min(5, len(knn_tabular.X_train)), random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(knn_tabular.predict_proba, background, silent=True)
        shap_vals = np.abs(explainer.shap_values(X_tab_df.values, nsamples=20, l1_reg="num_features(10)"))

    raw_shap_dict = {}
    psi_norm_dict = {}

    for var in TARGET_VARS:
        raw_col = VAR_TO_TABULAR_NAME[var]
        matched_cols = [c for c in X_tab_df.columns if c == raw_col or c.startswith(f"{raw_col}_")]
        if matched_cols:
            col_idx = [X_tab_df.columns.get_loc(c) for c in matched_cols]
            val = float(np.mean(shap_vals[:, col_idx]))
        else:
            val = 0.0

        raw_shap_dict[var] = val

        if population_max_shap and var in population_max_shap:
            max_v = population_max_shap[var] + EPS
            psi_norm_dict[var] = float(np.clip(val / max_v, 0.0, 1.0))
        else:
            psi_norm_dict[var] = val

    return psi_norm_dict, raw_shap_dict
