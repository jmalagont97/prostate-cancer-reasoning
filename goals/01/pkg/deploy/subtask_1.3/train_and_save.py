#!/usr/bin/env python3
"""
deploy/subtask_1.3/train_and_save.py

Trains and optimizes global 3D thresholds (tau_1*, tau_2*, tau_3*) for each of the 10
target clinical relevance variables on ALL N=88 usable labeled cohort cases (exp_19).
Serializes fitted thresholds, population SHAP maxes, and reveal mapping rules into model_subtask_1.3.pkl.
"""

import sys
import pickle
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

sys.path.insert(0, str(SUBTASK_1_1_DIR))
sys.path.insert(0, str(DEPLOY_DIR))

from knn_model import ConfidenceWeightedKNN
from predict_inference import load_model_bundle as load_subtask_1_1_bundle
from compute_shap_descriptors import compute_shap_attributions, TARGET_VARS, VAR_TO_TABULAR_NAME

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"

ORDINAL_WEIGHT_MAP = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}
EPS = 1e-10


def compute_moe_abs_weights(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present_classes = np.unique(y_true)
    errors = []
    for c in present_classes:
        mask = (y_true == c)
        diff = np.abs(y_pred[mask] - c) / 3.0
        errors.append(np.mean(diff))
    return float(np.mean(errors)) if errors else 0.0


def fit_exhaustive_3thresholds(psi_train, y_train):
    psi_sorted = np.sort(np.unique(psi_train))
    if len(psi_sorted) < 3:
        return 0.25, 0.50, 0.75

    candidates = [0.0]
    for i in range(len(psi_sorted) - 1):
        candidates.append((psi_sorted[i] + psi_sorted[i+1]) / 2.0)
    candidates.append(float(np.max(psi_train) + 1e-3))
    candidates = np.array(candidates, dtype=np.float64)

    tuples_list = []
    n_cand = len(candidates)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            for k in range(j + 1, n_cand):
                tuples_list.append((candidates[i], candidates[j], candidates[k]))

    if not tuples_list:
        return 0.25, 0.50, 0.75

    tuples = np.array(tuples_list, dtype=np.float64)  # shape (K, 3)

    tau1 = tuples[:, 0:1]  # shape (K, 1)
    tau2 = tuples[:, 1:2]  # shape (K, 1)
    tau3 = tuples[:, 2:3]  # shape (K, 1)

    psi_mat = np.asarray(psi_train, dtype=np.float64).reshape(1, -1)  # shape (1, N)
    y_mat = np.asarray(y_train, dtype=int).reshape(1, -1)

    preds = np.zeros((len(tuples), psi_mat.shape[1]), dtype=int)
    preds[psi_mat >= tau1] = 1
    preds[psi_mat >= tau2] = 2
    preds[psi_mat >= tau3] = 3

    present_classes = np.unique(y_mat[0])
    class_errors = []
    for c in present_classes:
        mask = (y_mat == c)
        err_c = np.mean(np.abs(preds[:, mask[0]] - c) / 3.0, axis=1) if np.any(mask) else np.zeros(len(tuples))
        class_errors.append(err_c)

    moe_vec = np.mean(class_errors, axis=0)
    best_idx = np.argmin(moe_vec)
    best_tuple = tuples[best_idx]

    return float(best_tuple[0]), float(best_tuple[1]), float(best_tuple[2])


def main():
    print("=" * 80)
    print("Subtask 1.3: Training & Optimizing Global 3D Thresholds per Variable (exp_19)")
    print("=" * 80)

    # 1. Load data
    inputs_df = pd.read_csv(DATA / "inputs.csv")
    gt_df = pd.read_csv(DATA / "ground_truth.csv")
    splits_df = pd.read_csv(DATA / "mccv_loocv_splits.csv")

    usable_df = splits_df[splits_df["cohort_status"] == "usable_labeled"].sort_values("case_id").reset_index(drop=True)
    usable_ids = usable_df["case_id"].tolist()
    N = len(usable_ids)
    print(f"  Optimizing thresholds on ALL {N} usable labeled cohort cases...")

    inputs_df = inputs_df.set_index("case_id").loc[usable_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[usable_ids].reset_index()

    # Load 1.1 model bundle
    bundle_1_1 = load_subtask_1_1_bundle()

    # 2. Extract raw SHAP attributions for all N=88 cases
    print("  Extracting raw SHAP attributions across N=88 cohort cases...")
    raw_shap_matrix = {var: [] for var in TARGET_VARS}

    for i in range(N):
        tab_data = inputs_df.iloc[i].to_dict()
        case_input = {"tabular": tab_data}
        _, raw_shap_dict = compute_shap_attributions(case_input, bundle_1_1)
        for var in TARGET_VARS:
            raw_shap_matrix[var].append(raw_shap_dict[var])

    # 3. Compute population max SHAP per variable
    population_max_shap = {}
    for var in TARGET_VARS:
        population_max_shap[var] = float(np.max(raw_shap_matrix[var]) + EPS)

    # Max-normalize SHAP values
    psi_norm_matrix = {}
    for var in TARGET_VARS:
        psi_norm_matrix[var] = np.array(raw_shap_matrix[var]) / population_max_shap[var]

    # 4. Optimize 3D thresholds per target variable
    print("  Optimizing 3D thresholds (tau_1*, tau_2*, tau_3*) per variable...")
    thresholds_per_variable = {}
    moe_per_variable = {}

    for var in TARGET_VARS:
        y_train = gt_df[f"target_code_weight_{var}"].values.astype(int)
        psi_tr = psi_norm_matrix[var]

        t1, t2, t3 = fit_exhaustive_3thresholds(psi_tr, y_train)
        thresholds_per_variable[var] = (t1, t2, t3)

        preds = np.zeros(N, dtype=int)
        preds[psi_tr >= t1] = 1
        preds[psi_tr >= t2] = 2
        preds[psi_tr >= t3] = 3
        moe_val = compute_moe_abs_weights(y_train, preds)
        moe_per_variable[var] = moe_val
        print(f"    - {var:12s}: tau = ({t1:.4f}, {t2:.4f}, {t3:.4f}) -> MOE_abs = {moe_val:.4f}")

    mean_moe_abs = float(np.mean(list(moe_per_variable.values())))
    print(f"  Fitted Cohort Mean MOE_abs (Pesos) = {mean_moe_abs:.4f}")

    # 5. Pack and Serialize Pipeline Bundle
    pipeline_bundle = {
        "subtask": "1.3_relevance_and_reveal_sequence",
        "model_name": "exp_19_shap_max_normalized_3d_thresholds",
        "target_variables": TARGET_VARS,
        "var_to_tabular_name": VAR_TO_TABULAR_NAME,
        "population_max_shap": population_max_shap,
        "thresholds_per_variable": thresholds_per_variable,
        "ordinal_weight_map": ORDINAL_WEIGHT_MAP,
        "fitted_mean_moe_abs": mean_moe_abs,
    }

    pkl_path = DEPLOY_DIR / "model_subtask_1.3.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    print(f"\n✓ Successfully optimized and serialized Subtask 1.3 model to: {pkl_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
