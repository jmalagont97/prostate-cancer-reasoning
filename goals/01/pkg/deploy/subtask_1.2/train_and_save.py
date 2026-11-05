#!/usr/bin/env python3
"""
deploy/subtask_1.2/train_and_save.py

Trains and optimizes global 2D thresholds (tau_1*, tau_2*) on Decision Risk Omega
for ALL N=88 usable labeled cohort cases (exp_18 winning configuration).
Serializes fitted thresholds into model_subtask_1.2.pkl.
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

sys.path.insert(0, str(SUBTASK_1_1_DIR))
sys.path.insert(0, str(DEPLOY_DIR))

from predict_inference import predict_subtask_1_1
from compute_risk_descriptor import compute_decision_risk

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"

CONFIDENCE_MAP_ORDINAL = {"uncertain": 0, "borderline": 1, "clear": 2}
ORDINAL_TO_LABEL = {0: "uncertain", 1: "borderline", 2: "clear"}


def compute_moe_abs_ordinal(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present_classes = np.unique(y_true)
    errors = []
    for c in present_classes:
        mask = (y_true == c)
        diff = np.abs(y_pred[mask] - c) / 2.0
        errors.append(np.mean(diff))
    return float(np.mean(errors)) if errors else 0.0


def fit_global_2d_thresholds(omega_scores, y_true_ordinal, min_preds_per_class=3):
    omega_sorted = np.sort(np.unique(omega_scores))
    if len(omega_sorted) < 2:
        return 0.20, 0.40

    candidates = [float(np.min(omega_scores) - 1e-3)]
    for i in range(len(omega_sorted) - 1):
        candidates.append((omega_sorted[i] + omega_sorted[i+1]) / 2.0)
    candidates.append(float(np.max(omega_scores) + 1e-3))
    candidates = np.array(candidates, dtype=np.float64)

    tuples_list = []
    n_cand = len(candidates)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            tuples_list.append((candidates[i], candidates[j]))

    if not tuples_list:
        return 0.20, 0.40

    tuples = np.array(tuples_list, dtype=np.float64)
    tau1 = tuples[:, 0:1] # shape (K, 1)
    tau2 = tuples[:, 1:2] # shape (K, 1)

    omega_mat = np.asarray(omega_scores, dtype=np.float64).reshape(1, -1) # shape (1, N)
    y_mat = np.asarray(y_true_ordinal, dtype=int).reshape(1, -1)

    preds = np.zeros((len(tuples), omega_mat.shape[1]), dtype=int)
    preds[omega_mat >= tau1] = 1
    preds[omega_mat >= tau2] = 2

    present_classes = np.unique(y_mat[0])
    class_errors = []

    for c in present_classes:
        mask = (y_mat == c)
        err_c = np.mean(np.abs(preds[:, mask[0]] - c) / 2.0, axis=1) if np.any(mask) else np.zeros(len(tuples))
        class_errors.append(err_c)

    moe_vec = np.mean(class_errors, axis=0)

    # Apply min recall constraint
    valid_mask = np.ones(len(tuples), dtype=bool)
    for c in present_classes:
        count_c = np.sum(preds == c, axis=1)
        valid_mask &= (count_c >= min_preds_per_class)

    moe_vec[~valid_mask] = 999.0

    best_idx = np.argmin(moe_vec)
    best_tuple = tuples[best_idx]
    return float(best_tuple[0]), float(best_tuple[1])


def main():
    print("=" * 80)
    print("Subtask 1.2: Training & Optimizing Global 2D Thresholds (exp_18)")
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
    print(f"  Optimizing thresholds on ALL {N} usable labeled cohort cases...")

    inputs_df = inputs_df.set_index("case_id").loc[usable_ids].reset_index()
    gt_df = gt_df.set_index("case_id").loc[usable_ids].reset_index()
    text_df = text_df.set_index("case_id").loc[usable_ids].reset_index()
    images_df = images_df.set_index("case_id").loc[usable_ids].reset_index()

    y_conf_labels = gt_df["target_confidence"].tolist()
    y_ordinal = np.array([CONFIDENCE_MAP_ORDINAL[c] for c in y_conf_labels], dtype=int)

    # 2. Compute continuous Omega risk for all N=88 cases
    print("  Extracting continuous Decision Risk Omega scores...")
    omega_scores = []
    for i in range(N):
        cid = usable_ids[i]
        tab_data = inputs_df.iloc[i].to_dict()
        text_data = text_df.iloc[i]["txt_full_prompt_narrative"]
        mri_cols = [c for c in images_df.columns if c.startswith("mri_emb_")]
        mri_data = images_df.iloc[i][mri_cols].values

        case_input = {"tabular": tab_data, "text": text_data, "mri_emb": mri_data}
        risk_res = compute_decision_risk(case_input, c_fn=0.65, lambda_param=0.00)
        omega_scores.append(risk_res["omega_risk_score"])

    omega_scores = np.array(omega_scores, dtype=np.float64)

    # 3. Fit Global 2D Thresholds
    tau1_star, tau2_star = fit_global_2d_thresholds(omega_scores, y_ordinal, min_preds_per_class=3)
    print(f"  Optimal Thresholds Found: tau_1* = {tau1_star:.4f}, tau_2* = {tau2_star:.4f}")

    # Evaluate fitted MOE_abs on cohort
    preds_ordinal = np.zeros(N, dtype=int)
    preds_ordinal[omega_scores >= tau1_star] = 1
    preds_ordinal[omega_scores >= tau2_star] = 2

    moe_abs = compute_moe_abs_ordinal(y_ordinal, preds_ordinal)
    print(f"  Fitted Cohort MOE_abs = {moe_abs:.4f}")

    # 4. Pack and Serialize Pipeline Bundle
    pipeline_bundle = {
        "subtask": "1.2_clinical_confidence",
        "model_name": "exp_18_decision_risk_global_2d_thresholds",
        "risk_parameters": {"c_fn": 0.65, "c_fp": 0.35, "lambda": 0.00},
        "thresholds": {"tau_1_star": tau1_star, "tau_2_star": tau2_star},
        "ordinal_mapping": CONFIDENCE_MAP_ORDINAL,
        "ordinal_labels": ORDINAL_TO_LABEL,
        "fitted_moe_abs": moe_abs,
    }

    pkl_path = DEPLOY_DIR / "model_subtask_1.2.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)

    print(f"\n✓ Successfully optimized and serialized Subtask 1.2 model to: {pkl_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
