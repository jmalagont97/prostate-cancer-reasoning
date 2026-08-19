#!/usr/bin/env python3
"""
deploy/subtask_1.2/predict_inference.py

Inference Engine & Agent Tool Interface for Subtask 1.2 (Clinical Confidence).
Loads model_subtask_1.2.pkl, computes continuous Decision Risk Omega,
and applies optimal 2D thresholds (tau_1*, tau_2*) to output confidence grades.
"""

import sys
import pickle
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

sys.path.insert(0, str(SUBTASK_1_1_DIR))
sys.path.insert(0, str(DEPLOY_DIR))

from compute_risk_descriptor import compute_decision_risk

DEFAULT_PKL_PATH = DEPLOY_DIR / "model_subtask_1.2.pkl"
_MODEL_BUNDLE_CACHE_1_2 = None


def load_model_bundle_1_2(pkl_path=None):
    global _MODEL_BUNDLE_CACHE_1_2
    if _MODEL_BUNDLE_CACHE_1_2 is not None and (pkl_path is None or pkl_path == DEFAULT_PKL_PATH):
        return _MODEL_BUNDLE_CACHE_1_2

    path = Path(pkl_path) if pkl_path else DEFAULT_PKL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}. Run train_and_save.py first.")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    _MODEL_BUNDLE_CACHE_1_2 = bundle
    return bundle


def predict_subtask_1_2(raw_case, pkl_path=None):
    """
    Inference API for Subtask 1.2 Agent Tool.

    Parameters:
      raw_case (dict): Patient data containing tabular, text, and mri_emb keys.
      pkl_path (str/Path, optional): Path to model_subtask_1.2.pkl

    Returns:
      dict: Structured prediction dictionary for LLM Agent Tool
    """
    model_bundle = load_model_bundle_1_2(pkl_path)

    risk_params = model_bundle["risk_parameters"]
    c_fn = risk_params["c_fn"]
    lambda_param = risk_params["lambda"]

    risk_info = compute_decision_risk(raw_case, c_fn=c_fn, lambda_param=lambda_param)
    omega = risk_info["omega_risk_score"]

    thresholds = model_bundle["thresholds"]
    tau1 = thresholds["tau_1_star"]
    tau2 = thresholds["tau_2_star"]

    if omega >= tau2:
        conf_ordinal = 2
        conf_label = "clear"
    elif omega >= tau1:
        conf_ordinal = 1
        conf_label = "borderline"
    else:
        conf_ordinal = 0
        conf_label = "uncertain"

    return {
        "subtask": "1.2_clinical_confidence",
        "confidence_ordinal": conf_ordinal,
        "confidence_label": conf_label,
        "decision_risk_score": float(np.round(omega, 4)),
        "thresholds": {
            "tau_1_uncertain_borderline": float(np.round(tau1, 4)),
            "tau_2_borderline_clear": float(np.round(tau2, 4))
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Subtask 1.2 Clinical Confidence Inference CLI")
    parser.add_argument("--case_id", type=str, help="Evaluate a specific patient case_id from dataset")
    parser.add_argument("--sample", action="store_true", help="Run a test sample prediction")
    args = parser.parse_args()

    ROOT = DEPLOY_DIR.parent.parent
    DATA = ROOT / "data" / "chimera26" / "preprocessed" / "task1"

    if args.case_id:
        inputs_df = pd.read_csv(DATA / "inputs.csv").set_index("case_id")
        text_df = pd.read_csv(DATA / "full_prompt_narrative.csv").set_index("case_id")
        images_df = pd.read_csv(DATA / "images.csv").set_index("case_id")

        if args.case_id not in inputs_df.index:
            print(f"Error: case_id '{args.case_id}' not found in dataset.")
            sys.exit(1)

        tab_data = inputs_df.loc[args.case_id].to_dict()
        text_data = text_df.loc[args.case_id, "txt_full_prompt_narrative"]
        mri_cols = [c for c in images_df.columns if c.startswith("mri_emb_")]
        mri_data = images_df.loc[args.case_id, mri_cols].values

        case_input = {"tabular": tab_data, "text": text_data, "mri_emb": mri_data}
        result = predict_subtask_1_2(case_input)
        print(f"\nInference Result for Patient Case '{args.case_id}':")
        print(json.dumps(result, indent=2))

    elif args.sample:
        print("\nRunning Sample Test Case Inference...")
        inputs_df = pd.read_csv(DATA / "inputs.csv").iloc[0]
        text_df = pd.read_csv(DATA / "full_prompt_narrative.csv").iloc[0]
        images_df = pd.read_csv(DATA / "images.csv").iloc[0]

        mri_cols = [c for c in images_df.index if c.startswith("mri_emb_")]
        case_input = {
            "tabular": inputs_df.to_dict(),
            "text": text_df["txt_full_prompt_narrative"],
            "mri_emb": images_df[mri_cols].values
        }
        result = predict_subtask_1_2(case_input)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
