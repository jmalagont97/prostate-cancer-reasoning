#!/usr/bin/env python3
"""
deploy/subtask_1.1/predict_inference.py

Inference Engine & Agent Tool Interface for Subtask 1.1 (Biopsy Decision).
Loads model_subtask_1.1.pkl, runs multimodal preprocessing and KNN inference,
and returns structured JSON output for an AI Agent.
"""

import sys
import pickle
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR))

from knn_model import ConfidenceWeightedKNN
from preprocess_input import preprocess_case, clean_text_narrative

DEFAULT_PKL_PATH = DEPLOY_DIR / "model_subtask_1.1.pkl"
_MODEL_BUNDLE_CACHE = None


def load_model_bundle(pkl_path=None):
    global _MODEL_BUNDLE_CACHE
    if _MODEL_BUNDLE_CACHE is not None and (pkl_path is None or pkl_path == DEFAULT_PKL_PATH):
        return _MODEL_BUNDLE_CACHE

    path = Path(pkl_path) if pkl_path else DEFAULT_PKL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}. Run train_and_save.py first.")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    _MODEL_BUNDLE_CACHE = bundle
    return bundle


def predict_subtask_1_1(raw_case, pkl_path=None):
    """
    Inference API for Subtask 1.1 Agent Tool.

    Parameters:
      raw_case (dict): Patient data containing tabular, text, and mri_emb keys.
      pkl_path (str/Path, optional): Path to model_subtask_1.1.pkl

    Returns:
      dict: Structured prediction dictionary for LLM Agent Tool
    """
    model_bundle = load_model_bundle(pkl_path)

    # 1. Preprocess raw patient inputs
    X_tab, X_mri, X_text = preprocess_case(raw_case, model_bundle)

    # 2. Extract predictions per modality
    knn_models = model_bundle["knn_models"]
    p_T = float(knn_models["tabular"].predict_proba(X_tab)[0])
    p_M = float(knn_models["mri"].predict_proba(X_mri)[0])
    p_X = float(knn_models["text"].predict_proba(X_text)[0])

    # 3. Compute Late Multimodal Fusion Probability (Equal Weights 1/3)
    weights = model_bundle["fusion_weights"]
    p_fused = float(weights["tabular"] * p_T + weights["mri"] * p_M + weights["text"] * p_X)

    threshold = model_bundle["decision_threshold"]
    decision_binary = 1 if p_fused >= threshold else 0
    decision_label = "RECOMMEND_BIOPSY" if decision_binary == 1 else "NO_BIOPSY"

    return {
        "subtask": "1.1_biopsy_decision",
        "biopsy_decision_binary": decision_binary,
        "biopsy_decision_label": decision_label,
        "fused_probability": round(p_fused, 4),
        "modality_probabilities": {
            "tabular_p_T": round(p_T, 4),
            "mri_p_M": round(p_M, 4),
            "text_p_X": round(p_X, 4)
        },
        "decision_threshold": threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Subtask 1.1 Biopsy Decision Inference CLI")
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
        result = predict_subtask_1_1(case_input)
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
        result = predict_subtask_1_1(case_input)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
