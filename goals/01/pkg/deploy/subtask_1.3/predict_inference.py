#!/usr/bin/env python3
"""
deploy/subtask_1.3/predict_inference.py

Inference Engine & Agent Tool Interface for Subtask 1.3 (Clinical Relevance & Section Reveal Sequence).
Loads model_subtask_1.3.pkl, computes max-normalized SHAP attributions from Subtask 1.1 model,
discretizes 10 ordinal relevance weights, and derives the section reveal sequence.
"""

import sys
import pickle
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

# Import predict_inference from Subtask 1.1 cleanly
spec_1_1 = importlib.util.spec_from_file_location("subtask_1_1_infer_mod", SUBTASK_1_1_DIR / "predict_inference.py")
subtask_1_1_mod = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(subtask_1_1_mod)
load_subtask_1_1_bundle = subtask_1_1_mod.load_model_bundle

sys.path.insert(0, str(DEPLOY_DIR))
from compute_shap_descriptors import compute_shap_attributions, TARGET_VARS

DEFAULT_PKL_PATH = DEPLOY_DIR / "model_subtask_1.3.pkl"
_MODEL_BUNDLE_CACHE_1_3 = None


def load_model_bundle_1_3(pkl_path=None):
    global _MODEL_BUNDLE_CACHE_1_3
    if _MODEL_BUNDLE_CACHE_1_3 is not None and (pkl_path is None or pkl_path == DEFAULT_PKL_PATH):
        return _MODEL_BUNDLE_CACHE_1_3

    path = Path(pkl_path) if pkl_path else DEFAULT_PKL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}. Run train_and_save.py first.")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    _MODEL_BUNDLE_CACHE_1_3 = bundle
    return bundle


def map_weights_to_reveal_sequence(pred_weights):
    sections = []
    rad_max = max(pred_weights["pirads"], pred_weights["psad"], pred_weights["vol"], pred_weights["cspca"])
    if rad_max >= 1:
        sections.append("radiology_report")
    if pred_weights["dre"] >= 1:
        sections.append("laboratory_results")
    if pred_weights["psa"] >= 1:
        sections.append("psa_trend")
    if pred_weights["fh"] >= 1:
        sections.append("family_history")
    if pred_weights["bx"] >= 1:
        sections.append("pathology_report")
    if pred_weights["comorbidity"] >= 1 or pred_weights["age"] >= 2:
        sections.append("previous_notes")
    return sections


def predict_subtask_1_3(raw_case, pkl_path=None):
    """
    Inference API for Subtask 1.3 Agent Tool.

    Parameters:
      raw_case (dict): Patient data containing tabular data.
      pkl_path (str/Path, optional): Path to model_subtask_1.3.pkl

    Returns:
      dict: Structured prediction dictionary for LLM Agent Tool
    """
    bundle_1_3 = load_model_bundle_1_3(pkl_path)
    bundle_1_1 = load_subtask_1_1_bundle()

    pop_max_shap = bundle_1_3["population_max_shap"]
    thresholds_dict = bundle_1_3["thresholds_per_variable"]
    weight_map = bundle_1_3["ordinal_weight_map"]

    # 1. Compute max-normalized SHAP values
    psi_norm_dict, _ = compute_shap_attributions(raw_case, bundle_1_1, pop_max_shap)

    # 2. Categorize 10 target relevance weights
    pred_weights = {}
    pred_labels = {}

    for var in TARGET_VARS:
        val = psi_norm_dict[var]
        t1, t2, t3 = thresholds_dict[var]

        if val >= t3:
            w = 3
        elif val >= t2:
            w = 2
        elif val >= t1:
            w = 1
        else:
            w = 0

        pred_weights[var] = int(w)
        pred_labels[var] = weight_map[int(w)]

    # 3. Map predicted weights to section reveal sequence
    reveal_sequence = map_weights_to_reveal_sequence(pred_weights)

    return {
        "subtask": "1.3_relevance_and_reveal_sequence",
        "relevance_weights": pred_weights,
        "relevance_labels": pred_labels,
        "reveal_sequence": reveal_sequence
    }


def main():
    parser = argparse.ArgumentParser(description="Subtask 1.3 Relevance & Reveal Sequence Inference CLI")
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
        result = predict_subtask_1_3(case_input)
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
        result = predict_subtask_1_3(case_input)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
