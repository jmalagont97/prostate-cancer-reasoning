#!/usr/bin/env python3
"""
experiments/exp_19/scripts/compare_baselines.py

Executes a side-by-side comparison between:
  1. Always Noted (1=noted for all variables)
  2. Always Majority Class (per-variable mode)
  3. exp_19 SOTA (Max-Normalized SHAP + Global 3D Thresholds)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data" / "chimera26" / "preprocessed" / "task1"

TARGET_VARS = ["age", "fh", "cspca", "pirads", "vol", "psa", "comorbidity", "psad", "dre", "bx"]
ALL_SECTIONS = ["radiology_report", "laboratory_results", "psa_trend", "previous_notes", "family_history", "pathology_report"]


def map_weights_to_reveal(pred_dict):
    sections = []
    rad_max = max(pred_dict["pirads"], pred_dict["psad"], pred_dict["vol"], pred_dict["cspca"])
    if rad_max >= 1:
        sections.append("radiology_report")
    if pred_dict["dre"] >= 1:
        sections.append("laboratory_results")
    if pred_dict["psa"] >= 1:
        sections.append("psa_trend")
    if pred_dict["fh"] >= 1:
        sections.append("family_history")
    if pred_dict["bx"] >= 1:
        sections.append("pathology_report")
    if pred_dict["comorbidity"] >= 1 or pred_dict["age"] >= 2:
        sections.append("previous_notes")
    return sections


def main():
    gt_df = pd.read_csv(DATA_DIR / "ground_truth.csv")
    splits_df = pd.read_csv(DATA_DIR / "mccv_loocv_splits.csv")

    usable_df = splits_df[splits_df["cohort_status"] == "usable_labeled"].sort_values("case_id").reset_index(drop=True)
    usable_ids = usable_df["case_id"].tolist()
    N = len(usable_ids)

    gt_df = gt_df.set_index("case_id").loc[usable_ids].reset_index()

    y_true_dict = {}
    for var in TARGET_VARS:
        y_true_dict[var] = gt_df[f"target_code_weight_{var}"].values.astype(int)

    gt_reveals = [json.loads(s) for s in gt_df["target_reveal_sequence_json"].tolist()]

    # 1. Baseline: Always Noted (1)
    moe_noted = []
    f1_weights_noted = []
    for var in TARGET_VARS:
        y = y_true_dict[var]
        pred = np.ones(N, dtype=int)
        classes = np.unique(y)
        errs = [np.mean(np.abs(pred[y == c] - c) / 3.0) for c in classes]
        moe_noted.append(np.mean(errs))
        f1_weights_noted.append(f1_score(y, pred, average="macro", zero_division=0))

    noted_weights = {var: 1 for var in TARGET_VARS}
    noted_reveal = map_weights_to_reveal(noted_weights)
    f1_sec_noted = []
    for sec in ALL_SECTIONS:
        y_s = [1 if sec in r else 0 for r in gt_reveals]
        p_s = [1 if sec in noted_reveal else 0] * N
        f1_sec_noted.append(f1_score(y_s, p_s, average="macro", zero_division=0))

    # 2. Baseline: Always Majority Class
    moe_maj = []
    f1_weights_maj = []
    maj_class_per_var = {}
    for var in TARGET_VARS:
        y = y_true_dict[var]
        vals, counts = np.unique(y, return_counts=True)
        maj = vals[np.argmax(counts)]
        maj_class_per_var[var] = int(maj)
        pred = np.full(N, maj, dtype=int)
        classes = np.unique(y)
        errs = [np.mean(np.abs(pred[y == c] - c) / 3.0) for c in classes]
        moe_maj.append(np.mean(errs))
        f1_weights_maj.append(f1_score(y, pred, average="macro", zero_division=0))

    maj_reveal = map_weights_to_reveal(maj_class_per_var)
    f1_sec_maj = []
    for sec in ALL_SECTIONS:
        y_s = [1 if sec in r else 0 for r in gt_reveals]
        p_s = [1 if sec in maj_reveal else 0] * N
        f1_sec_maj.append(f1_score(y_s, p_s, average="macro", zero_division=0))

    # Load exp_19 summary metrics
    summary_path = ROOT / "experiments" / "exp_19" / "results" / "summary.json"
    with open(summary_path, "r") as f:
        exp_19_data = json.load(f)["selected_config"]

    print("=" * 80)
    print("COMPARATIVA DE DESEMPEÑO DE MODELOS DE LA SUBTAREA 1.3 (LOO N=88)")
    print("=" * 80)
    print(f"{'Modelo / Estrategia':<30} | {'LOO F1 (Secciones)':<18} | {'LOO MOE_abs (Pesos)':<20} | {'LOO F1 (Pesos)':<15}")
    print("-" * 90)
    print(f"{'1. Always Noted (1=noted)':<30} | {np.mean(f1_sec_noted):<18.4f} | {np.mean(moe_noted):<20.4f} | {np.mean(f1_weights_noted):<15.4f}")
    print(f"{'2. Always Majority Class':<30} | {np.mean(f1_sec_maj):<18.4f} | {np.mean(moe_maj):<20.4f} | {np.mean(f1_weights_maj):<15.4f}")
    print(f"{'3. exp_19 SOTA (SHAP Kernel)':<30} | {exp_19_data['loo_section_f1_macro']:<18.4f} | {exp_19_data['loo_weights_moe_abs']:<20.4f} | {exp_19_data['loo_weights_f1_macro']:<15.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
