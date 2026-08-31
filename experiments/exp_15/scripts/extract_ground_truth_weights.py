#!/usr/bin/env python3
"""
Extracts 11 variable weights from raw reasoning JSONs and enriches ground_truth.csv.
Missing variables in a case default to 'not_used' in strict alignment with chimera_metrics.
"""

import json
from pathlib import Path
import pandas as pd

def main():
    root = Path(__file__).resolve().parents[3]
    gt_path = root / "data" / "processed" / "ground_truth.csv"
    gt_task2_path = root / "data" / "processed" / "task2" / "ground_truth_task2.csv"
    raw_dir = root / "data" / "raw" / "train_release" / "task2"

    df_gt = pd.read_csv(gt_path)
    print(f"[*] Loaded ground truth: {df_gt.shape[0]} cases.")

    expected_vars = [
        "age", "bx_gl_prim", "bx_gl_sec", "bx_isup", "comorbidity",
        "cspca", "ct", "fh", "pirads", "psa", "psad"
    ]

    records = []
    for pid in df_gt["pid"]:
        reasoning_file = raw_dir / pid / "prostate-treatment-decision-reasoning.json"
        if not reasoning_file.exists():
            raise FileNotFoundError(f"Missing reasoning file for {pid} at {reasoning_file}")
        
        with open(reasoning_file) as f:
            data = json.load(f)
        
        weights = data.get("variable_weights", {})
        row = {"pid": pid}
        for var_name in expected_vars:
            val = weights.get(var_name, "not_used")
            row[f"weight_{var_name}"] = str(val).strip().lower()
        records.append(row)

    df_weights = pd.DataFrame(records)
    # Merge or overwrite columns if they already exist
    cols_to_keep = [c for c in df_gt.columns if not c.startswith("weight_")]
    df_enriched = pd.merge(df_gt[cols_to_keep], df_weights, on="pid", how="inner")

    assert len(df_enriched) == len(df_gt) == 70, f"Row mismatch: {len(df_enriched)}"
    assert df_enriched.isna().sum().sum() == 0, "NaN values detected in enriched ground truth"

    # Save to both target locations
    df_enriched.to_csv(gt_path, index=False)
    gt_task2_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(gt_task2_path, index=False)

    print(f"[+] Successfully enriched ground truth files with 11 variable weights.")
    print(f"    - Columns: {list(df_enriched.columns)}")
    print(f"    - Shape: {df_enriched.shape}")

if __name__ == "__main__":
    main()
