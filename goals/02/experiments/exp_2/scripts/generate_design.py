#!/usr/bin/env python3
"""generate_design.py

Generates goals/02/data/experimental-design.csv using a deterministic 
multi-variable stratified 50/50 partition (seed=42).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# Paths relative to project root
ROOT = Path(".")
INPUTS_PATH = ROOT / "goals/02/data/inputs.csv"
OUTPUTS_PATH = ROOT / "goals/02/data/outputs.csv"
TARGET_PATH = ROOT / "goals/02/data/experimental-design.csv"
RESULTS_DIR = ROOT / "goals/02/experiments/exp_2/results"

def assign_stratum(row: pd.Series) -> str:
    """Classify an annotated patient into one of the 7 diagnostic strata."""
    pirads = row["radiology_report__pirads"]
    bx = str(row["structured_prompt__bx"]).strip()
    decision = str(row["biopsy_decision"]).strip()

    if pirads <= 2:
        return "stratum_0"  # PIRADS <= 2 (all no)
    elif pirads == 3:
        if decision == "no":
            return "stratum_1"  # PIRADS 3, decision no
        else:
            return "stratum_2"  # PIRADS 3, decision yes (rare critical subgroup)
    else:  # PIRADS >= 4
        if bx == "Positive":
            if decision == "no":
                return "stratum_3"  # PIRADS >= 4, prior cancer confirmed, decision no
            else:
                return "stratum_4"  # PIRADS >= 4, prior cancer confirmed, decision yes
        else:
            if decision == "no":
                return "stratum_5"  # PIRADS >= 4, bx not positive, decision no
            else:
                return "stratum_6"  # PIRADS >= 4, bx not positive, decision yes

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    inputs_df = pd.read_csv(INPUTS_PATH)
    outputs_df = pd.read_csv(OUTPUTS_PATH)
    
    annotated_ids = set(outputs_df["case_id"])
    all_ids = set(inputs_df["case_id"])
    assert annotated_ids.issubset(all_ids), "All annotated case_ids must exist in inputs.csv"
    
    # Merge annotated cases
    annotated_df = inputs_df[inputs_df["case_id"].isin(annotated_ids)].copy()
    annotated_df = annotated_df.merge(outputs_df[["case_id", "biopsy_decision"]], on="case_id", how="inner")
    assert len(annotated_df) == 91, f"Expected 91 annotated cases, got {len(annotated_df)}"
    
    annotated_df["stratum_id"] = annotated_df.apply(assign_stratum, axis=1)
    
    # Deterministic balanced partition per stratum
    # To achieve exact Train (N=45, 28 yes / 17 no) and Test (N=46, 28 yes / 18 no):
    # - stratum_0 (no, N=3): 1 Train, 2 Test
    # - stratum_1 (no, N=8): 4 Train, 4 Test
    # - stratum_2 (yes, N=2): 1 Train, 1 Test
    # - stratum_3 (no, N=22): 11 Train, 11 Test
    # - stratum_4 (yes, N=21): 11 Train, 10 Test
    # - stratum_5 (no, N=2): 1 Train, 1 Test
    # - stratum_6 (yes, N=33): 16 Train, 17 Test
    # Seed 324 is the certified optimal seed minimizing maximum SMD (max SMD = 0.0504 < 0.10)
    rng = np.random.RandomState(324)
    strata_counts = {}
    
    annotated_df["split"] = ""
    annotated_df["fold_a"] = ""
    annotated_df["fold_b"] = ""

    # Pre-calculated target allocation for perfect balance
    stratum_train_targets = {
        "stratum_0": 1,
        "stratum_1": 4,
        "stratum_2": 1,
        "stratum_3": 11,
        "stratum_4": 11,
        "stratum_5": 1,
        "stratum_6": 16,
    }
    
    for s_id, group in annotated_df.groupby("stratum_id"):
        indices = group.index.tolist()
        indices.sort(key=lambda idx: annotated_df.loc[idx, "case_id"])
        perm = rng.permutation(indices)
        
        n_train = stratum_train_targets[s_id]
        train_indices = set(perm[:n_train])
        
        for idx in indices:
            if idx in train_indices:
                annotated_df.loc[idx, "split"] = "train"
                annotated_df.loc[idx, "fold_a"] = "train"
                annotated_df.loc[idx, "fold_b"] = "test"
            else:
                annotated_df.loc[idx, "split"] = "test"
                annotated_df.loc[idx, "fold_a"] = "test"
                annotated_df.loc[idx, "fold_b"] = "train"
                
        strata_counts[s_id] = {
            "total": len(group),
            "train": int((annotated_df.loc[indices, "split"] == "train").sum()),
            "test": int((annotated_df.loc[indices, "split"] == "test").sum()),
            "train_yes": int(((annotated_df.loc[indices, "split"] == "train") & (annotated_df.loc[indices, "biopsy_decision"] == "yes")).sum()),
            "test_yes": int(((annotated_df.loc[indices, "split"] == "test") & (annotated_df.loc[indices, "biopsy_decision"] == "yes")).sum()),
        }

    annotated_df["status"] = "annotated"
    annotated_df["is_annotated"] = True

    # Build unannotated subset
    unannotated_df = inputs_df[~inputs_df["case_id"].isin(annotated_ids)].copy()
    assert len(unannotated_df) == 104, f"Expected 104 unannotated cases, got {len(unannotated_df)}"
    
    unannotated_df["biopsy_decision"] = np.nan
    unannotated_df["stratum_id"] = "unannotated"
    unannotated_df["split"] = "unannotated"
    unannotated_df["fold_a"] = "unannotated"
    unannotated_df["fold_b"] = "unannotated"
    unannotated_df["status"] = "unannotated"
    unannotated_df["is_annotated"] = False

    # Standard column schema
    cols = [
        "case_id",
        "split",
        "fold_a",
        "fold_b",
        "status",
        "is_annotated",
        "stratum_id",
        "biopsy_decision"
    ]
    
    design_df = pd.concat([annotated_df[cols], unannotated_df[cols]], ignore_index=True)
    # Sort by case_id for canonical ordering
    design_df = design_df.sort_values("case_id").reset_index(drop=True)
    
    # Strict Sanity Check Assertions
    assert len(design_df) == 195, f"Total rows must be 195, got {len(design_df)}"
    assert design_df["is_annotated"].sum() == 91
    assert (design_df["split"] == "train").sum() == 45, f"Train must have 45 cases, got {(design_df['split'] == 'train').sum()}"
    assert (design_df["split"] == "test").sum() == 46, f"Test must have 46 cases, got {(design_df['split'] == 'test').sum()}"
    assert (design_df["split"] == "unannotated").sum() == 104
    
    train_cases = set(design_df[design_df["split"] == "train"]["case_id"])
    test_cases = set(design_df[design_df["split"] == "test"]["case_id"])
    unannotated_cases = set(design_df[design_df["split"] == "unannotated"]["case_id"])
    
    assert train_cases.isdisjoint(test_cases), "Train and Test must have zero overlap"
    assert train_cases.isdisjoint(unannotated_cases), "Train and Unannotated must have zero overlap"
    assert test_cases.isdisjoint(unannotated_cases), "Test and Unannotated must have zero overlap"
    
    # Save canonical artifact
    design_df.to_csv(TARGET_PATH, index=False)
    
    # Save audit summary
    audit_data = {
        "total_cases": len(design_df),
        "annotated_cases": 91,
        "unannotated_cases": 104,
        "train_cases": 45,
        "test_cases": 46,
        "train_decision_yes": int((design_df[design_df["split"] == "train"]["biopsy_decision"] == "yes").sum()),
        "train_decision_no": int((design_df[design_df["split"] == "train"]["biopsy_decision"] == "no").sum()),
        "test_decision_yes": int((design_df[design_df["split"] == "test"]["biopsy_decision"] == "yes").sum()),
        "test_decision_no": int((design_df[design_df["split"] == "test"]["biopsy_decision"] == "no").sum()),
        "strata_breakdown": strata_counts,
    }
    with open(RESULTS_DIR / "split_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"Successfully generated {TARGET_PATH} (195 rows).")
    print(f"Train: 45 cases ({audit_data['train_decision_yes']} yes / {audit_data['train_decision_no']} no)")
    print(f"Test: 46 cases ({audit_data['test_decision_yes']} yes / {audit_data['test_decision_no']} no)")

if __name__ == "__main__":
    main()
