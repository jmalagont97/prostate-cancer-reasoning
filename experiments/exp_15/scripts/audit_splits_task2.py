#!/usr/bin/env python3
"""
Audit script for Task 2 MCCV-30 partitions and chimera_metrics compatibility.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure utils/pkg/chimera_metrics is accessible
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "utils" / "pkg" / "chimera_metrics"))

from chimera_metrics.core.base import norm_treatment_decision
from chimera_metrics.task2.decision import compute_treatment_decision_gate
from chimera_metrics.task2.reasoning import (
    compute_confidence_score,
    compute_variable_weights_score,
    compute_important_decisive_f1,
)

def main():
    root = Path(__file__).resolve().parents[3]
    gt_path = root / "data" / "processed" / "ground_truth.csv"
    input_path = root / "data" / "processed" / "input.csv"
    splits_path = root / "data" / "processed" / "task2" / "mccv_splits_30.csv"

    print("=================================================================")
    print("AUDITING TASK 2 MCCV-30 SPLITS & CHIMERA_METRICS COMPATIBILITY")
    print("=================================================================")

    df_gt = pd.read_csv(gt_path)
    df_in = pd.read_csv(input_path)
    df_splits = pd.read_csv(splits_path)

    assert len(df_gt) == 70, f"GT rows expected 70, got {len(df_gt)}"
    assert len(df_in) == 70, f"Input rows expected 70, got {len(df_in)}"
    assert len(df_splits) == 70, f"Splits rows expected 70, got {len(df_splits)}"
    assert df_gt["pid"].equals(df_in["pid"]), "PID alignment mismatch between input and GT"
    assert df_gt["pid"].equals(df_splits["pid"]), "PID alignment mismatch between GT and splits"

    print(f"[✓] Data integrity confirmed: 70 cases perfectly aligned.")

    # Check partitions balance
    for i in range(30):
        col = f"split_{i}"
        val_mask = df_splits[col] == "val"
        train_mask = df_splits[col] == "train"

        assert val_mask.sum() == 14, f"Split {i} val count {val_mask.sum()} != 14"
        assert train_mask.sum() == 56, f"Split {i} train count {train_mask.sum()} != 56"

        # Check exact decision quotas
        val_df = df_gt[val_mask]
        dec_series: pd.Series = pd.Series(val_df["treatment_decision"])
        dec_counts = dec_series.value_counts().to_dict()
        assert dec_counts.get("active_treatment", 0) == 6
        assert dec_counts.get("active_surveillance", 0) == 5
        assert dec_counts.get("continued_surveillance", 0) == 2
        assert dec_counts.get("watchful_waiting", 0) == 1

    print(f"[✓] Exact multi-class decision quotas verified across all 30 splits (6 / 5 / 2 / 1 in validation).")

    # Smoke test chimera_metrics on ground truth self-evaluation
    # 1. Official nested dict contract
    sample_gt_official = {
        "treatment_recommendation": {
            "primary": df_gt.iloc[0]["treatment_decision"]
        },
        "confidence": df_gt.iloc[0]["confidence"],
        "variable_weights": {
            c.replace("weight_", ""): df_gt.iloc[0][c]
            for c in df_gt.columns if c.startswith("weight_")
        }
    }

    # Perfect prediction test
    gate_score_official, _, _, _ = compute_treatment_decision_gate(sample_gt_official, sample_gt_official)
    gate_score_str, _, _, _ = compute_treatment_decision_gate(df_gt.iloc[0]["treatment_decision"], df_gt.iloc[0]["treatment_decision"])
    conf_score = compute_confidence_score(sample_gt_official, sample_gt_official)
    var_w_score = compute_variable_weights_score(sample_gt_official, sample_gt_official)
    imp_f1 = compute_important_decisive_f1(sample_gt_official, sample_gt_official)

    assert gate_score_official == 1.0, f"Official dict Gate score {gate_score_official} != 1.0"
    assert gate_score_str == 1.0, f"Raw string Gate score {gate_score_str} != 1.0"
    assert conf_score == 1.0, f"Confidence score {conf_score} != 1.0"
    assert var_w_score == 1.0, f"Weights score {var_w_score} != 1.0"
    assert imp_f1 == 1.0, f"Important/decisive F1 {imp_f1} != 1.0"

    print(f"[✓] chimera_metrics self-consistency validated (all subtasks exact match = 1.0).")
    print("=================================================================")
    print("ALL AUDIT CHECKS PASSED: Environment ready for subtask benchmarking.")
    print("=================================================================")

if __name__ == "__main__":
    main()
