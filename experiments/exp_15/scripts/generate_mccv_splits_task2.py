#!/usr/bin/env python3
"""
Generates 30 deterministic, multi-subtask stratified Monte Carlo Cross-Validation partitions for Task 2.
Stratification criteria:
1. Exact class balance on Treatment Decision (4 classes).
2. Earth Mover's Distance minimization on Confidence distribution (ordinal).
3. Earth Mover's Distance minimization on 11 Clinical Variable Weights (ordinal).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def compute_emd_ordinal(p_train: np.ndarray, p_val: np.ndarray) -> float:
    """Earth Mover's Distance (Wasserstein-1) for 1D ordinal discrete distributions."""
    cdf_train = np.cumsum(p_train)
    cdf_val = np.cumsum(p_val)
    return float(np.sum(np.abs(cdf_train - cdf_val)))

def generate_splits(
    gt_path: str = "data/processed/ground_truth.csv",
    output_path: str = "data/processed/task2/mccv_splits_30.csv",
    n_splits: int = 30,
    max_search_per_split: int = 2500,
) -> dict:
    gt_df = pd.read_csv(gt_path)
    n_total = len(gt_df)
    assert n_total == 70, f"Expected 70 cases, got {n_total}"

    # Group pids by treatment decision class
    decision_groups = {}
    for decision, grp in gt_df.groupby("treatment_decision"):
        decision_groups[decision] = grp["pid"].values

    # Validation quotas per decision class (targeting 14 val / 56 train = 20% / 80%)
    # active_treatment: 31 -> 6 val / 25 train
    # active_surveillance: 25 -> 5 val / 20 train
    # continued_surveillance: 12 -> 2 val / 10 train
    # watchful_waiting: 2 -> 1 val / 1 train (or round-robin preservation)
    val_quotas = {
        "active_treatment": 6,
        "active_surveillance": 5,
        "continued_surveillance": 2,
        "watchful_waiting": 1,
    }
    assert sum(val_quotas.values()) == 14, "Total validation cases must be 14"

    # Encode Confidence to ordinal 0, 1, 2
    conf_map = {"uncertain": 0, "borderline": 1, "clear": 2}
    conf_by_pid = {
        row["pid"]: conf_map.get(str(row["confidence"]).lower(), 1)
        for _, row in gt_df.iterrows()
    }

    # Encode 11 Variable Weights to ordinal 0, 1, 2, 3
    weight_map = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
    weight_cols = [c for c in gt_df.columns if c.startswith("weight_")]
    assert len(weight_cols) == 11, f"Expected 11 weight columns, got {len(weight_cols)}"

    weights_by_pid = {}
    for _, row in gt_df.iterrows():
        weights_by_pid[row["pid"]] = np.array([
            weight_map.get(str(row[c]).lower(), 0) for c in weight_cols
        ])

    split_matrix = {"pid": gt_df["pid"].tolist()}
    diagnostics = []

    print(f"[*] Optimizing {n_splits} multi-subtask stratified splits for Task 2 (70 cases: 56 train / 14 val)...")

    for split_idx in range(n_splits):
        best_loss = float("inf")
        best_val_pids = None
        best_diag = {}

        base_seed = 52000 + split_idx * 1000

        for trial in range(max_search_per_split):
            rng = np.random.default_rng(base_seed + trial)

            val_candidates = []
            for dec, quota in val_quotas.items():
                pids_dec = decision_groups[dec]
                selected = rng.choice(pids_dec, size=quota, replace=False)
                val_candidates.extend(selected)

            val_set = set(val_candidates)
            train_candidates = [p for p in gt_df["pid"] if p not in val_set]

            # 1. Confidence EMD
            train_conf_counts = np.zeros(3)
            val_conf_counts = np.zeros(3)
            for p in train_candidates:
                train_conf_counts[conf_by_pid[p]] += 1
            for p in val_candidates:
                val_conf_counts[conf_by_pid[p]] += 1

            p_train_conf = train_conf_counts / len(train_candidates)
            p_val_conf = val_conf_counts / len(val_candidates)
            emd_conf = compute_emd_ordinal(p_train_conf, p_val_conf)

            # 2. Variable Weights Mean EMD across 11 variables
            emd_weights_list = []
            for var_idx in range(11):
                train_w_counts = np.zeros(4)
                val_w_counts = np.zeros(4)
                for p in train_candidates:
                    train_w_counts[weights_by_pid[p][var_idx]] += 1
                for p in val_candidates:
                    val_w_counts[weights_by_pid[p][var_idx]] += 1
                
                p_train_w = train_w_counts / len(train_candidates)
                p_val_w = val_w_counts / len(val_candidates)
                emd_weights_list.append(compute_emd_ordinal(p_train_w, p_val_w))

            mean_emd_weights = float(np.mean(emd_weights_list))

            # Composite multi-subtask loss
            loss = emd_conf + 1.0 * mean_emd_weights

            if loss < best_loss:
                best_loss = loss
                best_val_pids = val_set
                best_diag = {
                    "split_index": split_idx,
                    "seed": base_seed + trial,
                    "loss": loss,
                    "emd_confidence": emd_conf,
                    "mean_emd_weights": mean_emd_weights,
                    "val_confidence_dist": p_val_conf.tolist(),
                    "train_confidence_dist": p_train_conf.tolist(),
                }

        # Build assignment column
        col_assignments = []
        for pid in split_matrix["pid"]:
            if pid in best_val_pids:
                col_assignments.append("val")
            else:
                col_assignments.append("train")

        split_matrix[f"split_{split_idx}"] = col_assignments
        diagnostics.append(best_diag)

    df_splits = pd.DataFrame(split_matrix)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(out_p, index=False)

    print(f"[+] Splits matrix saved to {output_path}")
    print(f"    - Dimensions: {df_splits.shape}")
    print(f"    - Mean EMD Confidence: {np.mean([d['emd_confidence'] for d in diagnostics]):.4f}")
    print(f"    - Mean EMD Weights: {np.mean([d['mean_emd_weights'] for d in diagnostics]):.4f}")

    return {
        "splits_csv": str(out_p),
        "diagnostics": diagnostics,
    }

def main():
    root = Path(__file__).resolve().parents[3]
    gt_path = root / "data" / "processed" / "ground_truth.csv"
    output_path = root / "data" / "processed" / "task2" / "mccv_splits_30.csv"
    diag_path = root / "experiments" / "exp_15" / "results" / "splits_diagnostics.json"

    res = generate_splits(gt_path=str(gt_path), output_path=str(output_path), n_splits=30)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump(res["diagnostics"], f, indent=2)
    print(f"[+] Diagnostics saved to {diag_path}")

if __name__ == "__main__":
    main()
