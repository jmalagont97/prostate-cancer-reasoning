#!/usr/bin/env python3
"""
Generates 30 deterministic, multi-subtask stratified Monte Carlo Cross-Validation partitions.
Saves the artifact to data/processed/task1/mccv_splits_30.csv.
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
    gt_path: str = "data/processed/task1/ground_truth_task1.csv",
    output_path: str = "data/processed/task1/mccv_splits_30.csv",
    n_splits: int = 30,
    n_val_yes: int = 11,
    n_val_no: int = 7,
    max_search_per_split: int = 2000,
) -> dict:
    gt_df = pd.read_csv(gt_path)
    labeled_df = gt_df[gt_df["decision"].notna()].copy()
    unlabeled_df = gt_df[gt_df["decision"].isna()].copy()

    yes_pids = labeled_df[labeled_df["decision"] == "yes"]["pid"].values
    no_pids = labeled_df[labeled_df["decision"] == "no"]["pid"].values
    unlabeled_pids = unlabeled_df["pid"].values

    assert len(yes_pids) == 56, f"Expected 56 yes cases, got {len(yes_pids)}"
    assert len(no_pids) == 35, f"Expected 35 no cases, got {len(no_pids)}"
    assert len(unlabeled_pids) == 104, f"Expected 104 unlabeled cases, got {len(unlabeled_pids)}"

    conf_map = {"uncertain": 0, "borderline": 1, "clear": 2}
    conf_by_pid = {
        row["pid"]: conf_map.get(str(row["confidence"]).lower(), 1)
        for _, row in labeled_df.iterrows()
    }

    # Factor active flags
    factor_cols = [c for c in labeled_df.columns if c.startswith("factor_")]
    factors_by_pid = {
        row["pid"]: np.array([1.0 if row[c] is True or row[c] == 1 else 0.0 for c in factor_cols])
        for _, row in labeled_df.iterrows()
    }

    split_matrix = {"pid": gt_df["pid"].tolist()}
    diagnostics = []

    print(f"[*] Optimizing {n_splits} multi-subtask stratified splits...")

    for split_idx in range(n_splits):
        best_score = float("inf")
        best_val_pids = None
        best_diag = {}

        base_seed = 42000 + split_idx * 1000

        for trial in range(max_search_per_split):
            rng = np.random.default_rng(base_seed + trial)

            val_yes = rng.choice(yes_pids, size=n_val_yes, replace=False)
            val_no = rng.choice(no_pids, size=n_val_no, replace=False)
            val_candidates = set(val_yes).union(set(val_no))
            train_candidates = [p for p in labeled_df["pid"] if p not in val_candidates]

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

            # 2. Critical Factors Frequency Discrepancy
            if factor_cols:
                train_factors = np.mean([factors_by_pid[p] for p in train_candidates], axis=0)
                val_factors = np.mean([factors_by_pid[p] for p in val_candidates], axis=0)
                factor_diff = float(np.mean(np.abs(train_factors - val_factors)))
            else:
                factor_diff = 0.0

            loss = emd_conf + 1.0 * factor_diff

            if loss < best_score:
                best_score = loss
                best_val_pids = val_candidates
                best_diag = {
                    "split_index": split_idx,
                    "seed": base_seed + trial,
                    "loss": loss,
                    "emd_conf": emd_conf,
                    "factor_discrepancy": factor_diff,
                    "val_confidence_dist": p_val_conf.tolist(),
                    "train_confidence_dist": p_train_conf.tolist(),
                }

        # Build assignment column
        col_assignments = []
        for pid in split_matrix["pid"]:
            if pid in unlabeled_pids:
                col_assignments.append("unlabeled")
            elif pid in best_val_pids:
                col_assignments.append("val")
            else:
                col_assignments.append("train")

        split_matrix[f"split_{split_idx}"] = col_assignments
        diagnostics.append(best_diag)

    # Save CSV
    out_df = pd.DataFrame(split_matrix)
    out_df.to_csv(output_path, index=False)
    print(f"[+] Splits saved successfully to: {output_path} (Shape: {out_df.shape})")

    return diagnostics


if __name__ == "__main__":
    results_dir = Path("experiments/exp_3/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    diag = generate_splits()

    with open(results_dir / "split_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    with open(results_dir / "version.txt", "w") as f:
        f.write("v1.0.0-mccv30-stratified\n")

    print("[+] Diagnostics written to experiments/exp_3/results/split_diagnostics.json")
