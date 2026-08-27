#!/usr/bin/env python3
"""
CHIMERA Task 2 -- 100-Split Stratified MCCV Generation (exp_10).

Generates data/processed/task2/mccv_splits_100.csv with columns:
  pid, split_0 .. split_99
values: 'train' | 'val'

Stratification guarantees per split:
  - 70% train / 30% val overall
  - watchful_waiting (n=2): AT LEAST 1 in train, AT MOST 1 in val
  - remaining 3 classes stratified via StratifiedShuffleSplit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"

N_SPLITS = 100
TRAIN_FRAC = 0.70
SEED = 42


def main() -> None:
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    print(f"Loaded ground truth: N={len(df_gt)}")
    print(df_gt["treatment_decision"].value_counts().to_string())

    # Separate the extreme minority class
    minority_class = "watchful_waiting"
    df_minor = df_gt[df_gt["treatment_decision"] == minority_class].reset_index(drop=True)
    df_rest = df_gt[df_gt["treatment_decision"] != minority_class].reset_index(drop=True)
    print(f"\nMinority ({minority_class}): {len(df_minor)} | Rest: {len(df_rest)}")

    if len(df_minor) < 2:
        raise ValueError("watchful_waiting has fewer than 2 cases; cannot guarantee training presence.")

    # Base stratified splits for the 3 remaining classes
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        train_size=TRAIN_FRAC,
        test_size=1.0 - TRAIN_FRAC,
        random_state=SEED,
    )

    rng = np.random.default_rng(SEED)
    split_matrix = {pid: [] for pid in df_gt["pid"]}

    for split_idx, (rest_train_idx, rest_val_idx) in enumerate(sss.split(df_rest, df_rest["treatment_decision"])):
        # Assign minority: 1 train, 1 val (alternate to balance exposure)
        min_order = [0, 1] if split_idx % 2 == 0 else [1, 0]
        minority_assignments = {
            df_minor.loc[min_order[0], "pid"]: "train",
            df_minor.loc[min_order[1], "pid"]: "val",
        }

        train_pids = set(df_rest.loc[rest_train_idx, "pid"])
        val_pids = set(df_rest.loc[rest_val_idx, "pid"])
        train_pids |= {p for p, a in minority_assignments.items() if a == "train"}
        val_pids |= {p for p, a in minority_assignments.items() if a == "val"}

        # Safety checks
        assert len(train_pids) + len(val_pids) == len(df_gt), \
            f"Split {split_idx}: partition does not cover all cases"
        assert df_minor.loc[0, "pid"] in train_pids or df_minor.loc[1, "pid"] in train_pids, \
            f"Split {split_idx}: no minority in train"

        for pid in df_gt["pid"]:
            split_matrix[pid].append("train" if pid in train_pids else "val")

    # Build output DataFrame
    out_df = pd.DataFrame({"pid": list(df_gt["pid"])})
    for split_idx in range(N_SPLITS):
        out_df[f"split_{split_idx}"] = [split_matrix[pid][split_idx] for pid in out_df["pid"]]

    out_path = DATA_DIR / "mccv_splits_100.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nPersisted: {out_path} ({out_df.shape[0]}x{out_df.shape[1]})")

    # Verification
    print("\nVerification:")
    for split_idx in range(N_SPLITS):
        col = f"split_{split_idx}"
        train_s = out_df.loc[out_df[col] == "train", "pid"]
        val_s = out_df.loc[out_df[col] == "val", "pid"]
        n_train, n_val = len(train_s), len(val_s)
        assert abs(n_train / (n_train + n_val) - TRAIN_FRAC) < 0.02, f"Split {split_idx} train ratio off"

        gt_train = df_gt[df_gt["pid"].isin(train_s)]["treatment_decision"]
        gt_val = df_gt[df_gt["pid"].isin(val_s)]["treatment_decision"]
        assert (gt_train == minority_class).sum() >= 1, f"Split {split_idx}: minority missing in train"
        assert (gt_val == minority_class).sum() <= 1, f"Split {split_idx}: >1 minority in val"
        for cls in df_gt["treatment_decision"].unique():
            assert (gt_train == cls).sum() >= 1 or cls == minority_class, \
                f"Split {split_idx}: class {cls} absent from train"

    print(f"  All {N_SPLITS} splits passed stratification checks.")

    # Per-class train/val counts (first 5 splits summarize)
    print("\nExample (split_0):")
    col = "split_0"
    train_s = out_df.loc[out_df[col] == "train", "pid"]
    print(df_gt[df_gt["pid"].isin(train_s)]["treatment_decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
