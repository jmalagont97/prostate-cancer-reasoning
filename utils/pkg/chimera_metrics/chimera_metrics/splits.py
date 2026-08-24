"""
Split management and verification utilities for standardized MCCV evaluation.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_mccv_splits(
    splits_csv_path: str = "data/processed/task1/mccv_splits_30.csv",
) -> pd.DataFrame:
    """Load and validate the standardized 30-fold MCCV split matrix."""
    path = Path(splits_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Standardized splits matrix not found at: {path.resolve()}")

    df = pd.read_csv(path)
    if "pid" not in df.columns:
        raise ValueError("Splits file must contain a 'pid' column.")

    split_cols = [c for c in df.columns if c.startswith("split_")]
    if len(split_cols) != 30:
        raise ValueError(f"Expected exactly 30 split columns ('split_0'..'split_29'), found {len(split_cols)}.")

    return df


def get_split_indices(
    splits_df: pd.DataFrame,
    split_index: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Retrieve train, val, and unlabeled patient ID lists for a specific split.

    Returns:
        (train_pids, val_pids, unlabeled_pids)
    """
    col = f"split_{split_index}"
    if col not in splits_df.columns:
        raise ValueError(f"Split column '{col}' not found in splits DataFrame.")

    train_pids = splits_df[splits_df[col] == "train"]["pid"].tolist()
    val_pids = splits_df[splits_df[col] == "val"]["pid"].tolist()
    unlabeled_pids = splits_df[splits_df[col] == "unlabeled"]["pid"].tolist()

    return train_pids, val_pids, unlabeled_pids
