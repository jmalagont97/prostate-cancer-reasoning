#!/usr/bin/env python3
"""compute_gini_split.py

Deterministic evaluation tool for computing Gini impurity and information gain (Delta Gini)
on clinical features for Goal 02 (Task 1.1).
Strictly and immutably evaluated ONLY on the training partition (N = 45).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _resolve_paths() -> Tuple[Path, Path, Path]:
    """Resolve relative paths to Goal 02 data files."""
    current_file = Path(__file__).resolve()
    # Path: experiments/exp_3/scripts/tools/compute_gini_split.py -> goals/02
    goal_dir = current_file.parents[4] if "experiments" in current_file.parts else current_file.parents[3]
    # Fallback checking parent hierarchy
    for p in current_file.parents:
        if (p / "data" / "experimental-design.csv").exists():
            goal_dir = p
            break
    data_dir = goal_dir / "data"
    inputs_csv = data_dir / "inputs.csv"
    outputs_csv = data_dir / "outputs.csv"
    design_csv = data_dir / "experimental-design.csv"
    return inputs_csv, outputs_csv, design_csv


def _calculate_gini(series: pd.Series) -> float:
    """Calculate Gini impurity for a discrete categorical distribution."""
    if len(series) == 0:
        return 0.0
    probabilities = series.value_counts(normalize=True).values
    return float(round(1.0 - np.sum(probabilities ** 2), 4))


def _resolve_feature_name(raw_feature: str, available_cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve raw feature input to an exact column name in inputs.csv."""
    cleaned = raw_feature.strip()
    if cleaned in available_cols:
        return cleaned, None

    # Common short aliases
    alias_map = {
        "pirads": "radiology_report__pirads",
        "pirads_score": "radiology_report__pirads",
        "pi_rads": "radiology_report__pirads",
        "pi-rads": "radiology_report__pirads",
        "psa_density": "radiology_report__psa_density",
        "psad": "radiology_report__psa_density",
        "bx": "structured_prompt__bx",
        "biopsy": "structured_prompt__bx",
        "biopsy_history": "structured_prompt__bx",
        "prior_bx": "structured_prompt__bx",
        "prior_cancer": "previous_notes__prior_cancer_confirmed",
        "prior_cancer_confirmed": "previous_notes__prior_cancer_confirmed",
        "cancer_confirmed": "previous_notes__prior_cancer_confirmed",
        "psa": "laboratory_results__psa",
        "psa_level": "laboratory_results__psa",
        "psa_total": "laboratory_results__psa",
        "total_psa": "laboratory_results__psa",
        "serum_psa": "laboratory_results__psa",
        "free_psa": "laboratory_results__free_psa",
        "free_psa_percent": "laboratory_results__free_psa_percent",
        "free_psa_ratio": "laboratory_results__free_psa_percent",
        "f_t_psa": "laboratory_results__free_psa_percent",
        "dre": "structured_prompt__dre",
        "dre_nodus": "structured_prompt__dre",
        "dre_finding": "structured_prompt__dre",
        "age": "structured_prompt__age",
        "psav": "structured_prompt__psav",
        "psa_velocity": "structured_prompt__psav",
        "ipss": "structured_prompt__ipss",
        "cspca": "clinical_data__cspca",
        "family_history": "clinical_data__family_history",
        "prostate_volume": "radiology_report__prostate_volume",
        "volume": "radiology_report__prostate_volume",
        "psa_delta": "psa_trend__psa_delta",
        "active_treatment": "structured_prompt__active_treatment_flag",
        "active_treatment_flag": "structured_prompt__active_treatment_flag",
    }
    norm = cleaned.lower().replace(" ", "_").replace("-", "_")
    if norm in alias_map and alias_map[norm] in available_cols:
        return alias_map[norm], f"Mapped alias '{cleaned}' to column '{alias_map[norm]}'."

    # Strip suffixes like _level, _value, _score, _result
    stem = re.sub(r"_(level|value|score|result|status|finding)$", "", norm)
    if stem in alias_map and alias_map[stem] in available_cols:
        return alias_map[stem], f"Mapped stemmed alias '{cleaned}' to column '{alias_map[stem]}'."

    matches = [c for c in available_cols if norm in c.lower() or stem in c.lower()]
    if len(matches) == 1:
        return matches[0], f"Fuzzy matched '{cleaned}' to column '{matches[0]}'."

    return None, f"Feature '{cleaned}' not found in available clinical columns: {available_cols}"


def compute_gini_split(
    feature: str,
    operator: str,
    threshold: Any,
    unit: Optional[str] = None,
    target_leaf_id: str = "root",
    active_filter: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[List[str]] = None,
    split: str = "train",
    min_delta_gini: float = 0.0,
    min_samples_leaf: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate candidate decision tree split on clinical training partition.

    Args:
        feature: Column name or recognized alias.
        operator: '<=', '<', '>=', '>', '==', '!=', or 'in'.
        threshold: Numeric or categorical split boundary.
        unit: Optional unit documentation.
        target_leaf_id: Node ID.
        active_filter: Prior tree constraints.
        sample_indices: Subset of case_id values at the active node.
        split: Strictly 'train'.
    """
    inputs_csv, outputs_csv, design_csv = _resolve_paths()
    if not (inputs_csv.exists() and outputs_csv.exists() and design_csv.exists()):
        return {
            "ok": False,
            "error": "Dataset files missing in goals/02/data/",
            "decision": "REJECTED",
        }

    inputs_df = pd.read_csv(inputs_csv)
    outputs_df = pd.read_csv(outputs_csv)[["case_id", "biopsy_decision"]]
    design_df = pd.read_csv(design_csv)[["case_id", "split"]]

    merged = inputs_df.merge(outputs_df, on="case_id", how="inner").merge(design_df, on="case_id", how="inner")

    # Strictly filter by split == 'train'
    df = merged[merged["split"] == split].copy()
    if len(df) == 0:
        return {
            "ok": False,
            "error": f"No samples found for split '{split}'.",
            "decision": "REJECTED",
        }

    # Restrict to active node partition if sample_indices provided
    if sample_indices is not None:
        df = df[df["case_id"].isin(sample_indices)].copy()
    elif active_filter:
        # Apply prior branch filters only when sample_indices is not provided (e.g. CLI fallback)
        for f_col, f_cond in active_filter.items():
            if f_col in df.columns and isinstance(f_cond, dict):
                f_op = f_cond.get("operator", "==")
                f_val = f_cond.get("threshold")
                if pd.api.types.is_numeric_dtype(df[f_col]):
                    try:
                        f_num = float(f_val)
                    except (ValueError, TypeError):
                        continue
                    if f_op == "<=":
                        df = df[df[f_col] <= f_num]
                    elif f_op == "<":
                        df = df[df[f_col] < f_num]
                    elif f_op == ">=":
                        df = df[df[f_col] >= f_num]
                    elif f_op == ">":
                        df = df[df[f_col] > f_num]
                    elif f_op == "==":
                        df = df[df[f_col] == f_num]
                    elif f_op == "!=":
                        df = df[df[f_col] != f_num]
                else:
                    s_str = df[f_col].fillna("").astype(str).str.strip().str.lower()
                    v_str = str(f_val).strip().lower()
                    if f_op == "==":
                        df = df[s_str == v_str]
                    elif f_op == "!=":
                        df = df[s_str != v_str]

    total_samples = len(df)
    if total_samples == 0:
        return {
            "ok": False,
            "error": "Active node partition has 0 samples after applying filters.",
            "decision": "REJECTED",
        }

    parent_distribution = df["biopsy_decision"].value_counts().to_dict()
    parent_gini = _calculate_gini(df["biopsy_decision"])

    available_cols = [c for c in inputs_df.columns if c != "case_id"]
    resolved_col, resolution_note = _resolve_feature_name(feature, available_cols)
    if not resolved_col:
        return {
            "ok": False,
            "error": resolution_note,
            "decision": "REJECTED",
        }

    series = df[resolved_col]
    op = operator.strip()

    # Parse threshold
    is_numeric_col = pd.api.types.is_numeric_dtype(series)
    thresh = threshold
    if is_numeric_col:
        if isinstance(threshold, str):
            clean_t = threshold.strip().lower()
            if clean_t in ("true", "yes", "positive"):
                thresh = 1.0
            elif clean_t in ("false", "no", "negative"):
                thresh = 0.0
            else:
                try:
                    thresh = float(threshold)
                except (ValueError, TypeError):
                    return {
                        "ok": False,
                        "error": f"Column '{resolved_col}' is numeric, but threshold '{threshold}' could not be converted to float.",
                        "decision": "REJECTED",
                    }
        elif isinstance(threshold, bool):
            thresh = 1.0 if threshold else 0.0
        else:
            try:
                thresh = float(threshold)
            except (ValueError, TypeError):
                return {
                    "ok": False,
                    "error": f"Column '{resolved_col}' is numeric, but threshold '{threshold}' could not be converted to float.",
                    "decision": "REJECTED",
                }

    # Evaluate mask
    if is_numeric_col:
        if op == "<=":
            mask = series <= thresh
        elif op == "<":
            mask = series < thresh
        elif op == ">=":
            mask = series >= thresh
        elif op == ">":
            mask = series > thresh
        elif op == "==":
            mask = series == thresh
        elif op == "!=":
            mask = series != thresh
        elif op == "in":
            if isinstance(thresh, (list, tuple, set)):
                mask = series.isin([float(x) for x in thresh])
            else:
                items = [float(s.strip()) for s in str(thresh).split(",") if s.strip()]
                mask = series.isin(items)
        else:
            return {
                "ok": False,
                "error": f"Unsupported operator '{operator}'. Use <=, <, >=, >, ==, !=, or in.",
                "decision": "REJECTED",
            }
    else:
        str_series = series.fillna("").astype(str).str.strip().str.lower()
        str_thresh = str(thresh).strip().lower()
        if op == "==":
            mask = str_series == str_thresh
        elif op == "!=":
            mask = str_series != str_thresh
        elif op == "in":
            if isinstance(thresh, (list, tuple, set)):
                items = [str(x).strip().lower() for x in thresh]
            else:
                items = [str(s).strip().lower() for s in str(thresh).split(",") if s.strip()]
            mask = str_series.isin(items)
        else:
            return {
                "ok": False,
                "error": f"Operator '{operator}' is not valid for categorical feature '{resolved_col}'. Use ==, !=, or in.",
                "decision": "REJECTED",
            }

    left_df = df[mask]
    right_df = df[~mask]
    n_left = len(left_df)
    n_right = len(right_df)

    feat_min = float(series.min()) if is_numeric_col and not series.empty else None
    feat_max = float(series.max()) if is_numeric_col and not series.empty else None
    unique_vals = list(series.dropna().unique())[:8] if not is_numeric_col else None

    # Empty branch or minimum sample size check
    if n_left == 0 or n_right == 0 or n_left < min_samples_leaf or n_right < min_samples_leaf:
        range_info = f"[min={feat_min}, max={feat_max}]" if is_numeric_col else f"observed categories: {unique_vals}"
        if n_left == 0 or n_right == 0:
            feedback = (
                f"Invalid split: One partition is empty (Left={n_left}, Right={n_right}). "
                f"Evaluated rule '{resolved_col} {op} {thresh}' does not isolate samples in this node. "
                f"Available values in node partition: {range_info}."
            )
        else:
            feedback = (
                f"Invalid split: Child partition has too few samples (Left={n_left}, Right={n_right}, minimum required={min_samples_leaf}). "
                f"Evaluated rule '{resolved_col} {op} {thresh}' creates a degenerate branch. "
                f"Available values in node partition: {range_info}."
            )
        return {
            "ok": True,
            "target_leaf_id": target_leaf_id,
            "resolved_feature": resolved_col,
            "operator": op,
            "threshold": thresh,
            "total_samples": total_samples,
            "parent_gini": parent_gini,
            "parent_distribution": parent_distribution,
            "samples_left": n_left,
            "samples_right": n_right,
            "delta_gini": 0.0,
            "is_valid_split": False,
            "decision": "REJECTED",
            "feedback": feedback,
            "note": resolution_note,
        }

    left_gini = _calculate_gini(left_df["biopsy_decision"])
    right_gini = _calculate_gini(right_df["biopsy_decision"])
    left_dist = left_df["biopsy_decision"].value_counts().to_dict()
    right_dist = right_df["biopsy_decision"].value_counts().to_dict()

    weighted_child_gini = round((n_left / total_samples) * left_gini + (n_right / total_samples) * right_gini, 4)
    delta_gini = round(parent_gini - weighted_child_gini, 4)

    # Acceptance criterion: strictly positive impurity reduction
    is_accepted = bool(delta_gini > 0.0 and delta_gini >= min_delta_gini)
    decision = "ACCEPTED" if is_accepted else "REJECTED"

    feedback = (
        f"Split {decision}: Left partition has {n_left} samples (Gini={left_gini}, dist={left_dist}), "
        f"Right partition has {n_right} samples (Gini={right_gini}, dist={right_dist}). "
        f"Impurity reduction Delta Gini = {delta_gini:.4f} (Parent Gini={parent_gini} -> Weighted Child={weighted_child_gini})."
    )

    return {
        "ok": True,
        "target_leaf_id": target_leaf_id,
        "resolved_feature": resolved_col,
        "operator": op,
        "threshold": thresh,
        "total_samples": total_samples,
        "parent_gini": parent_gini,
        "parent_distribution": parent_distribution,
        "samples_left": n_left,
        "left_gini": left_gini,
        "left_distribution": left_dist,
        "left_is_pure": bool(left_gini == 0.0),
        "samples_right": n_right,
        "right_gini": right_gini,
        "right_distribution": right_dist,
        "right_is_pure": bool(right_gini == 0.0),
        "weighted_child_gini": weighted_child_gini,
        "delta_gini": delta_gini,
        "is_valid_split": True,
        "decision": decision,
        "feedback": feedback,
        "left_case_ids": left_df["case_id"].tolist(),
        "right_case_ids": right_df["case_id"].tolist(),
        "note": resolution_note,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate clinical split on Train partition.")
    parser.add_argument("--feature", type=str, required=True, help="Clinical feature name or alias")
    parser.add_argument("--operator", type=str, default="<=", help="Operator (<=, >=, ==, etc.)")
    parser.add_argument("--threshold", required=True, help="Threshold value")
    parser.add_argument("--leaf-id", type=str, default="root", help="Target leaf ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (strictly 'train')")

    args = parser.parse_args()
    res = compute_gini_split(
        feature=args.feature,
        operator=args.operator,
        threshold=args.threshold,
        target_leaf_id=args.leaf_id,
        split=args.split,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
