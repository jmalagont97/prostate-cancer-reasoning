"""Orchestration script for exp_1: Master Dataset Extraction and Validation.

Calls build_master_dataset.build_master_datasets() on the raw chimera26
case directories and writes outputs to data/chimera26/preprocessed/task1/.

Validates that the regenerated CSVs match the expected dimensions:
  - inputs.csv:       (195, 1077)
  - ground_truth.csv: (195, 27)

Usage:
    python3 experiments/exp_1/scripts/run_preprocessing.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: allow importing build_master_dataset from the same scripts/ dir
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # experiments/exp_1/scripts/ -> project root

sys.path.insert(0, str(SCRIPT_DIR))
from build_master_dataset import build_master_datasets  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_preprocessing")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR    = PROJECT_ROOT / "data" / "chimera26" / "raw" / "task1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

EXPECTED_INPUTS_SHAPE = (195, 1077)
EXPECTED_GT_SHAPE     = (195, 27)


def validate_dataframe(df: pd.DataFrame, name: str, expected_shape: tuple[int, int]) -> dict:
    """Run basic integrity checks and return a validation report dict."""
    report: dict = {"name": name, "shape": list(df.shape), "expected_shape": list(expected_shape), "passed": []}
    errors = []

    # Shape check
    if df.shape == expected_shape:
        report["passed"].append("shape_match")
    else:
        errors.append(f"Shape mismatch: got {df.shape}, expected {expected_shape}")

    # No collection columns (lists / dicts serialized as strings that contain '[' or '{' in header)
    collection_cols = [c for c in df.columns if c.startswith(("_list", "_dict"))]
    if not collection_cols:
        report["passed"].append("no_collection_columns")
    else:
        errors.append(f"Found collection columns: {collection_cols}")

    # case_id present and unique
    if "case_id" in df.columns:
        if df["case_id"].nunique() == len(df):
            report["passed"].append("unique_case_ids")
        else:
            errors.append("Duplicate case_id values detected")
    else:
        errors.append("Missing 'case_id' column")

    report["errors"] = errors
    report["all_passed"] = len(errors) == 0
    return report


def diff_report(df_new: pd.DataFrame, df_ref: pd.DataFrame, name: str) -> dict:
    """Compare new vs reference CSV: shape, column set, and value-level equality."""
    result = {"name": name}

    result["shapes_match"] = df_new.shape == df_ref.shape
    result["new_shape"]    = list(df_new.shape)
    result["ref_shape"]    = list(df_ref.shape)

    new_cols = set(df_new.columns)
    ref_cols = set(df_ref.columns)
    result["missing_cols"]  = sorted(ref_cols - new_cols)
    result["extra_cols"]    = sorted(new_cols - ref_cols)
    result["columns_match"] = (new_cols == ref_cols)

    if result["shapes_match"] and result["columns_match"]:
        # Numeric value comparison (NaN-safe)
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
        max_diff = 0.0
        for col in numeric_cols:
            diff = (df_new[col].fillna(-9999) - df_ref[col].fillna(-9999)).abs().max()
            if diff > max_diff:
                max_diff = diff
        result["max_numeric_diff"] = float(max_diff)
        result["values_identical"]  = max_diff < 1e-6
    else:
        result["values_identical"] = False
        result["max_numeric_diff"]  = None

    return result


def main() -> None:
    log.info("=" * 60)
    log.info("EXP-1  Master Dataset Extraction — Replicability Check")
    log.info("=" * 60)

    if not RAW_DIR.exists():
        log.error("Raw data directory not found: %s", RAW_DIR)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Build new datasets
    # ------------------------------------------------------------------
    log.info("Building datasets from %s …", RAW_DIR)
    df_inputs_new, df_gt_new = build_master_datasets(RAW_DIR)
    log.info("Done. inputs: %s | ground_truth: %s", df_inputs_new.shape, df_gt_new.shape)

    # ------------------------------------------------------------------
    # 2. Validate structure
    # ------------------------------------------------------------------
    val_inputs = validate_dataframe(df_inputs_new, "inputs",       EXPECTED_INPUTS_SHAPE)
    val_gt     = validate_dataframe(df_gt_new,     "ground_truth", EXPECTED_GT_SHAPE)

    for rep in [val_inputs, val_gt]:
        status = "✓ PASS" if rep["all_passed"] else "✗ FAIL"
        log.info("[%s] %s  shape=%s", status, rep["name"], rep["shape"])
        for err in rep.get("errors", []):
            log.error("   → %s", err)

    # ------------------------------------------------------------------
    # 3. Compare against existing reference CSVs
    # ------------------------------------------------------------------
    ref_inputs_path = OUTPUT_DIR / "inputs.csv"
    ref_gt_path     = OUTPUT_DIR / "ground_truth.csv"
    diff_results = {}

    if ref_inputs_path.exists() and ref_gt_path.exists():
        log.info("Reference CSVs found — running diff …")
        df_inputs_ref = pd.read_csv(ref_inputs_path, low_memory=False)
        df_gt_ref     = pd.read_csv(ref_gt_path,     low_memory=False)

        diff_inputs = diff_report(df_inputs_new, df_inputs_ref, "inputs")
        diff_gt     = diff_report(df_gt_new,     df_gt_ref,     "ground_truth")
        diff_results = {"inputs": diff_inputs, "ground_truth": diff_gt}

        for d in [diff_inputs, diff_gt]:
            identical = d["values_identical"]
            status = "✓ IDENTICAL" if identical else "⚠ DIFFERS"
            log.info("[%s] %s  shapes_match=%s | columns_match=%s | max_numeric_diff=%s",
                     status, d["name"], d["shapes_match"], d["columns_match"], d.get("max_numeric_diff"))
            if d["missing_cols"]:
                log.warning("   Missing cols: %s", d["missing_cols"])
            if d["extra_cols"]:
                log.warning("   Extra cols:   %s", d["extra_cols"])
    else:
        log.warning("No reference CSVs found — skipping diff.")

    # ------------------------------------------------------------------
    # 4. Overwrite outputs with fresh data
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_inputs_new.to_csv(ref_inputs_path, index=False)
    df_gt_new.to_csv(ref_gt_path, index=False)
    log.info("Outputs written to %s", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 5. Save validation report
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "validation": {"inputs": val_inputs, "ground_truth": val_gt},
        "diff":        diff_results,
    }
    report_path = RESULTS_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Validation report saved: %s", report_path)

    # ------------------------------------------------------------------
    # 6. Exit code
    # ------------------------------------------------------------------
    all_ok = val_inputs["all_passed"] and val_gt["all_passed"]
    if all_ok:
        log.info("=" * 60)
        log.info("✓ EXP-1 REPLICABILITY CHECK PASSED")
        log.info("=" * 60)
        sys.exit(0)
    else:
        log.error("=" * 60)
        log.error("✗ EXP-1 REPLICABILITY CHECK FAILED — see errors above")
        log.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
