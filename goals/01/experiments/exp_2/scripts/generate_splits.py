"""exp_2: Clean Cohort Selection & Rigorous Validation Protocol (MCCV + LOOCV).

Regenerates the deterministic validation-split artifact
``data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`` (195 x 56) from
``inputs.csv`` and ``ground_truth.csv`` following ``experiments/exp_2/DESIGN.md``:

- Cohort status per case (``usable_labeled`` / ``unlabeled_test`` /
  ``excluded_missing_mri`` / ``excluded_missing_pirads``).
- 50 stratified MCCV splits (80% train / 20% val, seed=42).
- 88 LOOCV fold indices (0..87, sorted lexicographically by ``case_id``).

Every rule is validated with hard assertions against the DESIGN expectations
and against the source matrices. The script is deterministic: running it twice
in the same process must produce byte-identical split columns.

Usage:
    python3 experiments/exp_2/scripts/generate_splits.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("exp_2")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # experiments/exp_2/scripts/ -> project root
DATA_DIR = PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1"
EXP2_DIR = PROJECT_ROOT / "experiments" / "exp_2"
RESULTS_DIR = EXP2_DIR / "results"

INPUTS_PATH = DATA_DIR / "inputs.csv"
GT_PATH = DATA_DIR / "ground_truth.csv"
OUTPUT_PATH = DATA_DIR / "mccv_loocv_splits.csv"
REPORT_PATH = RESULTS_DIR / "validation_report.json"
MANIFEST_PATH = RESULTS_DIR / "data_manifest.csv"
GIT_COMMIT_PATH = RESULTS_DIR / "git_commit.txt"

# ---------------------------------------------------------------------------
# Protocol constants (experiments/exp_2/DESIGN.md)
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 50
TEST_SIZE = 0.2
TARGET = "target_biopsy_decision_binary"
MRI_PREFIX = "mri_emb_"
PIRADS_COL = "cli_pirads"
GT_TARGET = "target_biopsy_decision"

EXPECTED_COUNTS = {
    "usable_labeled": 88,
    "unlabeled_test": 102,
    "excluded_missing_mri": 4,
    "excluded_missing_pirads": 1,
}
EXPECTED_MISSING_MRI = {
    "PT-pseudo_4bfd4ec864d8",
    "PT-pseudo_4d54f04e26ae",
    "PT-pseudo_7dbdcd6f9064",
    "PT-pseudo_8636aa471ef7",
}
EXPECTED_MISSING_PIRADS = {"PT-pseudo_3646e0a2ae13"}
EXPECTED_IO_SHAPES = {INPUTS_PATH.name: (195, 1077), GT_PATH.name: (195, 27)}
EXPECTED_OUT_SHAPE = (195, 56)
EXPECTED_CLASSES = {0.0: 34, 1.0: 54}  # usable_labeled balance (yes=54 / no=34)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(INPUTS_PATH, keep_default_na=False, na_values=["", "NA"])
    gt = pd.read_csv(GT_PATH, keep_default_na=False, na_values=["", "NA"])
    return inputs, gt


def compute_flags(inputs: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if inputs["case_id"].duplicated().any():
        raise ValueError("inputs.csv contains duplicated case_id")
    if gt["case_id"].duplicated().any():
        raise ValueError("ground_truth.csv contains duplicated case_id")

    mri_cols = [c for c in inputs.columns if c.startswith(MRI_PREFIX)]
    if len(mri_cols) != 1024:
        raise ValueError(f"expected 1024 mri_emb_* columns, found {len(mri_cols)}")

    meta = inputs[["case_id"]].merge(gt[["case_id", GT_TARGET]], on="case_id", how="left")
    meta["has_gt"] = (meta[GT_TARGET].notna()).astype(int)
    meta["has_mri"] = (~inputs[mri_cols].isna().all(axis=1)).astype(int)
    meta["has_pirads"] = (~inputs[PIRADS_COL].isna()).astype(int)
    meta.drop(columns=[GT_TARGET], inplace=True)

    status = []
    for _, row in meta.iterrows():
        if row["has_mri"] == 0:
            status.append("excluded_missing_mri")
        elif row["has_pirads"] == 0:
            status.append("excluded_missing_pirads")
        elif row["has_gt"] == 0:
            status.append("unlabeled_test")
        else:
            status.append("usable_labeled")
    meta["cohort_status"] = status
    return meta


def generate_mccv_splits(meta: pd.DataFrame, gt: pd.DataFrame) -> dict[str, np.ndarray]:
    usable = meta[meta["cohort_status"] == "usable_labeled"]["case_id"].tolist()
    usable = sorted(usable)
    y = gt.set_index("case_id").loc[usable, TARGET].astype(float).to_numpy()
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("usable_labeled cohort target must be binary 0/1")

    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    splits: dict[str, np.ndarray] = {}
    arrays: list[np.ndarray] = []
    for _i, (_, val_idx) in enumerate(sss.split(np.zeros(len(usable)), y)):
        assignment = np.zeros(len(usable), dtype=np.int8)
        assignment[val_idx] = 1
        arrays.append(assignment)
    arrays = np.array(arrays)  # (50, 88)

    for k in range(N_SPLITS):
        splits[f"mccv_split_{k:02d}"] = arrays[k]
    return splits, usable


def build_output(meta: pd.DataFrame, splits: dict[str, np.ndarray], usable: list[str]) -> pd.DataFrame:
    out = meta.copy()
    out = out.set_index("case_id")

    fold = pd.Series(index=out.index, data=-1, dtype=np.int64)
    fold[usable] = np.arange(len(usable), dtype=np.int64)
    out["loocv_fold"] = fold

    for name, arr in splits.items():
        s = pd.Series(index=out.index, data=-1, dtype=np.int8)
        s[usable] = arr
        out[name] = s

    col_order = ["case_id", "cohort_status", "has_gt", "has_mri", "has_pirads", "loocv_fold"] + list(splits.keys())
    out = out.reset_index().sort_values("case_id").reset_index(drop=True)
    return out[col_order]


def validate(out: pd.DataFrame, meta: pd.DataFrame, usable: list[str], gt: pd.DataFrame) -> dict:
    checks: dict = {}

    # 1. Shape & columns
    checks["shape"] = {
        "actual": list(out.shape),
        "expected": list(EXPECTED_OUT_SHAPE),
        "passed": out.shape == EXPECTED_OUT_SHAPE,
    }

    # 2. Cohort counts
    counts = out["cohort_status"].value_counts().to_dict()
    checks["cohort_counts"] = {
        "actual": counts,
        "expected": EXPECTED_COUNTS,
        "passed": counts == EXPECTED_COUNTS,
    }

    # 3. Excluded case sets match DESIGN
    missing_mri = set(out.loc[out["cohort_status"] == "excluded_missing_mri", "case_id"])
    missing_pirads = set(out.loc[out["cohort_status"] == "excluded_missing_pirads", "case_id"])
    checks["exclusion_sets"] = {
        "missing_mri_actual": sorted(missing_mri),
        "missing_mri_expected": sorted(EXPECTED_MISSING_MRI),
        "missing_pirads_actual": sorted(missing_pirads),
        "missing_pirads_expected": sorted(EXPECTED_MISSING_PIRADS),
        "passed": missing_mri == EXPECTED_MISSING_MRI and missing_pirads == EXPECTED_MISSING_PIRADS,
    }

    # 4. LOOCV fold integrity
    u = out[out["cohort_status"] == "usable_labeled"]
    checks["loocv"] = {
        "min": int(u["loocv_fold"].min()),
        "max": int(u["loocv_fold"].max()),
        "unique": int(u["loocv_fold"].nunique()),
        "excluded_are_minus_one": bool((out[out["cohort_status"] != "usable_labeled"]["loocv_fold"] == -1).all()),
        "passed": u["loocv_fold"].min() == 0 and u["loocv_fold"].max() == 87 and u["loocv_fold"].nunique() == 88,
    }

    # 5. MCCV split integrity (70/18, stratified, excluded=-1)
    y_map = gt.set_index("case_id")[TARGET].astype(float)
    u_y = u["case_id"].map(y_map)
    class_balance = u_y.value_counts().sort_index().astype(int).to_dict()
    per_split = {}
    split_ok = True
    for name in [c for c in out.columns if c.startswith("mccv_split_")]:
        tr_yes = int(((out[name] == 0) & (u_y == 1.0)).sum())
        tr_no = int(((out[name] == 0) & (u_y == 0.0)).sum())
        va_yes = int(((out[name] == 1) & (u_y == 1.0)).sum())
        va_no = int(((out[name] == 1) & (u_y == 0.0)).sum())
        ok = (tr_yes + tr_no == 70) and (va_yes + va_no == 18) and tr_yes > 0 and tr_no > 0 and va_yes > 0 and va_no > 0
        split_ok = split_ok and ok
        per_split[name] = {"train": {"yes": tr_yes, "no": tr_no}, "val": {"yes": va_yes, "no": va_no}, "passed": ok}
    excluded_all_minus_one = bool((out[out["cohort_status"] == "usable_labeled"][[c for c in out.columns if c.startswith("mccv_split_")]] != -1).all().all())
    excluded_val = bool((out[out["cohort_status"] != "usable_labeled"][[c for c in out.columns if c.startswith("mccv_split_")]] == -1).all().all())
    checks["mccv"] = {
        "class_balance_usable": class_balance,
        "expected_class_balance": {str(k): v for k, v in EXPECTED_CLASSES.items()},
        "per_split": per_split,
        "all_splits_70_18_stratified": split_ok,
        "usable_no_minus_one": excluded_all_minus_one,
        "excluded_all_minus_one": excluded_val,
        "passed": split_ok and excluded_all_minus_one and excluded_val and class_balance == {0.0: 34, 1.0: 54},
    }

    # 6. Consistency with inputs.csv (source of truth)
    inputs = pd.read_csv(INPUTS_PATH, keep_default_na=False, na_values=["", "NA"])
    mri_cols = [c for c in inputs.columns if c.startswith(MRI_PREFIX)]
    inputs_u = inputs[inputs["case_id"].isin(u["case_id"])]
    consistency = {
        "case_ids_align_with_inputs": bool(set(out["case_id"]) == set(inputs["case_id"])),
        "usable_rows_have_mri": bool((~inputs_u[mri_cols].isna().all(axis=1)).all()),
        "usable_rows_have_pirads": bool((~inputs_u[PIRADS_COL].isna()).all()),
        "n_usable_with_gt": int((meta[meta["cohort_status"] == "usable_labeled"]["has_gt"] == 1).sum()),
    }
    consistency["passed"] = all(consistency.values())
    checks["consistency_with_inputs"] = consistency

    checks["all_passed"] = all(v.get("passed", False) for v in checks.values())
    return checks


def determinism_check(meta: pd.DataFrame, gt: pd.DataFrame) -> dict:
    splits1, usable1 = generate_mccv_splits(meta, gt)
    splits2, usable2 = generate_mccv_splits(meta, gt)
    identical = usable1 == usable2 and all(
        np.array_equal(splits1[k], splits2[k]) for k in splits1
    )
    return {"usable_identical": usable1 == usable2, "splits_identical": identical, "passed": identical}


def main() -> int:
    inputs, gt = load_sources()
    for name, expected in EXPECTED_IO_SHAPES.items():
        actual = (len(inputs), len(inputs.columns)) if name == "inputs.csv" else (len(gt), len(gt.columns))
        if actual != expected:
            raise ValueError(f"{name} shape {actual} != expected {expected}")
        log.info("source %s shape OK: %s", name, actual)

    meta = compute_flags(inputs, gt)
    splits, usable = generate_mccv_splits(meta, gt)
    det = determinism_check(meta, gt)
    log.info("determinism double-run identical: %s", det["passed"])

    out = build_output(meta, splits, usable)
    checks = validate(out, meta, usable, gt)
    report = {
        "protocol": {
            "seed": SEED,
            "n_mccv_splits": N_SPLITS,
            "mccv_train_frac": 1 - TEST_SIZE,
            "mccv_val_frac": TEST_SIZE,
            "n_loocv_folds": 88,
        },
        "determinism": det,
        "validations": checks,
    }
    if not checks["all_passed"]:
        log.error("validation failed")
        json.dump(report, open(REPORT_PATH, "w"), indent=2, ensure_ascii=False)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    report["outputs"] = {
        "splits_csv": str(OUTPUT_PATH),
        "validation_report": str(REPORT_PATH),
        "data_manifest": str(MANIFEST_PATH),
        "git_commit_txt": str(GIT_COMMIT_PATH),
    }
    json.dump(report, open(REPORT_PATH, "w"), indent=2, ensure_ascii=False)

    manifest = pd.DataFrame(
        [
            {"file": path.name, "path": str(path), "sha256": sha256(path)}
            for path in [INPUTS_PATH, GT_PATH, OUTPUT_PATH]
        ]
    )
    manifest.to_csv(MANIFEST_PATH, index=False)

    commit = current_git_commit()
    GIT_COMMIT_PATH.write_text(f"{commit}\n")

    log.info("wrote %s (%s)", OUTPUT_PATH, out.shape)
    log.info("wrote %s", REPORT_PATH)
    log.info("wrote %s", MANIFEST_PATH)
    log.info("git commit at execution: %s", commit)
    log.info("ALL VALIDATIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
