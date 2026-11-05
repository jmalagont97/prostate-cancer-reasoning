"""exp_3: Multimodal Input Reorganisation (main_tabular / narrative / images).

Regenerates, deterministically, three input-view CSVs in
``data/chimera26/preprocessed/task1/`` that reorganise ``inputs.csv`` for the
three-model pipeline (biopsy decision, clinical confidence, clinical relevance):

- ``main_tabular.csv``        -> case_id + every tabular variable of ``main``
  (all ``cli_*`` INCLUDING ``cli_fh_binary`` and ``cli_comorbidity_count``,
  all ``vit_*``, all ``path_hist_*``). No ``txt_*`` columns.
- ``full_prompt_narrative.csv``-> case_id + ``txt_full_prompt_narrative``
  (visible text from ``structured-prompt.json``).
- ``images.csv``              -> case_id + ``mri_emb_0`` … ``mri_emb_1023``.

All other ``inputs.csv`` information is intentionally ignored for this
experiment (psa_trend, labs, remaining txt_*, consolidated narrative).

Hard assertions enforce:
  1. every view holds exactly the 195 source cases, ``case_id`` unique;
  2. all views share the same case set;
  3. the 10 official relevance variables (age, fh, cspca, pirads, vol, psa,
     comorbidity, psad, dre, bx) are all present in ``main_tabular.csv``;
  4. no feature is duplicated across views;
  5. no ``ground_truth.csv`` column leaks into any view;
  6. values are byte-faithful to the source (CSV round-trip equality);
  7. the 4 MRI-missing cases keep all-``NaN`` rows in ``images.csv``;
  8. generation is deterministic (double-run identical);
  9. ``ground_truth.csv`` and ``mccv_loocv_splits.csv`` are byte-unchanged.

Usage:
    python3 experiments/exp_3/scripts/build_multimodal_views.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("exp_3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # experiments/exp_3/scripts/ -> project root
DATA_DIR = PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1"
EXP3_DIR = PROJECT_ROOT / "experiments" / "exp_3"
RESULTS_DIR = EXP3_DIR / "results"

INPUTS_PATH = DATA_DIR / "inputs.csv"
GT_PATH = DATA_DIR / "ground_truth.csv"
SPLITS_PATH = DATA_DIR / "mccv_loocv_splits.csv"

REPORT_PATH = RESULTS_DIR / "validation_report.json"
MANIFEST_PATH = RESULTS_DIR / "data_manifest.csv"
GIT_COMMIT_PATH = RESULTS_DIR / "git_commit.txt"

N_CASES = 195
N_MRI_MISSING = 4
EXPECTED_MISSING_MRI = {
    "PT-pseudo_4bfd4ec864d8",
    "PT-pseudo_4d54f04e26ae",
    "PT-pseudo_7dbdcd6f9064",
    "PT-pseudo_8636aa471ef7",
}

# Official relevance variables -> source columns in inputs.csv
RELEVANCE_VARIABLES = {
    "age": "cli_age",
    "fh": "cli_fh_binary",
    "cspca": "cli_cspca",
    "pirads": "cli_pirads",
    "vol": "cli_vol",
    "psa": "cli_psa",
    "comorbidity": "cli_comorbidity_count",
    "psad": "cli_psad",
    "dre": "cli_dre",
    "bx": "cli_bx",
}

# Expected shapes per view
EXPECTED_SHAPES = {
    "main_tabular": (195, 28),
    "full_prompt_narrative": (195, 2),
    "images": (195, 1025),
}


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


def build_views(inputs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {view_name: DataFrame} with case_id first, in source order."""
    cli = [c for c in inputs.columns if c.startswith("cli_")]
    vit = [c for c in inputs.columns if c.startswith("vit_")]
    ph = [c for c in inputs.columns if c.startswith("path_hist_")]
    mri = [c for c in inputs.columns if c.startswith("mri_emb_")]

    if len(cli) != 15:
        raise ValueError(f"expected 15 cli_* columns, found {len(cli)}")
    if len(vit) != 8:
        raise ValueError(f"expected 8 vit_* columns, found {len(vit)}")
    if len(ph) != 4:
        raise ValueError(f"expected 4 path_hist_* columns, found {len(ph)}")
    if len(mri) != 1024:
        raise ValueError(f"expected 1024 mri_emb_* columns, found {len(mri)}")

    views = {
        "main_tabular": inputs[["case_id"] + cli + vit + ph].copy(),
        "full_prompt_narrative": inputs[["case_id", "txt_full_prompt_narrative"]].copy(),
        "images": inputs[["case_id"] + mri].copy(),
    }
    return views


def validate(inputs: pd.DataFrame, gt: pd.DataFrame, views: dict[str, pd.DataFrame]) -> dict:
    checks: dict = {}

    # 1. Row count + unique case_id per view
    checks["rows_and_unique_ids"] = {
        name: {
            "n_rows": int(len(df)),
            "expected_rows": N_CASES,
            "case_id_unique": bool(df["case_id"].is_unique),
            "passed": len(df) == N_CASES and df["case_id"].is_unique,
        }
        for name, df in views.items()
    }

    # 2. Same case set across views
    case_sets = {name: set(df["case_id"]) for name, df in views.items()}
    ref_set = case_sets[next(iter(case_sets))]
    checks["shared_case_set"] = {
        "all_same": all(cs == ref_set for cs in case_sets.values()),
        "n_cases": len(ref_set),
    }

    # 3. The 10 official relevance variables present in main_tabular
    main_cols = set(views["main_tabular"].columns)
    missing = {k: v for k, v in RELEVANCE_VARIABLES.items() if v not in main_cols}
    checks["ten_relevance_variables_present"] = {
        "expected": sorted(RELEVANCE_VARIABLES),
        "missing": sorted(missing),
        "passed": not missing,
    }

    # 4. No feature duplicated across views
    seen: dict[str, str] = {}
    dups = []
    for name, df in views.items():
        for c in df.columns:
            if c == "case_id":
                continue
            if c in seen:
                dups.append((c, seen[c], name))
            else:
                seen[c] = name
    checks["no_duplicate_features"] = {"duplicates": dups, "passed": not dups}

    # 5. No ground_truth column leaks into any view
    gt_cols = set(gt.columns) - {"case_id"}
    leaked = {name: sorted(set(df.columns) & gt_cols)
              for name, df in views.items()}
    checks["no_gt_leak"] = {
        "leaked_columns": {k: v for k, v in leaked.items() if v},
        "passed": all(not v for v in leaked.values()),
    }

    # 6. Round-trip faithfulness per view
    roundtrip = {}
    for name, df in views.items():
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        back = pd.read_csv(io.StringIO(buf.getvalue()))
        try:
            pd.testing.assert_frame_equal(df, back)
            roundtrip[name] = {"passed": True, "shape": list(df.shape)}
        except AssertionError as exc:  # pragma: no cover
            roundtrip[name] = {"passed": False, "error": str(exc)[:300]}
    checks["round_trip_faithful"] = roundtrip

    # 7. MRI-missing cases are all-NaN in images view
    img = views["images"]
    mri_cols = [c for c in img.columns if c.startswith("mri_emb_")]
    all_nan = set(img.loc[img[mri_cols].isna().all(axis=1), "case_id"])
    checks["images_missing_mri"] = {
        "n_all_nan_rows": len(all_nan),
        "expected_n": N_MRI_MISSING,
        "all_nan_cases": sorted(all_nan),
        "expected_cases": sorted(EXPECTED_MISSING_MRI),
        "passed": len(all_nan) == N_MRI_MISSING and all_nan == EXPECTED_MISSING_MRI,
    }

    # 8. Expected shapes
    checks["expected_shapes"] = {
        name: {"actual": list(df.shape), "expected": list(EXPECTED_SHAPES[name]),
               "passed": df.shape == EXPECTED_SHAPES[name]}
        for name, df in views.items()
    }

    checks["all_passed"] = (
        all(v["passed"] for v in checks["rows_and_unique_ids"].values())
        and bool(checks["shared_case_set"]["all_same"])
        and bool(checks["ten_relevance_variables_present"]["passed"])
        and bool(checks["no_duplicate_features"]["passed"])
        and bool(checks["no_gt_leak"]["passed"])
        and all(v["passed"] for v in roundtrip.values())
        and bool(checks["images_missing_mri"]["passed"])
        and all(v["passed"] for v in checks["expected_shapes"].values())
    )
    return checks


def determinism_check(inputs: pd.DataFrame) -> dict:
    v1 = build_views(inputs)
    v2 = build_views(inputs)
    identical = list(v1) == list(v2) and all(
        pd.testing.assert_frame_equal(v1[k], v2[k]) is None for k in v1
    )
    return {"views_identical": identical, "passed": identical}


def main() -> int:
    log.info("=" * 60)
    log.info("EXP-3  Multimodal Input Reorganisation (3 views)")
    log.info("=" * 60)

    inputs = pd.read_csv(INPUTS_PATH)
    gt = pd.read_csv(GT_PATH)
    if inputs.shape != (195, 1077):
        raise ValueError(f"inputs.csv shape {inputs.shape} != (195, 1077)")
    if gt.shape != (195, 27):
        raise ValueError(f"ground_truth.csv shape {gt.shape} != (195, 27)")

    before_gt = sha256(GT_PATH)
    before_splits = sha256(SPLITS_PATH)

    views = build_views(inputs)
    det = determinism_check(inputs)
    log.info("determinism double-run identical: %s", det["passed"])

    checks = validate(inputs, gt, views)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in views.items():
        out_path = DATA_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)
        log.info("wrote %s  shape=%s", out_path.name, df.shape)

    after_gt = sha256(GT_PATH)
    after_splits = sha256(SPLITS_PATH)
    checks["untouched_artifacts"] = {
        "ground_truth_unchanged": before_gt == after_gt,
        "mccv_loocv_splits_unchanged": before_splits == after_splits,
    }

    all_passed = (
        det["passed"]
        and checks["all_passed"]
        and checks["untouched_artifacts"]["ground_truth_unchanged"]
        and checks["untouched_artifacts"]["mccv_loocv_splits_unchanged"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "views": {name: {"shape": list(df.shape)} for name, df in views.items()},
        "relevance_variables": RELEVANCE_VARIABLES,
        "determinism": det,
        "validations": checks,
        "outputs": {
            "views": [str(p) for p in written],
            "validation_report": str(REPORT_PATH),
            "data_manifest": str(MANIFEST_PATH),
            "git_commit_txt": str(GIT_COMMIT_PATH),
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    file_hashes = pd.DataFrame(
        [
            {"file": p.name, "path": str(p), "sha256": sha256(p)}
            for p in [INPUTS_PATH, GT_PATH, SPLITS_PATH] + written
        ]
    )
    file_hashes.to_csv(MANIFEST_PATH, index=False)
    GIT_COMMIT_PATH.write_text(f"{current_git_commit()}\n")

    if all_passed:
        log.info("=" * 60)
        log.info("✓ EXP-3 ALL VALIDATIONS PASSED")
        log.info("=" * 60)
        return 0
    log.error("✗ EXP-3 VALIDATION FAILED — see %s", REPORT_PATH)
    return 1


if __name__ == "__main__":
    sys.exit(main())
