#!/usr/bin/env python3
"""Deterministic raw-to-processed preprocessing for CHIMERA-Agent Task 3.

The script only reads data/raw/train_release/task3 by default and writes the
canonical processed representation to data/processed/task3. It does not
fit imputation, PCA, or a predictive model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_FILES = (
    "structured-prompt.json",
    "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
    "prostate-time-to-recurrence-or-last-follow-up.json",
    "prostate-modality-level-neural-representations.json",
)
MODALITIES = {
    "MRI image": (1, 1024),
    "Biopsy slide": (3, 960),
    "Prostatectomy slide": (3, 960),
}
FEATURE_COLUMNS = [
    "age", "psa", "dre_abnormal", "dre_bilateral", "dre_nodule",
    "dre_ct_stage_ordinal", "active_treatment_flag", "isup_grade_group",
    "gleason_primary", "gleason_secondary", "gleason_sum",
    "gleason_tertiary_present", "pt_stage_ordinal", "epe_present",
    "margins_positive", "svi_present", "lvi_present", "ln_assessed",
    "ln_positive",
]
LABEL_COLUMNS = ["case_id", "months_to_recurrence", "event"]
MANIFEST_COLUMNS = [
    "case_id", "all_expected_files_present", "clinical_status",
    "pathology_status", "label_status", "embedding_status", "n_mri_vectors",
    "n_biopsy_slides", "n_prostatectomy_slides", "error_count", "errors",
]

DRE_FINDING_RE = re.compile(
    r"Digital rectal examination (was abnormal bilaterally|was abnormal on the left|"
    r"was abnormal on the right|was abnormal|was normal|findings:)", re.I
)
STAGE_RE = re.compile(r"[pc]T[0-9][a-z]?", re.I)
ISUP_RE = re.compile(r"ISUP grade group (\d+)", re.I)
GLEASON_RE = re.compile(r"Gleason (\d+)\+(\d+)", re.I)
TERTIARY_RE = re.compile(r"tertiary pattern (\d+)", re.I)
PATH_STAGE_RE = re.compile(r"pathological stage (pT\w+)", re.I)
EPE_RE = re.compile(
    r"Extraprostatic extension was present|There was no extraprostatic extension", re.I
)
MARGINS_RE = re.compile(r"surgical margins were (positive|negative)", re.I)
SVI_RE = re.compile(r"the seminal vesicles were (not invaded|invaded)", re.I)
LVI_RE = re.compile(r"lymphovascular invasion was (present|absent)", re.I)
LN_RE = re.compile(
    r"lymph node metastasis was present|there was no lymph node metastasis|"
    r"no lymph nodes were removed", re.I
)


def json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stage_ordinal(value: str | None) -> float | None:
    if not value:
        return None
    match = re.fullmatch(r"[pc]T(\d)([a-z]?)", value.strip(), re.I)
    if not match:
        return None
    return float(int(match.group(1)) * 10 + {"": 0, "a": 1, "b": 2, "c": 3}.get(match.group(2).lower(), 0))


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def parse_dre(text: Any) -> dict[str, float | None]:
    if not isinstance(text, str) or not text.strip():
        return {"dre_abnormal": None, "dre_bilateral": None, "dre_nodule": None, "dre_ct_stage_ordinal": None}
    match = DRE_FINDING_RE.search(text)
    finding = match.group(1).lower() if match else None
    if finding == "was normal":
        abnormal, bilateral = 0.0, 0.0
    elif finding == "findings:":
        abnormal, bilateral = 1.0, 0.0
    elif finding:
        abnormal, bilateral = 1.0, float("bilaterally" in finding)
    else:
        abnormal, bilateral = None, None
    stage = STAGE_RE.search(text)
    return {
        "dre_abnormal": abnormal,
        "dre_bilateral": bilateral,
        "dre_nodule": float("nodule" in text.lower()),
        "dre_ct_stage_ordinal": stage_ordinal(stage.group(0)) if stage else None,
    }


def parse_active_treatment(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "no":
        return 0.0
    return 1.0


def parse_pathology(text: Any) -> tuple[dict[str, float | None], list[str]]:
    columns = [
        "isup_grade_group", "gleason_primary", "gleason_secondary", "gleason_sum",
        "gleason_tertiary_present", "pt_stage_ordinal", "epe_present",
        "margins_positive", "svi_present", "lvi_present", "ln_assessed", "ln_positive",
    ]
    result: dict[str, float | None] = {column: None for column in columns}
    errors: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return result, ["missing surgical_pathology_report"]

    match = ISUP_RE.search(text)
    result["isup_grade_group"] = as_float(match.group(1)) if match else None
    match = GLEASON_RE.search(text)
    if match:
        result["gleason_primary"] = as_float(match.group(1))
        result["gleason_secondary"] = as_float(match.group(2))
        result["gleason_sum"] = result["gleason_primary"] + result["gleason_secondary"]  # type: ignore[operator]
    result["gleason_tertiary_present"] = float(bool(TERTIARY_RE.search(text)))
    match = PATH_STAGE_RE.search(text)
    result["pt_stage_ordinal"] = stage_ordinal(match.group(1)) if match else None

    match = EPE_RE.search(text)
    if match:
        result["epe_present"] = float(match.group(0).lower().startswith("extraprostatic"))
    match = MARGINS_RE.search(text)
    if match:
        result["margins_positive"] = float(match.group(1).lower() == "positive")
    match = SVI_RE.search(text)
    if match:
        result["svi_present"] = float(match.group(1).lower() == "invaded")
    match = LVI_RE.search(text)
    if match:
        result["lvi_present"] = float(match.group(1).lower() == "present")
    match = LN_RE.search(text)
    if match:
        phrase = match.group(0).lower()
        if phrase == "no lymph nodes were removed":
            result["ln_assessed"], result["ln_positive"] = 0.0, None
        elif phrase == "lymph node metastasis was present":
            result["ln_assessed"], result["ln_positive"] = 1.0, 1.0
        else:
            result["ln_assessed"], result["ln_positive"] = 1.0, 0.0

    missing = [column for column in columns if result[column] is None]
    # When no nodes were removed, ln_positive is structurally not applicable,
    # not an extraction failure.
    if result["ln_assessed"] == 0.0 and "ln_positive" in missing:
        missing.remove("ln_positive")
    if missing:
        errors.append("unparsed pathology fields: " + ",".join(missing))
    return result, errors


def normalize_embeddings(raw: Any) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    errors: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    if not isinstance(raw, dict):
        return {}, {}, ["embedding payload is not an object"]
    for modality, (_, dimension) in MODALITIES.items():
        value = raw.get(modality, [])
        if value is None:
            value = []
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            errors.append(f"{modality}: non-numeric values")
            array = np.empty((0, dimension), dtype=np.float32)
        if array.size == 0:
            array = np.empty((0, dimension), dtype=np.float32)
        elif array.ndim == 1 and modality == "MRI image":
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != dimension:
            errors.append(f"{modality}: expected (*,{dimension}), got {tuple(array.shape)}")
            array = np.empty((0, dimension), dtype=np.float32)
        if not np.isfinite(array).all():
            errors.append(f"{modality}: non-finite values")
        arrays[modality] = array
        counts[modality] = int(array.shape[0])
    return arrays, counts, errors


def git_version(root: Path) -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        return f"commit={commit}\nworking_tree={'clean' if not status else 'modified'}\n"
    except (OSError, subprocess.CalledProcessError):
        return "git_version=unavailable\n"


def discover(root: Path) -> list[Path]:
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def build_records(case_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    features: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    embedding_rows = {name: [] for name in MODALITIES}

    for case_dir in case_dirs:
        case_id = case_dir.name
        errors: list[str] = []
        payloads: dict[str, Any] = {}
        present = True
        for filename in EXPECTED_FILES:
            path = case_dir / filename
            if not path.exists():
                present = False
                errors.append(f"missing {filename}")
                continue
            try:
                payloads[filename] = json_load(path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid {filename}: {exc}")

        structured = payloads.get("structured-prompt.json", {})
        clinical = payloads.get("prostate-time-to-recurrence-or-last-follow-up-clinical-data.json", {})
        label = payloads.get("prostate-time-to-recurrence-or-last-follow-up.json", {})
        raw_embeddings = payloads.get("prostate-modality-level-neural-representations.json", {})
        if not isinstance(structured, dict): structured = {}
        if not isinstance(clinical, dict): clinical = {}
        if not isinstance(label, dict): label = {}

        row: dict[str, Any] = {"case_id": case_id}
        row["age"] = as_float(structured.get("age"))
        row["psa"] = as_float(structured.get("psa"))
        row.update(parse_dre(structured.get("dre")))
        row["active_treatment_flag"] = parse_active_treatment(structured.get("active_treatment_prior_to_surgery"))
        pathology, pathology_errors = parse_pathology(clinical.get("surgical_pathology_report"))
        row.update(pathology)
        features.append(row)

        event = label.get("event")
        months = as_float(label.get("months_to_recurrence"))
        label_errors: list[str] = []
        if months is None or months < 0: label_errors.append("invalid months_to_recurrence")
        if event not in (0, 1): label_errors.append("invalid event")
        labels.append({"case_id": case_id, "months_to_recurrence": months, "event": event if event in (0, 1) else None})

        arrays, counts, embedding_errors = normalize_embeddings(raw_embeddings)
        for modality in MODALITIES:
            embedding_rows[modality].append(arrays[modality])
        errors.extend(pathology_errors + label_errors + embedding_errors)
        manifests.append({
            "case_id": case_id,
            "all_expected_files_present": int(present),
            "clinical_status": "ok" if not errors else "review",
            "pathology_status": "ok" if not pathology_errors else "review",
            "label_status": "ok" if not label_errors else "review",
            "embedding_status": "ok" if not embedding_errors else "review",
            "n_mri_vectors": counts.get("MRI image", 0),
            "n_biopsy_slides": counts.get("Biopsy slide", 0),
            "n_prostatectomy_slides": counts.get("Prostatectomy slide", 0),
            "error_count": len(errors),
            "errors": "; ".join(errors),
        })
    return features, labels, manifests, embedding_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(root: Path, report_root: Path, features: list[dict[str, Any]], labels: list[dict[str, Any]], manifests: list[dict[str, Any]], embedding_rows: dict[str, list[np.ndarray]], repo_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report_dir = report_root / "full_preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(root / "clinical_pathology_features.csv", features, ["case_id"] + FEATURE_COLUMNS)
    write_csv(root / "survival_labels.csv", labels, LABEL_COLUMNS)
    write_csv(root / "case_manifest.csv", manifests, MANIFEST_COLUMNS)

    n = len(features)
    def pad(name: str, dimension: int) -> np.ndarray:
        padded = np.full((n, MODALITIES[name][0], dimension), np.nan, dtype=np.float32)
        for index, array in enumerate(embedding_rows[name]):
            if len(array): padded[index, :len(array), :] = array
        return padded
    counts = np.asarray([[len(row) for row in embedding_rows[name]] for name in MODALITIES], dtype=np.int32)
    np.savez_compressed(
        root / "embeddings.npz",
        case_ids=np.asarray([row["case_id"] for row in features]),
        mri=pad("MRI image", 1024),
        biopsy=pad("Biopsy slide", 960),
        prostatectomy=pad("Prostatectomy slide", 960),
        mri_counts=counts[0], biopsy_counts=counts[1], prostatectomy_counts=counts[2],
    )
    schema = {
        "schema_version": "task3-preprocessed-v1",
        "source": "data/raw/train_release/task3",
        "case_id_order": "lexicographic",
        "features": FEATURE_COLUMNS,
        "labels": LABEL_COLUMNS,
        "embeddings": {"mri": [n, 1, 1024], "biopsy": [n, 3, 960], "prostatectomy": [n, 3, 960]},
        "padding": "NaN; use *_counts to recover valid rows",
        "pca_or_imputation_fitted": False,
    }
    (root / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    summary = {
        "n_cases": n,
        "n_events": sum(row["event"] == 1 for row in labels),
        "n_censored": sum(row["event"] == 0 for row in labels),
        "n_review": sum(row["error_count"] > 0 for row in manifests),
        "source": str(root.parent.parent / "raw" / "train_release" / "task3"),
        "output": str(root),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "version.txt").write_text(git_version(repo_root), encoding="utf-8")
    (report_dir / "manifest.csv").write_text((root / "case_manifest.csv").read_text(encoding="utf-8"), encoding="utf-8")


def validate(root: Path, report_root: Path) -> int:
    required = ["clinical_pathology_features.csv", "survival_labels.csv", "embeddings.npz", "case_manifest.csv", "schema.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        print("MISSING: " + ", ".join(missing))
        return 1
    with (root / "survival_labels.csv").open(encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    with (root / "clinical_pathology_features.csv").open(encoding="utf-8") as handle:
        features = list(csv.DictReader(handle))
    with (root / "case_manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    ids = [row["case_id"] for row in features]
    if ids != sorted(ids) or ids != [row["case_id"] for row in labels] or ids != [row["case_id"] for row in manifest]:
        print("FAIL: case_id alignment/order")
        return 1
    embedding_case_ids = set(np.load(root / "embeddings.npz")["case_ids"].astype(str).tolist())
    if set(ids) != embedding_case_ids:
        print("FAIL: embedding case_id alignment")
        return 1
    if any(row["event"] not in {"0", "1"} for row in labels):
        print("FAIL: invalid event")
        return 1
    if any(float(row["months_to_recurrence"]) < 0 for row in labels):
        print("FAIL: negative time")
        return 1
    if any(column in features[0] for column in ("months_to_recurrence", "event")):
        print("FAIL: label leakage into features")
        return 1
    print(f"VALID: {len(ids)} cases, {sum(row['event'] == '1' for row in labels)} events, {sum(int(row['error_count']) > 0 for row in manifest)} cases requiring review")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "extract", "validate", "all"), nargs="?", default="all")
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/train_release/task3"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/task3"))
    parser.add_argument("--report-root", type=Path, default=Path("experiments/exp_11/results"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    input_root = args.input_root if args.input_root.is_absolute() else repo_root / args.input_root
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    report_root = args.report_root if args.report_root.is_absolute() else repo_root / args.report_root
    case_dirs = discover(input_root)
    if not case_dirs:
        raise SystemExit(f"No case directories found under {input_root}")
    if args.mode == "inventory":
        for case in case_dirs: print(case.name)
        return 0
    if args.mode in ("extract", "all"):
        features, labels, manifests, embedding_rows = build_records(case_dirs)
        write_outputs(output_root, report_root, features, labels, manifests, embedding_rows, repo_root)
        print(f"WROTE: {len(features)} cases to {output_root}")
    if args.mode in ("validate", "all"):
        return validate(output_root, report_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
