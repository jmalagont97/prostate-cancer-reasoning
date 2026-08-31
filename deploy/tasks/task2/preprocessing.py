"""
Preprocessing and Ingestion Utilities for Task 2 (CHIMERA-Agent Challenge).

Extracts tabular variables, text summaries, and MRI embeddings from raw patient folders
matching the structure of data/raw/train_release/task2/T2-XXX/.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Canonical column names matching data/processed/task2/clinical_tabular_task2.csv
T2_NUMERICAL_COLS = [
    "age",
    "psa",
    "months",
    "psad",
    "psav",
    "psap",
    "vol",
    "cspca",
    "bx_isup",
    "bx_gl_prim",
    "bx_gl_sec",
    "pirads",
    "vitals_weight_kg",
    "vitals_height_cm",
    "vitals_bmi",
    "vitals_bp_sys",
    "vitals_bp_dia",
    "vitals_hr_bpm",
    "smoking_pack_years",
    "ipss_score",
    "allergies_count",
    "comorb_diabetes",
    "comorb_hypertension",
    "comorb_cardiovascular",
    "comorb_respiratory",
    "comorb_renal",
    "comorb_surgery",
    "comorb_cancer_other",
    "comorbidity_count",
]

T2_CATEGORICAL_COLS = [
    "dre",
    "bx",
    "ct",
]


def parse_numeric(val: Any) -> Optional[float]:
    """Safely parse float or int from string/number."""
    if val is None or val == "" or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r"[-+]?\d*\.?\d+", val)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return np.nan
    return np.nan


def parse_bp(bp_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse blood pressure string like '125/84 mmHg' into (systolic, diastolic)."""
    if not bp_str or not isinstance(bp_str, str):
        return np.nan, np.nan
    match = re.search(r"(\d+)\s*/\s*(\d+)", bp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def extract_raw_patient_payload_task2(
    raw_input: Union[str, Path, Dict[str, Any]]
) -> Tuple[str, Dict[str, Any], str, np.ndarray]:
    """
    Extracts tabular features, clinical text narrative, and MRI embedding from a raw Task 2 patient directory or dict.
    Returns:
      (case_id, tabular_features_dict, clinical_text_narrative, mri_embedding_1024d)
    """
    case_id = "unknown_case"
    sp: Dict[str, Any] = {}
    mri_vec = np.zeros(1024, dtype=float)

    if isinstance(raw_input, (str, Path)):
        p_dir = Path(raw_input)
        case_id = p_dir.name

        sp_file = p_dir / "structured-prompt.json"
        if sp_file.exists():
            with open(sp_file, "r", encoding="utf-8") as f:
                sp = json.load(f)

        emb_file = p_dir / "prostate-modality-level-neural-representations.json"
        if emb_file.exists():
            try:
                with open(emb_file, "r", encoding="utf-8") as f:
                    emb_data = json.load(f)
                mri_list = emb_data.get("MRI image", emb_data.get("mri", []))
                if mri_list and len(mri_list) > 0 and len(mri_list[0]) == 1024:
                    mri_vec = np.array(mri_list[0], dtype=float)
            except Exception:
                pass
    elif isinstance(raw_input, dict):
        case_id = str(raw_input.get("case_id", raw_input.get("pid", "unknown_case")))
        sp = raw_input.get("structured_prompt", raw_input.get("patient", raw_input))
        mri_list = raw_input.get("mri_representation", raw_input.get("mri", raw_input.get("mri_image", [])))
        if mri_list is not None and len(mri_list) > 0:
            if isinstance(mri_list[0], list):
                mri_vec = np.array(mri_list[0], dtype=float)
            else:
                mri_vec = np.array(mri_list, dtype=float)

    # 1. Parse Tabular Features
    features: Dict[str, Any] = {"pid": case_id}

    demographics = sp.get("demographics", {})
    features["age"] = parse_numeric(demographics.get("age") or demographics.get("patient_age"))
    features["vitals_weight_kg"] = parse_numeric(demographics.get("weight") or demographics.get("weight_kg"))
    features["vitals_height_cm"] = parse_numeric(demographics.get("height") or demographics.get("height_cm"))
    features["vitals_bmi"] = parse_numeric(demographics.get("bmi"))

    vitals = sp.get("vitals", {})
    sys_bp, dia_bp = parse_bp(vitals.get("blood_pressure") or vitals.get("bp"))
    features["vitals_bp_sys"] = sys_bp
    features["vitals_bp_dia"] = dia_bp
    features["vitals_hr_bpm"] = parse_numeric(vitals.get("pulse") or vitals.get("heart_rate") or vitals.get("hr"))

    hist = sp.get("history", {})
    features["smoking_pack_years"] = parse_numeric(hist.get("smoking_pack_years") or hist.get("pack_years") or 0.0)
    features["ipss_score"] = parse_numeric(hist.get("ipss_score") or hist.get("ipss") or np.nan)
    features["allergies_count"] = parse_numeric(len(hist.get("allergies", [])) if isinstance(hist.get("allergies"), list) else 0.0)

    comorbs = str(hist.get("comorbidities", "")).lower()
    features["comorb_diabetes"] = 1 if "diabet" in comorbs else 0
    features["comorb_hypertension"] = 1 if "hyperten" in comorbs or "hta" in comorbs else 0
    features["comorb_cardiovascular"] = 1 if any(k in comorbs for k in ["cardio", "coronar", "infarct", "heart"]) else 0
    features["comorb_respiratory"] = 1 if any(k in comorbs for k in ["copd", "epoc", "asthma", "respirat"]) else 0
    features["comorb_renal"] = 1 if any(k in comorbs for k in ["renal", "kidney", "nephro"]) else 0
    features["comorb_surgery"] = 1 if any(k in comorbs for k in ["surg", "cirug", "operat"]) else 0
    features["comorb_cancer_other"] = 1 if any(k in comorbs for k in ["cancer", "neoplas", "tumor"]) and "prostat" not in comorbs else 0
    features["comorbidity_count"] = sum([
        features["comorb_diabetes"],
        features["comorb_hypertension"],
        features["comorb_cardiovascular"],
        features["comorb_respiratory"],
        features["comorb_renal"],
        features["comorb_surgery"],
        features["comorb_cancer_other"],
    ])

    exam = sp.get("physical_exam", {})
    dre = str(exam.get("dre_finding", "")).lower()
    if "abnormal" in dre or "nodule" in dre or "induration" in dre:
        features["dre"] = "abnormal"
    elif "normal" in dre:
        features["dre"] = "normal"
    else:
        features["dre"] = "not_performed"

    labs = sp.get("laboratory", {})
    features["psa"] = parse_numeric(labs.get("psa") or labs.get("total_psa"))
    features["months"] = parse_numeric(labs.get("months") or 0.0)
    features["psad"] = parse_numeric(labs.get("psa_density"))
    features["psav"] = parse_numeric(labs.get("psa_velocity"))
    features["psap"] = parse_numeric(labs.get("free_psa_percentage") or labs.get("psa_percent"))

    imaging = sp.get("imaging", {})
    features["vol"] = parse_numeric(imaging.get("prostate_volume") or imaging.get("volume"))
    features["pirads"] = parse_numeric(imaging.get("pirads_score") or imaging.get("max_pirads") or imaging.get("pirads"))
    features["cspca"] = parse_numeric(imaging.get("cspca") or 1.0)

    bx = sp.get("biopsy_findings", {})
    features["bx_gl_prim"] = parse_numeric(bx.get("primary_gleason"))
    features["bx_gl_sec"] = parse_numeric(bx.get("secondary_gleason"))
    features["bx_isup"] = parse_numeric(bx.get("isup_grade") or bx.get("grade_group"))
    bx_m = str(bx.get("method", "systematic")).lower()
    features["bx"] = "targeted" if "target" in bx_m else ("systematic" if "system" in bx_m else "combined")

    staging = sp.get("staging", {})
    ct_val = str(staging.get("ct", "negative")).lower()
    features["ct"] = "positive" if "pos" in ct_val else "negative"

    # 2. Build Clinical Narrative Text
    text_chunks: List[str] = []
    if demographics:
        text_chunks.append(f"Demographics: {demographics}")
    if vitals:
        text_chunks.append(f"Vitals: {vitals}")
    if hist:
        text_chunks.append(f"History: {hist}")
    if exam:
        text_chunks.append(f"Physical Exam: {exam}")
    if labs:
        text_chunks.append(f"Laboratory: {labs}")
    if imaging:
        text_chunks.append(f"Imaging: {imaging}")
    if bx:
        text_chunks.append(f"Biopsy Findings: {bx}")
    if staging:
        text_chunks.append(f"Staging: {staging}")

    narrative = " | ".join(text_chunks) if text_chunks else "Patient clinical records."

    return case_id, features, narrative, mri_vec
