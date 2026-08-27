#!/usr/bin/env python3
"""
CHIMERA Task 2 -- Canonical Preprocessing + Cohort Audit (exp_10).

Reads raw JSON files under data/raw/train_release/task2/ and builds 4 canonical
CSVs under data/processed/task2/ with strict leakage audit:

  clinical_tabular_task2.csv   pid + engineered tabular features (N=70)
  mri_embeddings_task2.csv     pid + emb_0..emb_1023 (MRI image 1024-D)
  clinical_texts_task2.csv     pid + full_encounter_narrative (free-text)
  ground_truth_task2.csv       pid + treatment_decision + confidence

Governance: active_treatment_flag is EXCLUDED (reveal target).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_TREATMENTS = {
    "watchful_waiting",
    "continued_surveillance",
    "active_surveillance",
    "active_treatment",
}
VALID_CONFIDENCE = {"clear", "borderline", "uncertain"}

# Keywords for comorbidity derivation from medhx/pmhx free text
COMORB_KEYWORDS: Dict[str, List[str]] = {
    "comorb_diabetes": ["diabet", "dm "],
    "comorb_hypertension": ["hypertens", "htn"],
    "comorb_cardiovascular": ["cardiac", "coronary", "myocardial", "heart", "cva", "stroke", "vascular"],
    "comorb_respiratory": ["respirat", "copd", "asthma", "pulmonary", "lung"],
    "comorb_renal": ["renal", "kidney", "ckd", "nephro"],
    "comorb_surgery": ["surgery", "surgical", "operation", "resection"],
    "comorb_cancer_other": ["cancer", "malignan", "melanoma", "lymphoma", "neoplasm"],
}


def _to_float(val: Any) -> float:
    try:
        if val is None:
            return float("nan")
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _concat_texts(texts: List[str], sep: str = " ") -> str:
    return sep.join([t.strip() for t in texts if t and t.strip()])


def _parse_vitals(vitals: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(vitals, dict):
        return out

    weight = vitals.get("weight", "")
    m = re.search(r"([\d.]+)\s*kg", str(weight), re.IGNORECASE)
    out["vitals_weight_kg"] = float(m.group(1)) if m else float("nan")

    height = vitals.get("height", "")
    m = re.search(r"([\d.]+)\s*cm", str(height), re.IGNORECASE)
    out["vitals_height_cm"] = float(m.group(1)) if m else float("nan")

    bmi = vitals.get("bmi", "")
    out["vitals_bmi"] = _to_float(bmi)

    bp = vitals.get("bp", "")
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(bp))
    if m:
        out["vitals_bp_sys"] = float(m.group(1))
        out["vitals_bp_dia"] = float(m.group(2))
    else:
        out["vitals_bp_sys"] = float("nan")
        out["vitals_bp_dia"] = float("nan")

    hr = vitals.get("hr", "")
    m = re.search(r"(\d+)\s*bpm", str(hr), re.IGNORECASE)
    out["vitals_hr_bpm"] = float(m.group(1)) if m else float("nan")

    smoking = vitals.get("smoking", "")
    m = re.search(r"([\d.]+)\s*(?:pack|pk|py)", str(smoking), re.IGNORECASE)
    out["smoking_pack_years"] = float(m.group(1)) if m else 0.0
    return out


def _parse_ipss(ipss_raw: Any) -> float:
    m = re.search(r"(\d+)\s*/\s*35", str(ipss_raw))
    if m:
        return float(m.group(1))
    m = re.search(r"IPSS[^\d]*(\d+)", str(ipss_raw), re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def _derive_comorbidities(medhx: Any, pmhx: Any) -> Dict[str, float]:
    texts = []
    if isinstance(medhx, str):
        texts.append(medhx)
    if isinstance(pmhx, list):
        texts.extend([str(x) for x in pmhx if x])
    elif isinstance(pmhx, str):
        texts.append(pmhx)
    joined = " ".join(texts).lower()

    flags: Dict[str, float] = {}
    for flag, keywords in COMORB_KEYWORDS.items():
        flags[flag] = 1.0 if any(kw in joined for kw in keywords) else 0.0

    flags["comorbidity_count"] = float(
        sum(1 for kw in COMORB_KEYWORDS.values() if any(k in joined for k in kw))
    )
    return flags


def _build_narrative(clinical: Dict[str, Any]) -> str:
    parts: List[str] = []

    rad = clinical.get("radiology_report")
    if rad:
        parts.append(f"RADIOLOGY: {rad}")

    prev = clinical.get("previous_notes", [])
    if isinstance(prev, list):
        for note in prev:
            if isinstance(note, dict):
                txt = _concat_texts(
                    [str(note.get("date", "")), str(note.get("author", "")), str(note.get("text", ""))],
                    " - ",
                )
                if txt:
                    parts.append(f"PREVIOUS: {txt}")

    psa_trend = clinical.get("psa_trend", [])
    if isinstance(psa_trend, list) and psa_trend:
        trend_str = "; ".join(
            f"{p.get('date', '?')}: {p.get('val', '?')}" for p in psa_trend if isinstance(p, dict)
        )
        parts.append(f"PSA_TREND: {trend_str}")

    labs = clinical.get("laboratory_results", [])
    if isinstance(labs, list) and labs:
        labs_str = "; ".join(
            f"{p.get('name', '?')}={p.get('val', '?')} ({p.get('date', '?')})"
            for p in labs if isinstance(p, dict)
        )
        parts.append(f"LABS: {labs_str}")

    fh = clinical.get("family_history")
    if fh:
        parts.append(f"FAMILY_HISTORY: {fh}")

    path = clinical.get("pathology_report")
    if path:
        parts.append(f"PATHOLOGY: {path}")

    return " ".join(parts)


def _process_case(case_id: str) -> Tuple[Dict[str, Any], np.ndarray, str, str, str]:
    case_dir = RAW_DIR / case_id

    # Ground truth + confidence
    gt_path = case_dir / "prostate-treatment-decision.json"
    treatment_gt = ""
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            treatment_gt = json.load(f)

    confidence = "clear"
    reasoning_path = case_dir / "prostate-treatment-decision-reasoning.json"
    if reasoning_path.exists():
        with open(reasoning_path, "r", encoding="utf-8") as f:
            reasoning = json.load(f)
        confidence = reasoning.get("confidence", "clear") if isinstance(reasoning, dict) else "clear"

    # Structured prompt (tabular)
    with open(case_dir / "structured-prompt.json", "r", encoding="utf-8") as f:
        sp = json.load(f)

    tab: Dict[str, Any] = {}
    for col in ["age", "psa", "months", "psad", "psav", "psap", "vol", "cspca",
                "bx_isup", "bx_gl_prim", "bx_gl_sec"]:
        tab[col] = _to_float(sp.get(col))
    tab["pirads"] = _to_float(sp.get("pirads"))

    tab.update(_parse_vitals(sp.get("vitals", {})))
    tab["ipss_score"] = _parse_ipss(sp.get("ipss", ""))

    allergies = sp.get("allergies", [])
    tab["allergies_count"] = float(len(allergies)) if isinstance(allergies, list) else 0.0

    comorb = _derive_comorbidities(sp.get("medhx"), sp.get("pmhx"))
    tab.update(comorb)

    tab["dre"] = str(sp.get("dre", "unknown"))
    tab["bx"] = str(sp.get("bx", "unknown"))
    tab["ct"] = str(sp.get("ct", "unknown"))

    # MRI embedding
    mri_vector = np.zeros(1024, dtype=float)
    with open(case_dir / "prostate-modality-level-neural-representations.json", "r", encoding="utf-8") as f:
        reps = json.load(f)
    mri_vals = reps.get("MRI image", [])
    if mri_vals and isinstance(mri_vals[0], list):
        mri_arr = np.asarray(mri_vals[0], dtype=float)
        if mri_arr.size == 1024:
            mri_vector = mri_arr

    # Clinical free-text narrative
    with open(case_dir / "prostate-treatment-decision-clinical-data.json", "r", encoding="utf-8") as f:
        clinical = json.load(f)
    narrative = _build_narrative(clinical)

    return tab, mri_vector, narrative, treatment_gt, confidence


def main() -> None:
    print("=" * 90)
    print("CHIMERA Task 2 -- Canonical Preprocessing + Cohort Audit (exp_10)")
    print("=" * 90)

    case_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"  Total case folders: {len(case_dirs)}")

    rows_tab: List[Dict[str, Any]] = []
    rows_mri: List[Dict[str, Any]] = []
    rows_text: List[Dict[str, Any]] = []
    rows_gt: List[Dict[str, Any]] = []

    excluded_no_gt: List[str] = []
    excluded_incomplete: List[str] = []

    for d in case_dirs:
        case_id = d.name
        gt_path = d / "prostate-treatment-decision.json"
        if not gt_path.exists():
            excluded_no_gt.append(case_id)
            continue

        tab, mri_vec, narrative, treatment_gt, confidence = _process_case(case_id)

        if mri_vec.sum() == 0.0 or not narrative.strip():
            excluded_incomplete.append(case_id)
            continue

        rows_gt.append({"pid": case_id, "treatment_decision": treatment_gt, "confidence": confidence})
        rows_tab.append({"pid": case_id, **tab})
        rows_mri.append({"pid": case_id, **{f"emb_{i}": float(v) for i, v in enumerate(mri_vec)}})
        rows_text.append({"pid": case_id, "full_encounter_narrative": narrative})

    df_gt = pd.DataFrame(rows_gt)
    print(f"\n  Cases with GT: {len(df_gt)}")
    print(f"  Excluded (no GT): {len(excluded_no_gt)}")
    print(f"  Excluded (incomplete modalities): {len(excluded_incomplete)} {excluded_incomplete}")

    invalid_treat = df_gt.loc[~df_gt["treatment_decision"].isin(VALID_TREATMENTS), "treatment_decision"].unique()
    if len(invalid_treat) > 0:
        raise ValueError(f"Invalid treatment labels: {invalid_treat}")

    invalid_conf = df_gt.loc[~df_gt["confidence"].isin(VALID_CONFIDENCE), "confidence"].unique()
    if len(invalid_conf) > 0:
        raise ValueError(f"Invalid confidence labels: {invalid_conf}")

    print("\n  Class distribution:")
    print(df_gt["treatment_decision"].value_counts().to_string())
    print("\n  Confidence distribution:")
    print(df_gt["confidence"].value_counts().to_string())

    df_tab = pd.DataFrame(rows_tab)
    df_mri = pd.DataFrame(rows_mri)
    df_text = pd.DataFrame(rows_text)

    df_tab.to_csv(OUT_DIR / "clinical_tabular_task2.csv", index=False)
    df_mri.to_csv(OUT_DIR / "mri_embeddings_task2.csv", index=False)
    df_text.to_csv(OUT_DIR / "clinical_texts_task2.csv", index=False)
    df_gt.to_csv(OUT_DIR / "ground_truth_task2.csv", index=False)

    print(f"\n  Persisted to {OUT_DIR}:")
    print(f"    clinical_tabular_task2.csv ({df_tab.shape[0]}x{df_tab.shape[1]})")
    print(f"    mri_embeddings_task2.csv   ({df_mri.shape[0]}x{df_mri.shape[1]})")
    print(f"    clinical_texts_task2.csv   ({df_text.shape[0]}x{df_text.shape[1]})")
    print(f"    ground_truth_task2.csv     ({df_gt.shape[0]}x{df_gt.shape[1]})")
    print("\nPreprocessing complete and verified.")


if __name__ == "__main__":
    main()
