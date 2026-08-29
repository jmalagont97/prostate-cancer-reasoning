#!/usr/bin/env python3
"""
Preprocessing and Ingestion Pipeline for Task 1 (Pre-Biopsy Decision).
Extracts 4 canonical CSV representations for all 195 cohort patients:
  1. clinical_tabular.csv
  2. clinical_texts.csv
  3. mri_embeddings.csv (1024-d, with PT-pseudo_0bd72f429a8b imputed as NaN)
  4. ground_truth_task1.csv (195 rows, 91 labeled)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Constants and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_TASK1_DIR = PROJECT_ROOT / "data" / "raw" / "train_release" / "task1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "task1"
EXCLUDED_EMBEDDING_PID = "PT-pseudo_0bd72f429a8b"

ESSENTIAL_VARIABLES = [
    "age", "psa", "vol", "psad", "dre", "pirads", "fh", "comorbidity", "bx", "cspca"
]


def normalize_bx(val: Any) -> Optional[float]:
    """Normalize prior-biopsy status to 1, 0, or NaN while preserving missingness."""
    if val is None or (isinstance(val, str) and val.strip() == "None"):
        return np.nan
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized == "positive":
            return 1.0
        if normalized == "negative":
            return 0.0
    raise ValueError(f"Unsupported BX value: {val!r}; expected Positive, Negative, or None")


def parse_numeric(val: Any) -> Optional[float]:
    """Safely parse float or int from string/number."""
    if val is None or val == "" or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Extract first numeric float match
        match = re.search(r"[-+]?\d*\.?\d+", val)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return np.nan
    return np.nan


def parse_bp(bp_str: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Parse blood pressure string like '125/84 mmHg' into (systolic, diastolic)."""
    if not bp_str or not isinstance(bp_str, str):
        return np.nan, np.nan
    match = re.search(r"(\d+)\s*/\s*(\d+)", bp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def extract_patient_tabular(sp: Dict[str, Any], pid: str) -> Dict[str, Any]:
    """Extract structured tabular features from structured-prompt.json."""
    vitals = sp.get("vitals") or {}
    bp_sys, bp_dia = parse_bp(vitals.get("bp"))
    
    # Parse IPSS score
    ipss_raw = sp.get("ipss")
    ipss_score = parse_numeric(ipss_raw)

    return {
        "pid": pid,
        "case_id": sp.get("case_id", pid),
        "task": sp.get("task", 1),
        # 10 essential clinical variables (and derivatives)
        "age": parse_numeric(sp.get("age")),
        "psa": parse_numeric(sp.get("psa")),
        "vol": parse_numeric(sp.get("vol")),
        "psad": parse_numeric(sp.get("psad")),
        "dre": sp.get("dre") if sp.get("dre") is not None else np.nan,
        "pirads": sp.get("pirads") if sp.get("pirads") is not None else np.nan,
        "bx": normalize_bx(sp.get("bx")),
        "cspca": parse_numeric(sp.get("cspca")),
        "psav": parse_numeric(sp.get("psav")),
        "psap": parse_numeric(sp.get("psap")),
        "months": parse_numeric(sp.get("months")),
        "active_treatment_flag": parse_numeric(sp.get("active_treatment_flag")),
        "ipss_score": ipss_score,
        "ipss_raw": ipss_raw if ipss_raw is not None else np.nan,
        # Demographics and lifestyle
        "occupation": sp.get("occupation") if sp.get("occupation") is not None else np.nan,
        "marital": sp.get("marital") if sp.get("marital") is not None else np.nan,
        "living": sp.get("living") if sp.get("living") is not None else np.nan,
        "next_of_kin": sp.get("next_of_kin") if sp.get("next_of_kin") is not None else np.nan,
        "alcohol": sp.get("alcohol") if sp.get("alcohol") is not None else np.nan,
        "exercise": sp.get("exercise") if sp.get("exercise") is not None else np.nan,
        # Encounter metadata
        "enc_dept": sp.get("enc_dept") if sp.get("enc_dept") is not None else np.nan,
        "enc_date": sp.get("enc_date") if sp.get("enc_date") is not None else np.nan,
        "enc_ref": sp.get("enc_ref") if sp.get("enc_ref") is not None else np.nan,
        "enc_type": sp.get("enc_type") if sp.get("enc_type") is not None else np.nan,
        "mri_hospital": sp.get("mri_hospital") if sp.get("mri_hospital") is not None else np.nan,
        "ct": sp.get("ct") if sp.get("ct") is not None else np.nan,
        # Vitals
        "vitals_weight_kg": parse_numeric(vitals.get("weight")),
        "vitals_height_cm": parse_numeric(vitals.get("height")),
        "vitals_bmi": parse_numeric(vitals.get("bmi")),
        "vitals_bp_sys": bp_sys,
        "vitals_bp_dia": bp_dia,
        "vitals_hr_bpm": parse_numeric(vitals.get("hr")),
        "vitals_smoking": vitals.get("smoking") if vitals.get("smoking") is not None else np.nan,
        "vitals_flu_vax": vitals.get("flu_vax") if vitals.get("flu_vax") is not None else np.nan,
        "vitals_last_dental": vitals.get("last_dental") if vitals.get("last_dental") is not None else np.nan,
    }


def extract_patient_texts(sp: Dict[str, Any], pid: str) -> Dict[str, Any]:
    """Extract clinical text narratives strictly from structured-prompt.json."""
    note_sections = sp.get("note_sections") or []
    sec_dict = {sec.get("s"): sec.get("t") for sec in note_sections if isinstance(sec, dict)}
    
    chief_complaint = sec_dict.get("Chief complaint", np.nan)
    history = sec_dict.get("History", np.nan)
    physical_exam = sec_dict.get("Physical examination", np.nan)
    
    pmhx_list = sp.get("pmhx") or []
    pmhx_str = "; ".join(pmhx_list) if isinstance(pmhx_list, list) and pmhx_list else np.nan
    
    allergies_list = sp.get("allergies") or []
    allergies_str = "; ".join(allergies_list) if isinstance(allergies_list, list) and allergies_list else np.nan
    
    recent_other_list = sp.get("recent_other") or []
    recent_other_str = "; ".join(recent_other_list) if isinstance(recent_other_list, list) and recent_other_list else np.nan
    
    full_narrative_parts = []
    if pd.notna(chief_complaint): full_narrative_parts.append(f"Chief Complaint: {chief_complaint}")
    if pd.notna(history): full_narrative_parts.append(f"History: {history}")
    if pd.notna(physical_exam): full_narrative_parts.append(f"Physical Exam: {physical_exam}")
    if sp.get("notes"): full_narrative_parts.append(f"Summary Notes: {sp.get('notes')}")
    full_narrative = "\n\n".join(full_narrative_parts) if full_narrative_parts else np.nan

    return {
        "pid": pid,
        "case_id": sp.get("case_id", pid),
        "summary_notes": sp.get("notes") if sp.get("notes") is not None else np.nan,
        "chief_complaint": chief_complaint,
        "history": history,
        "physical_exam": physical_exam,
        "medhx": sp.get("medhx") if sp.get("medhx") is not None else np.nan,
        "meds": sp.get("meds") if sp.get("meds") is not None else np.nan,
        "pmhx": pmhx_str,
        "allergies": allergies_str,
        "recent_other": recent_other_str,
        "admin_notes": sp.get("admin") if sp.get("admin") is not None else np.nan,
        "full_encounter_narrative": full_narrative,
    }


def extract_patient_embeddings(patient_dir: Path, pid: str) -> Dict[str, Any]:
    """Extract 1024-d MRI embedding vector, imputing NaN for excluded/missing cases."""
    res = {"pid": pid, "case_id": pid}
    
    if pid == EXCLUDED_EMBEDDING_PID:
        for i in range(1024):
            res[f"emb_{i}"] = np.nan
        return res
        
    emb_file = patient_dir / "prostate-modality-level-neural-representations.json"
    if not emb_file.exists():
        for i in range(1024):
            res[f"emb_{i}"] = np.nan
        return res
        
    try:
        with open(emb_file, "r") as f:
            data = json.load(f)
        mri_list = data.get("MRI image", [])
        if mri_list and len(mri_list) > 0 and len(mri_list[0]) == 1024:
            vec = mri_list[0]
            for i, val in enumerate(vec):
                res[f"emb_{i}"] = float(val)
        else:
            for i in range(1024):
                res[f"emb_{i}"] = np.nan
    except Exception:
        for i in range(1024):
            res[f"emb_{i}"] = np.nan
            
    return res


def extract_patient_ground_truth(patient_dir: Path, pid: str) -> Dict[str, Any]:
    """Extract ground truth biopsy decision, confidence, variable weights, and rationales."""
    dec_file = patient_dir / "prostate-biopsy-decision.json"
    reas_file = patient_dir / "prostate-biopsy-decision-reasoning.json"
    
    res = {
        "pid": pid,
        "case_id": pid,
        "is_labeled": False,
        "decision": np.nan,
        "confidence": np.nan,
        "weight_age": np.nan,
        "weight_psa": np.nan,
        "weight_vol": np.nan,
        "weight_psad": np.nan,
        "weight_dre": np.nan,
        "weight_pirads": np.nan,
        "weight_fh": np.nan,
        "weight_comorbidity": np.nan,
        "weight_bx": np.nan,
        "weight_cspca": np.nan,
        "factors_decisive": np.nan,
        "factors_important": np.nan,
        "reveal_sequence": np.nan,
        "free_text": np.nan,
    }
    
    if not dec_file.exists():
        return res
        
    res["is_labeled"] = True
    try:
        with open(dec_file, "r") as f:
            res["decision"] = json.load(f).strip().lower()
    except Exception:
        pass
        
    if reas_file.exists():
        try:
            with open(reas_file, "r") as f:
                reas = json.load(f)
            res["confidence"] = reas.get("confidence", np.nan)
            
            vw = reas.get("variable_weights", {}) or {}
            decisive_factors = []
            important_factors = []
            
            for var in ESSENTIAL_VARIABLES:
                val = vw.get(var, "not_used")
                res[f"weight_{var}"] = val
                if val == "decisive":
                    decisive_factors.append(var)
                elif val == "important":
                    important_factors.append(var)
                    
            res["factors_decisive"] = ";".join(sorted(decisive_factors)) if decisive_factors else "none"
            res["factors_important"] = ";".join(sorted(important_factors)) if important_factors else "none"
            
            rev = reas.get("reveal_sequence", [])
            res["reveal_sequence"] = ";".join(rev) if isinstance(rev, list) else str(rev)
            res["free_text"] = reas.get("free_text", np.nan)
        except Exception:
            pass
            
    return res


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted([d for d in RAW_TASK1_DIR.iterdir() if d.is_dir()])
    total_cases = len(patient_dirs)
    print(f"[*] Starting extraction for {total_cases} cases in Task 1...")
    
    tabular_records = []
    text_records = []
    embedding_records = []
    ground_truth_records = []
    
    for p_dir in patient_dirs:
        pid = p_dir.name
        sp_file = p_dir / "structured-prompt.json"
        
        if not sp_file.exists():
            print(f"[!] Warning: structured-prompt.json missing for {pid}")
            continue
            
        with open(sp_file, "r") as f:
            sp = json.load(f)
            
        # 1. Tabular
        tabular_records.append(extract_patient_tabular(sp, pid))
        # 2. Text
        text_records.append(extract_patient_texts(sp, pid))
        # 3. Embeddings
        embedding_records.append(extract_patient_embeddings(p_dir, pid))
        # 4. Ground Truth
        ground_truth_records.append(extract_patient_ground_truth(p_dir, pid))
        
    df_tabular = pd.DataFrame(tabular_records)
    df_texts = pd.DataFrame(text_records)
    df_embeddings = pd.DataFrame(embedding_records)
    df_gt = pd.DataFrame(ground_truth_records)
    
    # Save CSVs
    path_tabular = OUTPUT_DIR / "clinical_tabular.csv"
    path_texts = OUTPUT_DIR / "clinical_texts.csv"
    path_embeddings = OUTPUT_DIR / "mri_embeddings.csv"
    path_gt = OUTPUT_DIR / "ground_truth_task1.csv"
    
    df_tabular.to_csv(path_tabular, index=False)
    df_texts.to_csv(path_texts, index=False)
    df_embeddings.to_csv(path_embeddings, index=False)
    df_gt.to_csv(path_gt, index=False)
    
    print("\n" + "="*80)
    print("EXTRACTION & VALIDATION SUMMARY REPORT")
    print("="*80)
    print(f"1. clinical_tabular.csv:   Shape {df_tabular.shape} -> {path_tabular}")
    print(f"2. clinical_texts.csv:     Shape {df_texts.shape} -> {path_texts}")
    print(f"3. mri_embeddings.csv:     Shape {df_embeddings.shape} -> {path_embeddings}")
    print(f"4. ground_truth_task1.csv: Shape {df_gt.shape} (Labeled cases: {df_gt['is_labeled'].sum()}) -> {path_gt}")
    print(f"BX distribution: {df_tabular['bx'].value_counts(dropna=False).to_dict()}")
    
    # Assertions
    assert len(df_tabular) == total_cases, "Row count mismatch in tabular data"
    assert len(df_texts) == total_cases, "Row count mismatch in text data"
    assert len(df_embeddings) == total_cases, "Row count mismatch in embeddings"
    assert len(df_gt) == total_cases, "Row count mismatch in ground truth"
    assert "bx" in df_tabular.columns, "BX column missing from tabular data"
    bx_values = set(df_tabular["bx"].dropna().unique())
    assert bx_values <= {0.0, 1.0}, f"Unexpected normalized BX values: {bx_values}"
    assert df_embeddings.shape[1] == 1026, f"Expected 1026 cols (pid, case_id + 1024 emb), got {df_embeddings.shape[1]}"
    
    # Validate NaN exclusion on PT-pseudo_0bd72f429a8b
    excluded_row = df_embeddings[df_embeddings["pid"] == EXCLUDED_EMBEDDING_PID]
    assert not excluded_row.empty, "Excluded PID not found in embeddings"
    emb_cols = [c for c in df_embeddings.columns if c.startswith("emb_")]
    assert excluded_row[emb_cols].isna().all().all(), f"Embeddings for {EXCLUDED_EMBEDDING_PID} must be all NaN"
    
    print("\n[+] All validation checks passed successfully. Pipeline complete.")
    print("="*80)


if __name__ == "__main__":
    main()
