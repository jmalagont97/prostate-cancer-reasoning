"""Master Data Parser and Feature Extraction Library for CHIMERA Task 1.

Parses raw case JSON files:
- structured-prompt.json
- prostate-biopsy-decision-clinical-data.json
- prostate-modality-level-neural-representations.json
- prostate-biopsy-decision.json
- prostate-biopsy-decision-reasoning.json

Extracts flattened scalar, numerical, categorical, binary, and narrative text features
preserving 100% of information across all 195 cases into master DataFrames.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# The 10 official relevance variables for Task 1
OFFICIAL_10_VARIABLES = [
    "age",
    "fh",
    "cspca",
    "pirads",
    "vol",
    "psa",
    "comorbidity",
    "psad",
    "dre",
    "bx",
]

WEIGHT_TO_CODE = {
    "not_used": 0.0,
    "noted": 1.0,
    "important": 2.0,
    "decisive": 3.0,
}

CONFIDENCE_TO_CODE = {
    "uncertain": 0.0,
    "borderline": 1.0,
    "clear": 2.0,
}


def _safe_float(val: Any) -> float:
    if val is None or val == "" or val == "NONE" or val == "Not reported":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _safe_int(val: Any) -> float:
    f = _safe_float(val)
    return f if np.isnan(f) else float(int(f))


def _extract_number(val: Any) -> float:
    """Extract first floating/integer number from a string like '68 kg' or '147/77'."""
    if val is None or val == "" or val == "NONE":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
    if match:
        return float(match.group(0))
    return np.nan


def _parse_bp(bp_str: Any) -> tuple[float, float]:
    """Parse BP string '147/77 mmHg' -> (147.0, 77.0)."""
    if not bp_str or not isinstance(bp_str, str):
        return np.nan, np.nan
    parts = re.findall(r"\d+", bp_str)
    if len(parts) >= 2:
        return float(parts[0]), float(parts[1])
    return np.nan, np.nan


def parse_single_case_inputs(case_dir: Path) -> dict[str, Any]:
    """Parse input JSON files for a single case directory into a flat scalar dictionary."""
    case_id = case_dir.name
    row: dict[str, Any] = {"case_id": case_id}

    # -------------------------------------------------------------------------
    # 1. READ structured-prompt.json
    # -------------------------------------------------------------------------
    sp_path = case_dir / "structured-prompt.json"
    sp_data: dict[str, Any] = {}
    if sp_path.exists():
        try:
            sp_data = json.loads(sp_path.read_text())
        except Exception as e:
            log.warning("Failed to load %s: %s", sp_path, e)

    # Scalar Clinical Features (cli_)
    row["cli_age"] = _safe_float(sp_data.get("age"))
    row["cli_psa"] = _safe_float(sp_data.get("psa"))
    row["cli_psap"] = _safe_float(sp_data.get("psap"))
    row["cli_psav"] = _safe_float(sp_data.get("psav"))
    row["cli_psad"] = _safe_float(sp_data.get("psad"))
    row["cli_vol"] = _safe_float(sp_data.get("vol"))
    row["cli_months"] = _safe_float(sp_data.get("months"))
    row["cli_pirads"] = _safe_float(sp_data.get("pirads"))
    row["cli_dre"] = str(sp_data.get("dre")) if sp_data.get("dre") is not None else np.nan
    row["cli_bx"] = str(sp_data.get("bx")) if sp_data.get("bx") is not None else np.nan
    row["cli_cspca"] = _safe_float(sp_data.get("cspca"))

    # Comorbidities (pmhx) & Allergies
    pmhx = sp_data.get("pmhx")
    if isinstance(pmhx, list):
        row["cli_comorbidity_count"] = float(len(pmhx))
        row["txt_comorbidities"] = ", ".join(str(item) for item in pmhx) if len(pmhx) > 0 else np.nan
    else:
        row["cli_comorbidity_count"] = 0.0
        row["txt_comorbidities"] = np.nan

    allergies = sp_data.get("allergies")
    if isinstance(allergies, list):
        row["cli_allergies_count"] = float(len(allergies))
        row["txt_allergies"] = ", ".join(str(item) for item in allergies) if len(allergies) > 0 else np.nan
    else:
        row["cli_allergies_count"] = 0.0
        row["txt_allergies"] = np.nan

    # IPSS Score
    ipss_raw = sp_data.get("ipss")
    row["cli_ipss_score"] = _extract_number(ipss_raw)

    # Vitals (vit_)
    vitals = sp_data.get("vitals") if isinstance(sp_data.get("vitals"), dict) else {}
    row["vit_weight_kg"] = _extract_number(vitals.get("weight"))
    row["vit_height_cm"] = _extract_number(vitals.get("height"))
    row["vit_bmi"] = _extract_number(vitals.get("bmi"))
    sbp, dbp = _parse_bp(vitals.get("bp"))
    row["vit_bp_systolic"] = sbp
    row["vit_bp_diastolic"] = dbp
    row["vit_heart_rate_bpm"] = _extract_number(vitals.get("hr"))

    smoking_str = str(vitals.get("smoking", "")) if vitals.get("smoking") is not None else ""
    if "Ex-smoker" in smoking_str:
        row["vit_smoking_status"] = "Ex-smoker"
    elif "Current" in smoking_str:
        row["vit_smoking_status"] = "Current"
    elif "Never" in smoking_str or "Non-smoker" in smoking_str:
        row["vit_smoking_status"] = "Never"
    else:
        row["vit_smoking_status"] = smoking_str if smoking_str else np.nan

    row["vit_smoking_pack_years"] = _extract_number(smoking_str)

    # Pathology History (path_hist_)
    row["path_hist_bx_isup"] = _safe_float(sp_data.get("bx_isup"))
    row["path_hist_bx_gl_prim"] = _safe_float(sp_data.get("bx_gl_prim"))
    row["path_hist_bx_gl_sec"] = _safe_float(sp_data.get("bx_gl_sec"))
    row["path_hist_bx_gl_tert"] = _safe_float(sp_data.get("bx_gl_tert"))

    # Prompt Text Sections (txt_)
    note_sections = sp_data.get("note_sections") if isinstance(sp_data.get("note_sections"), list) else []
    sec_dict = {}
    sec_texts = []
    for sec in note_sections:
        if isinstance(sec, dict):
            s_title = sec.get("s", "Section")
            s_text = sec.get("t", "")
            sec_dict[s_title] = s_text
            if s_text:
                sec_texts.append(f"{s_title}: {s_text}")

    row["txt_chief_complaint"] = sec_dict.get("Chief complaint", np.nan)
    row["txt_history"] = sec_dict.get("History", np.nan)
    row["txt_physical_examination"] = sec_dict.get("Physical examination", np.nan)
    row["txt_prompt_summary_notes"] = str(sp_data.get("notes")) if sp_data.get("notes") is not None else np.nan
    row["txt_full_prompt_narrative"] = " | ".join(sec_texts) if sec_texts else np.nan

    # -------------------------------------------------------------------------
    # 2. READ prostate-biopsy-decision-clinical-data.json
    # -------------------------------------------------------------------------
    cd_path = case_dir / "prostate-biopsy-decision-clinical-data.json"
    cd_data: dict[str, Any] = {}
    if cd_path.exists():
        try:
            cd_data = json.loads(cd_path.read_text())
        except Exception as e:
            log.warning("Failed to load %s: %s", cd_path, e)

    # Family History (cli_fh_binary & txt_family_history_narrative)
    fh_raw = cd_data.get("family_history")
    if fh_raw == "Yes":
        row["cli_fh_binary"] = 1.0
    elif fh_raw == "No":
        row["cli_fh_binary"] = 0.0
    else:
        row["cli_fh_binary"] = np.nan
    row["txt_family_history_narrative"] = str(fh_raw) if fh_raw is not None else np.nan

    # Radiology Report
    rad_rep = cd_data.get("radiology_report")
    row["txt_radiology_report"] = str(rad_rep) if rad_rep is not None else np.nan

    # Previous Notes
    prev_notes = cd_data.get("previous_notes")
    if isinstance(prev_notes, list):
        note_strs = []
        for n in prev_notes:
            if isinstance(n, dict):
                d_str = n.get("date", "")
                a_str = n.get("author", "")
                t_str = n.get("text", "")
                note_strs.append(f"[{d_str} - {a_str}] {t_str}")
        row["txt_previous_notes"] = " | ".join(note_strs) if note_strs else np.nan
    else:
        row["txt_previous_notes"] = np.nan

    # PSA Trend
    psa_trend = cd_data.get("psa_trend")
    if isinstance(psa_trend, list) and len(psa_trend) > 0:
        row["psa_tr_count"] = float(len(psa_trend))
        vals = [_extract_number(item.get("val")) for item in psa_trend if isinstance(item, dict)]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            row["psa_tr_first_val"] = vals[0]
            row["psa_tr_last_val"] = vals[-1]
            row["psa_tr_min"] = float(min(vals))
            row["psa_tr_max"] = float(max(vals))
            row["psa_tr_mean"] = float(np.mean(vals))
            row["psa_tr_delta"] = float(vals[-1] - vals[0])
            row["psa_tr_slope"] = float((vals[-1] - vals[0]) / max(len(vals) - 1, 1))
        else:
            row["psa_tr_first_val"] = np.nan
            row["psa_tr_last_val"] = np.nan
            row["psa_tr_min"] = np.nan
            row["psa_tr_max"] = np.nan
            row["psa_tr_mean"] = np.nan
            row["psa_tr_delta"] = np.nan
            row["psa_tr_slope"] = np.nan

        trend_texts = [f"{item.get('date')}: {item.get('val')}" for item in psa_trend if isinstance(item, dict)]
        row["txt_psa_trend_summary"] = " | ".join(trend_texts) if trend_texts else np.nan
    else:
        row["psa_tr_count"] = 0.0
        row["psa_tr_first_val"] = np.nan
        row["psa_tr_last_val"] = np.nan
        row["psa_tr_min"] = np.nan
        row["psa_tr_max"] = np.nan
        row["psa_tr_mean"] = np.nan
        row["psa_tr_delta"] = np.nan
        row["psa_tr_slope"] = np.nan
        row["txt_psa_trend_summary"] = np.nan

    # Laboratory Results
    lab_res = cd_data.get("laboratory_results")
    if isinstance(lab_res, list) and len(lab_res) > 0:
        lab_texts = []
        creat_val, hb_val, fpsa_val = np.nan, np.nan, np.nan
        for item in lab_res:
            if isinstance(item, dict):
                name = item.get("name", "")
                val_raw = item.get("val", "")
                lab_texts.append(f"{name}: {val_raw}")
                val_num = _extract_number(val_raw)
                if "Creatinine" in name:
                    creat_val = val_num
                elif "Hemoglobin" in name or "Hb" in name:
                    hb_val = val_num
                elif "Free PSA" in name:
                    fpsa_val = val_num

        row["lab_creatinine_mg_dl"] = creat_val
        row["lab_hemoglobin_g_dl"] = hb_val
        row["lab_free_psa_ng_ml"] = fpsa_val
        row["lab_free_total_ratio"] = (fpsa_val / row["cli_psa"]) if (not np.isnan(fpsa_val) and row["cli_psa"] > 0) else np.nan
        row["txt_laboratory_results_summary"] = " | ".join(lab_texts) if lab_texts else np.nan
    else:
        row["lab_creatinine_mg_dl"] = np.nan
        row["lab_hemoglobin_g_dl"] = np.nan
        row["lab_free_psa_ng_ml"] = np.nan
        row["lab_free_total_ratio"] = np.nan
        row["txt_laboratory_results_summary"] = np.nan

    # Master Consolidated Narrative
    ehr_parts = [
        row.get("txt_full_prompt_narrative"),
        row.get("txt_radiology_report"),
        row.get("txt_previous_notes"),
        row.get("txt_laboratory_results_summary"),
        row.get("txt_family_history_narrative"),
    ]
    ehr_parts = [p for p in ehr_parts if isinstance(p, str) and p.strip()]
    row["txt_consolidated_ehr_narrative"] = "\n\n".join(ehr_parts) if ehr_parts else np.nan

    # -------------------------------------------------------------------------
    # 3. READ prostate-modality-level-neural-representations.json
    # -------------------------------------------------------------------------
    nr_path = case_dir / "prostate-modality-level-neural-representations.json"
    mri_vector: list[float] | None = None
    if nr_path.exists():
        try:
            nr_data = json.loads(nr_path.read_text())
            mri_img = nr_data.get("MRI image")
            if isinstance(mri_img, list) and len(mri_img) > 0:
                first_elem = mri_img[0]
                if isinstance(first_elem, list) and len(first_elem) > 0:
                    mri_vector = [float(v) for v in first_elem]
                elif isinstance(first_elem, (int, float)):
                    mri_vector = [float(v) for v in mri_img]
        except Exception as e:
            log.warning("Failed to load %s: %s", nr_path, e)

    for idx in range(1024):
        col_name = f"mri_emb_{idx}"
        if mri_vector and idx < len(mri_vector):
            row[col_name] = mri_vector[idx]
        else:
            row[col_name] = np.nan

    return row


def parse_single_case_targets(case_dir: Path) -> dict[str, Any]:
    """Parse ground truth decision & reasoning target files for a single case."""
    case_id = case_dir.name
    row: dict[str, Any] = {"case_id": case_id}

    # 1. Decision File (Task 1A)
    dec_path = case_dir / "prostate-biopsy-decision.json"
    if dec_path.exists():
        try:
            dec_val = json.loads(dec_path.read_text())
            row["target_biopsy_decision"] = str(dec_val).strip().lower()
            row["target_biopsy_decision_binary"] = 1.0 if str(dec_val).strip().lower() == "yes" else 0.0
        except Exception:
            row["target_biopsy_decision"] = np.nan
            row["target_biopsy_decision_binary"] = np.nan
    else:
        row["target_biopsy_decision"] = np.nan
        row["target_biopsy_decision_binary"] = np.nan

    # 2. Reasoning File (Tasks 1B & 1C)
    reas_path = case_dir / "prostate-biopsy-decision-reasoning.json"
    if reas_path.exists():
        try:
            reas_data = json.loads(reas_path.read_text())

            # Confidence
            conf_val = str(reas_data.get("confidence", "")).strip().lower()
            row["target_confidence"] = conf_val if conf_val in CONFIDENCE_TO_CODE else np.nan
            row["target_confidence_code"] = CONFIDENCE_TO_CODE.get(conf_val, np.nan)

            # 10 Variable Weights
            weights_dict = reas_data.get("variable_weights", {}) if isinstance(reas_data.get("variable_weights"), dict) else {}

            for var_name in OFFICIAL_10_VARIABLES:
                w_str = str(weights_dict.get(var_name, "not_used")).strip().lower()
                row[f"target_weight_{var_name}"] = w_str if w_str in WEIGHT_TO_CODE else "not_used"
                row[f"target_code_weight_{var_name}"] = WEIGHT_TO_CODE.get(w_str, 0.0)

            # Free text & reveal sequence
            row["target_reasoning_free_text"] = str(reas_data.get("free_text")) if reas_data.get("free_text") is not None else np.nan
            row["target_reveal_sequence_json"] = json.dumps(reas_data.get("reveal_sequence", []))
        except Exception as e:
            log.warning("Failed to load reasoning %s: %s", reas_path, e)
            _fill_empty_targets(row)
    else:
        _fill_empty_targets(row)

    return row


def _fill_empty_targets(row: dict[str, Any]) -> None:
    row["target_confidence"] = np.nan
    row["target_confidence_code"] = np.nan
    for var_name in OFFICIAL_10_VARIABLES:
        row[f"target_weight_{var_name}"] = np.nan
        row[f"target_code_weight_{var_name}"] = np.nan
    row["target_reasoning_free_text"] = np.nan
    row["target_reveal_sequence_json"] = np.nan


def build_master_datasets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all raw case directories and return (df_inputs, df_ground_truth)."""
    case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    log.info("Found %d case directories in %s", len(case_dirs), raw_dir)

    input_rows = []
    target_rows = []

    for c_dir in case_dirs:
        input_rows.append(parse_single_case_inputs(c_dir))
        target_rows.append(parse_single_case_targets(c_dir))

    df_inputs = pd.DataFrame(input_rows)
    df_targets = pd.DataFrame(target_rows)

    return df_inputs, df_targets
