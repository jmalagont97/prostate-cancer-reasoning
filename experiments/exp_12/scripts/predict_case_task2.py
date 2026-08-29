#!/usr/bin/env python3
"""
Task-2 inference for one case using the exp_12 pickled models.

Loads the three subtask bundles, extracts canonical features for a raw case,
and produces the two official Task-2 output JSONs.

Usage:
  python predict_case_task2.py --case DIR [--outdir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP12_DIR = PROJECT_ROOT / "experiments" / "exp_12"
MODELS_DIR = EXP12_DIR / "models"


def _load_save_module() -> Any:
    """Load the module that defines the pickled classes and register as __main__."""
    spec = importlib.util.spec_from_file_location(
        "exp12_save", str(EXP12_DIR / "scripts" / "save_best_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp12_save"] = mod
    spec.loader.exec_module(mod)
    sys.modules["__main__"] = mod
    return mod


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Canonical feature extraction (mirror of preprocess_task2.py)
# ---------------------------------------------------------------------------

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
    out["vitals_bmi"] = _to_float(vitals.get("bmi"))
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
    flags["comorbidity_count"] = float(sum(1 for kw in COMORB_KEYWORDS.values() if any(k in joined for k in kw)))
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
                txt = _concat_texts([str(note.get("date", "")), str(note.get("author", "")), str(note.get("text", ""))], " - ")
                if txt:
                    parts.append(f"PREVIOUS: {txt}")
    psa_trend = clinical.get("psa_trend", [])
    if isinstance(psa_trend, list) and psa_trend:
        trend_str = "; ".join(f"{p.get('date', '?')}: {p.get('val', '?')}" for p in psa_trend if isinstance(p, dict))
        parts.append(f"PSA_TREND: {trend_str}")
    labs = clinical.get("laboratory_results", [])
    if isinstance(labs, list) and labs:
        labs_str = "; ".join(f"{p.get('name', '?')}={p.get('val', '?')} ({p.get('date', '?')})" for p in labs if isinstance(p, dict))
        parts.append(f"LABS: {labs_str}")
    fh = clinical.get("family_history")
    if fh:
        parts.append(f"FAMILY_HISTORY: {fh}")
    path = clinical.get("pathology_report")
    if path:
        parts.append(f"PATHOLOGY: {path}")
    return " ".join(parts)


def extract_case_features(case_dir: Path) -> Tuple[Dict[str, Any], np.ndarray, str]:
    """Return (tab_dict, mri_1024, narrative) for a raw case directory."""
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
    tab.update(_derive_comorbidities(sp.get("medhx"), sp.get("pmhx")))
    tab["dre"] = str(sp.get("dre", "unknown"))
    tab["bx"] = str(sp.get("bx", "unknown"))
    tab["ct"] = str(sp.get("ct", "unknown"))

    with open(case_dir / "prostate-modality-level-neural-representations.json", "r", encoding="utf-8") as f:
        reps = json.load(f)
    mri_vector = np.zeros(1024, dtype=float)
    mri_vals = reps.get("MRI image", [])
    if mri_vals and isinstance(mri_vals[0], list):
        mri_arr = np.asarray(mri_vals[0], dtype=float)
        if mri_arr.size == 1024:
            mri_vector = mri_arr

    with open(case_dir / "prostate-treatment-decision-clinical-data.json", "r", encoding="utf-8") as f:
        clinical = json.load(f)
    narrative = _build_narrative(clinical)

    return tab, mri_vector, narrative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Path to raw case directory")
    parser.add_argument("--outdir", default=None, help="Optional output directory for JSON files")
    args = parser.parse_args()

    _load_save_module()

    # Load bundles
    b1 = _load_pickle(MODELS_DIR / "subtask1_treatment_fuzzy_knn.pkl")
    b2 = _load_pickle(MODELS_DIR / "subtask2_confidence_fuzzy_knn.pkl")
    b3 = _load_pickle(MODELS_DIR / "subtask3_weights_ensemble.pkl")

    case_dir = Path(args.case)
    case_id = case_dir.name
    tab_dict, mri_vec, narrative = extract_case_features(case_dir)

    df_tab = pd.DataFrame([tab_dict])
    X_tab = b1["preprocessors"]["tab"].transform(df_tab)
    X_mri = b1["preprocessors"]["mri"].transform(mri_vec.reshape(1, -1))

    # Subtask 2.1: treatment
    p_tab = b1["models"]["tab"].predict_proba(X_tab)[0]
    p_mri = b1["models"]["mri"].predict_proba(X_mri)[0]
    X_txt = b1["preprocessors"]["text"].transform([narrative])
    p_txt = b1["models"]["text"].predict_proba(X_txt)[0]
    w1 = b1["fusion_weights"]
    p_fused = w1["w_tab"] * p_tab + w1["w_mri"] * p_mri + w1["w_txt"] * p_txt
    treatment = b1["class_order"][int(np.argmax(p_fused))]

    # Subtask 2.2: confidence
    p_tab_c = b2["models"]["tab"].predict_proba(b2["preprocessors"]["tab"].transform(df_tab))[0]
    p_mri_c = b2["models"]["mri"].predict_proba(b2["preprocessors"]["mri"].transform(mri_vec.reshape(1, -1)))[0]
    X_txt_c = b2["preprocessors"]["text"].transform([narrative])
    p_txt_c = b2["models"]["text"].predict_proba(X_txt_c)[0]
    w2 = b2["fusion_weights"]
    p_conf_fused = w2["w_tab"] * p_tab_c + w2["w_mri"] * p_mri_c + w2["w_txt"] * p_txt_c
    confidence = b2["class_order"][int(np.argmax(p_conf_fused))]

    # Subtask 2.3: variable weights
    prep23 = b3["preprocessors"]
    X_tab23 = prep23["tab"].transform(df_tab)
    X_mri23 = prep23["mri"].transform(mri_vec.reshape(1, -1))
    X_txt23 = prep23["text"].transform([narrative])
    X_joint = np.hstack([X_tab23, X_mri23, X_txt23])
    variable_weights: Dict[str, str] = {}
    for var in b3["variables"]:
        pred_idx = int(b3["models"][var].predict(X_joint)[0])
        variable_weights[var] = b3["weight_order"][pred_idx]

    # free_text rationale (deterministic summary)
    free_text = (
        f"Case {case_id}: treatment={treatment}, confidence={confidence}. "
        f"Key factors: " + ", ".join(
            f"{v}={w}" for v, w in variable_weights.items() if w in ("important", "decisive")
        ) + "."
    )

    decision_json = treatment
    reasoning_json = {
        "confidence": confidence,
        "variable_weights": variable_weights,
        "reveal_sequence": [],
        "free_text": free_text,
    }

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prostate-treatment-decision.json").write_text(json.dumps(decision_json), encoding="utf-8")
        (out / "prostate-treatment-decision-reasoning.json").write_text(json.dumps(reasoning_json, indent=2), encoding="utf-8")
        print(f"Wrote outputs to {out}")

    print(f"Case: {case_id}")
    print(f"  Treatment decision: {treatment}")
    print(f"  Confidence: {confidence}")
    print(f"  Variable weights: {variable_weights}")
    print(f"  Free text: {free_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
