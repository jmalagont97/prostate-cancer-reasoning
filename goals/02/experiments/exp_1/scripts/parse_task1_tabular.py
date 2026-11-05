import os
import json
import re
import pandas as pd
from pathlib import Path

raw_dir = Path("../../../../data/raw/latests/task1")
goal_data_dir = Path("../../data")
goal_data_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

cases = sorted([d for d in os.listdir(raw_dir) if (raw_dir / d).is_dir()])

inputs_rows = []
outputs_rows = []

for c in cases:
    case_path = raw_dir / c
    sp_path = case_path / "structured-prompt.json"
    cd_path = case_path / "prostate-biopsy-decision-clinical-data.json"
    dec_path = case_path / "prostate-biopsy-decision.json"
    reas_path = case_path / "prostate-biopsy-decision-reasoning.json"

    with open(sp_path) as f:
        sp = json.load(f)
    with open(cd_path) as f:
        cd = json.load(f)

    # 1. Radiology extraction
    rad = cd.get("radiology_report", "")
    
    m_pirads = re.search(r'PI-RADS\s*[:\-\s]\s*\"?([1-5])\"?', rad, re.IGNORECASE)
    pirads = int(m_pirads.group(1)) if m_pirads else None

    m_vol = re.search(r'(?:glandular volume|prostate volume|registers at)\s*(?:measures)?\s*[:\s]*\"?([0-9]+(?:\.[0-9]+)?)\"?', rad, re.IGNORECASE)
    if not m_vol:
        m_vol = re.search(r'\"?([0-9]+(?:\.[0-9]+)?)\"?\s*mL', rad, re.IGNORECASE)
    vol = float(m_vol.group(1)) if m_vol else None

    # PSA value from trend or lab
    psa_val = None
    if cd.get("psa_trend") and len(cd["psa_trend"]) > 0:
        psa_val = float(cd["psa_trend"][-1].get("val"))
    elif cd.get("laboratory_results"):
        for lab in cd["laboratory_results"]:
            if lab.get("name") == "PSA":
                val_str = str(lab.get("val")).replace("ng/mL", "").strip()
                try:
                    psa_val = float(val_str)
                except ValueError:
                    pass
                break

    m_psad = re.search(r'PSA density\s*[:\sof]*\"?([0-9]+(?:\.[0-9]+)?)\"?', rad, re.IGNORECASE)
    if m_psad:
        psad = float(m_psad.group(1))
    elif psa_val and vol and vol > 0:
        psad = round(psa_val / vol, 4)
    else:
        psad = None

    # 2. PSA trend dynamics
    earliest_psa = float(cd["psa_trend"][0].get("val")) if cd.get("psa_trend") else psa_val
    psa_delta = round(psa_val - earliest_psa, 4) if (psa_val is not None and earliest_psa is not None) else 0.0
    trend_count = len(cd.get("psa_trend", []))

    # 3. Lab results
    free_psa = None
    free_psa_pct = None
    for lab in cd.get("laboratory_results", []):
        name = lab.get("name", "")
        val = str(lab.get("val", ""))
        if name == "Free PSA":
            m_f = re.search(r'([0-9]+(?:\.[0-9]+)?)', val)
            if m_f: free_psa = float(m_f.group(1))
        elif name == "%Free PSA":
            m_fp = re.search(r'([0-9]+(?:\.[0-9]+)?)', val)
            if m_fp: free_psa_pct = float(m_fp.group(1))

    # 4. Previous notes
    notes = cd.get("previous_notes", [])
    notes_count = len(notes)
    notes_text = " ".join([n.get("text", "") for n in notes])
    prior_cancer_confirmed = 1 if re.search(r'(?:isup|cancer diagnosis|gleason|carcinoma)', notes_text, re.IGNORECASE) else 0

    # 5. Structured prompt
    age = sp.get("age")
    dre = sp.get("dre", "Normal")
    bx = sp.get("bx", "Not reported")
    ct = sp.get("ct", "Not reported")
    psav = sp.get("psav")
    ipss_str = str(sp.get("ipss", ""))
    m_ipss = re.search(r'([0-9]+)/35', ipss_str)
    ipss_score = int(m_ipss.group(1)) if m_ipss else None
    active_tx = 1 if sp.get("active_treatment_flag") or "prior PCa" in sp.get("enc_type", "") else 0

    # 6. Clinical data
    cspca = float(cd.get("cspca", 0.0))
    fh = cd.get("family_history", "No")

    # Assemble inputs row
    inputs_rows.append({
        "case_id": c,
        "structured_prompt__age": age,
        "structured_prompt__dre": dre,
        "structured_prompt__bx": bx,
        "structured_prompt__ct": ct,
        "structured_prompt__psav": psav,
        "structured_prompt__ipss": ipss_score,
        "structured_prompt__active_treatment_flag": active_tx,
        "radiology_report__pirads": pirads,
        "radiology_report__prostate_volume": vol,
        "radiology_report__psa_density": psad,
        "clinical_data__cspca": cspca,
        "clinical_data__family_history": fh,
        "psa_trend__latest_psa": psa_val,
        "psa_trend__earliest_psa": earliest_psa,
        "psa_trend__psa_delta": psa_delta,
        "psa_trend__readings_count": trend_count,
        "laboratory_results__psa": psa_val,
        "laboratory_results__free_psa": free_psa,
        "laboratory_results__free_psa_percent": free_psa_pct,
        "previous_notes__prior_cancer_confirmed": prior_cancer_confirmed,
        "previous_notes__notes_count": notes_count
    })

    # Assemble outputs row if labeled
    if dec_path.exists() and reas_path.exists():
        with open(dec_path) as f:
            dec = json.load(f)
        with open(reas_path) as f:
            reas = json.load(f)

        weights = reas.get("variable_weights", {})
        outputs_rows.append({
            "case_id": c,
            "biopsy_decision": dec,
            "confidence": reas.get("confidence"),
            "weight__age": weights.get("age", "not_used"),
            "weight__fh": weights.get("fh", "not_used"),
            "weight__cspca": weights.get("cspca", "not_used"),
            "weight__pirads": weights.get("pirads", "not_used"),
            "weight__vol": weights.get("vol", "not_used"),
            "weight__psa": weights.get("psa", "not_used"),
            "weight__comorbidity": weights.get("comorbidity", "not_used"),
            "weight__psad": weights.get("psad", "not_used"),
            "weight__dre": weights.get("dre", "not_used"),
            "weight__bx": weights.get("bx", "not_used"),
            "reveal_sequence": json.dumps(reas.get("reveal_sequence", [])),
            "free_text": reas.get("free_text", "")
        })

df_inputs = pd.DataFrame(inputs_rows)
df_outputs = pd.DataFrame(outputs_rows)

inputs_path = goal_data_dir / "inputs.csv"
outputs_path = goal_data_dir / "outputs.csv"
df_inputs.to_csv(inputs_path, index=False)
df_outputs.to_csv(outputs_path, index=False)

audit = {
    "total_input_cases": len(df_inputs),
    "total_output_cases": len(df_outputs),
    "inputs_columns": list(df_inputs.columns),
    "outputs_columns": list(df_outputs.columns),
    "inputs_null_counts": df_inputs.isnull().sum().to_dict(),
    "outputs_null_counts": df_outputs.isnull().sum().to_dict(),
    "biopsy_decision_counts": df_outputs["biopsy_decision"].value_counts().to_dict(),
    "pirads_counts": df_inputs["radiology_report__pirads"].value_counts().to_dict()
}

with open(results_dir / "dataset_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print(f"Extraction complete: {len(df_inputs)} inputs, {len(df_outputs)} outputs saved to {goal_data_dir}")
