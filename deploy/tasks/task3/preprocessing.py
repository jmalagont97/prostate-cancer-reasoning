"""Exact raw JSON adapter for the canonical Task 3 representation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age", "psa", "dre_abnormal", "dre_bilateral", "dre_nodule",
    "dre_ct_stage_ordinal", "active_treatment_flag", "isup_grade_group",
    "gleason_primary", "gleason_secondary", "gleason_sum",
    "gleason_tertiary_present", "pt_stage_ordinal", "epe_present",
    "margins_positive", "svi_present", "lvi_present", "ln_assessed", "ln_positive",
]
EXPECTED_FILES = (
    "structured-prompt.json",
    "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
    "prostate-time-to-recurrence-or-last-follow-up.json",
    "prostate-modality-level-neural-representations.json",
)
STAGE_RE = re.compile(r"[pc]T[0-9][a-z]?", re.I)
ISUP_RE = re.compile(r"ISUP grade group (\d+)", re.I)
GLEASON_RE = re.compile(r"Gleason (\d+)\+(\d+)", re.I)
PATH_STAGE_RE = re.compile(r"pathological stage (pT\w+)", re.I)
DRE_RE = re.compile(r"Digital rectal examination (was abnormal bilaterally|was abnormal on the left|was abnormal on the right|was abnormal|was normal|findings:)", re.I)


def _float(value: Any) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _stage(value: str | None) -> float | None:
    if not value:
        return None
    m = re.fullmatch(r"[pc]T(\d)([a-z]?)", value.strip(), re.I)
    if not m:
        return None
    return float(int(m.group(1)) * 10 + {"": 0, "a": 1, "b": 2, "c": 3}.get(m.group(2).lower(), 0))


def _dre(text: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"dre_abnormal": None, "dre_bilateral": None, "dre_nodule": None, "dre_ct_stage_ordinal": None}
    if not isinstance(text, str):
        return out
    m = DRE_RE.search(text)
    finding = m.group(1).lower() if m else ""
    if finding == "was normal": out.update(dre_abnormal=0.0, dre_bilateral=0.0)
    elif finding: out.update(dre_abnormal=1.0, dre_bilateral=float("bilaterally" in finding))
    out["dre_nodule"] = float("nodule" in text.lower())
    stage = STAGE_RE.search(text)
    out["dre_ct_stage_ordinal"] = _stage(stage.group(0)) if stage else None
    return out


def _pathology(text: Any) -> dict[str, Any]:
    out: dict[str, Any] = {k: None for k in FEATURE_COLUMNS[7:]}
    if not isinstance(text, str):
        return out
    m = ISUP_RE.search(text)
    out["isup_grade_group"] = _float(m.group(1)) if m else None
    m = GLEASON_RE.search(text)
    if m:
        primary, secondary = _float(m.group(1)), _float(m.group(2))
        out["gleason_primary"], out["gleason_secondary"] = primary, secondary
        out["gleason_sum"] = (primary or 0.0) + (secondary or 0.0)
    out["gleason_tertiary_present"] = float(bool(re.search(r"tertiary pattern (\d+)", text, re.I)))
    m = PATH_STAGE_RE.search(text)
    out["pt_stage_ordinal"] = _stage(m.group(1)) if m else None
    for key, pattern, positive in (
        ("epe_present", r"Extraprostatic extension was present|There was no extraprostatic extension", "extraprostatic extension was present"),
        ("margins_positive", r"surgical margins were (positive|negative)", "positive"),
        ("svi_present", r"the seminal vesicles were (not invaded|invaded)", "invaded"),
        ("lvi_present", r"lymphovascular invasion was (present|absent)", "present"),
    ):
        m = re.search(pattern, text, re.I)
        if m: out[key] = float((m.group(0) if key == "epe_present" else m.group(1)).lower() == positive)
    m = re.search(r"lymph node metastasis was present|there was no lymph node metastasis|no lymph nodes were removed", text, re.I)
    if m:
        phrase = m.group(0).lower()
        out["ln_assessed"], out["ln_positive"] = ((0.0, None) if phrase == "no lymph nodes were removed" else (1.0, float(phrase == "lymph node metastasis was present")))
    return out


def _read_case(case: str | Path | dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if isinstance(case, (str, Path)):
        directory = Path(case)
        payload = {name: json.loads((directory / name).read_text(encoding="utf-8")) for name in EXPECTED_FILES if (directory / name).exists()}
        return directory.name, payload.get(EXPECTED_FILES[0], {}), payload
    payload = case
    case_id = str(payload.get("case_id", payload.get("structured-prompt", {}).get("case_id", "unknown_case")))
    return case_id, payload.get("structured-prompt", payload.get("structured_prompt", payload)), payload


def transform_case(case: str | Path | dict[str, Any], require_label: bool = False) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any] | None]:
    case_id, structured, payload = _read_case(case)
    clinical = payload.get("prostate-time-to-recurrence-or-last-follow-up-clinical-data.json", payload.get("clinical_data", {}))
    row: dict[str, Any] = {"case_id": case_id, "age": _float(structured.get("age")), "psa": _float(structured.get("psa"))}
    row.update(_dre(structured.get("dre")))
    treatment = structured.get("active_treatment_prior_to_surgery")
    row["active_treatment_flag"] = None if treatment in (None, "") else float(str(treatment).strip().lower() != "no")
    row.update(_pathology(clinical.get("surgical_pathology_report")))
    raw_emb = payload.get("prostate-modality-level-neural-representations.json", payload.get("embeddings", {}))
    prostatectomy = np.asarray(raw_emb.get("Prostatectomy slide", raw_emb.get("prostatectomy", [])), dtype=float)
    if prostatectomy.size == 0: prostatectomy = np.empty((0, 960), dtype=float)
    if prostatectomy.ndim == 1: prostatectomy = prostatectomy.reshape(1, -1)
    if prostatectomy.ndim != 2 or prostatectomy.shape[1] != 960: raise ValueError("Prostatectomy embeddings must have shape (*, 960)")
    label = None
    raw_label = payload.get("prostate-time-to-recurrence-or-last-follow-up.json", payload.get("label"))
    if raw_label is not None:
        label = {"months_to_recurrence": _float(raw_label.get("months_to_recurrence")), "event": raw_label.get("event")}
    if require_label and (label is None or label["months_to_recurrence"] is None or label["event"] not in (0, 1)):
        raise ValueError(f"Missing valid Task 3 label for {case_id}")
    return pd.DataFrame([row], columns=["case_id"] + FEATURE_COLUMNS), {"prostatectomy": prostatectomy}, label
