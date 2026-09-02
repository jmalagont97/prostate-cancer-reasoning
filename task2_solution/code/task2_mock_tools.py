"""Plain-function mock tools reading data/processed/task2/clinical_notes.jsonl -- mirrors
mock_tools.py (task1) exactly, with one addition: get_pathology_report, since task2's clinical-data
file has 6 gated sections (task1 has 5; no biopsy exists yet at task1's decision point).
"""

from __future__ import annotations

import json
from pathlib import Path

_CLINICAL_NOTES: dict[str, dict] | None = None
TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task2"


def _load():
    global _CLINICAL_NOTES
    if _CLINICAL_NOTES is None:
        _CLINICAL_NOTES = {}
        with open(TASK_DIR / "clinical_notes.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                _CLINICAL_NOTES[rec["case_id"]] = rec
    return _CLINICAL_NOTES


def get_mri_report(case_id: str) -> str | None:
    return _load().get(case_id, {}).get("radiology_report")


def get_pathology_report(case_id: str) -> str | None:
    return _load().get(case_id, {}).get("pathology_report")


def get_psa_trend(case_id: str) -> list | None:
    return _load().get(case_id, {}).get("psa_trend")


def get_lab_results(case_id: str) -> list | None:
    return _load().get(case_id, {}).get("laboratory_results")


def get_previous_notes(case_id: str) -> list | None:
    return _load().get(case_id, {}).get("previous_notes")


def get_family_history(case_id: str) -> str | None:
    return _load().get(case_id, {}).get("family_history")


# tool function -> the reveal_sequence section name it corresponds to, confirmed from evaluate.py's
# SECTION_KEY_TO_REVEAL_NAME (section_s3-path -> pathology_report is the one entry task1 doesn't need).
TOOL_TO_SECTION = {
    "get_mri_report": "radiology_report",
    "get_pathology_report": "pathology_report",
    "get_psa_trend": "psa_trend",
    "get_lab_results": "laboratory_results",
    "get_previous_notes": "previous_notes",
    "get_family_history": "family_history",
}
