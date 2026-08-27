"""Extract task3 engineered features: (1) a cleaned parse of the existing structured `dre` free
text (Group B — it's not actually a clean category, see notebooks/task3_eda.ipynb §3's bar chart,
which is dominated by near-unique long strings once you look past the top few), and (2) genuinely
new post-surgical pathology findings parsed out of `surgical_pathology_report` free text (Group C) --
ISUP grade group, pathological T-stage, extraprostatic extension, margin status, seminal-vesicle
invasion, lymphovascular invasion, and lymph-node status.

Both text sources were confirmed in notebooks/task3_eda.ipynb §5 (and the follow-up scan below) to
be a small, closed, fully-templated vocabulary -- not free prose -- so a plain regex/keyword parse
gets 100% coverage with no NLP model needed. Every pattern here was validated against the *exact*
set of distinct phrasings actually present in the corpus (not guessed), the same way task1's
extract_features_task1.py's parsers were built.

Reads:
  data/processed/task3/structured.csv       (Group B source: dre, active_treatment_prior_to_surgery)
  data/processed/task3/clinical_notes.jsonl (Group C source: surgical_pathology_report)

Writes:
  data/processed/task3/engineered_features.csv  (case_id + all new columns, additive)
"""

import csv
import json
import re
from pathlib import Path

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
TASK = "task3"


def stage_ordinal(stage_str):
    """'pT4b' -> 42, 'cT2a' -> 21, 'cT3' -> 30, 'cTx' -> None (indeterminate, not a severity rung --
    mirrors task1's dre_not_done/pirads_na treatment: missingness-as-a-label pulled out, not coerced
    into the ordinal scale). Generic over both pT (surgical) and cT (DRE-derived) stages -- same
    T-number + optional subletter grammar in both, confirmed by direct corpus scan.
    """
    if not stage_str:
        return None
    m = re.match(r"[pc]T(\d)([a-z]?)", stage_str)
    if not m:
        return None
    num = int(m.group(1))
    sub_rank = {"": 0, "a": 1, "b": 2, "c": 3}.get(m.group(2), 0)
    return num * 10 + sub_rank


# --- Group B: cleaned re-parse of the existing `dre` structured column -----------------------
# Confirmed by direct corpus scan (6 distinct finding-phrasings + 1 free-text outlier row, 9
# distinct cT-stage values including cTx) -- not assumed.
DRE_FINDING_RE = re.compile(
    r"Digital rectal examination (was abnormal bilaterally|was abnormal on the left|"
    r"was abnormal on the right|was abnormal|was normal|findings:)"
)


def parse_dre(dre_text):
    """Returns (dre_abnormal, dre_bilateral, dre_nodule, dre_ct_stage_ordinal). All None if dre_text
    is missing (23/75 cases -- DRE not performed/recorded, a real missingness, not an error)."""
    if not dre_text:
        return None, None, None, None
    m = DRE_FINDING_RE.search(dre_text)
    finding = m.group(1) if m else None
    nodule = 1.0 if "nodule" in dre_text.lower() else 0.0
    if finding == "was normal":
        abnormal, bilateral = 0.0, 0.0
    elif finding == "findings:":  # the one free-text outlier row -- has a palpable nodule, no
        abnormal, bilateral = 1.0, 0.0  # explicit normal/abnormal phrase, treated as abnormal
    elif finding is not None:
        abnormal = 1.0
        bilateral = 1.0 if "bilaterally" in finding else 0.0
    else:
        abnormal, bilateral = None, None
    stage_m = re.search(r"cT\w+", dre_text)
    return abnormal, bilateral, nodule, stage_ordinal(stage_m.group(0) if stage_m else None)


def parse_active_treatment(s):
    """73/75 missing; the 2 non-missing values are 'no' and 'radiotherapy + cryotherapy' --
    near-zero variance either way, but encoded as a clean binary rather than left as raw text."""
    if not s:
        return 0.0  # unrecorded -- treated the same as an explicit "no" (no evidence of treatment)
    return 0.0 if s.strip().lower() == "no" else 1.0


# --- Group C: surgical_pathology_report -------------------------------------------------------
# Every phrasing below is the FULL closed vocabulary actually observed across all 75 cases
# (confirmed by direct regex scan over the corpus, see notebooks/task3_eda.ipynb's companion
# spot-check cell) -- not a guess at plausible phrasings.
ISUP_RE = re.compile(r"ISUP grade group (\d)")
GLEASON_RE = re.compile(r"Gleason (\d)\+(\d)")
TERTIARY_RE = re.compile(r"tertiary pattern (\d)")
STAGE_RE = re.compile(r"pathological stage (pT\w+)")
EPE_RE = re.compile(r"Extraprostatic extension was present|There was no extraprostatic extension")
MARGINS_RE = re.compile(r"surgical margins were (positive|negative)")
SVI_RE = re.compile(r"the seminal vesicles were (not invaded|invaded)")
LVI_RE = re.compile(r"lymphovascular invasion was (present|absent)")
LN_RE = re.compile(
    r"lymph node metastasis was present|there was no lymph node metastasis|no lymph nodes were removed"
)


def parse_surgical_pathology(text):
    out = {}
    if not text:
        return {k: None for k in (
            "isup_grade_group", "gleason_primary", "gleason_secondary", "gleason_sum",
            "gleason_tertiary_present", "pt_stage_ordinal", "epe_present", "margins_positive",
            "svi_present", "lvi_present", "ln_assessed", "ln_positive")}

    m = ISUP_RE.search(text)
    out["isup_grade_group"] = float(m.group(1)) if m else None

    m = GLEASON_RE.search(text)
    if m:
        out["gleason_primary"], out["gleason_secondary"] = float(m.group(1)), float(m.group(2))
        out["gleason_sum"] = out["gleason_primary"] + out["gleason_secondary"]
    else:
        out["gleason_primary"] = out["gleason_secondary"] = out["gleason_sum"] = None
    out["gleason_tertiary_present"] = 1.0 if TERTIARY_RE.search(text) else 0.0

    m = STAGE_RE.search(text)
    out["pt_stage_ordinal"] = stage_ordinal(m.group(1)) if m else None

    m = EPE_RE.search(text)
    out["epe_present"] = (1.0 if m and m.group(0).startswith("Extraprostatic") else
                           0.0 if m else None)

    m = MARGINS_RE.search(text)
    out["margins_positive"] = {"positive": 1.0, "negative": 0.0}.get(m.group(1)) if m else None

    m = SVI_RE.search(text)
    out["svi_present"] = {"invaded": 1.0, "not invaded": 0.0}.get(m.group(1)) if m else None

    m = LVI_RE.search(text)
    out["lvi_present"] = {"present": 1.0, "absent": 0.0}.get(m.group(1)) if m else None

    m = LN_RE.search(text)
    if m is None:
        out["ln_assessed"] = out["ln_positive"] = None
    elif m.group(0) == "no lymph nodes were removed":
        out["ln_assessed"], out["ln_positive"] = 0.0, None
    elif m.group(0) == "lymph node metastasis was present":
        out["ln_assessed"], out["ln_positive"] = 1.0, 1.0
    else:  # "there was no lymph node metastasis"
        out["ln_assessed"], out["ln_positive"] = 1.0, 0.0

    return out


def main():
    tdir = PROC_ROOT / TASK
    with open(tdir / "structured.csv", newline="", encoding="utf-8") as f:
        structured = {r["case_id"]: r for r in csv.DictReader(f)}
    clinical = {}
    with open(tdir / "clinical_notes.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            clinical[rec["case_id"]] = rec

    rows = []
    for case_id, srow in structured.items():
        out = {"case_id": case_id}
        abnormal, bilateral, nodule, ct_stage = parse_dre(srow.get("dre"))
        out["dre_abnormal"] = abnormal
        out["dre_bilateral"] = bilateral
        out["dre_nodule"] = nodule
        out["dre_ct_stage_ordinal"] = ct_stage
        out["active_treatment_flag"] = parse_active_treatment(srow.get("active_treatment_prior_to_surgery"))
        out.update(parse_surgical_pathology(clinical.get(case_id, {}).get("surgical_pathology_report")))
        rows.append(out)

    fieldnames = ["case_id"] + [k for k in rows[0] if k != "case_id"]
    out_path = tdir / "engineered_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Wrote {out_path} ({n} cases, {len(fieldnames) - 1} engineered columns)\n")
    print(f"{'column':28s} {'non_null':>9s} {'coverage':>9s}")
    for col in fieldnames[1:]:
        non_null = sum(1 for r in rows if r[col] is not None)
        print(f"{col:28s} {non_null:9d} {non_null / n:9.1%}")


if __name__ == "__main__":
    main()
