"""Extract task2 engineered features: (1) `ct` (clinical T-stage, e.g. "cT2a") parsed into an
ordinal, reusing task3's exact stage_ordinal() grammar (same [pc]T<digit><subletter?> notation,
confirmed by direct scan: 'cT1c','cT2','cT2a','cT2c','cT3','cT3a','cT3b','cTx' -- cTx -> None, an
indeterminate stage, not a severity rung), (2) `comorbidity_count`, the number of
semicolon-separated conditions in `pmhx` (100% coverage, always-available -- NOT gated behind any
reveal section; see notebooks/task2_eda.ipynb §3 and the grounding note below), (3) PSA-kinetics
and lab-panel features from `psa_trend`/`laboratory_results` (genuinely new signal, not redundant
with `structured.csv`'s `psa` column: confirmed directly that `psa` == `psa_trend`'s last reading
== the lab panel's own PSA entry, so only the *history* and the *other* lab values add anything),
(4) `family_history_present`, and (5) `radiology_report`/`pathology_report` staging findings --
extracapsular extension, seminal vesicle invasion, AI-vs-pathologist ISUP discordance, cribriform
pattern, perineural invasion. Group (5) is real free prose, not the closed templated sentences
task3's surgical-pathology text had -- every pattern below was iteratively confirmed against a
direct corpus scan (documented in experiments/exp_18/DESIGN.md), not guessed at plausible wording;
a naive first-pass regex on this corpus produced real false positives (e.g. matching "clinical
suspicion for extracapsular extension" in the exam's *indication* line, or missing the many suffix-
style negations like "Perineural invasion is absent") that were caught and fixed before writing
the final patterns here.

Reads:
  data/processed/task2/structured.csv
  data/processed/task2/clinical_notes.jsonl   (psa_trend, laboratory_results, family_history,
                                                radiology_report, pathology_report)

Writes:
  data/processed/task2/engineered_features.csv  (case_id + new columns, additive)
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
TASK = "task2"


def stage_ordinal(stage_str):
    """Ported verbatim from scripts/extract_features_task3.py -- same cT/pT grammar, confirmed
    directly against task2's own `ct` column (8 distinct values, all matching)."""
    if not stage_str:
        return None
    m = re.match(r"[pc]T(\d)([a-z]?)", stage_str)
    if not m:
        return None
    num = int(m.group(1))
    sub_rank = {"": 0, "a": 1, "b": 2, "c": 3}.get(m.group(2), 0)
    return num * 10 + sub_rank


def parse_comorbidity_count(pmhx_text):
    if not pmhx_text or not pmhx_text.strip():
        return 0.0
    return float(len(pmhx_text.split(";")))


# --- psa_trend: PSA-velocity features -----------------------------------------------------------
# 3-5 readings/case, always >=3 (100% coverage, confirmed in notebooks/task2_eda.ipynb §5). Dates
# are "Mon YYYY" strings, chronologically ordered (confirmed by direct scan). The final reading
# always equals structured.csv's `psa` -- only the *trajectory* is new signal.
def parse_psa_velocity(psa_trend):
    if not psa_trend or len(psa_trend) < 2:
        return {"psa_velocity_per_year": None, "psa_ratio_last_first": None}
    first, last = psa_trend[0], psa_trend[-1]
    d_first = datetime.strptime(first["date"], "%b %Y")
    d_last = datetime.strptime(last["date"], "%b %Y")
    years = (d_last - d_first).days / 365.25
    if years <= 0:
        return {"psa_velocity_per_year": None, "psa_ratio_last_first": None}
    velocity = (last["val"] - first["val"]) / years
    ratio = last["val"] / first["val"] if first["val"] > 0 else None
    return {"psa_velocity_per_year": velocity, "psa_ratio_last_first": ratio}


# --- laboratory_results: the panel beyond PSA itself ---------------------------------------------
# 100% coverage for every name except HbA1c (36/153, diabetics only -- left out, too sparse to be
# worth the missingness). eGFR is reported as a threshold ("<60"/">60"), not an exact value --
# encoded as a binary flag, not a fabricated number.
LAB_NUMERIC_FIELDS = {
    "Free PSA": "free_psa", "%Free PSA": "pct_free_psa", "Testosterone": "testosterone",
    "Creatinine": "creatinine", "Alkaline phosphatase": "alk_phos", "LDH": "ldh",
}
VALUE_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_lab_value(val_str):
    m = VALUE_RE.search(val_str)
    return float(m.group(0)) if m else None


def parse_labs(lab_list):
    out = {col: None for col in LAB_NUMERIC_FIELDS.values()}
    out["egfr_reduced"] = None
    if not lab_list:
        return out
    for lab in lab_list:
        name = lab.get("name")
        if name in LAB_NUMERIC_FIELDS:
            out[LAB_NUMERIC_FIELDS[name]] = parse_lab_value(lab["val"])
        elif name == "eGFR":
            out["egfr_reduced"] = 1.0 if lab["val"].startswith("<") else 0.0
    return out


# --- family_history: completes the 11-variable evaluator-scored vocabulary (task2_schema.
# TASK2_VARIABLES) as a usable numeric feature -- previously only consumed for tool-gating logic,
# never as a model input. Fully closed vocabulary, 3 values, 100% coverage (No 124, Yes 21,
# Unknown 8) -- "Unknown" kept as None alongside any raw-missing case, same convention as every
# other missingness in this project (task3's family_history_present in scripts/
# extract_features_task3.py uses the identical Yes/No/Unknown -> 1.0/0.0/None mapping).
FAMILY_HISTORY_MAP = {"Yes": 1.0, "No": 0.0}


def parse_family_history(fh_value):
    return FAMILY_HISTORY_MAP.get(fh_value)  # "Unknown" or missing -> None


# --- radiology_report: EPE / SVI (exp_18) ---------------------------------------------------------
# Real free prose, not task3's closed templated sentences -- confirmed by direct corpus scan
# (experiments/exp_18/DESIGN.md). Restricted to the FINDINGS/IMPRESSION sections: the "Clinical
# indication" line describes why the scan was ordered ("...clinical suspicion for extracapsular
# extension"), not the actual finding -- an early naive regex without this restriction produced
# false positives there. Negation is checked as a suffix immediately before the term (handles
# "No extracapsular extension", "...without evidence of...", etc.) -- a lookback window of 70 chars
# was enough for EPE's shorter sentences.
EPE_RE = re.compile(r"extra(?:capsular|prostatic) extension", re.IGNORECASE)
SVI_RE = re.compile(r"seminal vesicle (?:invasion|involvement|infiltration)", re.IGNORECASE)
NEG_PREFIX_CUES = ["no evidence of", "without", "no ", "not visualized", "not identified",
                    "not seen", "not observed", "no suspicious"]


def _results_section(radiology_text):
    m = re.search(r"FINDINGS", radiology_text)
    return radiology_text[m.start():] if m else radiology_text


def _prefix_negated_finding(text, term_re, window=70):
    """present (1.0) / absent (0.0) / None if the term never appears -- confirmed against a direct
    corpus scan, see EPE_RE/SVI_RE's docstring above."""
    body = _results_section(text)
    m = term_re.search(body)
    if not m:
        return None
    w = body[max(0, m.start() - window):m.start()].lower()
    return 0.0 if any(cue in w for cue in NEG_PREFIX_CUES) else 1.0


# --- pathology_report: AI-ISUP discordance / cribriform / perineural invasion (exp_18) ------------
AI_ISUP_RE = re.compile(r"AI-predicted ISUP grade group.*?grade group (\d)", re.IGNORECASE)
CRIBRIFORM_RE = re.compile(r"cribriform", re.IGNORECASE)
PNI_RE = re.compile(r"perineural invasion", re.IGNORECASE)

# Cribriform: purely prefix-negated in this corpus ("No cribriform ...") -- confirmed by direct
# scan, window=25 chars, strict endswith (not substring-anywhere, which produced false positives
# on an earlier pass by matching an unrelated "no" earlier in a longer preceding clause).
CRIBRIFORM_NEG_PREFIX = ["no", "without", "no evidence of", "not identified as"]

# Perineural invasion: this corpus mixes prefix ("No perineural invasion") AND suffix
# ("Perineural invasion is absent" / "...is identified") negation/affirmation -- confirmed by
# direct scan; task3's equivalent pathology sentences never needed a suffix check.
PNI_NEG_PREFIX = ["no", "without", "no evidence of"]
PNI_NEG_SUFFIX = ["is absent", "was absent", "absent", "is not observed", "is not identified",
                  "is not present", "is not seen", "was not observed", "was not identified",
                  "was not seen"]
PNI_POS_SUFFIX = ["is present", "present", "is identified", "is confirmed", "is observed",
                  "was identified", "was confirmed", "was observed"]


def parse_ai_isup(pathology_text, bx_isup):
    """AI-predicted ISUP minus the pathologist's own bx_isup (signed) -- the discordance is the
    genuinely new signal, confirmed not redundant with the already-included bx_isup itself."""
    m = AI_ISUP_RE.search(pathology_text or "")
    if not m or bx_isup is None:
        return None
    return float(m.group(1)) - float(bx_isup)


def parse_cribriform(pathology_text):
    text = pathology_text or ""
    m = CRIBRIFORM_RE.search(text)
    if not m:
        return None
    before = text[max(0, m.start() - 25):m.start()].lower().strip()
    return 0.0 if any(before.endswith(c) for c in CRIBRIFORM_NEG_PREFIX) else 1.0


def parse_perineural_invasion(pathology_text):
    text = pathology_text or ""
    m = PNI_RE.search(text)
    if not m:
        return None
    before = text[max(0, m.start() - 25):m.start()].lower().strip()
    if any(before.endswith(c) for c in PNI_NEG_PREFIX):
        return 0.0
    after = text[m.end():m.end() + 22].strip().lower()
    for cue in PNI_NEG_SUFFIX:
        if after.startswith(cue):
            return 0.0
    for cue in PNI_POS_SUFFIX:
        if after.startswith(cue):
            return 1.0
    return None  # mentioned but neither pattern matched -- honest miss, not a guess


def main():
    tdir = PROC_ROOT / TASK
    with open(tdir / "structured.csv", newline="", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))
    clinical = {}
    with open(tdir / "clinical_notes.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            clinical[rec["case_id"]] = rec

    rows = []
    for srow in rows_in:
        case_id = srow["case_id"]
        out = {"case_id": case_id}
        out["ct_ordinal"] = stage_ordinal(srow.get("ct"))
        out["comorbidity_count"] = parse_comorbidity_count(srow.get("pmhx"))
        crec = clinical.get(case_id, {})
        out.update(parse_psa_velocity(crec.get("psa_trend")))
        out.update(parse_labs(crec.get("laboratory_results")))
        out["family_history_present"] = parse_family_history(crec.get("family_history"))
        radiology_text = crec.get("radiology_report") or ""
        out["epe_present"] = _prefix_negated_finding(radiology_text, EPE_RE)
        out["svi_present"] = _prefix_negated_finding(radiology_text, SVI_RE)
        pathology_text = crec.get("pathology_report") or ""
        bx_isup_val = srow.get("bx_isup")
        out["ai_isup_discordance"] = parse_ai_isup(pathology_text, bx_isup_val if bx_isup_val else None)
        out["cribriform_present"] = parse_cribriform(pathology_text)
        out["perineural_invasion_present"] = parse_perineural_invasion(pathology_text)
        rows.append(out)

    fieldnames = ["case_id"] + [k for k in rows[0] if k != "case_id"]
    out_path = tdir / "engineered_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Wrote {out_path} ({n} cases, {len(fieldnames) - 1} engineered columns)\n")
    print(f"{'column':20s} {'non_null':>9s} {'coverage':>9s}")
    for col in fieldnames[1:]:
        non_null = sum(1 for r in rows if r[col] is not None)
        print(f"{col:20s} {non_null:9d} {non_null / n:9.1%}")


if __name__ == "__main__":
    main()
