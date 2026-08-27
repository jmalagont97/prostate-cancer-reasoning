"""Grounded template free-text generation for task3. NOT LLM-authored -- same constraint/pattern
as agent/src/free_text.py (task1): no LLM available in this environment; cites only values actually
present in the case row, never fabricates. Task3 has no variable_weights to rank, so the driving
factors here are simply the pathology fields with the strongest established prognostic weight
(ISUP grade, pT stage, margins, EPE, SVI, lymph-node status) when they're actually available.
"""

ISUP_LABEL = {1: "ISUP grade group 1", 2: "ISUP grade group 2", 3: "ISUP grade group 3",
              4: "ISUP grade group 4", 5: "ISUP grade group 5"}


def _pt_stage_label(ordinal):
    if ordinal is None:
        return None
    num, sub = divmod(int(ordinal), 10)
    letter = {0: "", 1: "a", 2: "b", 3: "c"}.get(sub, "")
    return f"pT{num}{letter}"


def build_free_text(row: dict, predicted_months: float) -> str:
    findings = []

    isup = row.get("isup_grade_group")
    if isup is not None and not (isinstance(isup, float) and isup != isup):  # not NaN
        findings.append(ISUP_LABEL.get(int(isup), f"ISUP grade group {int(isup)}"))

    stage = _pt_stage_label(row.get("pt_stage_ordinal"))
    if stage:
        findings.append(f"pathological stage {stage}")

    if row.get("margins_positive") == 1.0:
        findings.append("positive surgical margins")
    elif row.get("margins_positive") == 0.0:
        findings.append("negative surgical margins")

    if row.get("epe_present") == 1.0:
        findings.append("extraprostatic extension")

    if row.get("svi_present") == 1.0:
        findings.append("seminal vesicle invasion")

    if row.get("ln_assessed") == 1.0 and row.get("ln_positive") == 1.0:
        findings.append("positive lymph nodes")

    findings_str = "; ".join(findings) if findings else "the available pre- and post-surgical findings"
    risk_band = "high" if predicted_months < 24 else "moderate" if predicted_months < 60 else "low"

    return (
        f"Predicted time to biochemical recurrence (or last follow-up if none occurs) is "
        f"approximately {predicted_months:.1f} months, corresponding to a {risk_band}-risk "
        f"prognosis in this cohort. Key findings informing this estimate: {findings_str}. "
        f"(Template-generated summary of a survival-model prediction -- not LLM-authored; "
        f"see docs/model_training_plan.md and experiments/exp_3/.)"
    )
