"""Grounded template free-text generation for task2. NOT LLM-authored -- no LLM available in this
environment, same constraint as free_text.py (task1). Cites only values actually present in the
structured row, never fabricates.
"""

VARIABLE_LABELS = {
    "age": "age", "psa": "PSA", "psad": "PSA density", "pirads": "PI-RADS score",
    "cspca": "csPCa probability", "ct": "clinical T-stage",
    "bx_isup": "biopsy ISUP grade group", "bx_gl_prim": "biopsy Gleason primary pattern",
    "bx_gl_sec": "biopsy Gleason secondary pattern", "comorbidity": "comorbidity burden",
    "fh": "family history",
}

VALUE_FIELD = {
    "age": "age", "psa": "psa", "psad": "psad", "pirads": "pirads", "cspca": "cspca",
    "ct": "ct", "bx_isup": "bx_isup", "bx_gl_prim": "bx_gl_prim", "bx_gl_sec": "bx_gl_sec",
}

WEIGHT_RANK = {"decisive": 3, "important": 2, "noted": 1, "not_used": 0}

DECISION_LABELS = {
    "watchful_waiting": "watchful waiting",
    "active_surveillance": "active surveillance",
    "continued_surveillance": "continued surveillance",
    "active_treatment": "active treatment",
}


def build_free_text(structured_row: dict, decision: str, confidence: str,
                     variable_weights: dict) -> str:
    ranked = sorted(
        [(v, w) for v, w in variable_weights.items() if w in ("decisive", "important")],
        key=lambda vw: -WEIGHT_RANK[vw[1]],
    )[:4]

    if not ranked:
        drivers = "no single factor stood out as decisive or important"
    else:
        parts = []
        for var, weight in ranked:
            label = VARIABLE_LABELS.get(var, var)
            field = VALUE_FIELD.get(var)
            value = structured_row.get(field) if field else None
            parts.append(f"{label} ({value}, {weight})" if value is not None else f"{label} ({weight})")
        drivers = ", ".join(parts)

    verdict = DECISION_LABELS.get(decision, decision)
    return (
        f"Based on the available clinical picture, {verdict} is recommended, with {confidence} "
        f"confidence. The primary factors weighed were: {drivers}. "
        f"(Template-generated summary of model outputs -- not LLM-authored.)"
    )
