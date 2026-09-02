"""Honest task2 LOOCV: each of the 72 labeled cases scored by a model trained on the other 71
(the 81 unlabeled cases are excluded -- no ground truth to validate against, see
notebooks/task2_eda.ipynb §1). Uses task2_subtask_config.py's per-sub-task configs, mirrors
run_agent_loocv.py (task1) exactly. Writes the two-file GC-style outputs to disk.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from task2_free_text import build_free_text  # noqa: E402
from task2_mock_tools import (  # noqa: E402
    TOOL_TO_SECTION, get_family_history, get_lab_results, get_mri_report,
    get_pathology_report, get_previous_notes, get_psa_trend,
)
from task2_schema import (  # noqa: E402
    Confidence, Task2Output, TreatmentDecision, Weight, enforce_eligibility,
    normalise_to_full_shape,
)
from task2_subtask_config import compute_called_tools, default_configs  # noqa: E402
from task2_targets import (  # noqa: E402
    TASK2_VARIABLES, parse_confidence, parse_decision, parse_variable_weights,
)

EXP2_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "experiments" / "exp_2" / "scripts"
sys.path.insert(0, str(EXP2_SCRIPTS))
from common import build_pipeline  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task2"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "task2"

CONF_CODE_TO_LABEL = {0: "uncertain", 1: "borderline", 2: "clear"}
WEIGHT_CODE_TO_LABEL = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}
TOOL_CALLERS = {
    "get_mri_report": get_mri_report, "get_pathology_report": get_pathology_report,
    "get_psa_trend": get_psa_trend, "get_lab_results": get_lab_results,
    "get_previous_notes": get_previous_notes, "get_family_history": get_family_history,
}


def write_outputs(output: Task2Output):
    case_dir = OUTPUT_DIR / output.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "prostate-treatment-decision.json", "w") as f:
        json.dump(output.treatment_recommendation.value, f)
    reasoning = {
        "confidence": output.confidence.value,
        "variable_weights": {k: v.value for k, v in output.variable_weights.items()},
        "reveal_sequence": output.reveal_sequence,
        "free_text": output.reasoning,
    }
    with open(case_dir / "prostate-treatment-decision-reasoning.json", "w") as f:
        json.dump(reasoning, f, indent=2)


def _fit_predict_one(tr, va, numeric, categorical, model_factory, y_tr):
    pipe = build_pipeline(numeric, categorical)
    X_tr, X_va = pipe.fit_transform(tr), pipe.transform(va)
    if len(np.unique(y_tr)) < 2:
        return y_tr[0]
    m = model_factory()
    m.fit(X_tr, y_tr)
    return m.predict(X_va)[0]


def load_labeled_df():
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")  # only the 72 labeled cases
    df = structured.merge(engineered, on="case_id", how="inner").merge(labels, on="case_id", how="inner")
    numeric_cols = {"age", "psa", "psad", "pirads", "bx_isup", "bx_gl_prim", "bx_gl_sec", "cspca",
                     "ct_ordinal", "comorbidity_count", "psa_velocity_per_year",
                     "psa_ratio_last_first", "free_psa", "pct_free_psa", "testosterone",
                     "creatinine", "alk_phos", "ldh", "egfr_reduced", "family_history_present",
                     "epe_present", "svi_present", "ai_isup_discordance", "cribriform_present",
                     "perineural_invasion_present"}
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def main():
    df = load_labeled_df()

    configs = default_configs(df)
    called_tools = compute_called_tools(configs)
    reveal_sequence = [TOOL_TO_SECTION[t] for t in called_tools]
    print(f"n_labeled_cases={len(df)}")
    print(f"Reveal policy (union across all 3 sub-task configs): {called_tools}")

    y_decision = parse_decision(df)
    y_confidence = parse_confidence(df)
    y_weights = parse_variable_weights(df)  # (72, 11)

    n = len(df)
    n_ok = 0
    for i in range(n):
        case_id = df.iloc[i]["case_id"]
        for tool in called_tools:
            TOOL_CALLERS[tool](case_id)  # real calls; content unused in this baseline
        tr_idx = [j for j in range(n) if j != i]
        tr, va = df.iloc[tr_idx], df.iloc[[i]]

        dc = configs["decision"]
        decision_pred = str(_fit_predict_one(tr, va, dc["numeric"], dc["categorical"],
                                              dc["model_factory"], y_decision[tr_idx]))

        cc = configs["confidence"]
        conf_code = int(_fit_predict_one(tr, va, cc["numeric"], cc["categorical"],
                                          cc["model_factory"], y_confidence[tr_idx]))
        confidence_label = CONF_CODE_TO_LABEL[conf_code]

        wc = configs["variable_weights"]
        raw_weights = {}
        for v_idx, var in enumerate(TASK2_VARIABLES):
            code = int(_fit_predict_one(tr, va, wc["numeric"], wc["categorical"],
                                         wc["model_factory"], y_weights[tr_idx, v_idx]))
            raw_weights[var] = WEIGHT_CODE_TO_LABEL[code]

        grounded_weights = enforce_eligibility(raw_weights, called_tools)
        full_weights = normalise_to_full_shape(grounded_weights)

        structured_row = va.iloc[0].to_dict()
        free_text = build_free_text(structured_row, decision_pred, confidence_label, full_weights)

        output = Task2Output(
            case_id=case_id, treatment_recommendation=TreatmentDecision(decision_pred),
            confidence=Confidence(confidence_label),
            variable_weights={k: Weight(v) for k, v in full_weights.items()},
            reveal_sequence=reveal_sequence, reasoning=free_text,
        )
        write_outputs(output)
        n_ok += 1

    print(f"Wrote {n_ok} LOOCV (leakage-free) cases to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
