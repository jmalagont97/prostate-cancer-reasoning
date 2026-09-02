"""In-memory LOOCV + real-evaluate.py-function scoring for task2, generalized to vary any
sub-task's {numeric, categorical, model_factory, required_tools} -- mirrors variant_tester.py
(task1) exactly. No disk writes; test_variant() returns the full component breakdown so a swap's
trade-offs are visible, not just the headline ranking_score.
"""

import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from task2_free_text import build_free_text  # noqa: E402
from task2_mock_tools import TOOL_TO_SECTION  # noqa: E402
from task2_schema import enforce_eligibility, normalise_to_full_shape  # noqa: E402
from task2_subtask_config import compute_called_tools, merge_overrides  # noqa: E402
from task2_targets import (  # noqa: E402
    TASK2_VARIABLES, VALID_TREATMENT_DECISIONS, parse_confidence, parse_decision,
    parse_variable_weights,
)

EXP2_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "experiments" / "exp_2" / "scripts"
sys.path.insert(0, str(EXP2_SCRIPTS))
from common import build_pipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "CHIMERA-agent" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))
from evaluate import (  # noqa: E402
    compute_tool_score, confidence_score, decision_score, important_decisive_factor_score,
    section_grounding_score, variable_weight_score,
)
from sklearn.metrics import f1_score  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task2"

CONF_CODE_TO_LABEL = {0: "uncertain", 1: "borderline", 2: "clear"}
WEIGHT_CODE_TO_LABEL = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}
COMPONENT_WEIGHTS = {
    "confidence": 0.225, "var_weight": 0.275, "factor_f1": 0.175,
    "tool": 0.150, "section_grounding": 0.175,
}


def load_labeled_df():
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")
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


def _fit_predict_one(tr, va, numeric, categorical, model_factory, y_tr):
    pipe = build_pipeline(numeric, categorical)
    X_tr, X_va = pipe.fit_transform(tr), pipe.transform(va)
    if len(np.unique(y_tr)) < 2:
        return y_tr[0]
    m = model_factory()
    m.fit(X_tr, y_tr)
    return m.predict(X_va)[0]


def _score_case(gt: dict, pred: dict) -> dict:
    ds, gt_dec, pred_dec, d_reason = decision_score("treatment", gt, pred)
    row = {"decision_score": ds, "gt_decision": gt_dec, "pred_decision": pred_dec}
    if ds == 0.0:
        row["case_score"] = 0.0
        row["gate"] = "decision_failed"
        return row
    cs = confidence_score(gt, pred)
    vws = variable_weight_score(gt, pred)
    fs = important_decisive_factor_score(gt, pred)
    ts, _ = compute_tool_score(gt, pred, None)
    sgs, sg_details = section_grounding_score(pred)
    components = {"confidence": cs, "var_weight": vws, "factor_f1": fs, "tool": ts,
                  "section_grounding": sgs}
    score = sum((v if v is not None else 0.0) * COMPONENT_WEIGHTS[k] for k, v in components.items())
    row.update({
        "case_score": max(0.0, min(1.0, score)), "gate": "passed",
        "confidence_score": cs, "variable_weight_score": vws, "important_decisive_factor_score": fs,
        "tool_score": ts, "section_grounding_score": sgs,
        "ungrounded_variables": sg_details.get("ungrounded_variables", []),
    })
    return row


def test_variant(name: str, config_overrides: dict, df=None, verbose: bool = True) -> dict:
    if df is None:
        df = load_labeled_df()
    configs = merge_overrides(df, config_overrides)
    called_tools = compute_called_tools(configs)
    reveal_sequence = [TOOL_TO_SECTION[t] for t in called_tools]

    y_decision = parse_decision(df)
    y_confidence = parse_confidence(df)
    y_weights = parse_variable_weights(df)

    n = len(df)
    rows = []
    for i in range(n):
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

        gt_row = df.iloc[i]
        gt = {
            "treatment_recommendation": {"primary": gt_row["decision"]},
            "confidence": gt_row["confidence"],
            "variable_weights": {var: WEIGHT_CODE_TO_LABEL[y_weights[i, v_idx]]
                                  for v_idx, var in enumerate(TASK2_VARIABLES)},
            "reveal_sequence": [],
        }
        pred = {
            "treatment_recommendation": {"primary": decision_pred},
            "confidence": confidence_label,
            "variable_weights": full_weights,
            "reveal_sequence": reveal_sequence,
        }
        rows.append(_score_case(gt, pred))

    mean_case_score = mean(r["case_score"] for r in rows)
    y_true = [r["gt_decision"] for r in rows]
    y_pred = [r["pred_decision"] or "__missing__" for r in rows]
    decision_weighted_f1 = f1_score(
        y_true, y_pred, labels=sorted(VALID_TREATMENT_DECISIONS), average="weighted", zero_division=0,
    )
    ranking_score = (mean_case_score + decision_weighted_f1) / 2.0
    passed = [r for r in rows if r["gate"] == "passed"]

    result = {
        "name": name, "n_cases": n, "gate_pass_rate": len(passed) / n,
        "mean_case_score": mean_case_score, "decision_weighted_f1": decision_weighted_f1,
        "ranking_score": ranking_score, "called_tools": sorted(called_tools),
    }
    for comp in ["confidence_score", "variable_weight_score", "important_decisive_factor_score",
                 "tool_score", "section_grounding_score"]:
        vals = [r[comp] for r in passed if r.get(comp) is not None]
        result[f"mean_{comp}"] = mean(vals) if vals else None

    if verbose:
        print(f"{name}: ranking_score={ranking_score:.4f}  mean_case_score={mean_case_score:.4f}  "
              f"decision_f1={decision_weighted_f1:.4f}  tools={sorted(called_tools)}")
        print(f"  confidence={result['mean_confidence_score']:.3f}  "
              f"var_weight={result['mean_variable_weight_score']:.3f}  "
              f"factor_f1={result['mean_important_decisive_factor_score']:.3f}  "
              f"tool={result['mean_tool_score']:.3f}  "
              f"section_grounding={result['mean_section_grounding_score']:.3f}")
    return result


if __name__ == "__main__":
    df = load_labeled_df()
    test_variant("current_default (mri+path grounding)", {}, df=df)
