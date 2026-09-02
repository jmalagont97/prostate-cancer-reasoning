"""Scores agent/output/task2 against ground truth using the REAL evaluate.py scoring functions,
imported directly (not reimplemented). Mirrors local_eval.py (task1) with the two real differences
confirmed from evaluate.py source: decision_score/task_kind take task="treatment" (nested
treatment_recommendation.primary, not a flat field), and the leaderboard F1 is
support-weighted across the 4 treatment classes (decision_weighted_f1), not a single positive-class
F1 -- both read directly from compute_aggregate_metrics(), not re-derived.
"""

import json
import sys
from pathlib import Path
from statistics import mean

EVAL_DIR = Path(__file__).resolve().parent.parent / "external" / "CHIMERA-agent" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from evaluate import (  # noqa: E402
    VALID_TREATMENT_DECISIONS, compute_tool_score, confidence_score, decision_score,
    important_decisive_factor_score, section_grounding_score, variable_weight_score,
)
from sklearn.metrics import f1_score  # noqa: E402

RAW_TASK2 = Path(__file__).resolve().parent.parent / "data" / "train_release_v2" / "train_release" / "task2"
AGENT_OUTPUT = Path(__file__).resolve().parent / "output" / "task2"

COMPONENT_WEIGHTS = {  # confirmed from evaluate.py source, rationale-off renormalization
    "confidence": 0.225, "var_weight": 0.275, "factor_f1": 0.175,
    "tool": 0.150, "section_grounding": 0.175,
}


def load_pred_record(case_dir: Path) -> dict:
    decision = json.load(open(case_dir / "prostate-treatment-decision.json"))
    reasoning = json.load(open(case_dir / "prostate-treatment-decision-reasoning.json"))
    record = dict(reasoning)
    record["treatment_recommendation"] = {"primary": decision}
    return record


def load_gt_record(case_dir: Path) -> dict:
    decision = json.load(open(case_dir / "prostate-treatment-decision.json"))
    reasoning = json.load(open(case_dir / "prostate-treatment-decision-reasoning.json"))
    record = dict(reasoning)
    record["treatment_recommendation"] = {"primary": decision}
    return record


def score_case(case_id: str, gt: dict, pred: dict) -> dict:
    ds, gt_dec, pred_dec, d_reason = decision_score("treatment", gt, pred)
    row = {"case_id": case_id, "decision_score": ds, "gt_decision": gt_dec, "pred_decision": pred_dec}

    if ds == 0.0:
        row["case_score"] = 0.0
        row["gate"] = "decision_failed"
        row["reason"] = d_reason
        return row

    cs = confidence_score(gt, pred)
    vws = variable_weight_score(gt, pred)
    fs = important_decisive_factor_score(gt, pred)
    ts, t_reason = compute_tool_score(gt, pred, None)
    sgs, sg_details = section_grounding_score(pred)

    components = {
        "confidence": cs, "var_weight": vws, "factor_f1": fs, "tool": ts, "section_grounding": sgs,
    }
    score = sum((v if v is not None else 0.0) * COMPONENT_WEIGHTS[k] for k, v in components.items())
    row.update({
        "case_score": max(0.0, min(1.0, score)), "gate": "passed",
        "confidence_score": cs, "variable_weight_score": vws, "important_decisive_factor_score": fs,
        "tool_score": ts, "tool_reason": t_reason, "section_grounding_score": sgs,
        "ungrounded_variables": sg_details.get("ungrounded_variables", []),
    })
    return row


def main():
    case_dirs = sorted(p for p in AGENT_OUTPUT.iterdir() if p.is_dir())
    rows = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        gt_dir = RAW_TASK2 / case_id
        if not gt_dir.exists() or not (gt_dir / "prostate-treatment-decision.json").exists():
            print(f"[skip] no raw ground truth for {case_id}")
            continue
        gt = load_gt_record(gt_dir)
        pred = load_pred_record(case_dir)
        rows.append(score_case(case_id, gt, pred))

    mean_case_score = mean(r["case_score"] for r in rows)
    y_true = [r["gt_decision"] for r in rows]
    y_pred = [r["pred_decision"] or "__missing__" for r in rows]
    decision_weighted_f1 = f1_score(
        y_true, y_pred, labels=sorted(VALID_TREATMENT_DECISIONS), average="weighted", zero_division=0,
    )
    ranking_score = (mean_case_score + decision_weighted_f1) / 2.0

    passed = [r for r in rows if r["gate"] == "passed"]
    print(f"n_cases={len(rows)}  gate_passed={len(passed)}  gate_pass_rate={len(passed)/len(rows):.3f}")
    print(f"mean_case_score={mean_case_score:.3f}")
    print(f"decision_weighted_f1={decision_weighted_f1:.3f}")
    print(f"RANKING_SCORE={ranking_score:.3f}")
    print()

    from collections import Counter
    print("decision confusion (gt -> pred), all cases:")
    for gt_lab in sorted(VALID_TREATMENT_DECISIONS):
        preds_for_gt = [r["pred_decision"] for r in rows if r["gt_decision"] == gt_lab]
        c = Counter(preds_for_gt)
        print(f"  {gt_lab:24s} n={len(preds_for_gt):2d}  {dict(c)}")

    if passed:
        print()
        for comp in ["confidence_score", "variable_weight_score", "important_decisive_factor_score",
                     "tool_score", "section_grounding_score"]:
            vals = [r[comp] for r in passed if r.get(comp) is not None]
            if vals:
                print(f"  mean {comp}: {mean(vals):.3f}  (n={len(vals)})")
        n_ungrounded = sum(len(r.get("ungrounded_variables", [])) for r in passed)
        print(f"  total ungrounded-variable instances across all passed cases: {n_ungrounded}")

    out_path = Path(__file__).resolve().parent / "output" / "eval_results_task2.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "mean_case_score": mean_case_score,
                    "decision_weighted_f1": decision_weighted_f1, "ranking_score": ranking_score},
                   f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
