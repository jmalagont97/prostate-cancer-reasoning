#!/usr/bin/env python3
"""Evaluate deploy/agent Task-1 artifacts with the exp_1-compatible metrics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from chimera_metrics import (
    compute_biopsy_decision_gate,
    compute_confidence_score,
    compute_cost_aware_tool_efficiency,
    compute_important_decisive_f1,
    compute_section_grounding,
    compute_variable_weights_score,
    evaluate_task1_case,
)

ROOT = Path(__file__).resolve().parents[3]
GT_ROOT = ROOT / "data/raw/train_release/task1"
PRED_ROOT = ROOT / "experiments/exp_10/results/deploy_agent/outputs/task1"
OUT = ROOT / "experiments/exp_10/results/deploy_agent"
MAPPING = ROOT / "docs/chimera-agent-evalaution/evaluation/ground_truth/section_variable_mapping.json"


def load_case(cid: str) -> tuple[dict, dict]:
    gt_dir = GT_ROOT / cid
    pred_dir = PRED_ROOT / cid
    with (gt_dir / "prostate-biopsy-decision.json").open() as f:
        gt_decision = json.load(f)
    with (gt_dir / "prostate-biopsy-decision-reasoning.json").open() as f:
        gt_reasoning = json.load(f)
    with (pred_dir / "prostate-biopsy-decision.json").open() as f:
        pred_decision = json.load(f)
    with (pred_dir / "prostate-biopsy-decision-reasoning.json").open() as f:
        pred_reasoning = json.load(f)
    gt = {"biopsy_decision": gt_decision, **gt_reasoning}
    pred = {"biopsy_decision": pred_decision, **pred_reasoning}
    return gt, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with MAPPING.open() as f:
        mapping = json.load(f)
    case_ids = sorted(
        p.name for p in GT_ROOT.iterdir()
        if p.is_dir()
        and (p / "prostate-biopsy-decision.json").exists()
        and (p / "prostate-biopsy-decision-reasoning.json").exists()
        and (PRED_ROOT / p.name / "prostate-biopsy-decision.json").exists()
        and (PRED_ROOT / p.name / "prostate-biopsy-decision-reasoning.json").exists()
    )
    if len(case_ids) != 91:
        raise RuntimeError(f"Expected 91 paired cases, found {len(case_ids)}")

    results = []
    for i, cid in enumerate(case_ids, 1):
        gt, pred = load_case(cid)
        result = evaluate_task1_case(cid, gt, pred, section_mapping=mapping)
        results.append(result)
        print(f"[eval {i:02d}/91] {cid} gate={result.decision_gate:.0f} score={result.overall_case_score:.3f}", flush=True)

    y_true = [r.decision_gt for r in results]
    y_pred = [r.decision_pred for r in results]
    tp = sum(a == "yes" and b == "yes" for a, b in zip(y_true, y_pred))
    fp = sum(a == "no" and b == "yes" for a, b in zip(y_true, y_pred))
    fn = sum(a == "yes" and b == "no" for a, b in zip(y_true, y_pred))
    tn = sum(a == "no" and b == "no" for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    valid = lambda key: [getattr(r, key) for r in results if getattr(r, key) is not None]
    summary = {
        "n_cases": len(results),
        "n_decision_correct": sum(r.decision_gate == 1.0 for r in results),
        "decision_accuracy": mean(r.decision_gate for r in results),
        "decision_f1_yes": f1,
        "precision_yes": precision,
        "recall_yes": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mean_case_score": mean(r.overall_case_score for r in results),
        "ranking_score": 0.5 * mean(r.overall_case_score for r in results) + 0.5 * f1,
        "mean_case_score_among_gate_passed": mean([r.overall_case_score for r in results if r.decision_gate == 1.0]),
        "mean_confidence_score": mean(valid("confidence_score")),
        "mean_variable_weights_score": mean(valid("variable_weights_score")),
        "mean_important_decisive_f1": mean(valid("important_decisive_f1")),
        "mean_tool_efficiency_score": mean(valid("tool_efficiency_score")),
        "mean_section_grounding_score": mean(valid("section_grounding_score")),
    }
    (OUT / "case_metrics.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    (OUT / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
