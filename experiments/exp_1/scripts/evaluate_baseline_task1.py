#!/usr/bin/env python3
"""
Benchmark execution script for Task 1: Baseline vs Ground Truth Cohort.
Computes deterministic metrics, decision gates, reasoning alignment, and official ranking scores.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np

# Ensure chimera_metrics is accessible
from chimera_metrics import (
    ExperimentReporter,
    compute_biopsy_decision_gate,
    compute_confidence_score,
    compute_cost_aware_tool_efficiency,
    compute_important_decisive_f1,
    compute_section_grounding,
    compute_variable_weights_score,
    evaluate_task1_case,
)
from chimera_metrics.core.types import CONF_MAP, WEIGHT_MAP


def main():
    root = Path(__file__).resolve().parents[3]
    raw_task1_dir = root / "data" / "raw" / "train_release" / "task1"
    base_task1_dir = root / "data" / "base-line" / "task1"
    mapping_file = root / "docs" / "chimera-agent-evalaution" / "evaluation" / "ground_truth" / "section_variable_mapping.json"
    results_dir = root / "experiments" / "exp_1" / "results" / "baseline"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load section mapping
    section_mapping = None
    if mapping_file.exists():
        with open(mapping_file) as f:
            section_mapping = json.load(f)

    # Find paired cases
    all_raw_cases = sorted([d.name for d in raw_task1_dir.iterdir() if d.is_dir()])
    paired_cases: List[str] = []

    for cid in all_raw_cases:
        dec_file = raw_task1_dir / cid / "prostate-biopsy-decision.json"
        reas_file = raw_task1_dir / cid / "prostate-biopsy-decision-reasoning.json"
        pred_file = base_task1_dir / cid / "prediction.json"

        if dec_file.exists() and reas_file.exists() and pred_file.exists():
            paired_cases.append(cid)

    print(f"[*] Total raw cases in dataset: {len(all_raw_cases)}")
    print(f"[*] Paired ground truth & baseline cases: {len(paired_cases)}")

    # 1. Evaluate baseline predictions
    baseline_case_results = []
    oracle_case_results = []

    for cid in paired_cases:
        # Load GT
        with open(raw_task1_dir / cid / "prostate-biopsy-decision.json") as f:
            gt_dec = json.load(f)
        with open(raw_task1_dir / cid / "prostate-biopsy-decision-reasoning.json") as f:
            gt_reas = json.load(f)
        gt = {"biopsy_decision": gt_dec, **gt_reas}

        # Load Pred
        with open(base_task1_dir / cid / "prediction.json") as f:
            pred = json.load(f)

        # Baseline evaluation
        res_baseline = evaluate_task1_case(cid, gt, pred, section_mapping=section_mapping)
        baseline_case_results.append(res_baseline)

        # Oracle evaluation (upper bound)
        res_oracle = evaluate_task1_case(cid, gt, gt, section_mapping=section_mapping)
        oracle_case_results.append(res_oracle)

    # Aggregate computations for baseline
    n_cases = len(baseline_case_results)
    y_true = [r.decision_gt for r in baseline_case_results]
    y_pred = [r.decision_pred for r in baseline_case_results]

    correct_decisions = sum(int(r.decision_gate == 1.0) for r in baseline_case_results)
    decision_accuracy = correct_decisions / n_cases

    # Positive class F1 (biopsy = 'yes')
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "yes")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "yes")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "no")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "no")

    precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_yes = (2 * precision_yes * recall_yes / (precision_yes + recall_yes)) if (precision_yes + recall_yes) > 0 else 0.0

    mean_case_score = mean(r.overall_case_score for r in baseline_case_results)
    mean_gate = mean(r.decision_gate for r in baseline_case_results)

    valid_conf = [r.confidence_score for r in baseline_case_results if r.confidence_score is not None]
    valid_weights = [r.variable_weights_score for r in baseline_case_results if r.variable_weights_score is not None]
    valid_f1 = [r.important_decisive_f1 for r in baseline_case_results if r.important_decisive_f1 is not None]
    valid_tool = [r.tool_efficiency_score for r in baseline_case_results if r.tool_efficiency_score is not None]
    valid_grounding = [r.section_grounding_score for r in baseline_case_results if r.section_grounding_score is not None]

    # Cases that passed gate
    passed_cases = [r for r in baseline_case_results if r.decision_gate == 1.0]
    mean_among_passed = mean(r.overall_case_score for r in passed_cases) if passed_cases else 0.0

    # Official Task 1 Ranking Score: 0.5 * mean_case_score + 0.5 * f1_yes
    ranking_score = 0.5 * mean_case_score + 0.5 * f1_yes

    aggregate_summary = {
        "n_cases": n_cases,
        "n_decision_correct": correct_decisions,
        "n_decision_incorrect": n_cases - correct_decisions,
        "decision_accuracy": decision_accuracy,
        "decision_f1_yes": f1_yes,
        "precision_yes": precision_yes,
        "recall_yes": recall_yes,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mean_case_score": mean_case_score,
        "ranking_score": ranking_score,
        "decision_gate_pass_rate": mean_gate,
        "mean_case_score_among_gate_passed": mean_among_passed,
        "mean_confidence_score": mean(valid_conf) if valid_conf else None,
        "mean_variable_weights_score": mean(valid_weights) if valid_weights else None,
        "mean_important_decisive_f1": mean(valid_f1) if valid_f1 else None,
        "mean_tool_efficiency_score": mean(valid_tool) if valid_tool else None,
        "mean_section_grounding_score": mean(valid_grounding) if valid_grounding else None,
    }

    # Oracle summary
    oracle_mean_case_score = mean(r.overall_case_score for r in oracle_case_results)
    oracle_summary = {
        "n_cases": len(oracle_case_results),
        "mean_case_score": oracle_mean_case_score,
        "ranking_score": 1.0,
        "decision_accuracy": 1.0,
        "decision_f1_yes": 1.0,
    }

    # Save artifacts
    with open(results_dir / "case_metrics.json", "w") as f:
        json.dump([asdict(r) for r in baseline_case_results], f, indent=2)

    with open(results_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    with open(results_dir / "oracle_metrics.json", "w") as f:
        json.dump(oracle_summary, f, indent=2)

    with open(results_dir / "version.txt", "w") as f:
        f.write("exp_1: Task 1 Baseline Benchmarking against Ground Truth Release\n")
        f.write(f"Cohort Size: {n_cases} paired cases\n")
        f.write("Evaluation Package: chimera_metrics v0.1.0\n")

    print("\n" + "=" * 80)
    print("TASK 1 BASELINE BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Evaluated Cases:              {n_cases}")
    print(f"Decision Gate Accuracy:             {decision_accuracy:.4f} ({correct_decisions}/{n_cases})")
    print(f"Biopsy F1-Score (Positive Class):   {f1_yes:.4f} (Prec: {precision_yes:.4f}, Rec: {recall_yes:.4f})")
    print(f"Confusion Matrix (TP/FP/FN/TN):     {tp}/{fp}/{fn}/{tn}")
    print(f"Mean Case Composite Score:          {mean_case_score:.4f}")
    print(f"Mean Score (Passed Gates Only):     {mean_among_passed:.4f}")
    print(f"--- Reasoning Component Breakdown ---")
    print(f"  * Confidence Ordinal Score:       {aggregate_summary['mean_confidence_score']:.4f}")
    print(f"  * Variable Weights MAE Score:     {aggregate_summary['mean_variable_weights_score']:.4f}")
    print(f"  * Decisive Factors Set-F1:        {aggregate_summary['mean_important_decisive_f1']:.4f}")
    print(f"  * Cost-Aware Tool Precision:      {aggregate_summary['mean_tool_efficiency_score']:.4f}")
    print(f"  * Section Grounding Score:        {aggregate_summary['mean_section_grounding_score']:.4f}")
    print(f"-------------------------------------")
    print(f"OFFICIAL TASK 1 RANKING SCORE (R_1): {ranking_score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
