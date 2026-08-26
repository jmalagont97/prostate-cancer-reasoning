"""Dataset and Experiment Aggregation Summaries for CHIMERA-Agent."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from chimera_metrics.core.base import norm_event, norm_months
from chimera_metrics.core.types import (
    Task1CaseResult,
    Task2CaseResult,
    Task3CaseResult,
    Task3CohortResult,
)
from chimera_metrics.task1.decision import compute_biopsy_decision_gate
from chimera_metrics.task1.grounding import (
    compute_cost_aware_tool_efficiency,
    compute_section_grounding,
)
from chimera_metrics.task1.reasoning import (
    compute_confidence_score,
    compute_important_decisive_f1,
    compute_variable_weights_score,
)
from chimera_metrics.task2.decision import compute_treatment_decision_gate
from chimera_metrics.task3.concordance import compute_concordance_index
from chimera_metrics.task3.survival_scores import (
    compute_recurrence_event_score,
    compute_recurrence_time_score,
)
from chimera_metrics.task3.time_dependent_auc import compute_time_dependent_auc


def evaluate_task1_case(
    case_id: str,
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    section_mapping: Optional[Dict[str, Any]] = None,
) -> Task1CaseResult:
    """Evaluates a single Task 1 case across all metrics."""
    gate_score, gt_dec, pr_dec, gate_msg = compute_biopsy_decision_gate(gt, pred)
    conf_score = compute_confidence_score(gt, pred)
    var_w_score = compute_variable_weights_score(gt, pred)
    f1_score = compute_important_decisive_f1(gt, pred)
    tool_score, tool_details = compute_cost_aware_tool_efficiency(gt, pred)
    ground_score, ground_details = compute_section_grounding(pred, section_mapping)

    # Composite case score: decision gate * average of reasoning components
    reasoning_components = [c for c in (conf_score, var_w_score, f1_score, tool_score, ground_score) if c is not None]
    avg_reasoning = mean(reasoning_components) if reasoning_components else 0.0
    overall_score = gate_score * avg_reasoning

    return Task1CaseResult(
        case_id=case_id,
        decision_gate=gate_score,
        decision_gt=gt_dec,
        decision_pred=pr_dec,
        confidence_score=conf_score,
        variable_weights_score=var_w_score,
        important_decisive_f1=f1_score,
        tool_efficiency_score=tool_score,
        section_grounding_score=ground_score,
        overall_case_score=overall_score,
        details={
            "gate_message": gate_msg,
            "tool_details": tool_details,
            "grounding_details": ground_details,
        },
    )


def evaluate_task2_case(
    case_id: str,
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    section_mapping: Optional[Dict[str, Any]] = None,
) -> Task2CaseResult:
    """Evaluates a single Task 2 case across all metrics."""
    gate_score, gt_dec, pr_dec, gate_msg = compute_treatment_decision_gate(gt, pred)
    conf_score = compute_confidence_score(gt, pred)
    var_w_score = compute_variable_weights_score(gt, pred)
    f1_score = compute_important_decisive_f1(gt, pred)
    tool_score, tool_details = compute_cost_aware_tool_efficiency(gt, pred)
    ground_score, ground_details = compute_section_grounding(pred, section_mapping)

    reasoning_components = [c for c in (conf_score, var_w_score, f1_score, tool_score, ground_score) if c is not None]
    avg_reasoning = mean(reasoning_components) if reasoning_components else 0.0
    overall_score = gate_score * avg_reasoning

    return Task2CaseResult(
        case_id=case_id,
        decision_gate=gate_score,
        decision_gt=gt_dec,
        decision_pred=pr_dec,
        confidence_score=conf_score,
        variable_weights_score=var_w_score,
        important_decisive_f1=f1_score,
        tool_efficiency_score=tool_score,
        section_grounding_score=ground_score,
        overall_case_score=overall_score,
        details={
            "gate_message": gate_msg,
            "tool_details": tool_details,
            "grounding_details": ground_details,
        },
    )


def evaluate_task3_cohort(
    gt_list: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    case_ids: Optional[List[str]] = None,
) -> Tuple[List[Task3CaseResult], Task3CohortResult]:
    """Evaluates a full cohort for Task 3 (survival & time-to-event)."""
    n = len(gt_list)
    if case_ids is None:
        case_ids = [f"case_{i}" for i in range(n)]
    if len(pred_list) != n or len(case_ids) != n:
        raise ValueError("gt_list, pred_list, and case_ids must have equal lengths")

    case_results: List[Task3CaseResult] = []
    times: List[float] = []
    preds: List[float] = []
    events: List[int] = []

    for cid, gt, pr in zip(case_ids, gt_list, pred_list):
        ev_score = compute_recurrence_event_score(gt, pr)
        tm_score = compute_recurrence_time_score(gt, pr)

        g_ev = gt.get("event")
        p_ev = pr.get("event")
        g_tm = norm_months(gt.get("months_to_recurrence"))
        p_tm = norm_months(pr.get("months_to_recurrence"))
        g_ev = norm_event(gt.get("event"))
        p_ev = norm_event(pr.get("event"))

        if g_tm is not None and p_tm is not None and g_ev in (0, 1):
            times.append(g_tm)
            preds.append(p_tm)
            events.append(g_ev)

        case_results.append(
            Task3CaseResult(
                case_id=cid,
                event_score=ev_score,
                time_score=tm_score,
                event_gt=g_ev,
                event_pred=p_ev,
                time_gt=g_tm,
                time_pred=p_tm,
            )
        )

    c_index = compute_concordance_index(times, preds, events) if len(times) > 0 else None
    td_auc = compute_time_dependent_auc(times, preds, events) if len(times) > 0 else {}

    valid_events = [r.event_score for r in case_results if r.event_score is not None]
    valid_times = [r.time_score for r in case_results if r.time_score is not None]

    cohort_summary = Task3CohortResult(
        c_index=c_index,
        td_auc=td_auc,
        mean_event_score=mean(valid_events) if valid_events else None,
        mean_time_score=mean(valid_times) if valid_times else None,
        num_cases=n,
    )

    return case_results, cohort_summary
