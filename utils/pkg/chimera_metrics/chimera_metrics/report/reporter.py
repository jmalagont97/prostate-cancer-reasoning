"""Formatted ASCII and Markdown Experiment Reporting."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from chimera_metrics.core.types import (
    Task1CaseResult,
    Task2CaseResult,
    Task3CaseResult,
    Task3CohortResult,
)


class ExperimentReporter:
    """Generates clean ASCII and Markdown evaluation summaries for research experiments."""

    @staticmethod
    def format_task1_summary(results: List[Task1CaseResult]) -> str:
        lines: List[str] = [
            "+----------------------------------------------------------------------------------------------------------------+",
            "| Task 1: Pre-Biopsy Decision Evaluation Summary                                                                 |",
            "+------------+--------+-------------+-------------+------------+------------+------------+-----------+-----------+",
            "| Case ID    | Gate   | GT Dec      | Pred Dec    | Conf Score | Weight MAE | Factors F1 | Tool Prec | Overall   |",
            "+------------+--------+-------------+-------------+------------+------------+------------+-----------+-----------+",
        ]
        for r in results:
            g_str = f"{r.decision_gate:.2f}"
            gt_d = (r.decision_gt or "-")[:11]
            pr_d = (r.decision_pred or "-")[:11]
            conf = f"{r.confidence_score:.2f}" if r.confidence_score is not None else "  - "
            w_mae = f"{r.variable_weights_score:.2f}" if r.variable_weights_score is not None else "  - "
            f1 = f"{r.important_decisive_f1:.2f}" if r.important_decisive_f1 is not None else "  - "
            tool = f"{r.tool_efficiency_score:.2f}" if r.tool_efficiency_score is not None else "  - "
            ovr = f"{r.overall_case_score:.2f}"
            lines.append(
                f"| {r.case_id:<10} | {g_str:>6} | {gt_d:<11} | {pr_d:<11} | {conf:>10} | {w_mae:>10} | {f1:>10} | {tool:>9} | {ovr:>9} |"
            )
        lines.append("+------------+--------+-------------+-------------+------------+------------+------------+-----------+-----------+")

        # Aggregate averages
        if results:
            avg_gate = sum(r.decision_gate for r in results) / len(results)
            avg_ovr = sum(r.overall_case_score for r in results) / len(results)
            lines.append(f"Mean Decision Gate: {avg_gate:.4f} | Mean Overall Case Score: {avg_ovr:.4f}")

        return "\n".join(lines)

    @staticmethod
    def format_task2_summary(results: List[Task2CaseResult]) -> str:
        lines: List[str] = [
            "+----------------------------------------------------------------------------------------------------------------+",
            "| Task 2: Treatment Recommendation Evaluation Summary                                                            |",
            "+------------+--------+------------------+------------------+------------+------------+------------+------------+",
            "| Case ID    | Gate   | GT Rec           | Pred Rec         | Conf Score | Weight MAE | Factors F1 | Overall    |",
            "+------------+--------+------------------+------------------+------------+------------+------------+------------+",
        ]
        for r in results:
            g_str = f"{r.decision_gate:.2f}"
            gt_d = (r.decision_gt or "-")[:16]
            pr_d = (r.decision_pred or "-")[:16]
            conf = f"{r.confidence_score:.2f}" if r.confidence_score is not None else "  - "
            w_mae = f"{r.variable_weights_score:.2f}" if r.variable_weights_score is not None else "  - "
            f1 = f"{r.important_decisive_f1:.2f}" if r.important_decisive_f1 is not None else "  - "
            ovr = f"{r.overall_case_score:.2f}"
            lines.append(
                f"| {r.case_id:<10} | {g_str:>6} | {gt_d:<16} | {pr_d:<16} | {conf:>10} | {w_mae:>10} | {f1:>10} | {ovr:>10} |"
            )
        lines.append("+------------+--------+------------------+------------------+------------+------------+------------+------------+")

        if results:
            avg_gate = sum(r.decision_gate for r in results) / len(results)
            avg_ovr = sum(r.overall_case_score for r in results) / len(results)
            lines.append(f"Mean Decision Gate: {avg_gate:.4f} | Mean Overall Case Score: {avg_ovr:.4f}")

        return "\n".join(lines)

    @staticmethod
    def format_task3_summary(
        case_results: List[Task3CaseResult],
        cohort_result: Task3CohortResult,
    ) -> str:
        lines: List[str] = [
            "+------------------------------------------------------------------------------------+",
            "| Task 3: Post-Prostatectomy BCR Prognosis & Survival Analysis Summary               |",
            "+------------+------------+------------+-------------+--------------+----------------+",
            "| Case ID    | Event GT   | Event Pred | Time GT (m) | Time Pred(m) | Time Score     |",
            "+------------+------------+------------+-------------+--------------+----------------+",
        ]
        for r in case_results:
            e_gt = str(r.event_gt) if r.event_gt is not None else "-"
            e_pr = str(r.event_pred) if r.event_pred is not None else "-"
            t_gt = f"{r.time_gt:.1f}" if r.time_gt is not None else "-"
            t_pr = f"{r.time_pred:.1f}" if r.time_pred is not None else "-"
            t_sc = f"{r.time_score:.3f}" if r.time_score is not None else "-"
            lines.append(
                f"| {r.case_id:<10} | {e_gt:>10} | {e_pr:>10} | {t_gt:>11} | {t_pr:>12} | {t_sc:>14} |"
            )
        lines.append("+------------+------------+------------+-------------+--------------+----------------+")

        c_idx_str = f"{cohort_result.c_index:.4f}" if cohort_result.c_index is not None else "N/A"
        lines.append(f"Cohort Harrell's C-index: {c_idx_str}")
        lines.append("Cumulative/Dynamic Time-Dependent AUC (IPCW):")
        for h, val in cohort_result.td_auc.items():
            val_str = f"{val:.4f}" if val is not None else "N/A"
            lines.append(f"  - Horizon {int(h)} months: {val_str}")

        return "\n".join(lines)
