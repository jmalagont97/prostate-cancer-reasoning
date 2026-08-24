"""
chimera_metrics: Standardized Evaluation Metrics Suite for CHIMERA-Agent Challenge (MICCAI 2026).
"""

from chimera_metrics.core import (
    BaseMetric,
    BiopsyDecision,
    TreatmentDecision,
    ConfidenceLevel,
    ImportanceWeight,
    MetricResult,
    Task1CaseResult,
    Task2CaseResult,
    Task3CaseResult,
    Task3CohortResult,
    CONF_MAP,
    WEIGHT_MAP,
    WEIGHT_ALIAS,
    IMPORTANT_OR_DECISIVE,
    SECTION_KEY_TO_REVEAL_NAME,
)

from chimera_metrics.task1 import (
    BiopsyDecisionGateMetric,
    Task1ReasoningMetric,
    ToolEfficiencyMetric,
    SectionGroundingMetric,
    compute_biopsy_decision_gate,
    compute_confidence_score,
    compute_variable_weights_score,
    extract_important_decisive_set,
    compute_set_f1,
    compute_important_decisive_f1,
    compute_tool_efficiency,
    compute_cost_aware_tool_efficiency,
    compute_section_grounding,
)

from chimera_metrics.task2 import (
    TreatmentDecisionGateMetric,
    Task2ReasoningMetric,
    compute_treatment_decision_gate,
    compute_confidence_score as compute_treatment_confidence_score,
    compute_variable_weights_score as compute_treatment_variable_weights_score,
    compute_important_decisive_f1 as compute_treatment_important_decisive_f1,
    compute_section_grounding as compute_treatment_section_grounding,
    compute_tool_efficiency as compute_treatment_tool_efficiency,
)

from chimera_metrics.task3 import (
    HarrellCIndexMetric,
    TimeDependentAUCMetric,
    RecurrenceCaseScoreMetric,
    compute_concordance_index,
    compute_time_dependent_auc,
    compute_recurrence_event_score,
    compute_recurrence_time_score,
)

from chimera_metrics.judge import (
    LLMRationaleJudge,
    DECISION_RATIONALE_RUBRIC,
    RECURRENCE_RATIONALE_RUBRIC,
)

from chimera_metrics.report import (
    ExperimentReporter,
    evaluate_task1_case,
    evaluate_task2_case,
    evaluate_task3_cohort,
)

from chimera_metrics.splits import (
    load_mccv_splits,
    get_split_indices,
)

__version__ = "0.1.0"
