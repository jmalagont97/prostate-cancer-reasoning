# `chimera_metrics`

Standardized Evaluation Metrics Suite for the **CHIMERA-Agent Challenge** (MICCAI 2026).

---

## Installation

Install in editable mode within your Python research environment:

```bash
pip install -e utils/pkg/chimera_metrics
```

---

## Features

- **Task 1 (Pre-Biopsy Decision):**
  - Binary decision gate (`compute_biopsy_decision_gate`)
  - Ordinal confidence distance (`compute_confidence_score`)
  - Multivariable weights MAE (`compute_variable_weights_score`)
  - Decisive & important factors set-F1 (`compute_important_decisive_f1`)
  - Asymmetric cost-aware tool efficiency (`compute_cost_aware_tool_efficiency`)
  - EHR section grounding check (`compute_section_grounding`)

- **Task 2 (Treatment Recommendation):**
  - Multiclass decision gate for active surveillance / watchful waiting / active treatment (`compute_treatment_decision_gate`)
  - Multi-variable weighting across 17 clinical variables
  - Decision reasoning and grounding checks

- **Task 3 (Biochemical Recurrence Survival Prognosis):**
  - Binary recurrence event match (`compute_recurrence_event_score`)
  - Censoring-aware time closeness (`compute_recurrence_time_score`)
  - Harrell's Concordance Index (`compute_concordance_index`)
  - Inverse Probability of Censoring Weighting (IPCW) dynamic AUC across 12, 24, 36, 60-month horizons (`compute_time_dependent_auc`)

- **Experiment Reporting:**
  - Automated ASCII and Markdown tabular report generators (`ExperimentReporter`).

- **Mathematical Documentation:**
  - Full formal derivations and mappings available in [`EQUATIONS.md`](./EQUATIONS.md).

---

## Quick Example

```python
from chimera_metrics import evaluate_task1_case, ExperimentReporter

gt = {
    "biopsy_decision": "yes",
    "confidence": "clear",
    "variable_weights": {"pirads": "decisive", "psa": "important", "age": "noted"},
    "reveal_sequence": ["radiology_report", "psa_trend"]
}

pred = {
    "biopsy_decision": "yes",
    "confidence": "borderline",
    "variable_weights": {"pirads": "decisive", "psa": "important", "age": "not_used"},
    "reveal_sequence": ["radiology_report", "psa_trend"]
}

result = evaluate_task1_case(case_id="BX_01", gt=gt, pred=pred)
print(ExperimentReporter.format_task1_summary([result]))
```
