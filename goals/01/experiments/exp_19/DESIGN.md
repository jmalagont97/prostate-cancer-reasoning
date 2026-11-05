# Experiment Design: SHAP / Distance-Attribution Exhaustive Threshold Optimization for Clinical Relevance (exp_19)

**Experiment**: `experiments/exp_19/`  
**Project**: `pathology-reasoning` (CHIMERA Task 1 — Subtask 1.3)  
**Date**: 2026-08-18  
**Author**: Experto en Machine Learning, Creación de Agentes y Razonamiento  
**Status**: Draft  

---

## 1. Hypothesis

Extracting continuous local feature attributions \(\psi_{i, k}\) (via KNN distance contributions and SHAP Kernel explainer) from the frozen tabular decision model (\(T_{21}\) `ConfidenceWeightedKNN`) for the 10 official variables, and discretizing each variable's attribution using **Global 3-Threshold Optimization** \((\tau_{1, k}^*, \tau_{2, k}^*, \tau_{3, k}^*)\) — directly minimizing training-fold Balanced Ordinal Error (\(\text{MOE}_{\text{abs}}\)) and maximizing \(\text{F1}_{\text{macro}}\) — accurately predicts the 10 clinical relevance target weights (\(\text{MOE}_{\text{abs}}^{\text{weights}} < 0.35\), vs baseline $0.500$) and section reveal sequences (\(\text{F1}_{\text{macro}}^{\text{sections}} > 0.65\)).

### Null Hypothesis
Feature attribution descriptors \(\psi_k\) discretized via global threshold optimization do not achieve lower average \(\text{MOE}_{\text{abs}}\) across the 10 relevance variables than the majority prevalence baseline (\(\text{MOE}_{\text{abs}} \ge 0.500\)).

---

## 2. Experimental Setup

### Cohort & Dataset
- **Cohort:** `usable_labeled` cases ($N=88$) from `data/chimera26/preprocessed/task1/inputs.csv` and `ground_truth.csv`.
- **Splits:** 50 MCCV splits (70 train / 18 val) + 88 LOO folds from `mccv_loocv_splits.csv` (Seed 42).
- **Target 1 (10 Relevance Weights):** `target_code_weight_*` for 10 official variables:
  `age`, `fh`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`.
  Categories: `0 = not_used`, `1 = noted`, `2 = important`, `3 = decisive`.
- **Target 2 (Section Reveal Sequence):** `target_reveal_sequence_json` containing sections:
  `radiology_report`, `laboratory_results`, `psa_trend`, `previous_notes`, `family_history`, `pathology_report`.

---

## 3. Mathematical & Algorithmic Formulation

### 1. Tabular Decision Model & Feature Attribution \(\psi_{i, k}\)
Using the frozen 21-variable tabular model (\(T_{21}\) exp_5 `ConfidenceWeightedKNN`, $k=1$, metric=cosine):
For patient $i$ and variable $k \in \{1, \dots, 10\}$:
- **Method A (KNN Distance Contribution):**
  \[
  \psi_{i, k}^{\text{dist}} = w_{\text{knn}} \cdot |x_{i, k} - x_{\text{nn}, k}| \quad \in [0, \infty)
  \]
- **Method B (SHAP Value):**
  \[
  \psi_{i, k}^{\text{shap}} = |\text{SHAP}(x_{i, k})| \quad \in [0, \infty)
  \]
  computed via `shap.KernelExplainer` on the decision probability output $p_{\text{biopsy}}$.

### 2. Global Exhaustive 3-Threshold Optimizer per Target Variable
For each target variable $k$:
Given training attribution scores \(\psi_{\text{train}, k}\) and ground truth codes \(y_{\text{train}, k} \in \{0, 1, 2, 3\}\):
- Extract candidate split points along the sorted continuous spectrum of \(\psi_{k}\).
- Evaluate all valid 3-threshold tuples \((\tau_1, \tau_2, \tau_3)\) with \(0 < \tau_1 < \tau_2 < \tau_3 < \max(\psi_k)\).
- Categorize \(\psi_k \to \hat{c}_{i, k} \in \{0, 1, 2, 3\}\):
  \[
  \hat{c}_{i, k}(\tau_1, \tau_2, \tau_3) = \begin{cases}
  0 \quad (\text{not\_used}) & \text{if } \psi_{i, k} < \tau_1 \\
  1 \quad (\text{noted}) & \text{if } \tau_1 \le \psi_{i, k} < \tau_2 \\
  2 \quad (\text{important}) & \text{if } \tau_2 \le \psi_{i, k} < \tau_3 \\
  3 \quad (\text{decisive}) & \text{if } \psi_{i, k} \ge \tau_3
  \end{cases}
  \]
- Select optimal thresholds \((\tau_{1, k}^*, \tau_{2, k}^*, \tau_{3, k}^*)\) on train:
  \[
  \arg\min_{(\tau_1, \tau_2, \tau_3)} \Big[ \text{MOE}_{\text{abs}}\big(y_{\text{train}, k}, \hat{c}_k\big) - 0.001 \cdot \text{F1}_{\text{macro}}\big(y_{\text{train}, k}, \hat{c}_k\big) \Big]
  \]

### 3. Threshold Modes & Section Mapping
- **Threshold Mode:** `exact_free` vs `exact_balanced_min_recall` (minimum 2 training predictions per present class).
- **Section Reveal Sequence Derivation:**
  - `radiology_report` \(\iff \max(\hat{c}_{\text{pirads}}, \hat{c}_{\text{psad}}, \hat{c}_{\text{vol}}, \hat{c}_{\text{cspca}}) \ge 1\)
  - `laboratory_results` \(\iff \hat{c}_{\text{dre}} \ge 1\)
  - `psa_trend` \(\iff \hat{c}_{\text{psa}} \ge 1\)
  - `family_history` \(\iff \hat{c}_{\text{fh}} \ge 1\)
  - `pathology_report` \(\iff \hat{c}_{\text{bx}} \ge 1\)
  - `previous_notes` \(\iff \hat{c}_{\text{comorbidity}} \ge 1 \text{ or } \hat{c}_{\text{age}} \ge 2\)

---

## 4. File Layout for This Experiment

```
experiments/exp_19/
├── DESIGN.md                  ← This research design document
├── IMPLEMENTATION.md          ← Implementation plan (to be generated in plan mode)
├── scripts/
│   └── run_shap_relevance_experiment.py  ← Self-contained runner script
├── results/
│   ├── summary.json           ← Overall sweep summary and selected config
│   ├── evaluation_scorecard.csv  ← Detailed per-condition MCCV scorecard
│   ├── per_variable_metrics.csv  ← Detailed metrics for each of the 10 variables
│   ├── predictions_mccv.csv   ← Out-of-fold predictions across 50 MCCV splits
│   ├── predictions_loo.csv    ← Out-of-fold predictions across 88 LOO folds
│   └── confusion_matrices.json← Aggregated confusion matrices per target
└── reports/
    └── figures/
        ├── confusion_matrices_10_variables.png
        └── section_f1_confusion.png
```

---

## 5. Baselines & Experimental Conditions

### Baselines
| Baseline | Description | \(\text{MOE}_{\text{abs}}\) Target Range |
| :--- | :--- | :---: |
| **Always Noted (1.0)** | Predict constant `noted` (1.0) for all variables | $\approx 0.500$ |
| **Class Prevalencia** | Predict majority class per variable on train fold | $\approx 0.480$ |

### Proposed Conditions (Grid Search)
1. **Attribution Engine:** `knn_distance_attribution` vs `shap_kernel`.
2. **Threshold Search Mode:** `exact_free` vs `exact_balanced_min_recall`.
3. **Attribution Scaling:** Raw \(\psi_k\) vs Max-normalized \(\psi_k / \max(\psi)\).

Total conditions: $2 \times 2 \times 2 = 8 \text{ conditions}$.

---

## 6. Evaluation Protocol & Selection Cascade

### Primary Metric (Relevance Weights)
\[
\text{MOE}_{\text{abs}}^{\text{weights}} = \frac{1}{10} \sum_{k=1}^{10} \left[ \frac{1}{C_k} \sum_{c \in C_k} \frac{1}{N_{c, k}} \sum_{i: y_{i, k}=c} \frac{|\hat{c}_{i, k} - c|}{3} \right] \quad (\text{Normalized Ordinal Error in } [0, 1])
\]

### Selection Cascade (MCCV)
1. Strictly superior to baseline (\(\text{MOE}_{\text{abs}}^{\text{weights}} < \text{Baseline}\)).
2. Minimize \(\text{MOE}_{\text{abs}}^{\text{mean}}\) and \(\text{MOE}_{\text{abs}}^{\text{pooled}}\) across 10 variables.
3. Secondary tiebreaker: Maximize \(\text{F1}_{\text{macro}}^{\text{weights}}\) and Section \(\text{F1}_{\text{macro}}^{\text{sections}}\).
4. Final audit on 88 LOO folds for out-of-fold metrics and confusion matrices.

---

## 7. Reproducibility Checklist

- [x] Random seeds fixed (`seed=42`)
- [x] Canonical splits file referenced (`data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`)
- [x] Out-of-fold predictions logged to `results/`
- [x] Environment frozen (`histo-DL`)
- [x] Git commit hash logged to `results/git_commit.txt` before execution

---

## 8. Next Steps

1. Approve this experiment design (`experiments/exp_19/DESIGN.md`).
2. Update `experiments/INDEX.md` to register `exp_19`.
3. In **plan mode**, request an **implementation plan** to be saved as `experiments/exp_19/IMPLEMENTATION.md`.
