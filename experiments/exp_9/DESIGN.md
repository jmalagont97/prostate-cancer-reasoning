# Experiment Design: Out-of-Fold Diagnostic Confidence Prediction via Composite Reliability Index (ICI) & Meta-Threshold Decision Trees
**Experiment**: experiments/exp_9/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
Mapping the Composite Reliability Index ($ICI$), computed out-of-fold across the optimal unimodal models (Tabular `exp_5`, MRI `exp_6`, Text `exp_7`), to diagnostic confidence levels using decision tree boundary thresholds ensemble-averaged over 100 MCCV splits ($\bar{\tau}_1, \bar{\tau}_2$) will generalize effectively in a frozen Leave-One-Out Cross-Validation (LOOCV) evaluation, accurately predicting medical diagnostic confidence (`clear`, `borderline`, `uncertain`) without data leakage or optimism bias.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
  - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` target column)
  - Cohort: $N=91$ labeled complete-case patients with valid `confidence` annotations (`clear`: 58, `borderline`: 18, `uncertain`: 15).
- **Validation Strategy**:
  - **Phase A (Meta-Threshold Learning - 100 MCCV splits)**: Train 1D Decision Trees on $ICI$ per split, extract decision boundary cut-points $(\tau_{1, s}, \tau_{2, s})$, and calculate mean meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
  - **Phase B (Frozen LOOCV Evaluation - 91 Folds)**: Predict out-of-fold unimodal probabilities, compute $ICI$, and classify confidence applying frozen $(\bar{\tau}_1, \bar{\tau}_2)$ WITHOUT re-fitting decision trees during LOOCV.

## 3. Mathematical Formulation of the Composite Reliability Index (ICI)
For each patient $i$, given out-of-fold predicted probabilities $p_{\text{tab}}^{(i)}, p_{\text{mri}}^{(i)}, p_{\text{text}}^{(i)} \in [0, 1]$ from the optimal unimodal models:

1. **Ensemble Mean Probability**:
   $$\bar{p}^{(i)} = \frac{p_{\text{tab}}^{(i)} + p_{\text{mri}}^{(i)} + p_{\text{text}}^{(i)}}{3}$$

2. **Inter-Modality Standard Deviation (Epistemic Disagreement)**:
   $$\sigma_p^{(i)} = \sqrt{\frac{1}{3} \sum_{m \in \{\text{tab}, \text{mri}, \text{text}\}} \left(p_m^{(i)} - \bar{p}^{(i)}\right)^2} \in [0.0, 0.50]$$

3. **Certitude Margin (Aleatoric Certainty)**:
   $$\delta_{\text{margin}}^{(i)} = \left| \bar{p}^{(i)} - 0.50 \right| \in [0.0, 0.50]$$

4. **Composite Reliability Index ($ICI$)**:
   $$ICI^{(i)} = \left(2 \cdot \delta_{\text{margin}}^{(i)}\right) \cdot \left(1 - 2 \cdot \sigma_p^{(i)}\right) \in [0.0, 1.0]$$

Where:
- $ICI \to 1.0 \implies$ High certitude AND unanimous inter-modality consensus (`clear` confidence).
- $ICI \to 0.0 \implies$ High aleatoric uncertainty ($\bar{p} \approx 0.50$) OR severe inter-modality conflict ($\sigma_p \gg 0$) (`borderline` or `uncertain` confidence).

## 4. File Layout for This Experiment
```
experiments/exp_9/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← MCCV meta-threshold learning & frozen LOOCV evaluation script
├── results/
│   ├── meta_thresholds.json           ← averaged decision boundaries (bar_tau_1, bar_tau_2)
│   ├── loocv_confidence_metrics.json  ← 3-class F1-Macro, Accuracy, Spearman rho
│   └── loocv_confidence_predictions.csv ← patient-level probabilities, ICI, predicted & ground-truth confidence
└── reports/
    ├── figures/
    │   ├── decision_tree_thresholds.png ← distribution of learned thresholds across 100 splits
    │   └── confusion_matrix_3class.png  ← 3x3 confusion matrix of confidence predictions
    └── summary.md             ← write-up of confidence prediction results
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 3-class Macro-F1 score under frozen LOOCV.
- **Secondary Metrics**: Accuracy, Spearman rank correlation ($\rho$), 3x3 Confusion Matrix.
- **Decision Rule**:
  $$\hat{Y}_{\text{conf}}^{(i)} = \begin{cases} \text{uncertain} & \text{si } ICI^{(i)} < \bar{\tau}_1 \\ \text{borderline} & \text{si } \bar{\tau}_1 \le ICI^{(i)} < \bar{\tau}_2 \\ \text{clear} & \text{si } ICI^{(i)} \ge \bar{\tau}_2 \end{cases}$$

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Meta-thresholds saved in `results/meta_thresholds.json`
- [ ] Predictions and metrics logged to `results/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run the experiment.
