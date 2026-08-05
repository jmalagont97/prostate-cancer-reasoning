# Experiment Design: Out-of-Fold Diagnostic Confidence Prediction via Dynamic Fold-Level LOOCV Decision Tree Thresholding
**Experiment**: experiments/exp_11/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Dynamically fitting 1D decision tree cut-points $(\tau_1^{(i)}, \tau_2^{(i)})$ with balanced class weighting (`class_weight='balanced'`) locally within each Leave-One-Out Cross-Validation (LOOCV) fold on the 87 training cases will eliminate the boundary oversmoothing (difuminado por promediado) caused by static MCCV meta-averaging, allowing decision thresholds to adapt to out-of-fold $ICI$ topology and significantly improving 3-class Macro-F1 without introducing data leakage.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
  - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` target column)
  - Cohort: $N=88$ labeled complete-case patients with valid `confidence` annotations (`clear`: 56, `borderline`: 18, `uncertain`: 14).
- **Validation Harness (End-to-End Pure LOOCV - 88 Folds)**:
  - For each fold $i \in \{1, \dots, 88\}$:
    - Train Tabular, MRI, and Text KNN models on 87 training cases.
    - Generate out-of-fold probabilities $p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}$ for the 87 training cases.
    - Compute $ICI_{\text{train}}^{(i)}$ for the 87 training cases.
    - Fit `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI_{\text{train}}^{(i)}$ vs `confidence` on the 87 training cases.
    - Extract local fold-level cut-points $(\tau_1^{(i)}, \tau_2^{(i)})$.
    - Predict unimodal probabilities for test patient $i$, compute $ICI_{\text{test}}^{(i)}$.
    - Classify test patient $i$ applying local fold thresholds $(\tau_1^{(i)}, \tau_2^{(i)})$.

## 3. Mathematical Formulation of the Dynamic LOOCV Protocol
For each LOOCV fold $i$:
1. **Unimodal Probability Prediction**: $p_{\text{tab}}^{(i)}, p_{\text{mri}}^{(i)}, p_{\text{text}}^{(i)} \in [0, 1]$
2. **Composite Reliability Index ($ICI$)**:
   $$\bar{p} = \frac{p_{\text{tab}} + p_{\text{mri}} + p_{\text{text}}}{3}$$
   $$\sigma_p = \sqrt{\frac{1}{3} \sum_{m} \left(p_m - \bar{p}\right)^2}, \quad \delta_{\text{margin}} = |\bar{p} - 0.50|$$
   $$ICI = (2 \cdot \delta_{\text{margin}}) \cdot (1 - 2 \cdot \sigma_p) \in [0.0, 1.0]$$
3. **Dynamic Local Decision Rule**:
   $$\hat{Y}_{\text{conf}}^{(i)} = \begin{cases} \text{uncertain} & \text{si } ICI_{\text{test}}^{(i)} < \tau_1^{(i)} \\ \text{borderline} & \text{si } \tau_1^{(i)} \le ICI_{\text{test}}^{(i)} < \tau_2^{(i)} \\ \text{clear} & \text{si } ICI_{\text{test}}^{(i)} \ge \tau_2^{(i)} \end{cases}$$

## 4. File Layout for This Experiment
```
experiments/exp_11/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← pure dynamic LOOCV evaluation script
├── results/
│   ├── dynamic_thresholds_per_fold.csv  ← fold-by-fold learned thresholds (tau_1_i, tau_2_i)
│   ├── loocv_confidence_metrics.json  ← 3-class F1-Macro, Accuracy, Spearman rho
│   └── loocv_confidence_predictions.csv ← patient-level probabilities, ICI, predicted & ground-truth confidence
└── reports/
    ├── figures/
    │   ├── dynamic_thresholds_evolution.png ← fold-by-fold threshold stability plot
    │   └── confusion_matrix_3class.png      ← 3x3 confusion matrix of confidence predictions
    └── summary.md             ← write-up of dynamic LOOCV confidence prediction results
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 3-class Macro-F1 score under dynamic LOOCV.
- **Secondary Metrics**: Accuracy, Spearman rank correlation ($\rho$), 3x3 Confusion Matrix.

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Dynamic thresholds saved in `results/dynamic_thresholds_per_fold.csv`
- [ ] Predictions and metrics logged to `results/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run the experiment.
