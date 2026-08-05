# Experiment Design: Balanced Diagnostic Confidence Prediction via Class-Weighted ICI Meta-Thresholding
**Experiment**: experiments/exp_10/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Incorporating balanced class weighting (`class_weight='balanced'`) during 1D Decision Tree training in Phase A (100 MCCV splits) will shift the composite reliability index ($ICI$) decision meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$ toward minority confidence categories (`uncertain`: 14, `borderline`: 18, `clear`: 56), significantly improving 3-class out-of-fold Macro-F1 (target $\ge 0.50$) and minority recall under frozen Leave-One-Out Cross-Validation (LOOCV) without data leakage or optimism bias.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
  - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` target column)
  - Cohort: $N=88$ labeled complete-case patients with valid `confidence` annotations (`clear`: 56, `borderline`: 18, `uncertain`: 14).
- **Validation Strategy**:
  - **Phase A (Class-Weighted Meta-Threshold Learning - 100 MCCV splits)**:
    - Compute out-of-fold unimodal probabilities $p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}$ on validation split $s$.
    - Calculate $ICI = (2 \cdot \delta_{\text{margin}}) \cdot (1 - 2 \cdot \sigma_p)$.
    - Fit `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI$ vs `confidence` on training split $s$.
    - Extract decision boundary cut-points $(\tau_{1, s}, \tau_{2, s})$ and compute ensemble mean meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
  - **Phase B (Frozen LOOCV Evaluation - 88 Folds)**: Predict out-of-fold unimodal probabilities, compute $ICI$, and classify confidence applying frozen $(\bar{\tau}_1, \bar{\tau}_2)$ WITHOUT re-fitting decision trees during LOOCV.

## 3. Mathematical Formulation & Class Weighting
For each patient $i$:
1. **Mean Probability**: $\bar{p}^{(i)} = \frac{p_{\text{tab}}^{(i)} + p_{\text{mri}}^{(i)} + p_{\text{text}}^{(i)}}{3}$
2. **Inter-Modality Standard Deviation**: $\sigma_p^{(i)} = \sqrt{\frac{1}{3} \sum_{m} \left(p_m^{(i)} - \bar{p}^{(i)}\right)^2}$
3. **Certitude Margin**: $\delta_{\text{margin}}^{(i)} = |\bar{p}^{(i)} - 0.50|$
4. **Composite Reliability Index ($ICI$)**:
   $$ICI^{(i)} = \left(2 \cdot \delta_{\text{margin}}^{(i)}\right) \cdot \left(1 - 2 \cdot \sigma_p^{(i)}\right) \in [0.0, 1.0]$$
5. **Inverse Class Weights in Phase A**:
   $$w_k = \frac{N_{\text{train}}}{3 \cdot N_{k, \text{train}}}$$
   where $w_{\text{clear}} \approx 0.52$, $w_{\text{borderline}} \approx 1.63$, and $w_{\text{uncertain}} \approx 2.10$.

## 4. File Layout for This Experiment
```
experiments/exp_10/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← class-weighted MCCV meta-threshold learning & frozen LOOCV evaluation script
├── results/
│   ├── meta_thresholds.json           ← class-weighted averaged decision boundaries (bar_tau_1, bar_tau_2)
│   ├── loocv_confidence_metrics.json  ← 3-class F1-Macro, Accuracy, Spearman rho
│   └── loocv_confidence_predictions.csv ← patient-level probabilities, ICI, predicted & ground-truth confidence
└── reports/
    ├── figures/
    │   ├── decision_tree_thresholds.png ← distribution of learned balanced thresholds across 100 splits
    │   └── confusion_matrix_3class.png  ← 3x3 confusion matrix of confidence predictions
    └── summary.md             ← write-up of balanced confidence prediction results
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
