# Experiment Design: Diagnostic Confidence Prediction via Class-Weighted Fuzzy ICI Meta-Thresholding (MCCV & LOOCV)
**Experiment**: experiments/exp_17/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Computing the Inter-Modality Conflict Index ($ICI_{\text{fuzzy}}$) from continuous calibrated soft probabilities ($\tilde{p}_{\text{tab}}, \tilde{p}_{\text{mri}}, \tilde{p}_{\text{text}}$) of unimodal Fuzzy KNN models, and training a 1D Class-Weighted Decision Tree (`class_weight='balanced'`) over 100 MCCV splits to learn meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$, will yield continuous $ICI$ distributions with smooth cut-points, improving 3-class diagnostic confidence prediction out-of-fold Macro-F1 (baseline `exp_10`: LOOCV Macro-F1 = **0.3691**) under LOOCV evaluation.

## 2. Experimental Setup
- **Input Features & Modality Sources**:
  - Continuous soft probabilities $\mathbf{\tilde{p}}_i = [\tilde{p}_{\text{tab}, i}, \tilde{p}_{\text{mri}, i}, \tilde{p}_{\text{text}, i}]^T$ from `exp_13` (Tabular), `exp_14` (MRI), and `exp_15` (Text) Fuzzy KNN models.
- **Fuzzy Inter-Modality Conflict Index ($ICI_{\text{fuzzy}}$) Mathematical Equation**:
  Given the vector of continuous calibrated soft probabilities for patient $i$, $\mathbf{\tilde{p}}_i = [\tilde{p}_{\text{tab}, i}, \tilde{p}_{\text{mri}, i}, \tilde{p}_{\text{text}, i}]^T \in [0.0, 1.0]^3$, we compute the mean inter-modality prediction $\bar{p}_i$:
  $$\bar{p}_i = \frac{\tilde{p}_{\text{tab}, i} + \tilde{p}_{\text{mri}, i} + \tilde{p}_{\text{text}, i}}{3}$$
  The explicit Inter-Modality Conflict Index ($ICI_{\text{fuzzy}, i}$) measures the variance across modality predictions:
  $$ICI_{\text{fuzzy}, i} = \frac{1}{3} \left[ (\tilde{p}_{\text{tab}, i} - \bar{p}_i)^2 + (\tilde{p}_{\text{mri}, i} - \bar{p}_i)^2 + (\tilde{p}_{\text{text}, i} - \bar{p}_i)^2 \right] = \frac{1}{3} \sum_{m \in \{\text{tab}, \text{mri}, \text{text}\}} (\tilde{p}_{m, i} - \bar{p}_i)^2$$
- **Target Variable**:
  - 3-Class Expert Confidence annotation from `clinical_reasoning.csv` (`confidence`: `uncertain`=0, `borderline`=1, `clear`=2).
- **Validation Harness (Identical to `exp_10`)**:
  - **Phase A (100-Split MCCV Meta-Threshold Learning)**:
    - For each split $s \in [1..100]$, fit a `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI_{\text{fuzzy}, \text{train}}$.
    - Log learned decision cut-points $(\tau_{1, s}, \tau_{2, s})$.
    - Compute average meta-thresholds: $\bar{\tau}_1 = \frac{1}{100}\sum \tau_{1, s}$ and $\bar{\tau}_2 = \frac{1}{100}\sum \tau_{2, s}$.
  - **Phase B (Frozen LOOCV Evaluation - 88 Folds)**:
    - Freeze meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
    - For each fold $i \in [1..88]$ in LOOCV, assign 3-class prediction:
      - $ICI_{\text{fuzzy}, i} < \bar{\tau}_1 \implies \text{uncertain}$
      - $\bar{\tau}_1 \le ICI_{\text{fuzzy}, i} < \bar{\tau}_2 \implies \text{borderline}$
      - $ICI_{\text{fuzzy}, i} \ge \bar{\tau}_2 \implies \text{clear}$
    - Compute out-of-fold 3-class Macro-F1, Accuracy, Spearman Rank Correlation ($\rho$), and 3x3 confusion matrix.

## 3. File Layout for This Experiment
```
experiments/exp_17/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← Fuzzy ICI calculation + MCCV Decision Tree meta-thresholding + LOOCV script
├── results/
│   ├── meta_thresholds.json         ← learned balanced cut-points (\bar{\tau}_1, \bar{\tau}_2)
│   ├── loocv_confidence_metrics.json← 3-class out-of-fold metrics
│   ├── oof_confidence_predictions.csv ← out-of-fold predictions
│   └── git_commit.txt               ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── decision_tree_thresholds.png ← MCCV threshold distribution & histogram
    │   └── confusion_matrix_3class.png  ← 3x3 confusion matrix
    └── summary.md                   ← final report contrasting exp_17 vs exp_10
```

## 4. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 3-class Macro-F1 under LOOCV.
- **Baseline to Beat**: `exp_10` Class-Weighted ICI Meta-Thresholding (LOOCV 3-Class Macro-F1 = **0.3691**, Accuracy = **39.77%**).
- **Secondary Metrics**: Accuracy, Spearman Rank Correlation ($\rho$), Per-Class Precision/Recall.

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for Decision Tree fitting)
- [ ] Config and scripts saved in `scripts/`
- [ ] Meta-thresholds logged to `results/meta_thresholds.json`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_17`.
