# Experiment Design: Diagnostic Confidence Prediction via Class-Weighted Hybrid Composite ICI Meta-Thresholding (MCCV & LOOCV)
**Experiment**: experiments/exp_19/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Computing the Composite Reliability Index ($ICI_{\text{hybrid}}$) from the **Hybrid Multimodal Probabilities** ($\tilde{p}_{\text{tab\_fuzzy}}$ from Tabular Fuzzy KNN `exp_13`, $p_{\text{mri\_hard}}$ from MRI Hard KNN `exp_6`, and $p_{\text{text\_hard}}$ from Text Hard KNN `exp_7`), and training a 1D Class-Weighted Decision Tree (`class_weight='balanced'`) over 100 MCCV splits to learn meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$, will combine the calibrated soft margin of tabular data with the sharp modal separation of MRI/Text, improving 3-class diagnostic confidence prediction out-of-fold Macro-F1 over `exp_10` (**0.3691**) and `exp_17` (**0.4470**) under LOOCV evaluation.

## 2. Experimental Setup
- **Input Features & Modality Sources**:
  - Out-of-fold probabilities from `exp_18`: $\mathbf{p}_i = [\tilde{p}_{\text{tab\_fuzzy}, i}, p_{\text{mri\_hard}, i}, p_{\text{text\_hard}, i}]^T$.
- **Explicit Composite Hybrid Reliability Index ($ICI_{\text{hybrid}}$) Equation**:
  Given patient $i$'s hybrid probability vector, compute mean inter-modality prediction $\bar{p}_i$, inter-modality standard deviation $\sigma_{p, i}$, and certitude margin $\delta_i$:
  $$\bar{p}_i = \frac{\tilde{p}_{\text{tab\_fuzzy}, i} + p_{\text{mri\_hard}, i} + p_{\text{text\_hard}, i}}{3}$$
  $$\sigma_{p, i} = \sqrt{\frac{1}{3} \left[ (\tilde{p}_{\text{tab\_fuzzy}, i} - \bar{p}_i)^2 + (p_{\text{mri\_hard}, i} - \bar{p}_i)^2 + (p_{\text{text\_hard}, i} - \bar{p}_i)^2 \right]}$$
  $$\delta_i = |\bar{p}_i - 0.50|$$
  $$ICI_{\text{hybrid}, i} = \left( 2.0 \cdot |\bar{p}_i - 0.50| \right) \cdot \left( 1.0 - 2.0 \cdot \sigma_{p, i} \right)$$
- **Target Variable**:
  - 3-Class Expert Confidence annotation from `clinical_reasoning.csv` (`confidence`: `uncertain`=0, `borderline`=1, `clear`=2).
- **Validation Harness (Identical to `exp_10` and `exp_17`)**:
  - **Phase A (100-Split MCCV Meta-Threshold Learning)**:
    - For each split $s \in [1..100]$, fit a `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI_{\text{hybrid}, \text{train}}$.
    - Log learned decision cut-points $(\tau_{1, s}, \tau_{2, s})$.
    - Compute average meta-thresholds: $\bar{\tau}_1 = \frac{1}{100}\sum \tau_{1, s}$ and $\bar{\tau}_2 = \frac{1}{100}\sum \tau_{2, s}$.
  - **Phase B (Frozen LOOCV Evaluation - 88 Folds)**:
    - Freeze meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
    - For each fold $i \in [1..88]$ in LOOCV, assign 3-class prediction:
      - $ICI_{\text{hybrid}, i} < \bar{\tau}_1 \implies \text{uncertain}$
      - $\bar{\tau}_1 \le ICI_{\text{hybrid}, i} < \bar{\tau}_2 \implies \text{borderline}$
      - $ICI_{\text{hybrid}, i} \ge \bar{\tau}_2 \implies \text{clear}$
    - Compute out-of-fold 3-class Macro-F1, Accuracy, Spearman Rank Correlation ($\rho$), and 3x3 confusion matrix.

## 3. File Layout for This Experiment
```
experiments/exp_19/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← Hybrid ICI calculation + MCCV Decision Tree meta-thresholding + LOOCV script
├── results/
│   ├── meta_thresholds.json         ← learned balanced cut-points (\bar{\tau}_1, \bar{\tau}_2)
│   ├── loocv_confidence_metrics.json← 3-class out-of-fold metrics
│   ├── oof_confidence_predictions.csv ← out-of-fold predictions
│   └── git_commit.txt               ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── decision_tree_thresholds.png ← MCCV threshold distribution & histogram
    │   └── confusion_matrix_3class.png  ← 3x3 confusion matrix
    └── summary.md                   ← final report contrasting exp_19 vs exp_10 and exp_17
```

## 4. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 3-class Macro-F1 under LOOCV.
- **Baselines to Beat**:
  - `exp_10` Class-Weighted Hard Composite ICI (LOOCV 3-Class Macro-F1 = **0.3691**, Accuracy = **39.77%**).
  - `exp_17` Class-Weighted All-Fuzzy Composite ICI (LOOCV 3-Class Macro-F1 = **0.4470**, Accuracy = **57.95%**).
- **Secondary Metrics**: Accuracy, Spearman Rank Correlation ($\rho$), Per-Class Precision/Recall.

## 5. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for Decision Tree fitting)
- [ ] Config and scripts saved in `scripts/`
- [ ] Meta-thresholds logged to `results/meta_thresholds.json`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 6. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_19`.
