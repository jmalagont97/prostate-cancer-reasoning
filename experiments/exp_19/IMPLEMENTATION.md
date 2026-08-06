# Implementation Plan: Diagnostic Confidence Prediction via Class-Weighted Hybrid Composite ICI Meta-Thresholding (MCCV & LOOCV)
**Experiment**: experiments/exp_19/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_19/scripts/train.py`
This script implements the Class-Weighted 1D Decision Tree Meta-Thresholding pipeline on Hybrid Composite ICI:

1. **Load Out-of-Fold Predictions & Target Data**:
   - Out-of-fold probabilities ($\tilde{p}_{\text{tab\_fuzzy}}, p_{\text{mri\_hard}}, p_{\text{text\_hard}}$) from `experiments/exp_18/results/oof_predictions.csv`.
   - Expert Confidence annotations from `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column: `uncertain`=0, `borderline`=1, `clear`=2).
   - MCCV Split Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).

2. **Compute Composite Hybrid Reliability Index ($ICI_{\text{hybrid}}$)**:
   - Calculate $p_{\text{mean}, i} = \frac{\tilde{p}_{\text{tab\_fuzzy}, i} + p_{\text{mri\_hard}, i} + p_{\text{text\_hard}, i}}{3}$.
   - Calculate $\sigma_{p, i} = \sqrt{\frac{1}{3} [(\tilde{p}_{\text{tab\_fuzzy}, i} - p_{\text{mean}, i})^2 + (p_{\text{mri\_hard}, i} - p_{\text{mean}, i})^2 + (p_{\text{text\_hard}, i} - p_{\text{mean}, i})^2]}$.
   - Calculate $\delta_i = |p_{\text{mean}, i} - 0.50|$.
   - Calculate $ICI_{\text{hybrid}, i} = (2.0 \cdot |p_{\text{mean}, i} - 0.50|) \cdot (1.0 - 2.0 \cdot \sigma_{p, i})$.

3. **Phase A (100 MCCV Splits Decision Tree Meta-Thresholding)**:
   - For each split $s \in [1..100]$, train `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI_{\text{hybrid}, \text{train}}$.
   - Extract split thresholds $\tau_{1, s}, \tau_{2, s}$.
   - Compute average meta-thresholds $\bar{\tau}_1, \bar{\tau}_2$.

4. **Phase B (Frozen LOOCV Evaluation - 88 Folds)**:
   - Freeze meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
   - Predict 3-class confidence out-of-fold for each patient:
     - $ICI_{\text{hybrid}, i} < \bar{\tau}_1 \implies \text{uncertain}$
     - $\bar{\tau}_1 \le ICI_{\text{hybrid}, i} < \bar{\tau}_2 \implies \text{borderline}$
     - $ICI_{\text{hybrid}, i} \ge \bar{\tau}_2 \implies \text{clear}$
   - Compute 3-class Macro-F1, Accuracy, Spearman $\rho$, and 3x3 confusion matrix.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_19/scripts/train.py
```
