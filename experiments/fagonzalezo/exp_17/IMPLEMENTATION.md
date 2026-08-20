# Implementation Plan: Diagnostic Confidence Prediction via Class-Weighted Fuzzy ICI Meta-Thresholding (MCCV & LOOCV)
**Experiment**: experiments/exp_17/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_17/scripts/train.py`
This script implements the Class-Weighted 1D Decision Tree Meta-Thresholding pipeline on Fuzzy ICI:

1. **Load Out-of-Fold Predictions & Target Data**:
   - Out-of-fold soft probabilities ($\tilde{p}_{\text{tab}}, \tilde{p}_{\text{mri}}, \tilde{p}_{\text{text}}$) from `experiments/exp_16/results/oof_predictions.csv`.
   - Expert Confidence annotations from `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column: `uncertain`=0, `borderline`=1, `clear`=2).
   - MCCV Split Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).

2. **Compute Continuous Fuzzy Inter-Modality Conflict Index ($ICI_{\text{fuzzy}}$)**:
   - Compute mean prediction across modalities for patient $i$:
     $$\bar{p}_i = \frac{\tilde{p}_{\text{tab}, i} + \tilde{p}_{\text{mri}, i} + \tilde{p}_{\text{text}, i}}{3}$$
   - Calculate explicit inter-modality variance:
     $$ICI_{\text{fuzzy}, i} = \frac{1}{3} \left[ (\tilde{p}_{\text{tab}, i} - \bar{p}_i)^2 + (\tilde{p}_{\text{mri}, i} - \bar{p}_i)^2 + (\tilde{p}_{\text{text}, i} - \bar{p}_i)^2 \right]$$

3. **Phase A (100 MCCV Splits Decision Tree Meta-Thresholding)**:
   - For each split $s \in [1..100]$, train `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` on $ICI_{\text{fuzzy}, \text{train}}$.
   - Extract split thresholds $\tau_{1, s}, \tau_{2, s}$.
   - Compute average meta-thresholds $\bar{\tau}_1, \bar{\tau}_2$.

4. **Phase B (Frozen LOOCV Evaluation - 88 Folds)**:
   - Freeze meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$.
   - Predict 3-class confidence out-of-fold for each patient:
     - $ICI_{\text{fuzzy}, i} < \bar{\tau}_1 \implies \text{uncertain}$
     - $\bar{\tau}_1 \le ICI_{\text{fuzzy}, i} < \bar{\tau}_2 \implies \text{borderline}$
     - $ICI_{\text{fuzzy}, i} \ge \bar{\tau}_2 \implies \text{clear}$
   - Compute 3-class Macro-F1, Accuracy, Spearman $\rho$, and 3x3 confusion matrix.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_17/scripts/train.py
```
