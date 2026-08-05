# Implementation Plan: Out-of-Fold Diagnostic Confidence Prediction via Composite Reliability Index (ICI) & Meta-Threshold Decision Trees
**Experiment**: experiments/exp_9/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_9/scripts/train.py`
This script implements the two-phase experimental harness for predicting medical diagnostic confidence (`confidence` column in `clinical_reasoning.csv`):

1. **Load & Align Datasets**:
   - Tabular: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - MCCV Design: `experiments/exp_4/results/mccv_design.csv`.
   - Filter to the $N=91$ labeled complete-case cohort with valid `confidence` annotations (`clear`: 58, `borderline`: 18, `uncertain`: 15).

2. **Phase A: Meta-Threshold Learning over 100 MCCV Splits**:
   - For each split $s \in \{1, \dots, 100\}$:
     - Fit Tabular KNN (`exp_5` config), MRI EmbedKit KNN (`exp_6` config), and Text spaCy PCA KNN (`exp_7` config) on training split.
     - Predict out-of-fold probabilities $p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}$ on validation split.
     - Compute Composite Reliability Index $ICI$:
       $$ICI = (2 \cdot |\bar{p} - 0.50|) \cdot (1 - 2 \cdot \sigma_p)$$
     - Fit `DecisionTreeClassifier(max_depth=2, random_state=42)` mapping $ICI \to \text{confidence}$ on training split.
     - Extract decision boundary cut-points $(\tau_{1, s}, \tau_{2, s})$.
   - Compute mean meta-thresholds across 100 splits:
     $$\bar{\tau}_1 = \frac{1}{100} \sum \tau_{1, s}, \quad \bar{\tau}_2 = \frac{1}{100} \sum \tau_{2, s}$$
   - Save meta-thresholds to `experiments/exp_9/results/meta_thresholds.json`.

3. **Phase B: Frozen LOOCV Evaluation (91 Folds)**:
   - Execute Leave-One-Out loop across all 91 patients.
   - For each fold $i$:
     - Fit unimodal models on 90 training cases, predict out-of-fold probabilities for held-out patient $i$.
     - Calculate $ICI^{(i)}$.
     - Classify confidence applying frozen meta-thresholds $(\bar{\tau}_1, \bar{\tau}_2)$ WITHOUT re-fitting decision trees.
   - Compute out-of-fold 3-class Macro-F1, Accuracy, Spearman rank correlation ($\rho$), and 3x3 confusion matrix.
   - Save metrics to `results/loocv_confidence_metrics.json` and predictions to `results/loocv_confidence_predictions.csv`.

4. **Visualizations & Summary Report**:
   - Generate distribution plot of learned thresholds across 100 splits to `reports/figures/decision_tree_thresholds.png`.
   - Generate 3x3 confusion matrix plot to `reports/figures/confusion_matrix_3class.png`.
   - Generate summary report `reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_9/scripts/train.py
```
