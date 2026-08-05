# Implementation Plan: Out-of-Fold Diagnostic Confidence Prediction via Dynamic Fold-Level LOOCV Decision Tree Thresholding
**Experiment**: experiments/exp_11/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_11/scripts/train.py`
This script implements the pure end-to-end LOOCV dynamic thresholding pipeline for predicting medical diagnostic confidence (`confidence` column in `clinical_reasoning.csv`):

1. **Load & Align Datasets**:
   - Tabular: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - Filter to the $N=88$ labeled complete-case cohort with valid `confidence` annotations (`clear`: 56, `borderline`: 18, `uncertain`: 14).

2. **Pure Dynamic LOOCV Loop (88 Folds)**:
   - For each fold $i \in \{1, \dots, 88\}$:
     - **Step 1**: Train Tabular, MRI, and Text KNN models on 87 training cases.
     - **Step 2**: Generate unimodal predictions for the 87 training cases and compute $ICI_{\text{train}}$.
     - **Step 3**: Fit `DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)` dynamically on $ICI_{\text{train}}$ vs `confidence` of the 87 training cases.
     - **Step 4**: Extract dynamic local thresholds $(\tau_1^{(i)}, \tau_2^{(i)})$ for fold $i$.
     - **Step 5**: Predict probabilities for held-out patient $i$, compute $ICI_{\text{test}}^{(i)}$.
     - **Step 6**: Classify patient $i$ applying dynamic local fold thresholds $(\tau_1^{(i)}, \tau_2^{(i)})$.

3. **Evaluation & Artifact Generation**:
   - Save fold-by-fold learned thresholds to `experiments/exp_11/results/dynamic_thresholds_per_fold.csv`.
   - Compute out-of-fold 3-class Macro-F1, Accuracy, Spearman rank correlation ($\rho$), and 3x3 confusion matrix.
   - Save metrics to `results/loocv_confidence_metrics.json` and predictions to `results/loocv_confidence_predictions.csv`.
   - Plot threshold evolution across folds to `reports/figures/dynamic_thresholds_evolution.png`.
   - Plot 3x3 confusion matrix to `reports/figures/confusion_matrix_3class.png`.
   - Generate summary report `reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_11/scripts/train.py
```
