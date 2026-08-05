# Implementation Plan: Out-of-Fold Diagnostic Confidence Prediction via Multimodal Probability State Vector p = [p_tab, p_mri, p_text]
**Experiment**: experiments/exp_12/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_12/scripts/train.py`
This script implements the 3D probability state vector LOOCV pipeline for predicting medical diagnostic confidence (`confidence` column in `clinical_reasoning.csv`):

1. **Load & Align Datasets**:
   - Tabular: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`.
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - Filter to the $N=88$ labeled complete-case cohort with valid `confidence` annotations (`clear`: 56, `borderline`: 18, `uncertain`: 14).

2. **LOOCV 3D Probability Vector Loop (88 Folds)**:
   - For each fold $i \in \{1, \dots, 88\}$:
     - **Step 1**: Train Tabular, MRI, and Text KNN models on 87 training cases.
     - **Step 2**: Generate unimodal output probability vector $\mathbf{p}_{\text{train}} = [p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}] \in \mathbb{R}^{87 \times 3}$ for the 87 training cases.
     - **Step 3**: Fit `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)` on $\mathbf{p}_{\text{train}}$ vs `confidence` of the 87 training cases.
     - **Step 4**: Predict unimodal probability vector $\mathbf{p}_{\text{test}}^{(i)} \in \mathbb{R}^3$ for held-out test patient $i$.
     - **Step 5**: Classify test patient $i$ applying the fold's trained decision tree model.
     - **Step 6**: Track feature importances for $[p_{\text{tab}}, p_{\text{mri}}, p_{\text{text}}]$.

3. **Evaluation & Artifact Generation**:
   - Save feature importances to `experiments/exp_12/results/feature_importances.json`.
   - Compute out-of-fold 3-class Macro-F1, Accuracy, Spearman rank correlation ($\rho$), and 3x3 confusion matrix.
   - Save metrics to `results/loocv_confidence_metrics.json` and predictions to `results/loocv_confidence_predictions.csv`.
   - Plot feature importances bar chart to `reports/figures/feature_importance_bar.png`.
   - Plot 3x3 confusion matrix to `reports/figures/confusion_matrix_3class.png`.
   - Generate summary report `reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_12/scripts/train.py
```
