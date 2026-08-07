# Implementation Plan: Multivariate SHAP Vector Input Decision Tree Clinical Relevance Attribution
**Experiment**: experiments/exp_22/ · **Project**: pathology-reasoning · **Date**: 2026-08-06 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_22/scripts/train.py`
This script implements multivariate SHAP decision tree training and independent out-of-fold evaluation per feature directly in LOOCV:

1. **Load Preprocessed Tabular Features & Urologist Reasoning Annotations**:
   - Tabular Features: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`weight_*` columns for `age`, `psa`, `vol`, `pirads`, `psad`, `dre`).
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - MCCV Design: `experiments/exp_4/results/mccv_design.csv`.

2. **Compute LOOCV Out-of-Fold SHAP Shapley Vectors ($N=88$ Labeled Cases)**:
   - For each LOOCV fold $i$, train `DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')` on 87 cases.
   - Use `shap.KernelExplainer` on the fitted model to compute Shapley vectors $\boldsymbol{\Phi}^{(i)} \in \mathbb{R}^6$ for all 6 features on validation sample $i$.
   - Extract absolute Shapley magnitudes $|\boldsymbol{\Phi}^{(i)}|$.

3. **LOOCV Multivariate Decision Tree Training & Out-of-Fold Prediction ($N=88$)**:
   - For each fold $i \in \{1, \dots, 88\}$:
     1. Build matrix of training SHAP vectors $\mathbf{X}_{\text{tr}} = |\boldsymbol{\Phi}^{(\text{train})}| \in \mathbb{R}^{87 \times 6}$.
     2. For each target feature $j \in \{\text{age}, \text{psa}, \text{vol}, \text{pirads}, \text{psad}, \text{dre}\}$:
        Fit a multivariate `DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)` taking full 6D matrix $\mathbf{X}_{\text{tr}}$ to predict $y_j^{(\text{train})} \in \{0, 1, 2, 3\}$.
     3. Predict patient $i$'s out-of-fold weight classes $\hat{y}_{j, i}$ using validation sample 6D SHAP vector $\mathbf{x}_{\text{val}} = |\boldsymbol{\Phi}^{(i)}|$.

4. **Independent Multi-Metric Evaluation per Feature**:
   - **4-Class Macro-F1 Score**: Calculated independently for each feature $j$.
   - **Spearman Rank Correlation ($\rho$) & p-value**: Tested independently for each feature $j$.
   - **4x4 Confusion Matrix PNG Images**: Exported for each feature $j$ (`cm_age.png`, `cm_psa.png`, `cm_vol.png`, `cm_pirads.png`, `cm_psad.png`, `cm_dre.png`) and combined into 2x3 grid PNG (`confusion_matrices_all_features.png`).
   - **Accuracy, Sensitivity, Specificity**: Calculated per class for each feature $j$.
   - **Comparative Histograms**: Export `histograms_combined_f1_rho_exp22.png`.
   - **JSON Output Files**: Save `results/feature_attribution_metrics.json` and `results/feature_confusion_matrices.json`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_22/scripts/train.py
```
