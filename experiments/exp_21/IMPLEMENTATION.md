# Implementation Plan: Clinical Feature Relevance Attribution via SHAP Shapley Values
**Experiment**: experiments/exp_21/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_21/scripts/train.py`
This script implements SHAP Shapley value extraction, two-phase meta-threshold learning, and out-of-fold evaluation:

1. **Load Preprocessed Tabular Features & Urologist Reasoning Annotations**:
   - Tabular Features: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`weight_*` columns).
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - MCCV Design: `experiments/exp_4/results/mccv_design.csv`.

2. **Compute Out-of-Fold SHAP Shapley Values ($N=88$ Labeled Cases)**:
   - For each LOOCV fold $i$, fit MinMaxScaler + OneHotEncoder DRE on $N-1$ training cases, train `DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')`.
   - Use `shap.KernelExplainer` on the fitted model to compute Shapley values $\phi_{j, i}$ for all 8 features.
   - Extract absolute Shapley magnitude $|\phi_{j, i}|$.

3. **Phase A: 100 MCCV Split Meta-Threshold Learning**:
   - Over the 100 MCCV train splits, fit 1D `DecisionTreeClassifier(max_depth=3, class_weight='balanced')` on train set $|\phi_{j, s}^{(\text{train})}|$ to learn 4-class thresholds $(\bar{\tau}_1^{(j)}, \bar{\tau}_2^{(j)}, \bar{\tau}_3^{(j)})$ per feature.

4. **Phase B: Out-of-Fold LOOCV Evaluation ($N=88$)**:
   - Apply frozen meta-thresholds to predict discrete importance classes (`not_used`=0, `noted`=1, `important`=2, `decisive`=3).
   - Compute Spearman rank correlation ($\rho$) and p-value per feature against ground truth.
   - Compute 4-Class Macro-F1, Accuracy, Sensitivity, and Specificity per class.
   - Generate and export individual 4x4 confusion matrix PNG images (`cm_{col}.png`) and 2x4 grid PNG image (`confusion_matrices_all_features.png`).
   - Save JSON metrics to `results/feature_attribution_metrics.json` and `results/feature_confusion_matrices.json`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_21/scripts/train.py
```
