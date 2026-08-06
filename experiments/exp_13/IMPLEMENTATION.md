# Implementation Plan: Tabular Fuzzy KNN Sweep & LOOCV (Uncertainty-Guided Soft Targets)
**Experiment**: experiments/exp_13/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_13/scripts/train.py`
This script implements the Fuzzy KNN grid search and LOOCV evaluation pipeline on tabular clinical data:

1. **Load & Align Datasets**:
   - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv` (`biopsy_decision` column).
   - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column).
   - MCCV Split Design: `experiments/exp_4/results/mccv_design.csv` (100 splits).

2. **Construct Uncertainty-Guided Soft Targets ($\tilde{y}_j$)**:
   - Expert certainty weights derived from `confidence`:
     - `clear`: $c_j = 1.00$
     - `borderline`: $c_j = 0.50$
     - `uncertain`: $c_j = 0.25$
     - Default (if unannotated): $c_j = 1.00$
   - Continuous soft target formula:
     - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j$
     - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j$

3. **Preprocessing**:
   - Fit `MinMaxScaler` on numerical features (`age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`) per split/fold.
   - Fit `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `dre` per split/fold.

4. **Phase A (100 MCCV Splits Grid Search)**:
   - Sweep 48 configurations of `sklearn.neighbors.KNeighborsRegressor(n_neighbors=k, weights=w, metric=m)`.
   - For each split $s \in \{1, \dots, 100\}$: fit Fuzzy KNN Regressor on training soft targets $\tilde{y}_{\text{train}}$, predict continuous probabilities $\tilde{p}_{\text{val}}$, threshold at $\tilde{p} \ge 0.50$, and compute validation Macro-F1.
   - Save all grid search results to `results/grid_search_results.csv` and select optimal parameters $(k^*, w^*, m^*)$ into `results/best_hparams.json`.

5. **Phase B (LOOCV 88 Folds Final Evaluation)**:
   - Fit optimal Fuzzy KNN Regressor $(k^*, w^*, m^*)$ on 87 training soft targets $\tilde{y}_{\text{train}}$, predict continuous soft probability $\tilde{p}_i$ for held-out patient $i$.
   - Threshold at $\tilde{p}_i \ge 0.50 \implies \hat{y}_i = 1$.
   - Compute out-of-fold Macro-F1, Accuracy, Sensitivity, Specificity, AUROC, Brier Score, and 2x2 confusion matrix.

6. **Artifact Generation**:
   - Save metrics to `results/loocv_metrics.json` and out-of-fold soft predictions to `results/oof_predictions.csv`.
   - Plot grid search curves to `reports/figures/grid_search_curves.png`.
   - Plot 2x2 confusion matrix to `reports/figures/confusion_matrix.png`.
   - Plot ROC curve to `reports/figures/roc_curve.png`.
   - Generate summary report `reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_13/scripts/train.py
```
