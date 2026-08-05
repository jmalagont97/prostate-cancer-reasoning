# Implementation Plan: MRI KNN Representation Sweep & LOOCV
**Experiment**: experiments/exp_6/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_6/scripts/train.py`
This script will implement the representations pre-computation, KNN sweep, and final LOOCV evaluation:
1. **Load Data**:
   - MRI Embeddings from `data/chimera26/preprocessed/task1/mri_embeddings.csv`.
   - Biopsy decision labels from `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Split partitions from `experiments/exp_4/results/mccv_design.csv`.
2. **Exclusion Audit**:
   - Align patients and drop the 5 incomplete patients, leaving 190 complete cases (88 labeled, 102 unlabeled).
3. **Data Representation Pre-Computation per Split (Phase A)**:
   - To make the grid search fast, we compute the representations exactly once per MCCV split:
     - **Raw**: Scale training set using `MinMaxScaler` and transform validation set.
     - **PCA**: Apply PCA fit on the Raw train set (retaining components for explaining $\ge 90\%$ variance).
     - **EmbedKit Unsupervised**: Fit `EmbedKit(mode="self_supervised", target_dim="auto", epochs=60, random_state=42)` on Raw train features and project both train and validation splits. Record the latent dimension resolved by the diagnostics package for each split.
     - **EmbedKit Supervised**: Fit `EmbedKit(mode="supervised", target_dim="auto", epochs=60, random_state=42)` on Raw train features and project both train and validation splits. Record the latent dimension resolved by the diagnostics package for each split.
     - **Correlation Pruning**: Compute Pearson correlation on train Raw features. For $\theta \in [0.70, 0.80, 0.90, 0.95]$, greedily select features such that no two features have $|r| > \theta$. Project train and validation splits.
4. **Grid Search Sweep over 100 Splits**:
   - Sweep combinations: representation $\times$ $k$ $\times$ weights $\times$ metric.
   - For each combination:
     - Run over the 100 splits, training KNN on train and predicting on validation.
     - Save average validation metrics (Macro-F1, accuracy, sensitivity, specificity) to `experiments/exp_6/results/grid_search_results.csv`.
   - Identify the best configuration maximizing mean validation Macro-F1.
   - Save the best config parameters (including the resolved target dimensions summary if EmbedKit is selected) to `experiments/exp_6/results/best_hparams.json`.
5. **Leave-One-Out Cross-Validation (LOOCV) final evaluation (Phase B)**:
   - Freeze the best representation technique and KNN parameters.
   - If the best representation is an EmbedKit mode, freeze `target_dim` to the mode (most frequent value) of the latent dimensions resolved across the 100 MCCV splits of the winning configuration in Phase A.
   - Run LOOCV loop over the 88 complete labeled patients:
     - For each of the 88 folds, train the optimal representation transformation on 87 cases, project training and the 1 validation case.
     - Fit KNN on the projected 87 training cases and predict label and probability for the validation case.
   - Calculate out-of-fold metrics (Macro-F1, accuracy, sensitivity, specificity).
   - Save metrics to `experiments/exp_6/results/loocv_metrics.json` and out-of-fold predictions to `experiments/exp_6/results/loocv_predictions.csv`.
6. **Reports & Visualizations**:
   - Generate validation curves comparing representations under best KNN parameters to `reports/figures/grid_search_curves.png`.
   - Generate `reports/figures/confusion_matrix.png` for LOOCV predictions.
   - Generate summary report `experiments/exp_6/reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_6/scripts/train.py
```
