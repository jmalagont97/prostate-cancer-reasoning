# Implementation Plan: Tabular KNN Grid Search & LOOCV
**Experiment**: experiments/exp_5/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_5/scripts/train.py`
This script will implement the grid search sweep and final LOOCV evaluation:
1. **Load Data**:
   - Tabular features from `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`.
   - Biopsy decision labels from `data/chimera26/preprocessed/task1/biopsy_decision.csv`.
   - Split partitions from `experiments/exp_4/results/mccv_design.csv`.
2. **Filter Complete Cases**:
   - Retain only the 190 complete cases aligned with `mccv_design.csv`.
   - Separate into 88 labeled cases (`biopsy_decision` in `['yes', 'no']`) and 102 unlabeled test cases.
3. **Preprocessing Pipeline (per split/fold)**:
   - For train features:
     - Numerical features (`age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`) fit and scaled using `MinMaxScaler`.
     - Categorical features (`dre`) fit and encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
     - Concatenate scaled numerical and encoded categorical matrices.
   - Transform validation/test features using the fitted transformers.
4. **Phase A: Grid Search (100 MCCV Splits)**:
   - Parameter grid:
     - `n_neighbors`: `[1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25]`
     - `weights`: `['uniform', 'distance']`
     - `metric`: `['euclidean', 'manhattan', 'cosine']`
   - For each parameter set:
     - Run over all 100 splits (training on samples with index `0`, evaluating on samples with index `1`).
     - Save average validation metrics (Macro-F1, accuracy, sensitivity, specificity) to `experiments/exp_5/results/grid_search_results.csv`.
   - Save the best configuration (maximizing mean validation Macro-F1) to `experiments/exp_5/results/best_hparams.json`.
5. **Phase B: LOOCV Final Evaluation**:
   - Retrieve optimal hyperparameters.
   - Initialize LOOCV loop over the 88 labeled cases.
   - For each fold (from 0 to 87):
     - Train on 87 cases, predict label and class probability for the 1 validation case.
     - Ensure fit/transform parameters are computed strictly on the training 87 cases.
   - Calculate final OOF validation metrics: Macro-F1, accuracy, sensitivity, specificity.
   - Save LOOCV metrics to `experiments/exp_5/results/loocv_metrics.json` and out-of-fold predictions to `experiments/exp_5/results/loocv_predictions.csv`.
6. **Reports & Visualizations**:
   - Generate `reports/figures/grid_search_curves.png` plotting validation metrics against $k$ for different distances/weights.
   - Generate `reports/figures/confusion_matrix.png` displaying the confusion matrix for final LOOCV predictions.
   - Generate a comprehensive summary report in `experiments/exp_5/reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_5/scripts/train.py
```
