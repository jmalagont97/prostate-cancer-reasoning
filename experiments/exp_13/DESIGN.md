# Experiment Design: Tabular Fuzzy KNN Sweep & LOOCV (Uncertainty-Guided Soft Targets)
**Experiment**: experiments/exp_13/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Replacing binary hard targets ($y \in \{0, 1\}$) with expert uncertainty-guided soft targets ($\tilde{y} \in [0.0, 1.0]$) inside a Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) trained on tabular clinical data will attenuate dogmatic voting from uncertain neighbor cases, smoothing decision boundaries in ambiguous feature space regions and significantly improving out-of-fold biopsy prediction Macro-F1 (baseline `exp_5`: LOOCV Macro-F1 = **0.6333**) and probability calibration without data leakage.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv` (`biopsy_decision` column, $N=195$)
  - Clinical Reasoning Annotations: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column)
  - Cohort: $N=88$ labeled complete-case cohort for LOOCV final evaluation, matching `exp_5`.
- **Soft Target Formulation ($\tilde{y}_j$)**:
  - Expert certainty weights derived from `confidence`:
    - `clear`: $c_j = 1.00$
    - `borderline`: $c_j = 0.50$
    - `uncertain`: $c_j = 0.25$
  - Continuous soft target mapping:
    - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j \in [0.625, 1.00]$
    - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j \in [0.00, 0.375]$
- **Preprocessing**:
  - Numerical Features (`age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`): `MinMaxScaler`
  - Categorical Features (`dre`): `OneHotEncoder`
- **Validation Harness**:
  - **Phase A (MCCV Grid Search - 100 Splits)**: Evaluate 48 hyperparameter configurations over 100 Monte Carlo splits (`exp_4` design). Select optimal hyperparameters $(k^*, w^*, m^*)$ maximizing mean validation Macro-F1.
  - **Phase B (LOOCV Final Evaluation - 88 Folds)**: Evaluate optimal Fuzzy KNN configuration in Leave-One-Out Cross-Validation. Decision threshold $\tilde{p} \ge 0.50 \implies \hat{y} = 1$.

## 3. Hyperparameter Sweep Grid (Identical to `exp_5`)
- `n_neighbors`: $[1, 3, 5, 7, 9, 11, 15, 21]$
- `weights`: `['uniform', 'distance']`
- `metric`: `['euclidean', 'manhattan', 'cosine']`
- **Total Configurations**: $8 \times 2 \times 3 = 48$ models.

## 4. File Layout for This Experiment
```
experiments/exp_13/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← MCCV grid search & LOOCV evaluation script
├── results/
│   ├── best_hparams.json             ← optimal Fuzzy KNN parameters (k*, w*, m*)
│   ├── grid_search_results.csv       ← mean Macro-F1 across 48 configurations
│   ├── loocv_metrics.json            ← LOOCV metrics (Macro-F1, Accuracy, AUROC, etc.)
│   ├── oof_predictions.csv           ← out-of-fold soft probabilities & predictions
│   └── git_commit.txt                ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png    ← MCCV hyperparameter curves
    │   ├── confusion_matrix.png      ← LOOCV 2x2 confusion matrix
    │   └── roc_curve.png             ← LOOCV ROC curve
    └── summary.md                    ← final report contrasting exp_13 vs exp_5
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 2-class Macro-F1 under LOOCV.
- **Baseline to Beat**: `exp_5` Tabular Standard KNN (MCCV Mean Macro-F1 = **0.6218**, LOOCV Macro-F1 = **0.6333**, Accuracy = **68.18%**, Sensitivity = **0.8519**, Specificity = **0.4118**).
- **Secondary Metrics**: Accuracy, Sensitivity (Recall), Specificity, AUROC, Brier Score (calibration).

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Config and scripts saved in `scripts/`
- [ ] Grid search results logged to `results/grid_search_results.csv`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_13`.
