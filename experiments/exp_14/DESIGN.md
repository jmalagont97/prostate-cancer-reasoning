# Experiment Design: MRI Fuzzy KNN Representation Sweep & LOOCV (Uncertainty-Guided Soft Targets)
**Experiment**: experiments/exp_14/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Applying noise-filtering or metric-alignment projection methods (Raw, PCA 90%, EmbedKit Unsupervised, or EmbedKit Supervised) to 1024-dimensional MRI embeddings combined with a Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) trained on uncertainty-guided soft targets ($\tilde{y} \in [0.0, 1.0]$) will attenuate dogmatic voting from uncertain neighbor cases, improving out-of-fold biopsy prediction Macro-F1 (baseline `exp_6`: LOOCV Macro-F1 = **0.5335**) and probability calibration without data leakage.

## 2. Experimental Setup
- **Dataset**:
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv` (1024 features)
  - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv` (`biopsy_decision` column)
  - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column)
  - Complete cases: $N=88$ labeled complete-case cohort for LOOCV final evaluation, matching `exp_6`.
- **Soft Target Formulation ($\tilde{y}_j$)**:
  - Expert certainty weights derived from `confidence`:
    - `clear`: $c_j = 1.00$
    - `borderline`: $c_j = 0.50$
    - `uncertain`: $c_j = 0.25$
  - Continuous soft target mapping:
    - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j \in [0.625, 1.00]$
    - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j \in [0.00, 0.375]$
- **MRI Representation Strategies (Identical to `exp_6`)**:
  1. **Raw**: Scaled to $[0, 1]$ via `MinMaxScaler`.
  2. **PCA (90%)**: Fit PCA on training split, select components conserving $\ge 90\%$ cumulative variance.
  3. **EmbedKit Unsupervised**: MLP Projector with `target_dim=384` trained using contrastive loss (60 epochs, seed 42).
  4. **EmbedKit Supervised**: MLP Projector with `target_dim=384` trained using supervised CombinedLoss with soft targets (60 epochs, seed 42).
- **Validation Harness**:
  - **Phase A (MCCV Grid Search - 100 Splits)**: Evaluate 4 representation strategies across KNN parameter grid ($k \in [1..21]$, weights `uniform`/`distance`, metrics `euclidean`/`cosine`) over 100 Monte Carlo splits (`exp_4` design). Select optimal representation + hyperparameters maximizing mean validation Macro-F1.
  - **Phase B (LOOCV Final Evaluation - 88 Folds)**: Evaluate optimal Fuzzy KNN configuration in Leave-One-Out Cross-Validation. Decision threshold $\tilde{p} \ge 0.50 \implies \hat{y} = 1$.

## 3. Hyperparameter Sweep Grid
- Representation: `['raw', 'pca', 'embedkit_unsup', 'embedkit_sup']`
- `n_neighbors`: $[1, 3, 5, 7, 9, 11, 15, 21]$
- `weights`: `['uniform', 'distance']`
- `metric`: `['euclidean', 'cosine']`

## 4. File Layout for This Experiment
```
experiments/exp_14/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← MCCV grid search & LOOCV evaluation script
├── results/
│   ├── best_hparams.json             ← optimal representation & Fuzzy KNN parameters
│   ├── grid_search_results.csv       ← mean Macro-F1 across representation configurations
│   ├── loocv_metrics.json            ← LOOCV metrics (Macro-F1, Accuracy, AUROC, etc.)
│   ├── oof_predictions.csv           ← out-of-fold soft probabilities & predictions
│   └── git_commit.txt                ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png    ← MCCV hyperparameter curves per representation
    │   ├── confusion_matrix.png      ← LOOCV 2x2 confusion matrix
    │   └── roc_curve.png             ← LOOCV ROC curve
    └── summary.md                    ← final report contrasting exp_14 vs exp_6
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 2-class Macro-F1 under LOOCV.
- **Baseline to Beat**: `exp_6` MRI Standard KNN (MCCV Mean Macro-F1 = **0.5469**, LOOCV Macro-F1 = **0.5335**, Accuracy = **56.82%**, Sensitivity = **0.6852**, Specificity = **0.3824**).
- **Secondary Metrics**: Accuracy, Sensitivity (Recall), Specificity, AUROC, Brier Score (calibration).

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for PCA and EmbedKit training)
- [ ] Config and scripts saved in `scripts/`
- [ ] Grid search results logged to `results/grid_search_results.csv`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_14`.
