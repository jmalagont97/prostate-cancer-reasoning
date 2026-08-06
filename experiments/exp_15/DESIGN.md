# Experiment Design: Clinical Text TF-IDF Fuzzy KNN Representation & Vocabulary Sweep (MCCV) & LOOCV Evaluation
**Experiment**: experiments/exp_15/ · **Project**: pathology-reasoning · **Date**: 2026-08-05 · **Status**: Complete

---

## 1. Hypothesis
Optimizing the vocabulary size (`max_features`) and dimensionality reduction representations (Raw, PCA 90%, EmbedKit Unsupervised, EmbedKit Supervised) of TF-IDF clinical prompt narratives, combined with a Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) trained on uncertainty-guided soft targets ($\tilde{y} \in [0.0, 1.0]$), will attenuate dogmatic voting from uncertain neighbor cases, smoothing decision boundaries in sparse text space and improving out-of-fold biopsy prediction Macro-F1 (baseline `exp_7`: LOOCV Macro-F1 = **0.6988**) and probability calibration without data leakage.

## 2. Experimental Setup
- **Dataset**:
  - Clinical Text Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv` (`clinical_prompt_text` column)
  - Biopsy Decision Target: `data/chimera26/preprocessed/task1/biopsy_decision.csv` (`biopsy_decision` column)
  - Clinical Reasoning Target: `data/chimera26/preprocessed/task1/clinical_reasoning.csv` (`confidence` column)
  - Cohort: $N=88$ labeled complete-case cohort for LOOCV final evaluation, matching `exp_7`.
- **Soft Target Formulation ($\tilde{y}_j$)**:
  - Expert certainty weights derived from `confidence`:
    - `clear`: $c_j = 1.00$
    - `borderline`: $c_j = 0.50$
    - `uncertain`: $c_j = 0.25$
  - Continuous soft target mapping:
    - Positive Biopsy ($y_j = 1$): $\tilde{y}_j = 0.50 + 0.50 \cdot c_j \in [0.625, 1.00]$
    - Negative Biopsy ($y_j = 0$): $\tilde{y}_j = 0.50 - 0.50 \cdot c_j \in [0.00, 0.375]$
- **Preprocessing & Text Feature Extraction Pipeline (spaCy `en_core_web_sm`)**:
  - Convert text to lowercase, filter non-alphanumeric tokens (`token.is_alpha`), remove English stop words (`token.is_stop`), apply morphological lemmatization (`token.lemma_`).
  - `TfidfVectorizer` fit strictly on training partition of each split/fold with vocabulary sweep `max_features` $\in [100, 300, 500, 1000, \text{None}]$. Apply L2 normalization.
- **Text Representation Strategies (Identical to `exp_7`)**:
  1. **Raw**: L2-normalized TF-IDF features directly.
  2. **PCA (90%)**: Fit PCA on training split, select components conserving $\ge 90\%$ cumulative variance.
  3. **EmbedKit Unsupervised**: MLP Projector with `target_dim=384` trained using contrastive loss (60 epochs, seed 42).
  4. **EmbedKit Supervised**: MLP Projector with `target_dim=384` trained using supervised CombinedLoss with soft targets (60 epochs, seed 42).
- **Validation Harness**:
  - **Phase A (MCCV Grid Search - 100 Splits)**: Sweep 5 vocabulary sizes $\times$ 4 representation strategies $\times$ 32 KNN parameter settings over 100 Monte Carlo splits (`exp_4` design). Select optimal configuration maximizing mean validation Macro-F1.
  - **Phase B (LOOCV Final Evaluation - 88 Folds)**: Evaluate optimal Fuzzy KNN configuration in Leave-One-Out Cross-Validation. Decision threshold $\tilde{p} \ge 0.50 \implies \hat{y} = 1$.

## 3. Hyperparameter Sweep Grid
- `max_features`: `[100, 300, 500, 1000, None]`
- Representation: `['raw', 'pca', 'embedkit_unsup', 'embedkit_sup']`
- `n_neighbors`: $[1, 3, 5, 7, 9, 11, 15, 21]$
- `weights`: `['uniform', 'distance']`
- `metric`: `['euclidean', 'cosine']`

## 4. File Layout for This Experiment
```
experiments/exp_15/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← vocabulary + representation sweep + LOOCV script
├── results/
│   ├── best_hparams.json             ← optimal vocabulary size, representation & Fuzzy KNN parameters
│   ├── grid_search_results.csv       ← mean Macro-F1 across all text configurations
│   ├── loocv_metrics.json            ← LOOCV metrics (Macro-F1, Accuracy, AUROC, etc.)
│   ├── oof_predictions.csv           ← out-of-fold soft probabilities & predictions
│   └── git_commit.txt                ← recorded git commit hash
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png    ← MCCV hyperparameter curves per vocabulary size
    │   ├── confusion_matrix.png      ← LOOCV 2x2 confusion matrix
    │   └── roc_curve.png             ← LOOCV ROC curve
    └── summary.md                    ← final report contrasting exp_15 vs exp_7
```

## 5. Evaluation Protocol & Decision Rules
- **Primary Metric**: Out-of-fold 2-class Macro-F1 under LOOCV.
- **Baseline to Beat**: `exp_7` Text Standard KNN (MCCV Mean Macro-F1 = **0.6329**, LOOCV Macro-F1 = **0.6988**, Accuracy = **71.59%**, Sensitivity = **0.7778**, Specificity = **0.6176**).
- **Secondary Metrics**: Accuracy, Sensitivity (Recall), Specificity, AUROC, Brier Score (calibration).

## 6. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for TF-IDF, PCA, and EmbedKit training)
- [ ] Config and scripts saved in `scripts/`
- [ ] Grid search results logged to `results/grid_search_results.csv`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 7. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and execute `exp_15`.
