# Experiment Design: MRI KNN Representation Sweep (MCCV) & LOOCV Evaluation
**Experiment**: experiments/exp_6/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
Applying noise-filtering or metric-alignment projection methods (PCA, unsupervised EmbedKit, supervised EmbedKit, or Correlation Pruning) to 1024-dimensional raw MRI embeddings will mitigate high-dimensional metric pathologies (such as hubness and loss of contrast), resulting in higher out-of-fold generalization performance under a K-Nearest Neighbors classifier compared to raw embeddings, when optimized over 100 MCCV splits and evaluated via LOOCV.

## 2. Experimental Setup
- **Dataset**:
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv` (1024 features)
  - Targets: `data/chimera26/preprocessed/task1/biopsy_decision.csv`
  - Complete cases: 190 patients (excluding the 5 audit failures).
- **Validation Splitting**:
  - Phase A: 100-split Monte Carlo Cross-Validation (`experiments/exp_4/results/mccv_design.csv`). Labeled cohort ($N=88$ complete cases) partitioned into 70 train and 18 validation.
  - Phase B: Leave-One-Out Cross-Validation (88 folds) over the 88 labeled complete cases.
- **Preprocessing Pipeline (per split/fold)**:
  - Input features are scaled using `MinMaxScaler` onto $[0, 1]$ interval.
  - To prevent data leakage, all representation learners (PCA, EmbedKit projections, Correlation calculations) must be fit strictly on the training partition of each split/fold and applied to the validation/test partition.
- **Data Representation Techniques**:
  1. **Raw**: No reduction. Scale features to $[0, 1]$.
  2. **PCA (90%)**: Fit PCA on training split, select components conserving $\ge 90\%$ cumulative variance, project train and validation sets.
  3. **EmbedKit Unsupervised**: MLP Projector with `target_dim="auto"` (auto-detected via EmbedKit diagnostics TwoNN) trained using self-supervised contrastive combined loss. Fit dynamically per training split (60 epochs, seed 42).
  4. **EmbedKit Supervised**: MLP Projector with `target_dim="auto"` (auto-detected via EmbedKit diagnostics TwoNN) trained using supervised CombinedLoss (with SupConLoss) using target biopsy labels. Fit dynamically per training split (60 epochs, seed 42).
  5. **Correlation Pruning**: Compute Pearson correlation matrix on training splits. Drop highly collinear features ($|r| > \theta$) sweeping correlation threshold $\theta \in [0.70, 0.80, 0.90, 0.95]$.
- **Classifier Sweep Space (KNN)**:
  - Neighbors $k \in \{1, 3, 5, 7, 9, 11, 15, 21\}$.
  - Weights: `['uniform', 'distance']`.
  - Metric: `['euclidean', 'cosine']`.

## 3. File Layout for This Experiment
```
experiments/exp_6/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← grid search sweep + LOOCV script
├── results/
│   ├── grid_search_results.csv ← metrics per representation and parameter combination over 100 splits
│   ├── best_hparams.json       ← best selected representation and hyperparameters
│   ├── loocv_metrics.json      ← final out-of-fold metrics of Phase B
│   └── loocv_predictions.csv   ← final out-of-fold predictions
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png  ← validation metric curves across representations
    │   └── confusion_matrix.png     ← confusion matrix of final LOOCV
    └── summary.md             ← write-up of results and optimal representation
```

## 4. Baselines
| Baseline | Config file | Expected metric range |
|----------|------------|----------------------|
| Unimodal Tabular KNN (exp_5 LOOCV) | `experiments/exp_5/results/loocv_metrics.json` | Macro-F1 = 0.6333 |
| Unimodal MRI KNN (legacy exp_15 5-fold) | N/A | Macro-F1 $\sim$ 0.54 |

## 5. Proposed Conditions
| Condition ID | Representation | Validation Strategy | Search Space |
|:---|:---|:---|:---|
| **COND-01-Raw** | Raw [0, 1] | 100 MCCV Splits | $k$, weights, metric |
| **COND-02-PCA** | PCA (variance $\ge 90\%$) | 100 MCCV Splits | $k$, weights, metric |
| **COND-03-EmbedKit-Unsup** | Unsupervised contrastive MLP (auto dim) | 100 MCCV Splits | $k$, weights, metric |
| **COND-04-EmbedKit-Sup** | Supervised Triplet MLP (auto dim) | 100 MCCV Splits | $k$, weights, metric |

| **COND-05-Corr-Prune** | Collinear drop ($|r| > \theta$) | 100 MCCV Splits | $\theta \in [0.7, 0.8, 0.9, 0.95]$, $k$, weights, metric |
| **COND-06-LOOCV-Eval** | Optimal Best Representation + KNN | LOOCV (88 folds) | Frozen optimal configuration |

## 6. Evaluation Protocol
- **Primary Metric**: Macro-F1 score.
- **Secondary Metrics**: Accuracy, Sensitivity, Specificity.
- **Phase A (Sweep)**:
  - For each representation technique and KNN configuration, run the 100 MCCV split evaluations.
  - For EmbedKit conditions (`COND-03` and `COND-04`), the dynamic latent dimension determined by `target_dim="auto"` must be recorded and logged for each split.
  - Average the metrics across all 100 splits.
  - Choose the configuration (Representation + $k$ + weights + metric) maximizing mean validation Macro-F1.
- **Phase B (LOOCV)**:
  - Freeze the optimal configuration. If an EmbedKit representation is selected, its latent dimension parameter (`target_dim`) for LOOCV will be frozen to the mode (most frequent value) of the latent dimensions resolved across the 100 MCCV splits of the winning configuration in Phase A.
  - Execute a final Leave-One-Out validation loop (88 folds) over the labeled cohort.
  - Calculate out-of-fold predictions, probabilities, metrics, and save outputs.

## 7. Risks & Mitigations
- **Risk: EmbedKit Execution Time**: Training contrastive MLPs dynamically on 100 splits of 70 training samples can be slow if done sequentially.
  - *Mitigation*: Set epochs to a reasonable limit (e.g. 60 epochs) and optimize the PyTorch loop without unnecessary logging. Ensure GPU/CPU batch operations are fast.
- **Risk: PCA Variance Collapse**: If the features are extremely correlated, PCA explaining 90% variance might yield very few components (e.g., $<5$ components), which could discard predictive detail.
  - *Mitigation*: Programmatically log the number of selected PCA components per split to analyze component dimensionality.

## 8. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for PCA, PyTorch weights, EmbedKit training)
- [ ] Training script placed under `scripts/`
- [ ] Output metrics and plots saved to `results/` and `reports/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 9. Next Steps
1. Review and accept this experiment plan (hypothesis, representation methods, LOOCV evaluation).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run it to search representations and execute final LOOCV evaluation.
