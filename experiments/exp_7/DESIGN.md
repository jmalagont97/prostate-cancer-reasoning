# Experiment Design: Clinical Text TF-IDF KNN Representation & Vocabulary Sweep (MCCV) & LOOCV Evaluation
**Experiment**: experiments/exp_7/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
Optimizing the vocabulary size (`max_features`) of TF-IDF representations for clinical prompt narratives, combined with metric-alignment/dimensionality reduction methods (PCA, unsupervised EmbedKit, supervised EmbedKit, or Correlation Pruning), will reduce sparse high-dimensional text noise and hubness, yielding higher out-of-fold generalization performance under a K-Nearest Neighbors classifier compared to raw TF-IDF features, when evaluated across 100 MCCV splits and LOOCV.

## 2. Experimental Setup
- **Dataset**:
  - Clinical Text Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv` (`clinical_prompt_text` column)
  - Targets: `data/chimera26/preprocessed/task1/biopsy_decision.csv`
  - Complete cases: 190 patients (excluding the 5 audit failures).
- **Validation Splitting**:
  - Phase A: 100-split Monte Carlo Cross-Validation (`experiments/exp_4/results/mccv_design.csv`). Labeled cohort ($N=88$ complete cases) partitioned into 70 train and 18 validation per split.
  - Phase B: Leave-One-Out Cross-Validation (88 folds) over the 88 labeled complete cases.
- **Preprocessing & Text Feature Extraction Pipeline**:
  - **spaCy NLP Preprocessing Pipeline (`en_core_web_sm`)**: Convert text to lower case, filter non-alphanumeric tokens and special characters (`token.is_alpha`), remove English stop words (`token.is_stop`), and apply morphological lemmatization (`token.lemma_`).
  - `TfidfVectorizer` fit strictly on the training partition of each split/fold and applied to validation/test partitions.
  - Vocabulary size sweep (`max_features`): `[100, 300, 500, 1000, None]` (where `None` uses all vocabulary terms).
  - L2 normalization applied to output TF-IDF vectors.
- **Data Representation Techniques (applied to TF-IDF vectors)**:
  1. **Raw**: L2-normalized TF-IDF features directly.
  2. **PCA (90%)**: Fit PCA on training split, select components conserving $\ge 90\%$ cumulative variance, project train and validation sets.
  3. **EmbedKit Unsupervised**: MLP Projector with `target_dim="auto"` (auto-detected via EmbedKit diagnostics `TwoNN`) trained using self-supervised contrastive combined loss. Fit dynamically per training split (60 epochs, seed 42).
  4. **EmbedKit Supervised**: MLP Projector with `target_dim="auto"` (auto-detected via EmbedKit diagnostics `TwoNN`) trained using supervised CombinedLoss (with SupConLoss) using target biopsy labels. Fit dynamically per training split (60 epochs, seed 42).
  5. **Correlation Pruning**: Compute Pearson correlation matrix on training splits. Drop highly collinear features ($|r| > \theta$) sweeping correlation threshold $\theta \in [0.70, 0.80, 0.90, 0.95]$.
- **Classifier Sweep Space (KNN)**:
  - Neighbors $k \in \{1, 3, 5, 7, 9, 11, 15, 21\}$.
  - Weights: `['uniform', 'distance']`.
  - Metric: `['euclidean', 'cosine']`.

## 3. File Layout for This Experiment
```
experiments/exp_7/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← vocabulary + representation sweep + LOOCV script
├── results/
│   ├── grid_search_results.csv ← metrics per max_features, representation, and KNN combination over 100 splits
│   ├── best_hparams.json       ← best selected vocabulary size, representation, and KNN parameters
│   ├── loocv_metrics.json      ← final out-of-fold metrics of Phase B
│   └── loocv_predictions.csv   ← final out-of-fold predictions
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png  ← validation metric curves across vocabulary sizes and representations
    │   └── confusion_matrix.png     ← confusion matrix of final LOOCV
    └── summary.md             ← write-up of results and optimal configuration
```

## 4. Baselines
| Baseline | Config file | Expected metric range |
|----------|------------|----------------------|
| Unimodal Tabular KNN (exp_5 LOOCV) | `experiments/exp_5/results/loocv_metrics.json` | Macro-F1 = 0.6333 |
| Unimodal MRI KNN (exp_6 LOOCV) | `experiments/exp_6/results/loocv_metrics.json` | Macro-F1 = 0.5335 |

## 5. Proposed Conditions
| Condition ID | Representation | Vocabulary Size Sweep | Validation Strategy | Search Space |
|:---|:---|:---|:---|:---|
| **COND-01-Raw** | Raw TF-IDF | `[100, 300, 500, 1000, None]` | 100 MCCV Splits | $k$, weights, metric |
| **COND-02-PCA** | PCA (variance $\ge 90\%$) | `[100, 300, 500, 1000, None]` | 100 MCCV Splits | $k$, weights, metric |
| **COND-03-EmbedKit-Unsup** | Unsupervised contrastive MLP (auto dim) | `[100, 300, 500, 1000, None]` | 100 MCCV Splits | $k$, weights, metric |
| **COND-04-EmbedKit-Sup** | Supervised Triplet MLP (auto dim) | `[100, 300, 500, 1000, None]` | 100 MCCV Splits | $k$, weights, metric |
| **COND-05-Corr-Prune** | Collinear drop ($|r| > \theta$) | `[100, 300, 500, 1000, None]` | 100 MCCV Splits | $\theta \in [0.7, 0.8, 0.9, 0.95]$, $k$, weights, metric |
| **COND-06-LOOCV-Eval** | Optimal Best Representation + KNN | Optimal `max_features` | LOOCV (88 folds) | Frozen optimal configuration |

## 6. Evaluation Protocol
- **Primary Metric**: Macro-F1 score.
- **Secondary Metrics**: Accuracy, Sensitivity, Specificity.
- **Phase A (Sweep)**:
  - For each combination of vocabulary size (`max_features`), representation technique, and KNN hyperparameter set, evaluate across all 100 MCCV splits.
  - For EmbedKit conditions (`COND-03` and `COND-04`), log the dynamic latent dimension determined by `target_dim="auto"` for each split.
  - Select the optimal configuration maximizing mean validation Macro-F1.
- **Phase B (LOOCV)**:
  - Freeze the optimal configuration (`max_features`, representation method, $k$, weights, metric).
  - If an EmbedKit representation is selected, freeze `target_dim` to the mode (most frequent value) of the latent dimensions resolved across the 100 MCCV splits of the winning configuration in Phase A.
  - Execute a final Leave-One-Out validation loop (88 folds) over the 88 complete labeled cases.
  - Compute and save out-of-fold metrics, predictions, confusion matrix, and summary reports.

## 7. Risks & Mitigations
- **Risk: Small Vocabulary Sparsity**: If `max_features` is set too low (e.g. 100), important clinical terms might be truncated, degrading classification accuracy.
  - *Mitigation*: Include `max_features=None` (unrestricted vocabulary) in the sweep to benchmark against full TF-IDF representations.
- **Risk: Execution Time of Grid Search**: Sweeping 5 vocabulary sizes $\times$ 8 representation variants $\times$ 32 KNN settings $\times$ 100 splits can be computationally expensive if not structured efficiently.
  - *Mitigation*: Pre-compute TF-IDF vectorization and representation projections per split before evaluating KNN parameter grids in memory.

## 8. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for TF-IDF, PCA, PyTorch weights, EmbedKit training)
- [ ] Training script placed under `scripts/`
- [ ] Output metrics and plots saved to `results/` and `reports/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 9. Next Steps
1. Review and accept this experiment plan (hypothesis, vocabulary + representation sweep, LOOCV evaluation).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run it to execute the search and LOOCV evaluation.
