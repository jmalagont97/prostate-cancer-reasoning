# Experiment 9 — KNN + PCA Dimensionality Reduction on MRI Embedding

**Project**: pathology-reasoning
**Task**: CHIMERA Task 1 — Prostate Biopsy Decision
**Date**: 2026-08-16
**Status**: Complete
**Tier**: Standard
**Category**: hyperparameter sweep
**Runtime**: CPU, short (PCA on 70×1024 is fast; 32,400 KNN evaluations total)

---

## 1. Hypothesis

Applying PCA dimensionality reduction to the raw 1024-dimensional MRI embedding
before KNN classification improves MCCV `F1_macro` by at least 0.02 over the
unpruned baseline (`no_pca`: MCCV `F1_macro` ~0.550), while maintaining the
official `F1_yes` metric.

This is an exploratory hypothesis. PCA provides a data-adaptive linear projection
that retains maximal variance in fewer dimensions, which should mitigate the
curse of dimensionality that degrades KNN in the 1024D space with only 70
training cases.

## 2. Scope

- **Included**: PCA on MRI embedding + KNN classification.
- **Excluded**: StandardScaler, MinMaxScaler, L2 normalization, correlation
  pruning, tabular features, text features, model ensembles, threshold tuning.

### 2.1 Why PCA Instead of Spearman Pruning (exp_8)

exp_8 tested Spearman correlation pruning — a feature-selection method that
retains one representative per correlated cluster. This failed: all τ thresholds
degraded performance vs. `no_prune`. PCA is a fundamentally different approach:
it creates a new basis that captures maximal variance across *all* original
dimensions, rather than discarding dimensions entirely. PCA may succeed where
correlation pruning failed because it preserves the full geometric structure of
the embedding in a lower-dimensional subspace.

## 3. Input Data

| File | Shape | Description |
|------|-------|-------------|
| `data/chimera26/preprocessed/task1/images.csv` | 195 × 1025 | case_id + 1024 MRI embedding dims |
| `data/chimera26/preprocessed/task1/ground_truth.csv` | 195 × 27 | target + confidence labels |
| `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` | 195 × 54 | frozen splits (MCCV 50 + LOO 88) |

### 3.1 Cohort

- Filter to 88 cases with `cohort_status == "usable_labeled"`.
- Class balance: 54 yes / 34 no (~61/39%).

### 3.2 MRI Embedding

- 1024 float dimensions (`mri_emb_0` … `mri_emb_1023`).
- Range: [-28.85, 10.22]; no NaN/Inf in usable cohort.
- **No scaling applied** — raw embedding, same as exp_6.

## 4. PCA Protocol

### 4.1 Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `whiten` | `False` | Preserve variance structure; whitening normalizes scale across components, which may alter the geometry KNN exploits |
| `svd_solver` | `full` | Deterministic; exact SVD for small matrices (70×1024) |
| `random_state` | N/A | SVD is deterministic with `full` solver |
| `n_components` | grid (§5) | Hyperparameter — the variable under test |

### 4.2 Leak-Safe Execution

PCA is fit **exclusively on the training fold** in both MCCV and LOO:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=d, svd_solver="full", whiten=False)
X_train_pca = pca.fit_transform(X_train)   # fit + transform on train
X_test_pca  = pca.transform(X_test)        # transform only on test
```

No global PCA, no test-set statistics in fit, no data leakage.

### 4.3 Maximum Valid Components

With 70 training cases per MCCV fold, the centered data matrix has rank ≤ 69.
For `n_components > 69`, PCA with `svd_solver="full"` silently caps at the
matrix rank. The grid includes `n_components=69` as the maximum valid value
under MCCV, ensuring we test the regime where PCA retains all available
directions.

### 4.4 LOO: No Intersection

Unlike exp_8 where selected dimension sets were intersected across MCCV folds,
LOO in exp_9 uses **no intersection**. Each LOO fold refits PCA on 87 training
cases (rank ≤ 86), producing a different rotation. The `n_components` value
is frozen from MCCV selection, but the basis vectors differ per fold. This is
the correct protocol: intersecting PCA bases across folds has no meaningful
geometric interpretation.

## 5. Experimental Conditions

| Condition | `n_components` | Description |
|-----------|----------------|-------------|
| `no_pca` | — | Baseline: all 1024 dims, no PCA (replicates exp_6) |
| `pca_1` | 1 | Extreme compression → 1D projection |
| `pca_23` | 23 | Low-medium regime (~2% of original dims) |
| `pca_46` | 46 | Medium-high regime (~4.5% of original dims) |
| `pca_69` | 69 | Maximum valid components in MCCV (rank-limited) |

The grid is linearly spaced across the valid range: `[1, 23, 46, 69]` — covering
extreme compression through rank-limited full PCA in 4 steps.

## 6. KNN Grid

Same 72 configurations as exp_4–exp_8:

| Parameter | Values |
|-----------|--------|
| `n_neighbors` | 1, 3, 5, 7, 9, 11, 15, 21, 31 |
| `metric` | euclidean, cosine |
| `weights` | uniform, distance |
| `variant` | standard, confidence_weighted |

9 × 2 × 2 × 2 = 72 per condition.

### 6.1 Confidence-Weighted Variant

Neighbor weights multiplied by confidence: clear=1.0, borderline=0.5, uncertain=0.25.
Used only during training (query case never appears as a neighbor). With k=1, the
weight variant is mathematically equivalent across standard/distance/confidence.

## 7. Evaluation Protocol

### 7.1 MCCV (Search)

- 50 stratified splits, 70 train / 18 validation.
- Per-fold: fit PCA on train → transform train+val → evaluate all 72 KNN configs.
- Aggregate: mean ± std of `F1_macro` across 50 splits per config.
- Selection: best `F1_macro`, tie-break `F1_yes` > balanced accuracy > MCC.
- Total: 9 conditions × 72 configs × 50 splits = **32,400 evaluations**.

### 7.2 LOO (Sanity Check)

- 88 folds, one held-out case each.
- Per-fold: fit PCA on 87 training cases → transform train+test.
- `n_components` frozen from MCCV selection.
- No intersection of PCA bases across folds.
- Single config: the MCCV global winner.
- Official metric: `F1_yes`.

### 7.3 Selection Criterion

- **Primary (local)**: `F1_macro` (MCCV mean).
- **Guardrail (official)**: `F1_yes`.

This is a documented deviation from `docs/EVALUATION.md` where `F1_yes` is primary.
The deviation is maintained for consistency with exp_4–exp_8.

## 8. Baselines

| Reference | MCCV F1_macro | LOO F1_macro | Use |
|-----------|---------------|--------------|-----|
| exp_6 (raw 1024D) | 0.550 | 0.573 | Primary external baseline |
| exp_7 (standardized) | 0.514 | 0.511 | Context |
| exp_8 (Spearman pruning) | 0.550 (no_prune) | 0.573 (no_prune) | Context |
| exp_9 `no_pca` | recalculated | recalculated | Internal matched baseline |
| exp_5 (tabular pruning) | 0.667 | 0.689 | Cross-modality context |

`no_pca` must be executed within exp_9 to ensure identical process, order, and artefacts.

## 9. Dimensionality Analysis

For each `n_components` condition, report across 50 MCCV folds:

- Explained variance ratio per component.
- Cumulative explained variance ratio (mean across folds).
- Rank of the centered training matrix per fold (informational).
- Comparison of `no_pca` vs PCA conditions on the same fold.

## 10. Decision Rules

Compare against exp_9 `no_pca` (internal) and exp_6 (external):

- **PCA beneficial**: `F1_macro` improvement ≥ 0.02 AND no `F1_yes` drop > 0.02 AND non-trivial dimensionality reduction.
- **Compression neutral**: `F1_macro` within ±0.02 but with significantly fewer dimensions.
- **PCA harmful**: `F1_macro` drop ≥ 0.02 OR `F1_yes` drop > 0.02.
- **Inconclusive**: all differences within ±0.02 with no clear dimensional advantage.

## 11. Artefacts

```
experiments/exp_9/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_mri_pca_knn_experiment.py
├── results/
│   ├── summary_selection.json
│   ├── config_log.json
│   ├── pca_report.json
│   ├── explained_variance_<condition>.csv
│   └── <condition>_<config>/
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       ├── oof_predictions_mccv.csv
│       ├── oof_predictions_loo.csv
│       ├── hyperparameters.json
│       ├── pca_log.json
│       └── validation_report.json
└── reports/
    ├── figures/
    │   ├── confusion_matrices.{png,pdf}
    │   └── explained_variance.{png,pdf}
    └── summary.md
```

## 12. Risks

1. **PCA destroys geometry for k=1**: PCA rotates the embedding space. The
   optimal metric for raw embedding (euclidean, k=1 from exp_6) may no longer
   be optimal after PCA. The KNN grid covers both euclidean and cosine to
   capture this.
2. **n_components=69 ≈ full rank**: With 70 training cases, PCA with 69
   components retains 98.5% of the centered matrix. Differences between
   `pca_69` and `no_pca` may be negligible.
3. **Explained variance ≠ predictive power**: PCA maximizes variance, not
   class separability. Low-variance directions may carry discriminative signal.
4. **LOO refits PCA per fold**: Each LOO fold produces a different basis, so
   predictions are not strictly comparable across folds in the same way as
   `no_pca`. This is inherent to the method and correctly handled.
5. **Multiple testing**: 648 combinations selected per experimental condition.
   MCCV differences are comparative evidence, not independent statistical tests.

## 13. Reproducibility

- Hash input files before execution.
- Record git commit.
- Run on tmux session 0 with real-time output.
- Save per-fold PCA explained variance, confusion matrices.
- No random seed required (deterministic splits, deterministic SVD with `full` solver).
