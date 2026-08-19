# Experiment 8 — KNN + Spearman Correlation Pruning on MRI Embedding

**Project**: pathology-reasoning  
**Task**: CHIMERA Task 1 — Prostate Biopsy Decision  
**Date**: 2026-08-16  
**Status**: Draft  
**Tier**: Standard (auto)  
**Category**: ablation / hyperparameter sweep  
**Runtime**: CPU, potentially long (Spearman matrix ~524K pairs per fold)

---

## 1. Hypothesis

Applying Spearman correlation pruning (hierarchical clustering, complete linkage, medoid
selection) to the raw 1024-dimensional MRI embedding reduces redundancy and improves
MCCV `F1_macro` by at least 0.02 over the unpruned baseline (exp_6: MCCV F1_macro=0.550),
while maintaining the official `F1_yes` metric.

This is an exploratory hypothesis. The 1024 embedding components have no clinical
interpretation; pruning is treated as a geometric dimensionality reduction, not variable
selection.

## 2. Scope

- **Included**: Spearman pruning on MRI embedding + KNN classification.
- **Excluded**: PCA, StandardScaler, L2 normalization, tabular features, text features,
  model ensembles, threshold tuning.

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

## 4. Correlation Pruning

### 4.1 Association Matrix

For each MCCV training fold (70 cases), compute the full 1024 × 1024 Spearman
association matrix:

```python
from scipy.stats import spearmanr
rho, _ = spearmanr(X_train)  # X_train: (70, 1024)
A = np.abs(rho)              # absolute association, shape (1024, 1024)
```

This computes ~523,776 unique pairwise associations. With scipy's `spearmanr` on
a (70, 1024) matrix, the full computation is vectorized and takes a few seconds.

**No essential variables** — all 1024 components are treated equally. No variable is
protected from removal.

**No missingness filtering** — MRI embeddings have zero NaN values in the usable cohort.

**No categorical encoding** — all components are numeric.

### 4.2 Hierarchical Clustering

Convert association to distance:

```
D_ij = 1 - |ρ_Spearman(mri_emb_i, mri_emb_j)|
```

Apply complete linkage clustering on the condensed distance matrix:

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

D = 1.0 - A
np.fill_diagonal(D, 0)
D = np.maximum(D, 0)
condensed = squareform(D, checks=False)
Z = linkage(condensed, method="complete")
```

### 4.3 Threshold Cutting

For each threshold `τ`, cut the dendrogram at distance `D_cut = 1 - τ`:

```python
labels = fcluster(Z, t=D_cut, criterion="distance")
```

Within each resulting cluster, retain only the **medoid** (component with minimum
mean distance to all other members of the cluster). All other components in the
cluster are discarded.

In case of ties, the first component by index order (`mri_emb_0` < `mri_emb_1` < …)
is selected.

### 4.4 Why Medoids (Not Essential Variables)

Unlike exp_5 where clinical variables had clear semantic roles, the 1024 MRI embedding
dimensions are abstract encoder outputs with no individual clinical meaning. There is
no principled basis to protect any subset. The medoid rule is uniform and objective:
the most central representative of each correlated cluster.

## 5. Experimental Conditions

| Condition | Threshold (τ) | D_cut | Description |
|-----------|---------------|-------|-------------|
| `no_prune` | — | — | Baseline: all 1024 dims (replicates exp_6) |
| `tau_0.30` | 0.30 | 0.70 | Aggressive pruning |
| `tau_0.60` | 0.60 | 0.40 | Moderate pruning |
| `tau_0.80` | 0.80 | 0.20 | Conservative pruning |
| `tau_0.90` | 0.90 | 0.10 | Minimal pruning |

These thresholds replicate the grid executed in exp_5 (`[0.30, 0.60, 0.80, 0.90]`),
ensuring comparability.

## 6. KNN Grid

Same 72 configurations as exp_4–exp_7:

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
- Per-fold: compute Spearman matrix on train → prune → evaluate all 72 KNN configs.
- Aggregate: mean ± std of `F1_macro` across 50 splits per config.
- Selection: best `F1_macro`, tie-break `F1_yes` > balanced accuracy > MCC.
- Total: 5 conditions × 72 configs × 50 splits = 18,000 evaluations.

### 7.2 LOO (Sanity Check)

- 88 folds, one held-out case each.
- Feature set: intersection of dimension sets selected across 50 MCCV splits for the
  winning condition.
- If the intersection is empty → mark condition as non-evaluable, do not fall back to
  full 1024 dims.
- Single config: the MCCV global winner.
- Official metric: `F1_yes`.

### 7.3 Selection Criterion

- **Primary (local)**: `F1_macro` (MCCV mean).
- **Guardrail (official)**: `F1_yes`.

This is a documented deviation from `docs/EVALUATION.md` where `F1_yes` is primary.
The deviation is maintained for consistency with exp_4–exp_7.

## 8. Baselines

| Reference | MCCV F1_macro | LOO F1_macro | Use |
|-----------|---------------|--------------|-----|
| exp_6 (raw 1024D) | 0.550 | 0.573 | Primary external baseline |
| exp_7 (standardized) | 0.514 | 0.511 | Context |
| exp_8 `no_prune` | recalculated | recalculated | Internal matched baseline |
| exp_5 (tabular pruning) | 0.667 | 0.689 | Cross-modality context |

`no_prune` must be executed within exp_8 to ensure identical process, order, and artefacts.

## 9. Reduction Analysis

For each `τ`, report across 50 MCCV folds:

- Number of dimensions retained per fold.
- Mean, min, max of retained dimensions.
- Union of retained dimensions across all folds.
- Intersection of retained dimensions across all folds.
- Reduction rate: `R = 1 - d_retained / 1024`.

For LOO: use the MCCV intersection (same protocol as exp_5).

## 10. Decision Rules

Compare against exp_8 `no_prune` (internal) and exp_6 (external):

- **Pruning beneficial**: `F1_macro` improvement ≥ 0.02 AND no `F1_yes` drop > 0.02 AND non-trivial dimensionality reduction.
- **Compression neutral**: `F1_macro` within ±0.02 but with significantly fewer dimensions.
- **Pruning harmful**: `F1_macro` drop ≥ 0.02 OR `F1_yes` drop > 0.02.
- **Inconclusive**: all differences within ±0.02 with no clear dimensional advantage.

## 11. Artefacts

```
experiments/exp_8/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_mri_embedding_pruning_experiment.py
├── results/
│   ├── summary_selection.json
│   ├── config_log.json
│   ├── pruning_report.json
│   ├── feature_frequency_<condition>.csv
│   ├── clusters_<condition>.json
│   └── <condition>_<config>/
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       ├── oof_predictions_mccv.csv
│       ├── oof_predictions_loo.csv
│       ├── hyperparameters.json
│       ├── pruning_log.json
│       └── validation_report.json
└── reports/
    ├── figures/
    │   └── confusion_matrices.{png,pdf}
    └── summary.md
```

## 12. Risks

1. **Base-dependence**: Spearman correlations depend on the coordinate basis of the
   embedding. A rotation would change which components are correlated. Results do not
   represent rotation-invariant properties.
2. **High dimensionality, small n**: The association matrix is estimated from 70
   observations over 1024 variables. Correlations may be unstable.
3. **Over-pruning**: Low τ may eliminate too many dimensions or produce an empty
   LOO intersection.
4. **Multiple testing**: 360 combinations selected per experimental condition. MCCV
   differences are comparative evidence, not independent statistical tests.
5. **Computational cost**: ~524K pairwise Spearman evaluations per fold × 50 folds ×
   5 conditions. May dominate runtime relative to KNN fitting.
6. **Documented discrepancy**: exp_5's DESIGN.md documented τ ∈ [0.80, 0.85, 0.90, 0.95]
   but the runner executed [0.30, 0.60, 0.80, 0.90]. exp_8 explicitly replicates the
   executed grid.

## 13. Reproducibility

- Hash input files before execution.
- Record git commit.
- Run on tmux session 0 with real-time output.
- Save per-fold selected dimension sets, pruning logs, confusion matrices.
- No random seed required (deterministic splits, deterministic clustering).
