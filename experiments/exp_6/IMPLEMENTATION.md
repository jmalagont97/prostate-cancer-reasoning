# Implementation Plan: KNN on Full MRI Embedding (exp_6)
**Experiment**: experiments/exp_6/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved

---

## 1. Overview & Objective

Implement `experiments/exp_6/scripts/run_knn_image_embedding_experiment.py`: a single
self-contained script that evaluates KNN classification on the full 1024-dimensional
MRI embedding, with no dimensionality reduction and no feature-wise scaling.

The script:

1. Loads MRI embeddings, target, confidence weights, and frozen splits.
2. Searches 72 KNN configurations via 50 MCCV splits.
3. Selects the best configuration by Macro-F1.
4. Evaluates the selected configuration once via 88-fold LOO.
5. Generates confusion matrix figures (MCCV pooled + LOO, absolute + normalized).
6. Writes all artefacts to `experiments/exp_6/results/` and figures to `reports/figures/`.

## 2. File Structure

```
experiments/exp_6/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_image_embedding_experiment.py     ← single execution entry point
├── results/
│   ├── <config_name>/
│   │   ├── metrics_mccv.json
│   │   ├── metrics_loo.json
│   │   ├── hyperparameters.json
│   │   ├── oof_predictions_mccv.csv
│   │   ├── oof_predictions_loo.csv
│   │   ├── validation_report.json
│   │   └── git_commit.txt
│   └── summary_selection.json
└── reports/
    ├── figures/
    │   ├── confusion_matrices.png
    │   └── confusion_matrices.pdf
    └── summary.md
```

## 3. Script Architecture

### 3.1 Data loading

- `images.csv` → X_raw (columns `mri_emb_0` … `mri_emb_1023`, shape 195 × 1024)
- `ground_truth.csv` → y (`target_biopsy_decision_binary`), confidence (`target_confidence`)
- `mccv_loocv_splits.csv` → train/val masks per split, LOO fold assignments

Filter to 88 `usable_labeled` cases via `cohort_status`.

### 3.2 Preprocessing

No preprocessing is applied. The script:

1. Selects the 1024 `mri_emb_*` columns from `images.csv`.
2. Converts to `float64`.
3. Validates all values are finite (no NaN, no Inf) → abort if violated.
4. Does NOT apply MinMaxScaler, standardization, L2 normalization, PCA, pruning, or
   any other transformation.

This differs from `exp_4`/`exp_5` where `MinMaxScaler` was fitted per fold on training
data. For MRI embeddings, the raw coordinate space is used directly.

### 3.3 KNN variants

Reused from `exp_4` (`ConfidenceWeightedKNN` class unchanged). See
`experiments/exp_4/scripts/run_knn_experiment.py` lines 159–211.

### 3.4 MCCV loop

```python
for split_idx in range(50):
    X_train = X_raw[train_idx]  # (70, 1024), no preprocessing
    X_val = X_raw[val_idx]      # (18, 1024)

    for config in grid:
        model = train(config, X_train, y_train, conf_train)
        y_pred, y_prob = model.predict(X_val), model.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_prob)
        store(split_idx, config, metrics)
```

### 3.5 LOO evaluation

```python
best_config = select_best(config_log)
for fold_idx in range(88):
    test_idx = ...; train_idx = ...
    X_train = X_raw[train_idx]  # (87, 1024)
    X_test = X_raw[test_idx]    # (1, 1024)
    # train, predict one case
```

### 3.6 Confusion matrix figures

Generated via `matplotlib` + `seaborn`. Four subplots in a 2×2 grid:

| | Raw counts | Normalized (by true class) |
|---|---|---|
| **MCCV pooled** | 900 accumulated predictions | Row-wise percentages |
| **LOO** | 88 predictions | Row-wise percentages |

Labels: `no` (0), `yes` (1). Annotations: counts + percentages on normalized panels.
Saved as both PNG (300 dpi) and PDF.

### 3.7 Metrics computation

Reused from `exp_5` (includes both `brier` and `brier_score`).

## 4. Configuration name format

```python
config_name = f"knn_n{k}_metric{metric}_weights{weight}_variant{variant}"
# e.g.: knn_n1_metriccosine_weightsuniform_variantstandard
```

Note: no `condition` prefix (unlike `exp_5` which added `tau_XX_`).

## 5. Hard validations

| Check | Rule |
|---|---|
| Input shape | `images.csv`: 195 × 1025 |
| Usable cases | 88 rows where `cohort_status == usable_labeled` |
| MRI matrix | 88 × 1024, all finite |
| MCCV splits | 50, each with train=70, val=18, both classes present |
| LOO folds | 88, each exactly 1 test case |
| No leakage | Raw embeddings used; no scaler fit on validation data |
| Configs tried | 72 unique configurations |
| Selected config | Exactly 1 for LOO |

## 6. Execution

```bash
conda activate histo-DL && python3 experiments/exp_6/scripts/run_knn_image_embedding_experiment.py
```

Estimated runtime: ~10–20 min (3,600 evaluations on 1024-dim data, CPU-only KNN).

## 7. Environment

- conda `histo-DL` (Python 3.11.15)
- scikit-learn 1.9.0, pandas 3.0.3, numpy 1.26.4, scipy 1.17.1
- matplotlib 3.10.9, seaborn 0.13.2
