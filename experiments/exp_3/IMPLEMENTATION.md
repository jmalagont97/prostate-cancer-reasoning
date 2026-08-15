# Implementation Plan: Tabular KNN Baseline (exp_3)

**Experiment**: experiments/exp_3/ · **Project**: pathology-reasoning · **Date**: 2026-08-15 · **Status**: Approved

---

## 1. Overview

Two scripts implement and report the experiment defined in `DESIGN.md`:

- `scripts/knn_baseline.py` — data loading, per-fold preprocessing, KNN predictors, MCCV (128 configs), selection by Macro-F1, LOO for the selected config, and all CSV/JSON outputs.
- `scripts/plot_results.py` — reads the CSVs and produces the confusion-matrix figures.

Execution entry point: `python experiments/exp_3/scripts/knn_baseline.py` (then `python experiments/exp_3/scripts/plot_results.py`), in conda env `histo-DL`.

## 2. Module Layout

```
experiments/exp_3/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── __init__.py
│   ├── knn_baseline.py
│   └── plot_results.py
├── results/
│   ├── data_manifest.csv
│   ├── mccv_summary.csv
│   ├── mccv_fold_metrics.csv
│   ├── mccv_oof_predictions.csv
│   ├── loo_predictions.csv
│   ├── confusion_matrices_mccv.csv
│   ├── confusion_matrix_loo.csv
│   ├── classification_report.csv
│   ├── baseline_metrics.csv
│   └── selected_config/
│       ├── hyperparameters.json
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       └── git_commit.txt
└── reports/
    ├── summary.md          (written after execution)
    └── figures/
        ├── confusion_matrix_mccv_counts.png
        ├── confusion_matrix_mccv_normalized.png
        ├── confusion_matrix_loo_counts.png
        ├── confusion_matrix_loo_normalized.png
        └── confusion_matrix_mccv_vs_loo.png
```

## 3. knn_baseline.py

### 3.1 Constants
- `PROJECT_ROOT = Path(__file__).resolve().parents[3]` (robust; do not hardcode).
- Data paths + `SHA256` manifest writing.
- `CONTINUOUS_COLS` (36) and `CATEGORICAL_COLS = ["cli_bx", "cli_dre", "vit_smoking_status"]` derived from `inputs_tabular.csv` schema.
- `CONFIDENCE_MAP = {2.0: 1.0, 1.0: 0.5, 0.0: 0.25}` (clear/borderline/uncertain).
- `GRID`: distances × rules × weights × k → 128 entries.
- `N_JOBS = 22`.

### 3.2 Data loading
`load_data()` reads with `keep_default_na=False, na_values=[], dtype=str`; merges splits, features, and targets on `case_id`; keeps `cohort_status == usable_labeled`; builds `y` (yes=1/no=0) and `conf_w` (per train case). Asserts: 88 rows, 54/34 balance, 50 splits × val=18, 88 LOO folds.

### 3.3 TabularTransformer
- `fit(df_train)`: per continuous col, min/max over **observed** train values only; per categorical col, sorted categories observed in train. Stores nothing global.
- `transform(df) -> (X, col_names)`: continuous scaled `[0,1]` (missing→0.0, constant→1.0), categorical one-hot (missing→0 block), `missing_<var>` for all 39. Returns `float64`, NaN-free (asserted).
- Cosine guard: asserts no zero-norm row in transformed matrices.

### 3.4 Predictors
`knn_predict(Xtr, Xva, ytr, conf_w_tr, cfg) -> (p_yes, y_pred)`:
- distance matrix via `scipy.spatial.distance.cdist` (euclidean / cityblock / minkowski p=3 / cosine).
- stable argsort; take top-k.
- weights: uniform or `1/max(d,1e-12)`; rule fuzzy multiplies by neighbor `conf_w`.
- `p_yes = Σw·y / Σw` (divide-by-zero guarded); `y_pred = p_yes ≥ 0.5`.

### 3.5 Metrics
`fold_metrics(y_true, p_yes, y_pred) -> dict` with accuracy, balanced_accuracy, sensitivity, specificity, precision_yes, F1_yes, F1_no, Macro_F1, MCC, Brier, ROC_AUC, PR_AUC (the last two NaN if single-class fold), tp/fp/tn/fn.

### 3.6 MCCV
For each (config, split): fit transform on train 70, predict val 18. Returns fold metrics + OOF rows + confusion matrix. Run via `joblib.Parallel(n_jobs=22)`. Aggregates per config (mean/std/min/max/n_valid). Writes `mccv_fold_metrics.csv`, `mccv_summary.csv`, `mccv_oof_predictions.csv`, `confusion_matrices_mccv.csv`.

Baselines `always_yes`/`always_no` evaluated per fold on the same y_true; aggregated to `baseline_metrics.csv`.

### 3.7 Selection
Rank by `Macro_F1_mean`; tie-break `F1_yes_mean`, then `balanced_accuracy_mean`. Selected config written to `selected_config/hyperparameters.json` + `git_commit.txt` (`git log -1 --format="%H %s"`).

### 3.8 LOO (selected config)
88 folds; per fold fit transform on 87, predict 1. Per-fold metrics (n=1-safe) + pooled ROC_AUC/PR_AUC over all 88. Writes `loo_predictions.csv`, `confusion_matrix_loo.csv`, `metrics_loo.json`, and `classification_report.csv` (pooled MCCV + pooled LOO, per class + macro/weighted via sklearn `classification_report` internals).

## 4. plot_results.py

Reads `confusion_matrices_mccv.csv` (selected config), `confusion_matrix_loo.csv`, `classification_report.csv`. Produces the 5 figures with matplotlib (`imshow`, blue colormap, annotations, `dpi=200`). MCCV figures state explicitly in the title that they aggregate 900 validation events across 50 splits; LOO figures state one prediction per patient (88).

## 5. Verification after execution

- Row counts: `mccv_summary.csv` = 128; `mccv_fold_metrics.csv` = 6400; `mccv_oof_predictions.csv` = 115200; `loo_predictions.csv` = 88; `confusion_matrices_mccv.csv` = 25600.
- `X` transform has 0 NaNs and finite values.
- Selected config appears at rank 1 of `mccv_summary.csv`.
- Figures exist and are non-empty.
- `data_manifest.csv` hashes match the frozen files.
