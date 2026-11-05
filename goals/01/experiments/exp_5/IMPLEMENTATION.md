# Implementation Plan: KNN Classifier + Correlation Pruning (exp_5)
**Experiment**: experiments/exp_5/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved
**Revision**: v3 — corrected fuzzy v2 + selection rule F1_macro→Brier (2026-08-17)

---

## 1. Overview & Objective

Implement `experiments/exp_5/scripts/run_knn_pruning_experiment.py`: a single
self-contained script that extends `exp_4` with a correlation-pruning hyperparameter.
The script:

1. Loads tabular data, target, confidence weights, and frozen splits.
2. For each pruning condition (no_prune, tau 0.80/0.85/0.90/0.95) and each of 72 KNN
   configurations, evaluates over 50 MCCV splits (fold-local pruning).
3. Selects the best condition + config by Macro-F1.
4. For the winning condition, computes the LOO intersection from MCCV selected sets.
5. Evaluates the best config via 88-fold LOO with the fixed intersection feature set.
6. Writes all artefacts to `experiments/exp_5/results/`.

## 2. File Structure

```
experiments/exp_5/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_pruning_experiment.py
├── results/
│   ├── summary_selection.json
│   ├── pruning_report.json
│   ├── feature_frequency_<tau>.csv
│   ├── clusters_<tau>.json
│   ├── loo_intersection_<tau>.json
│   └── <condition>_<config>/
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       ├── oof_predictions_mccv.csv
│       ├── oof_predictions_loo.csv
│       ├── hyperparameters.json
│       ├── pruning_log.json
│       └── validation_report.json
└── reports/
    └── summary.md
```

## 3. Script Architecture

### 3.1 Pruning module (new)

```python
CORRELATION_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

ESSENTIAL_VARS = [
    "cli_age", "cli_fh_binary", "cli_cspca", "cli_pirads",
    "cli_vol", "cli_psa", "cli_comorbidity_count", "cli_psad",
    "cli_dre", "cli_bx",
]

def compute_spearman_variable_association(X_raw, categorical_cols, numeric_cols, min_cat_n=5):
    """Variable-level Spearman association matrix."""
    # Pre-fill categoricals NaN→"0" for association computation (sentinel excluded)
    # For categoricals: pd.get_dummies → drop "0" columns → abs(Spearman)
    # For numeric pairs: abs(Spearman) on pairwise-complete rows
    # Output: 26×26 matrix (variable-level)
    ...

def select_representatives(association_matrix, essential_vars):
    """Cluster + select reps. Returns list of selected original variable names."""
    # D = 1 - A
    # linkage(method='complete')
    # Cut at each tau
    # Rep rule: essential→keep all ess; no essential→medoid
    ...
```

### 3.2 Preprocessing (modified from exp_4)

```python
def build_features_pruned(X_raw, train_idx, selected_vars, categorical_cols):
    """Preprocess, keeping only selected original variables + their indicators."""
    # 1. Drop features >50% NaN in train (same as exp_4)
    # 2. Create missingness indicators
    # 3. Fill NaN→0 / NaN→"0"
    # 4. Keep only selected_vars + their __is_missing indicators
    # 5. One-hot encode categoricals (from selected_vars)
    # 6. MinMax scale numerics (from selected_vars)
    # Returns: DataFrame, ohe, scaler, dropped_by_threshold
```

### 3.3 KNN variants

Reused from `exp_4` (ConfidenceWeightedKNN class with v2 probability-smoothing).

**Fuzzy v2 formula**: `q_j = 0.5 + c_j × (y_j − 0.5)`, then `p = Σ(w_dist·q)/Σ(w_dist)`.
Confidence only softens the label; it does NOT multiply the distance weight.

### 3.4 MCCV loop

```python
for condition in [no_prune, tau_080, tau_085, tau_090, tau_095]:
    for split_idx in range(50):
        # Pruning on raw training data (if tau)
        if condition == "no_prune":
            selected = all_vars
        else:
            selected = select_representatives(
                compute_spearman_variable_association(X_raw_train, ...),
                ESSENTIAL_VARS,
            )
        # Standard preprocessing with selected_vars filter
        X_train, ohe, scaler = build_features_pruned(X_raw, train_idx, selected, ...)
        X_val = build_features_infer_pruned(X_raw, selected, drop_cols, ohe, scaler, val_idx)

        for config in grid:
            # train, predict, compute_metrics
```

### 3.5 LOO evaluation

```python
# After MCCV: compute intersection per condition
for condition in pruned_conditions:
    intersection[condition] = set.intersection(*mccv_selected_vars[condition])

# Run LOO only for best condition + best config
best_selected = intersection[best_condition]
for fold_idx in range(88):
    # build_features_pruned with best_selected (fixed)
    # standard train/predict
```

### 3.6 Feature reporting

Per condition, per split: record which original variables were selected.
At end: frequency table, cluster logs, intersection logs.

## 4. Configuration name format

```python
config_name = f"{condition}_knn_n{k}_metric{metric}_weights{weight}_variant{variant}"
# e.g.: tau_0.85_knn_n1_metriceuclidean_weightsuniform_variantstandard
```

## 5. Hard validations

| Check | Rule |
|---|---|
| Input shape | `main_tabular.csv`: 195 × 28 |
| Usable cases | 88 rows where `cohort_status == usable_labeled` |
| MCCV splits | 50, each with train=70, val=18, both classes present |
| LOO folds | 88, each exactly 1 test case |
| No leakage | Pruning uses train indices only |
| Conditions | 5 (no_prune + 4 thresholds) |
| Configs per condition | 72 |
| Essential vars always present | Verified per split |

## 6. Execution

```bash
tmux send-keys -t 0 "conda activate histo-DL && python3 experiments/exp_5/scripts/run_knn_pruning_experiment.py" C-m
```

## 7. Environment

- conda `histo-DL` (Python 3.11.15)
- scikit-learn 1.9.0, pandas 3.0.3, numpy, scipy 1.17.1
