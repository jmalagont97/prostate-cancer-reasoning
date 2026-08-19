# Implementation Plan: KNN Classifier on All Tabular Variables (exp_4)
**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved
**Revision**: v3 — corrected fuzzy v2 + selection rule F1_macro→Brier (2026-08-17)

---

## 1. Overview & Objective

Implement `experiments/exp_4/scripts/run_knn_experiment.py`: a single self-contained script that:

1. Loads tabular data, target, confidence weights, and frozen splits.
2. Searches 72 KNN configurations via 50 MCCV splits.
3. Selects the best configuration by Macro-F1.
4. Evaluates the selected configuration once via 88-fold LOO.
5. Writes all artefacts to `experiments/exp_4/results/`.

## 2. File Structure

```
experiments/exp_4/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_experiment.py     ← single execution entry point
├── results/
│   ├── <config_name>/
│   │   ├── metrics_mccv.json
│   │   ├── metrics_loo.json
│   │   ├── config_log.json
│   │   ├── oof_predictions_mccv.csv
│   │   ├── oof_predictions_loo.csv
│   │   ├── hyperparameters.json
│   │   └── validation_report.json
│   └── summary_selection.json
└── reports/
    └── summary.md
```

## 3. Script Architecture

### 3.1 Data loading

- `main_tabular.csv` → X_raw (all columns except `case_id`)
- `ground_truth.csv` → y (`target_biopsy_decision_binary`), confidence (`target_confidence`)
- `mccv_loocv_splits.csv` → train/val masks per split, LOO fold assignments

### 3.2 Preprocessing pipeline

```python
def build_features(X_raw, train_mask):
    """Leak-safe preprocessing fitting only on train_mask indices."""
    X = X_raw.copy()

    # 1. Identify missing > 50% in train
    missing_rates = X.loc[train_mask].isna().mean()
    drop_cols = missing_rates[missing_rates > 0.5].index.tolist()

    # 2. For retained columns: add indicator + replace NaN with 0
    for col in X.columns:
        if col in drop_cols:
            continue
        indicator = f"{col}__is_missing"
        X[indicator] = X[col].isna().astype(int)
        X[col] = X[col].fillna(0)

    # 3. Drop excluded columns + indicators
    for col in drop_cols:
        X = X.drop(columns=[col, f"{col}__is_missing"], errors="ignore")

    # 4. Identify categorical vs numeric
    categorical_cols = ["cli_dre", "cli_bx", "cli_fh_binary", "vit_smoking_status"]
    # Recalculate after drops
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # 5. One-hot encode categoricals
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = pd.DataFrame(
        ohe.fit_transform(X[categorical_cols]),
        index=X.index,
        columns=ohe.get_feature_names_out(categorical_cols),
    )

    # 6. MinMax scale numerics (fit on train)
    scaler = MinMaxScaler()
    X_num = X[numeric_cols].copy()
    X_num.iloc[train_mask] = scaler.fit_transform(X_num.iloc[train_mask])
    X_num.iloc[~train_mask] = scaler.transform(X_num.iloc[~train_mask])

    # 7. Combine
    X_out = pd.concat([X_num, X_cat], axis=1)
    return X_out, ohe, scaler
```

### 3.3 KNN variants

```python
class ConfidenceWeightedKNN:
    """KNN with fuzzy probability-smoothing (v2)."""
    def __init__(self, n_neighbors, metric, use_distance_weight):
        ...

    def fit(self, X, y, confidence_weights):
        ...

    def predict_proba(self, X):
        # For each query, find k neighbors
        # Compute smoothed label: q_j = 0.5 + c_j * (y_j - 0.5)
        # Aggregate: p = sum(w_dist * q) / sum(w_dist)
        ...
```

**Fuzzy v2 formula**: `q_j = 0.5 + c_j × (y_j − 0.5)`, then `p = Σ(w_dist·q)/Σ(w_dist)`.
Confidence only softens the label; it does NOT multiply the distance weight.

Standard KNN uses `sklearn.neighbors.KNeighborsClassifier` directly.

### 3.4 MCCV loop

```python
for split_idx in range(50):
    train_mask = splits[f"mccv_split_{split_idx:02d}"] == 0
    val_mask = splits[f"mccv_split_{split_idx:02d}"] == 1

    X_train, ohe, scaler = build_features(X_raw, train_mask)
    X_val = build_features_val(X_raw, train_mask, ohe, scaler)

    for config in grid:
        model = train(config, X_train, y_train, confidence_train)
        y_pred, y_prob = model.predict(X_val), model.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_prob)
        store(split_idx, config, metrics)
```

### 3.5 Metrics computation

```python
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_yes": f1_score(y_true, y_pred, pos_label=1),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "precision_yes": precision_score(y_true, y_pred, pos_label=1),
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": 1 - ((y_prob - y_true) ** 2).mean(),      # historical compat (higher=better)
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),  # conventional (lower=better)
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
```

### 3.6 LOO evaluation

After selecting best config:
```python
best_config = select_best(config_log)
for fold_idx in range(88):
    test_case = splits[splits.loocv_fold == fold_idx].case_id.values[0]
    train_mask = splits.loocv_fold != fold_idx
    # preprocess, train, predict one case
```

## 4. Configuration name format

```python
config_name = f"knn_n{k}_metric{metric}_weights{weights}_variant{variant}"
# e.g.: knn_n11_metriceuclidean_weightsdistance_variantstandard
```

## 5. Hard validations

| Check | Rule |
|---|---|
| Input shape | `main_tabular.csv`: 195 × 28 |
| Usable cases | 88 rows where `cohort_status == usable_labeled` |
| MCCV splits | 50, each with train=70, val=18, both classes present |
| LOO folds | 88 folds, each exactly 1 test case |
| No leakage | Preprocessing fit on train indices only |
| Configs tried | 72 unique configurations |
| Selected config | Exactly 1 for LOO |

## 6. Execution

```bash
python3 experiments/exp_4/scripts/run_knn_experiment.py
```

## 7. Environment

- conda `histo-DL` (Python 3.11.15)
- scikit-learn 1.9.0, pandas 3.0.3, numpy, scipy
