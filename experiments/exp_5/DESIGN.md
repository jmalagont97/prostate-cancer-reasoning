# Experiment Design: KNN Classifier + Correlation Pruning (exp_5)
**Experiment**: experiments/exp_5/
**Project**: pathology-reasoning
**Date**: 2026-08-16
**Author**: Antigravity & Principal Investigator
**Status**: Approved
**Revision**: v3 — corrected fuzzy v2 + selection rule F1_macro→Brier (2026-08-17)
**Legacy results**: `results_legacy_fuzzy_v1/` (fuzzy_v1_relative_vote)

---

## 1. Hypothesis

Correlation pruning based on Spearman association and hierarchical clustering
(complete linkage) reduces the number of input variables for KNN without degrading
classification performance, and may improve `F1_macro` relative to the unpruned
baseline (exp_4).

The pruning threshold is treated as a hyperparameter searched alongside the KNN
configuration.

---

## 2. Experimental Setup

### 2.1 Data

- **Feature matrix**: `data/chimera26/preprocessed/task1/main_tabular.csv` (195 × 28)
  - 27 predictor variables (15 `cli_*`, 8 `vit_*`, 4 `path_hist_*`) + `case_id`
- **Target**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_biopsy_decision_binary`
- **Confidence weights**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_confidence`
- **Splits**: `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (frozen)

### 2.2 Cohort

- **N = 88** `usable_labeled` cases
- **Class balance**: 54 yes / 34 no
- **Confidence distribution**: clear=56, borderline=18, uncertain=14

### 2.3 Excluded features

| Feature | Reason |
|---|---|
| `path_hist_bx_gl_tert` | 98.9% missing in usable cohort (>50% threshold) |

After exclusion: **26 predictor variables** retained (before missingness indicators).

---

## 3. Correlation Pruning

### 3.1 Association matrix

All associations are Spearman (unified statistic). The matrix has one row/column per
**original variable** (not per dummy).

| Variable types | Association |
|---|---|
| numeric vs. numeric | `abs(Spearman)` |
| categorical vs. numeric | `max` over all dummies of `abs(Spearman(dummy, numeric))` |
| categorical vs. categorical | `max` over all dummy pairs of `abs(Spearman)` |

Sentinel categories (`"0"`) and missingness indicators (`__is_missing`) are **excluded**
from the association matrix to avoid conflating missingness patterns with true redundancy.

### 3.2 Minimum category prevalence

Categories with `n < 5` observations in the training fold are excluded from the association
computation for that fold.

### 3.3 Hierarchical clustering

- Distance: `D[i,j] = 1 - A[i,j]`
- Method: `scipy.cluster.hierarchy.linkage(method='complete')`
- Cut threshold: `A[i,j] >= tau` ↔ `D[i,j] <= 1 - tau`

### 3.4 Representative selection (per cluster)

**Essential (evaluable) variables** — always protected:

| Variable | Column |
|---|---|
| age | `cli_age` |
| fh | `cli_fh_binary` |
| cspca | `cli_cspca` |
| pirads | `cli_pirads` |
| vol | `cli_vol` |
| psa | `cli_psa` |
| comorbidity | `cli_comorbidity_count` |
| psad | `cli_psad` |
| dre | `cli_dre` |
| bx | `cli_bx` |

Rules (in order):
1. If cluster contains ≥ 1 essential variable → keep ALL essential variables as representatives.
2. If cluster contains no essential variable → keep the medoid (lowest mean distance to others; tie-break: lowest missing rate, then lexicographic).
3. Non-representative variables are pruned (along with their dummies and missingness indicators).

### 3.5 Grid of pruning thresholds

| Condition | `tau` | Description |
|---|---:|---|
| `no_prune` | — | Baseline (all 26 variables; replicates exp_4) |
| `tau_0.80` | 0.80 | Aggressive pruning |
| `tau_0.85` | 0.85 | Moderate-pruning |
| `tau_0.90` | 0.90 | Conservative pruning |
| `tau_0.95` | 0.95 | Minimal pruning |

---

## 4. KNN Configurations

| Hyperparameter | Values | Count |
|---|---|---|
| `n_neighbors` | 1, 3, 5, 7, 9, 11, 15, 21, 31 | 9 |
| Distance metric | `euclidean`, `cosine` | 2 |
| Spatial weighting | `uniform`, `distance` (inverse) | 2 |
| Confidence variant | `standard`, `confidence_weighted` | 2 |

**Grid**: 9 × 2 × 2 × 2 = **72 configurations** per pruning condition.

**Total grid**: 5 conditions × 72 = **360 combinations**.

### 4.1 Confidence-weighted fuzzy (v2)

Neighbor labels are softened toward 0.5 by clinical confidence:

```
q_j = 0.5 + c_j × (y_j − 0.5)

c_j:
    clear      → 1.00  (q_j = y_j)
    borderline → 0.50  (q_j = 0.75 if y=1, 0.25 if y=0)
    uncertain  → 0.25  (q_j = 0.625 if y=1, 0.375 if y=0)

p(yes | x) = Σ [w_dist(j) × q_j] / Σ [w_dist(j)]
```

Confidence does NOT multiply the distance weight; it only softens the label.

---

## 5. Preprocessing (per fold, leak-safe)

### 5.1 Feature-level missingness (fit on train only)

1. Compute per-feature missing rate on the **training** fold.
2. Drop any feature where missing rate > 50%.

### 5.2 Missingness encoding (no imputation)

For every retained feature:

1. Detect NaN values.
2. Create `<feature>__is_missing` binary indicator: 1 if NaN, 0 otherwise.
3. Replace NaN with **0** in the original feature.

### 5.3 Categorical encoding

Categorical features are one-hot encoded via `OneHotEncoder(handle_unknown='ignore',
sparse_output=False)`:

| Feature | Notes |
|---|---|
| `cli_dre` | NaN → `"0"` |
| `cli_bx` | NaN → `"0"` |
| `cli_fh_binary` | NaN → `"0"` |
| `vit_smoking_status` | NaN → `"0"` |

### 5.4 Scaling

- `MinMaxScaler` applied to all numeric features (continuous + ordinal + one-hot + indicators).
- Fit exclusively on the training fold; transform validation/test.

---

## 6. Validation Protocol

### 6.1 MCCV (selection)

- 50 splits, frozen in `mccv_loocv_splits.csv` (80/20, stratified, seed=42).
- **Pruning is fold-local**: for each split, the association matrix, clustering, and
  representative selection are computed from the 70 training cases only.
- All 360 combinations evaluated per split.

### 6.2 Selection rule (v3: lexicographic F1→Brier)

1. **Primary**: highest mean `F1_macro` across 50 MCCV splits.
2. **Tie-break 1**: lowest mean `brier_score` (conventional: `mean((p-y)^2)`, lower = better).
3. **Tie-break 2**: highest mean `F1_yes`.
4. **Tie-break 3**: highest mean balanced accuracy.
5. **Tie-break 4**: highest mean MCC.

### 6.3 LOO (sanity check)

For each pruning condition, compute the **intersection** of selected variable sets across
all 50 MCCV folds → fixed feature set for all 88 LOO folds.

- Only the **single best configuration** from MCCV (over all 360 combinations) is
  evaluated via LOO.
- The LOO feature set is fixed by the winning pruning condition's MCCV intersection.
- Preprocessing re-fit on 87 training cases per fold.

---

## 7. Metrics

All metrics from `docs/EVALUATION.md` §3.1:

| Metric | Role |
|---|---|
| **Macro-F1** | Primary selection criterion → brier_score (tie-break) |
| **F1_yes** | Official primary (guardrail) |
| Balanced accuracy | Balance guardrail |
| MCC | Correlation coefficient |
| Sensitivity (recall of yes) | Secondary |
| Specificity (recall of no) | Secondary |
| Precision of yes | Secondary |
| Decision accuracy | Secondary |
| PR-AUC | Threshold-free ranking |
| ROC-AUC | Threshold-free ranking |
| `brier` (= `1 - mean((p-y)²)`) | Historical compat. (exp_4 convention; higher=better) |
| `brier_score` (= `mean((p-y)²)`) | Calibration (conventional; lower=better) |
| ECE | Calibration |

---

## 8. Artefacts

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

---

## 9. Next Steps

1. Implement `experiments/exp_5/scripts/run_knn_pruning_experiment.py`.
2. Execute MCCV search (360 combinations × 50 splits).
3. Select best condition + config.
4. Compute LOO intersection for winning condition.
5. Execute LOO (88 folds).
6. Generate summary report.
7. Update `experiments/INDEX.md` and hidden logbooks.
