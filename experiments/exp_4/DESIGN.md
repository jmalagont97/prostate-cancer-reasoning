# Experiment Design: KNN Classifier on All Tabular Variables (exp_4)
**Experiment**: experiments/exp_4/
**Project**: pathology-reasoning
**Date**: 2026-08-16
**Author**: Antigravity & Principal Investigator
**Status**: Approved
**Revision**: v3 — corrected fuzzy v2 + selection rule F1_macro→Brier (2026-08-17)
**Legacy results**: `results_legacy_fuzzy_v1/` (fuzzy_v1_relative_vote), `results_v2_old_selector/` (fuzzy v2 + old selector)

---

## 1. Hypothesis

A KNN classifier applied to all available tabular variables in `main_tabular.csv`
(excluding `case_id` and features exceeding the 50% missingness threshold), using
MinMax scaling, one-hot encoding, deterministic zero-replacement with missingness
indicators, distance metric search (euclidean / cosine), weighting mode search
(uniform / inverse-distance), and a confidence-weighted fuzzy variant (where
neighbor labels are softened toward 0.5 by clinical confidence: clear=1.0,
borderline=0.5, uncertain=0.25), can predict `target_biopsy_decision_binary`
competitively in the N=88 `usable_labeled` cohort.

**Selection criterion**: Lexicographic: (1) highest `F1_macro`, (2) lowest `brier_score`
conventional = `mean((p-y)^2)`, (3) highest `F1_yes`, (4) highest balanced accuracy,
(5) highest MCC. `docs/EVALUATION.md` designates `F1_yes` as the frozen primary
metric; both are reported.

### 1.1 Fuzzy formulation (corrected in v2)

The v1 implementation used a relative-vote normalization that cancelled the
confidence at k=1. The corrected v2 formulation treats confidence as a
probability-smoothing operator on each neighbor's label:

```
q_j = 0.5 + c_j × (y_j − 0.5)
```

where `c_j` ∈ {1.0 (clear), 0.5 (borderline), 0.25 (uncertain)} and `y_j` ∈ {0, 1}.

The final probability aggregates only using geometric distance weights:

```
p(yes | x) = Σ [w_dist(j) × q_j] / Σ [w_dist(j)]
```

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

## 3. Preprocessing (per fold, leak-safe)

### 3.1 Feature-level missingness (fit on train only)

1. Compute per-feature missing rate on the **training** fold.
2. Drop any feature where missing rate > 50% (both original and its `__is_missing` indicator).

### 3.2 Missingness encoding (no imputation)

For every retained feature:

1. Detect NaN values.
2. Create `<feature>__is_missing` binary indicator: 1 if NaN, 0 otherwise.
3. Replace NaN with **0** in the original feature.

### 3.3 Categorical encoding

Categorical features are one-hot encoded via `OneHotEncoder(handle_unknown='ignore',
sparse_output=False)`:

| Feature | Type | Notes |
|---|---|---|
| `cli_dre` | Categorical | Values: Normal/Nodus/Abnormal/Suspicious/Not done |
| `cli_bx` | Categorical | Values: Positive/Negative/NaN→`"0"` |
| `cli_fh_binary` | Categorical | Values: 0.0/1.0/NaN→`"0"` |
| `vit_smoking_status` | Categorical | Values: Ex-smoker/Non-smoker/Current smoker |

### 3.4 Scaling

- `MinMaxScaler` applied to **all numeric features** (continuous + ordinal + one-hot + indicators).
- Fit exclusively on the training fold; transform validation/test.

---

## 4. KNN Configurations

Each configuration combines four hyperparameters:

| Hyperparameter | Values | Count |
|---|---|---|
| `n_neighbors` | 1, 3, 5, 7, 9, 11, 15, 21, 31 | 9 |
| Distance metric | `euclidean`, `cosine` | 2 |
| Spatial weighting | `uniform`, `distance` (inverse) | 2 |
| Confidence variant | `standard`, `confidence_weighted` | 2 |

**Total grid**: 9 × 2 × 2 × 2 = **72 configurations**.

### 4.1 KNN standard (rigid)

Standard `sklearn.neighbors.KNeighborsClassifier` with `weights=uniform` or `weights=distance`.

### 4.2 KNN confidence-weighted (fuzzy v2)

Custom prediction logic using probability-smoothing:

```
For each neighbor j in training set:
    # Smoothed label probability
    q_j = 0.5 + c_j × (y_j − 0.5)

    # Geometric weight only (distance-based or uniform)
    w_dist(j) = 1                           if weights=uniform
    w_dist(j) = 1 / max(d_j, epsilon)       if weights=distance

    # Confidence values
    c_j:
        target_confidence[j] == 'clear'      → 1.00  (q_j = y_j)
        target_confidence[j] == 'borderline' → 0.50  (q_j = 0.75 if y=1, 0.25 if y=0)
        target_confidence[j] == 'uncertain'  → 0.25  (q_j = 0.625 if y=1, 0.375 if y=0)

p(yes | x) = Σ [w_dist(j) × q_j] / Σ [w_dist(j)]
prediction = 1 if p(yes) ≥ 0.5 else 0
```

Neighbor confidence is used **only for training neighbors**, never for the query case.
Confidence does NOT multiply the distance weight; it only softens the label.

---

## 5. Validation Protocol

### 5.1 MCCV (selection)

- 50 splits, frozen in `mccv_loocv_splits.csv` (80/20, stratified, seed=42).
- Train: 70 cases. Validation: 18 cases.
- Preprocessing fit exclusively on the 70 training cases.
- All 72 configurations evaluated per split.
- Metric stored per configuration per split.

### 5.2 Selection rule (v3: lexicographic F1→Brier)

1. **Primary**: highest mean `F1_macro` across 50 MCCV splits.
2. **Tie-break 1**: lowest mean `brier_score` (conventional: `mean((p-y)^2)`, lower = better).
3. **Tie-break 2**: highest mean `F1_yes`.
4. **Tie-break 3**: highest mean balanced accuracy.
5. **Tie-break 4**: highest mean MCC.

### 5.3 LOO (sanity check)

- 88 folds, fixed by `loocv_fold`.
- **Only the single best configuration** from MCCV.
- Hyperparameters frozen; no tuning on LOO.
- Preprocessing re-fit on 87 training cases per fold.
- Output: out-of-fold predictions (binary + probability) for all 88 cases.

---

## 6. Metrics

All metrics from `docs/EVALUATION.md` §3.1 (biopsy decision):

| Metric | Role |
|---|---|
| **Macro-F1** | Primary selection criterion (local) |
| **F1_yes** | Official primary (reported as guardrail) |
| Balanced accuracy | Tie-break / balance guardrail |
| MCC | Correlation coefficient |
| Sensitivity (recall of yes) | Secondary |
| Specificity (recall of no) | Secondary |
| Precision of yes | Secondary |
| Decision accuracy | Secondary |
| PR-AUC | Threshold-free ranking |
| ROC-AUC | Threshold-free ranking |
| `brier` (= `1 - mean((p-y)^2)`) | Historical compat. (higher=better) |
| `brier_score` (= `mean((p-y)^2)`) | Calibration / selection tie-break (lower=better) |
| ECE | Calibration |
| Classification report | Diagnostic |
| Confusion matrix | Diagnostic |

- ROC-AUC, PR-AUC reported as `NaN` for folds with only one class present.
- No subtask 1.2/1.3/1.4 metrics computed (this experiment predicts only 1.1).

---

## 7. Artefacts

```
experiments/exp_4/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_experiment.py
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

---

## 8. Next Steps

1. Implement `experiments/exp_4/scripts/run_knn_experiment.py`.
2. Execute MCCV search over 72 configurations.
3. Select best configuration by Macro-F1.
4. Execute single LOO run with best configuration.
5. Generate summary report.
6. Update `experiments/INDEX.md` and hidden logbooks.
