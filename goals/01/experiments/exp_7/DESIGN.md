# Experiment Design: KNN on Standardized MRI Embedding (exp_7)
**Experiment**: experiments/exp_7/
**Project**: pathology-reasoning
**Date**: 2026-08-16
**Author**: Principal Investigator & Co-Investigator
**Status**: Approved

---

## 1. Hypothesis

Applying `StandardScaler` (z-score standardization per column) to the full 1024-dimensional
MRI embedding before KNN classification improves performance relative to the raw embedding
baseline (`exp_6`: MCCV `F1_macro=0.5497`).

## 2. Experimental Setup

### 2.1 Data

- **Feature matrix**: `data/chimera26/preprocessed/task1/images.csv` (195 × 1025)
  - `case_id` + `mri_emb_0` … `mri_emb_1023` (1024 components)
- **Target**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_biopsy_decision_binary`
- **Confidence weights**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_confidence`
- **Splits**: `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (frozen)

### 2.2 Cohort

- **N = 88** `usable_labeled` cases
- **Class balance**: 54 yes / 34 no
- **Confidence distribution**: clear=56, borderline=18, uncertain=14

### 2.3 Input characteristics

- Same raw embeddings as `exp_6`: range ≈ [-28.85, 10.22]
- After per-fold StandardScaler: mean ≈ 0, std ≈ 1 per column in training data

## 3. Preprocessing (per fold, leak-safe)

### 3.1 Input validation (hard)

1. `images.csv` shape: exactly 195 × 1025
2. Usable MRI matrix: exactly 88 × 1024
3. Column names: exactly `mri_emb_0` … `mri_emb_1023`
4. All values finite (no NaN, no Inf) → abort if violated

### 3.2 Per-fold standardization

For each MCCV split / LOO fold:

1. Fit `StandardScaler` on training cases only (column-wise: `\mu_j`, `\sigma_j`).
2. Transform training data: `z_ij = (x_ij - \mu_j) / \sigma_j`.
3. Apply identical `\mu_j`, `\sigma_j` to validation/test data.
4. No PCA, no pruning, no dimension removal.

Columns with zero variance in training are set to zero after transformation.

## 4. KNN Configurations

| Hyperparameter | Values | Count |
|---|---|---|
| `n_neighbors` | 1, 3, 5, 7, 9, 11, 15, 21, 31 | 9 |
| Distance metric | `euclidean`, `cosine` | 2 |
| Spatial weighting | `uniform`, `distance` (inverse) | 2 |
| Confidence variant | `standard`, `confidence_weighted` | 2 |

**Grid**: 9 × 2 × 2 × 2 = **72 configurations**.
**Total**: 72 configs × 50 MCCV splits = **3,600 evaluations**.

## 5. Validation Protocol

### 5.1 MCCV (selection)

- 50 splits, frozen in `mccv_loocv_splits.csv` (80/20, stratified, seed=42).
- Train: 70 cases. Validation: 18 cases.
- `StandardScaler` fitted per split on training data only.
- All 72 configurations evaluated per split.

### 5.2 Selection rule

1. **Primary**: highest mean Macro-F1 across 50 MCCV splits.
2. **Tie-break 1**: highest mean F1_yes.
3. **Tie-break 2**: highest mean balanced accuracy.
4. **Tie-break 3**: highest mean MCC.

### 5.3 LOO (sanity check)

- 88 folds, fixed by `loocv_fold`.
- **Only the single best configuration** from MCCV.
- `StandardScaler` fitted on 87 training cases per fold.
- Hyperparameters frozen; no tuning on LOO.

## 6. Metrics

| Metric | Role |
|---|---|
| **Macro-F1** | Primary selection criterion (local) |
| **F1_yes** | Official primary (guardrail) |
| Balanced accuracy | Balance guardrail |
| MCC | Correlation coefficient |
| Sensitivity (recall of yes) | Secondary |
| Specificity (recall of no) | Secondary |
| Precision of yes | Secondary |
| Decision accuracy | Secondary |
| PR-AUC | Threshold-free ranking |
| ROC-AUC | Threshold-free ranking |
| Brier (1 - Brier) | Historical compat. |
| Brier score (conv.) | Calibration (lower=better) |
| ECE | Calibration |
| Classification report | Diagnostic |
| Confusion matrix | Diagnostic |

## 7. Confusion Matrix Figures

Generated as explicit PNG + PDF figures:

1. **MCCV pooled** (absolute counts): 900 predictions accumulated across 50 splits.
2. **MCCV pooled** (normalized by true class): row-wise percentages.
3. **LOO** (absolute counts): 88 predictions, one per case.
4. **LOO** (normalized by true class): row-wise percentages.

## 8. Artefacts

```
experiments/exp_7/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_std_mri_experiment.py
├── results/
│   ├── summary_selection.json
│   ├── config_log.json
│   └── <config_name>/
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       ├── hyperparameters.json
│       ├── oof_predictions_mccv.csv
│       ├── oof_predictions_loo.csv
│       ├── validation_report.json
│       └── git_commit.txt
└── reports/
    ├── figures/
    │   ├── confusion_matrices.png
    │   └── confusion_matrices.pdf
    └── summary.md
```

## 9. Expected Results & Decision Rules

| Outcome | F1_macro (MCCV) | Interpretation |
|---|---|---|
| Standardized MRI >> exp_6 | > 0.5497 + 0.02 | Standardization helps → use standardized embedding in future fusion |
| Standardized MRI ≈ exp_6 | ±0.02 of 0.5497 | Standardization neutral → report both, use raw for simplicity |
| Standardized MRI < exp_6 | < 0.5497 - 0.02 | Standardization harmful → raw embedding confirmed |

Baseline: `exp_6` MCCV `F1_macro = 0.5497`, LOO `F1_macro = 0.5731`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Standardization may destroy meaningful magnitude information | Compare against `exp_6` raw baseline |
| Zero-variance columns in some folds | Set to zero post-scaling; retain column for dimensional consistency |
| Cosine metric may interact with centering | Grid includes both euclidean and cosine; report both |

## 11. Reproducibility Checklist

- [ ] Random seeds: N/A (KNN is deterministic given frozen splits)
- [ ] Git commit hash recorded
- [ ] Environment: conda `histo-DL` (Python 3.11.15, sklearn 1.9.0, pandas 3.0.3)
