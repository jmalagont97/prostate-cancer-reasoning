# Experiment Design: KNN on Full MRI Embedding (exp_6)
**Experiment**: experiments/exp_6/
**Project**: pathology-reasoning
**Date**: 2026-08-16
**Author**: Principal Investigator & Co-Investigator
**Status**: Complete

---

## 1. Hypothesis

A KNN classifier applied directly to the full 1024-dimensional MRI embedding
(`images.csv`), without any PCA, dimensionality reduction, pruning, or feature-wise
scaling, can predict `target_biopsy_decision_binary` with performance comparable
to or exceeding the tabular KNN baseline (`exp_5`: MCCV `F1_macro=0.6665`).

## 2. Experimental Setup

### 2.1 Data

- **Feature matrix**: `data/chimera26/preprocessed/task1/images.csv` (195 × 1025)
  - `case_id` + `mri_emb_0` … `mri_emb_1023` (1024 components)
- **Target**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_biopsy_decision_binary`
- **Confidence weights**: `data/chimera26/preprocessed/task1/ground_truth.csv` → `target_confidence`
- **Splits**: `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (frozen)

### 2.2 Cohort

- **N = 88** `usable_labeled` cases (4 cases without MRI excluded by `exp_2`)
- **Class balance**: 54 yes / 34 no
- **Confidence distribution**: clear=56, borderline=18, uncertain=14

### 2.3 Input characteristics

- Raw embeddings: range ≈ [-28.85, 10.22], row norms 40.72–72.70
- No NaN, no infinite values in the usable cohort
- No feature-wise scaling applied (MinMaxScaler explicitly excluded)

## 3. Preprocessing (per fold, leak-safe)

### 3.1 Input validation (hard)

1. `images.csv` shape: exactly 195 × 1025
2. Usable MRI matrix: exactly 88 × 1024
3. Column names: exactly `mri_emb_0` … `mri_emb_1023`
4. All values finite (no NaN, no Inf) → abort if violated

### 3.2 Feature extraction

- Select `mri_emb_0` … `mri_emb_1023` columns from `images.csv`
- Convert to `float64`
- No MinMax scaling, no standardization, no L2 normalization
- No PCA, no pruning, no feature selection, no dimension removal
- The KNN operates on the raw 1024-dimensional coordinate space

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
- MinMaxScaler is NOT applied; raw embeddings used directly.
- All 72 configurations evaluated per split.

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
- Output: out-of-fold predictions (binary + probability) for all 88 cases.

## 6. Metrics

All metrics from `docs/EVALUATION.md` §3.1:

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

Each figure includes axis labels `no` / `yes`, count annotations, percentage annotations
on normalized versions, and a title identifying modality and validation type.

## 8. Artefacts

```
experiments/exp_6/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_knn_image_embedding_experiment.py
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
| MRI >> Tabular | >= 0.6865 | MRI alone is superior → proceed to multimodal fusion |
| MRI ≈ Tabular | 0.6465–0.6865 | MRI has comparable signal → include in fusion |
| MRI < Tabular | < 0.6465 | MRI alone insufficient under this KNN → still include in fusion |

Note: even if MRI alone is inferior, it may still contribute positively in multimodal
fusion (`exp_8`).

## 10. Risks

| Risk | Mitigation |
|---|---|
| 1024 dimensions vs 70 training cases | Report MCCV variability; do not interpret LOO as external validation |
| Concentration of distances in high dim | Compare cosine vs euclidean within grid |
| Feature-wise scaling may help | Documented as future experiment; this experiment intentionally tests raw embeddings |
| `F1_yes` differs from selection criterion | Report both; document selection by `F1_macro` |
| Confusion matrix labeled as independent | MCCV pooled explicitly labeled as 900 accumulated predictions |

## 11. Reproducibility Checklist

- [ ] Random seeds: N/A (KNN is deterministic given frozen splits)
- [ ] Dataset version recorded (SHA-256 of `images.csv`, `ground_truth.csv`, `mccv_loocv_splits.csv`)
- [ ] Git commit hash recorded
- [ ] Environment: conda `histo-DL` (Python 3.11.15, sklearn 1.9.0, pandas 3.0.3, scipy 1.17.1)
- [ ] Working tree state documented

## 12. Next Steps

1. Review and accept this experiment plan.
2. Produce implementation plan (saved as `IMPLEMENTATION.md`).
3. Implement `run_knn_image_embedding_experiment.py`.
4. Execute MCCV search, LOO evaluation, and figure generation.
5. Compare results with `exp_4` and `exp_5` tabular baselines.
6. Update `experiments/INDEX.md` and hidden logbooks.
