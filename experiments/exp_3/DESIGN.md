# Experiment Design: Tabular KNN Baseline for CHIMERA Task 1.1 (exp_3)

**Experiment**: experiments/exp_3/ · **Project**: pathology-reasoning · **Date**: 2026-08-15 · **Status**: Complete

---

## 1. Hypothesis

A KNN classifier trained on all 39 tabular features of the `usable_labeled` cohort (N=88), with missingness-aware preprocessing (no statistical imputation) and evaluated over the 50 frozen MCCV splits, produces a transparent decision baseline for subtask 1.1 (`biopsy_decision`) that is fully characterized by an exhaustive grid (4 distances × 2 rules × 2 weights × 8 k-values = 128 configurations). The best configuration is selected by **Macro-F1** on MCCV, then sanity-checked with 88-fold LOO.

## 2. Experimental Setup

- **Feature matrix**: `data/chimera26/preprocessed/task1/inputs_tabular.csv` (195 × 40, incl. `case_id` + 39 variables). Subset of `inputs.csv` verified column-by-column and value-by-value.
- **Targets**: `data/chimera26/preprocessed/task1/ground_truth.csv` — `target_biopsy_decision` ∈ {`yes`, `no`}.
- **Cohort**: 88 `usable_labeled` cases (54 `yes` / 34 `no`). All 39 variables used, no pruning.
- **Splits (frozen)**: `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`. MCCV = 50 splits of 70 train / 18 val; LOO = 88 folds.
- **Selection metric**: Macro-F1 = (F1_yes + F1_no)/2. Tie-break: F1_yes, then balanced accuracy.
  - **NOTE (explicit deviation)**: `docs/EVALUATION.md` §3.1 defines `F1_yes` as the primary 1.1 metric for the campaign. For **this** experiment the primary selection metric is **Macro-F1** (approved by the PI). `F1_yes` is still reported everywhere and used as the first tie-break. The global protocol is not silently modified; this is an exp_3-specific rule recorded here and in `IMPLEMENTATION.md`.
- **Threshold**: fixed at 0.5 for all configs. No threshold scanning on LOO or validation-pooled data.
- **LOO**: computed once, only for the MCCV-selected config, as a sanity check. It never influences selection.

## 3. Missingness & Preprocessing (no imputation)

- Read as strings (`keep_default_na=False, na_values=[], dtype=str`); missing = empty string.
- **Continuous (36 vars)**: Min-Max scaling fit **only on observed train values**. Missing → structural `0.0`; observed → `clip((x-min)/(max-min), 0, 1)`; constant feature (min==max) → observed `1.0`. A `missing_<var>` indicator distinguishes structural 0 from observed 0.
- **Categorical (3 vars: `cli_bx`, `cli_dre`, `vit_smoking_status`)**: one-hot with categories observed in train. `cli_bx="None"` is a **valid category** (prior biopsy = none), not a missing value. Missing → all-zero block + `missing_<var>=1`. Unknown categories in validation → all-zero block + `missing_<var>=0` (no new columns).
- All transforms fitted per training fold and applied to validation. No global statistics.
- `cli_fh_binary`, `vit_smoking_pack_years`, `path_hist_bx_*` are numeric with true empty-string missingness; they stay missingness-encoded (no imputation).

## 4. Grid: 128 Configurations

| Dimension | Values |
|---|---|
| distance | euclidean, manhattan, minkowski_p3, cosine |
| rule | rigid, fuzzy_confidence |
| weight | uniform, inverse_distance |
| k | 1, 3, 5, 7, 9, 11, 15, 21 |

- **Rigid**: `w_i = distance_weight_i`; **Fuzzy**: `w_i = confidence_weight_i × distance_weight_i`, where confidence comes from the neighbor's `target_confidence_code` (clear=1.0, borderline=0.5, uncertain=0.25), train-side only.
- `p_yes = Σ w_i y_i / Σ w_i` (binary weighted vote); `y_pred = 1[p_yes ≥ 0.5]`.
- `inverse_distance`: `w = 1/max(d, 1e-12)`. Neighbor selection: stable argsort (ties → lower train index).
- Cosine: `1 - cos_sim`; rows have strictly positive L2 norm by construction (missingness indicators contribute to the norm); validated at transform time.

## 5. Evaluation Protocol

Per fold (MCCV val subset): accuracy, balanced accuracy, sensitivity (recall yes), specificity (recall no), precision yes, F1_yes, F1_no, Macro-F1, MCC, Brier, ROC-AUC, PR-AUC. ROC-AUC/PR-AUC are NaN when a fold has a single class; `n_valid_splits` reported.

Per config aggregation over 50 MCCV folds: mean/std/min/max/n_valid.

**LOO (selected config)**: per-fold metrics defined at n=1 (accuracy, F1_yes, F1_no, Macro-F1, sensitivity, specificity, precision, MCC, Brier) + pooled (all 88) ROC-AUC and PR-AUC. The pooled confusion matrix is the 2×2 over the 88 held-out predictions.

**Baselines**: `always_yes` and `always_no`, computed on the same folds (MCCV per-fold aggregate + LOO pooled).

## 6. Outputs

### CSVs (results/ — canonical tabular outputs)
- `mccv_summary.csv` — one row per config (128): means/std/min/max/n_valid for the full metric suite + elapsed.
- `mccv_fold_metrics.csv` — one row per (config, split): per-fold metrics + tp/fp/tn/fn.
- `mccv_oof_predictions.csv` — one row per (config, split, val case): `config_id, split_id, case_id, y_true, p_yes, y_pred`.
- `loo_predictions.csv` — 88 rows for the selected config.
- `confusion_matrices_mccv.csv` — long format: `config_id, split_id, true_label, pred_label, count` (all configs, all splits).
- `confusion_matrix_loo.csv` — 2×2 for the selected config.
- `classification_report.csv` — per-class precision/recall/F1/support for the selected config (MCCV pooled + LOO pooled) + macro/weighted rows.
- `baseline_metrics.csv` — `always_yes`/`always_no` (MCCV aggregate + LOO pooled).
- `data_manifest.csv` — data files with sizes + sha256, cohort counts, and frozen protocol parameters.

### Selected config (results/selected_config/)
- `hyperparameters.json`, `metrics_mccv.json`, `metrics_loo.json`, `git_commit.txt`.

### Figures (reports/figures/) — selected config only
- `confusion_matrix_mccv_counts.png` — MCCV aggregated over val events (50 splits × 18 = 900 events), NOT per-patient.
- `confusion_matrix_mccv_normalized.png` — row-normalized (support=1).
- `confusion_matrix_loo_counts.png` — 88 pooled LOO predictions (one per patient).
- `confusion_matrix_loo_normalized.png` — row-normalized.
- `confusion_matrix_mccv_vs_loo.png` — side-by-side comparison.

## 7. Decision Rules

1. Rank the 128 configs by MCCV mean Macro-F1; tie-break by F1_yes, then balanced accuracy. Record the full ranking.
2. The winner is the exp_3 baseline; LOO is run for it only.
3. If the winner beats both majority baselines on Macro-F1, the baseline is declared adequate for campaign anchoring; the MCCV→LOO gap is reported as a diagnostic (KNN small-sample instability), never as a reason to change the config.

## 8. Risks & Mitigations

- **Selection leak / global statistics**: transformers fit per train fold only (asserted). Neighbors restricted to train cases (KNN lazy, by design).
- **Cosine zero-norm**: impossible here (indicators give nonzero norm); asserted at transform.
- **MCCV vs LOO gap**: expected for KNN at N=88 (87 train in LOO vs 70 in MCCV); LOO is diagnostic only.
- **Unversioned inputs**: `inputs_tabular.csv` provenance recorded in `data_manifest.csv` (hash) and DESIGN; it is an exact subset of `inputs.csv`.
- **Working tree not clean**: recorded in `git_commit.txt` at run time; results carry the commit hash of the code used.

## 9. Reproducibility Checklist

- [x] Split file frozen (never regenerated), hash recorded.
- [x] Cohort = 88 usable_labeled; class balance 54/34 recorded.
- [x] Seed: none required (deterministic KNN + frozen splits); joblib parallelism does not affect results.
- [x] All preprocessing per train fold.
- [x] No imputation.
- [x] Environment: conda `histo-DL` (numpy 1.26.4, pandas 3.0.3, scikit-learn 1.9.0, matplotlib 3.10.9).
- [x] Data hashes in `results/data_manifest.csv`.
- [x] Git commit hash at `results/selected_config/git_commit.txt`.
