# Implementation Notes: exp_3

## Files

```
experiments/exp_3/scripts/
├── common.py       # load_task3(): merges structured.csv + engineered_features.csv + labels.csv;
│                    # CONDITIONS dict: {name: {numeric: [...], categorical: [...]}}
└── run_cox.py       # for each condition: LOOCV (75-fold refit) + 5-fold (splits.json), both scored
                      # with lifelines.utils.concordance_index against the SAME evaluator-equivalent
                      # formula evaluate.py uses for task3's ranking_score

experiments/exp_3/results/
└── cox_comparison.json   # per-condition {loocv_c_index, five_fold_mean_c_index, five_fold_per_fold}
```

## Design choices

- **Raw `dre`/`active_treatment_prior_to_surgery` one-hot for `baseline_aspublished`**: reproduces
  `scripts/fit_baseline_models.py::FEATURES["task3"]` exactly (same columns, same
  `OneHotEncoder(handle_unknown="ignore")`), so its 5-fold number is a direct sanity check against
  the already-published 0.641, not a re-derivation.
- **CoxPHFitter penalizer=0.1**, matching `fit_baseline_models.py` — held constant across conditions
  so any difference in C-index is attributable to the feature set, not a regularization change.
- **LOOCV pooling**: with n=1 held out per fold there is no fold-local C-index (concordance needs
  pairs) — the standard fix, used here, is to collect every held-out case's predicted partial hazard
  across all 75 folds and compute ONE global concordance_index at the end. This is the same
  "pool LOOCV predictions, then score once" pattern `exp_1`/`exp_2` used for task1's AUROC/F1.
- **Missing-value imputation**: median for numeric engineered columns (dre_ct_stage_ordinal,
  gleason/pT fields where applicable), constant "missing" category for anything still categorical.
  `ln_positive`'s structural missingness (not assessed → no positive/negative to report) is imputed
  as its own value rather than treated as an error — consistent with task1's `pirads_na`/`dre_not_done`
  flag-based treatment of structural missingness.
- Small-fold Cox convergence failures (possible at n≈15/fold for 5-fold, or n=74 for LOOCV) are caught
  and reported, not silently skipped without a count — same as `fit_baseline_models.py::cv_cox`.
