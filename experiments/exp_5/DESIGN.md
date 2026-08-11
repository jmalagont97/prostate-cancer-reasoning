# Experiment Design: 8-Model Search for Variable-Weight Prediction
**Experiment**: experiments/exp_5/ · **Project**: challenge_chimera_2 · **Date**: 2026-08-10 · **Status**: Complete

## Hypothesis
Weights has only ever used plain OvR logistic regression across `exp_1`–`exp_4`, unlike
decision/confidence which got the full `exp_3` 8-model search. Extending that same search
(SVM, Random Forest, XGBoost, Extra Trees, MLP, Gaussian Naive Bayes, kNN, KDM) to the
variable-weight target — for both official (full-frame) and restricted (per-factor) scope —
finds a model/scope combination that clearly beats the incumbent best (`weights_official_flags`,
`exp_2`, mean ordinal error 0.585) and ideally the naive per-factor baseline (≈0.401–0.413).

## Setup
- Same 8 models, same hyperparameters, reused unchanged from `experiments/exp_3/scripts/{models.py,cv_utils.py}`.
  `cv_utils.repeated_cv_proba()` is already generic over `n_classes` (inferred from `y`), so the
  4-class weight target (`not_used`/`noted`/`important`/`decisive`) needs no new plumbing — same
  code path already used for confidence's 3-class target.
- **Feature set**: `exp_3`'s 19-column with-MRI frame (`select_exp3_feature_frame`) for
  **official** scope, per this session's explicit choice (not `exp_4`'s no-MRI frame, despite it
  scoring marginally better for weights with logistic regression — testing whether that holds
  for other model families rather than assuming it).
- **Restricted scope**: unchanged per-factor groups from `exp_2`/`exp_3`
  (`features.restricted_feature_group()`), MRI excluded from every group (same reasoning as
  before — no factor's weight is specifically MRI-grounded). Re-tested with all 8 models per
  this session's explicit choice, despite restricted scope losing to official in every prior
  experiment (`exp_2`, `exp_3`, `exp_4`) — this checks whether that finding is specific to
  logistic regression or holds across model families too.
- **Targets**: same 9 in-scope factors as always (`fh` excluded — separate tool-revealed source).
- **Conditions**: 16 total — `weights_official_{svm,rf,xgb,extratrees,mlp,nb,knn,kdm}` (8) +
  `weights_restricted_{svm,rf,xgb,extratrees,mlp,nb,knn,kdm}` (8). Each condition applies one
  model family uniformly across all 9 factors (not a per-factor model search — that would be
  9×8=72 independent choices, too fine-grained to interpret cleanly) and reports one aggregate
  `mean_ordinal_error`/`mean_decisive_set_f1` across the 9 factors, same shape as every prior
  weights condition.

## Metric & Decision
Same metrics as `exp_1`–`exp_4`: mean ordinal error (primary) and mean decisive-set F1
(secondary) across the 9 in-scope factors, plus per-factor breakdown. Compare each condition
against the naive per-factor baseline and against `exp_2`'s incumbent best (0.585). Git commit —
N/A, not a git repository, same as every experiment in this project.

## Risks
- **Restricted scope's smallest groups are single columns** (e.g. `age` → just `cli_age`,
  `pirads` → just `cli_pirads`) — some model families (SVM, RF, XGBoost, MLP) may behave
  degenerately on a 1-dimensional input (effectively a threshold rule). Not a bug if it happens;
  worth noting in the report rather than being surprised by it.
- **16 conditions × 9 factors for restricted scope = up to 144 individual model fits** (vs. 8 for
  official scope, one per condition) — meaningfully heavier than any prior experiment; expect
  longer runtime, plan to background the restricted-scope runs.

## Next Step
Implement directly (Lean-ish tier — reuses `exp_3`'s registry/harness unchanged, same pattern as
`exp_4`); no separate implementation plan needed given how mechanical the extension is.
