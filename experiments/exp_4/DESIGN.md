# Experiment Design: Clinical-Only Features (No MRI-PCA) — exp_3 Ablation
**Experiment**: experiments/exp_4/ · **Project**: challenge_chimera_2 · **Date**: 2026-08-10 · **Status**: Draft

## Hypothesis
Removing the 2-component MRI-embedding PCA from `exp_3`'s 19-column feature set (leaving the 16
clinical + comorbidity-flag columns only) does not meaningfully change decision/confidence
results — isolating whether `exp_3`'s MRI-PCA addition was actually helping, since `exp_3` never
tested a matched no-MRI condition with its 8-model comparison (only `exp_1`'s older, different
feature set had a no-MRI vs. with-MRI ablation).

## Setup
- Same 8 models as `exp_3` (SVM, RF, XGBoost, Extra Trees, MLP, Naive Bayes, kNN, KDM), same
  hyperparameters, reused directly from `experiments/exp_3/scripts/models.py` and `cv_utils.py`
  — no changes to either file.
- Feature set: `exp_3`'s 19-column frame minus `mri_pca_0`/`mri_pca_1`/`mri_missing` = 16 columns
  (10 clinical incl. `psa`+`psad` PSA family, 6 comorbidity flags). Same `N_REPEATS`/CV settings
  as `exp_3`.
- Targets: decision (8 models) + confidence (8 models) — the two targets `exp_3` model-searched.
  Weights (official/restricted) and reveal also re-run on the 16-column frame for completeness,
  same single-logistic-model pattern as `exp_2`/`exp_3`.
- Total: 18 conditions (8 decision + 8 confidence + 2 weights + 1 reveal — one fewer than exp_3
  since weights/reveal here isn't crossed with anything new, matching exp_3's own scope for those
  targets).

## Metric & Decision
Same metrics as `exp_3` (F1/ROC-AUC for decision, ordinal distance for confidence, mean ordinal
error/decisive-F1 for weights, set precision for reveal). Compare each condition directly against
its `exp_3` (with-MRI) counterpart, same model, same everything else. Record git commit — N/A,
not a git repository, same as every other experiment in this project.

## Next Step
Lean-tier — no separate implementation plan needed given how small the change is (one feature-
frame line, otherwise pure reuse of `exp_3`'s registry/harness). Implement directly.
