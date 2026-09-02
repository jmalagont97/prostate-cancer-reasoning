# exp_13 Design: Ordinal-Aware Models for confidence / variable_weights

## Background

exp_12's per-sub-task classification reports surfaced a real gap: `confidence`'s kNN classifier
never once predicts "borderline" (0/14 recall), defaulting to the majority class "clear" — yet it
scores 0.914 on `confidence_score`. Tracing why, directly from `evaluate.py`: `confidence_score`
and `variable_weight_score` are both L1 ordinal-distance formulas (`1 - |true-pred|/max_dist`), not
classification accuracy — confirmed from source, not assumed. None of exp_12's 8-model roster
(logistic, elastic_net, svm_rbf, knn, random_forest, extra_trees, naive_bayes, mlp) is ordinal-aware
— all are nominal classifiers under the hood, optimizing 0/1 accuracy regardless of how the
evaluator actually scores their output. `decision` is excluded from this experiment: both its gate
(`decision_score`) and its leaderboard contribution (`decision_weighted_f1`) are confirmed nominal
(exact-match), so an ordinal treatment doesn't apply there even though the 4 categories have a
rough real-world intensity ordering.

## Hypothesis

A model that explicitly targets the ordinal structure — regression on the integer code, rounded and
clipped to the valid range at prediction time, rather than classification — should score higher on
`confidence_score`/`variable_weight_score` (and plausibly `ranking_score`) than any of exp_12's
nominal classifiers, since it directly optimizes something aligned with the L1-distance scoring
formula instead of 0/1 accuracy. `mord` (the standard ordinal-classifier library) is unavailable in
this environment; regression-then-round is the dependency-free equivalent.

## Method

Three regressor families, wrapped in a `RoundedRegressor` (fits a regressor, rounds+clips
predictions to `[0, max_code]` at inference — a drop-in `model_factory`, no harness changes needed):
`KNeighborsRegressor` (the direct ordinal analog of the current `knn` classifier), `Ridge`, and
`RandomForestRegressor`. Swept independently for `confidence` (max_code=2) and `variable_weights`
(max_code=3) via `task2_variant_tester.py`, holding everything else at the exp_11/12-promoted
default. Classification reports pulled for the best candidate on each sub-task to check whether the
specific weak spot (borderline-blindness, decisive-recall) actually improved, not just the aggregate
score.

## What would change the promoted config

Only if an ordinal variant beats the current default on `ranking_score` by a bootstrap-significant
margin, same bar as every other experiment. Equally important here: even a small ranking_score gain
should be checked against the per-class classification report before promoting, since exp_12 already
showed the aggregate score can hide a rare-class regression.
