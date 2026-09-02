# exp_21 Design: KDMRegressor for `decision`

## Background

Requested directly: test `KDMRegressor` (built and smoke-tested in exp_20, never run against a
real target) on `decision`. Flagged honestly before running: `decision_score`/
`decision_weighted_f1` are both confirmed nominal from `evaluate.py` (exact match, no partial
credit for a clinically-close wrong category) — so treating `decision` as an ordinal regression
target is a real mismatch with how the sub-task is actually scored, in the same family as exp_13's
regress-and-round attempt (which failed for a different, genuinely-ordinal target). Tested anyway,
per the explicit request — empirical, not assumed.

## Method

The 4 treatment categories encoded on a clinically-motivated least-to-most-interventional integer
scale (0=`watchful_waiting`, 1=`active_surveillance`, 2=`continued_surveillance`,
3=`active_treatment`) — not derived from the evaluator, a modeling choice made explicit rather than
hidden. `KDMRegressorForDecision` wraps `KDMRegressor` (exp_20's wrapper) to fit/predict on this
code, rounding and clipping predictions to `[0, 3]` at inference (same `RoundedRegressor` pattern
exp_13 used). Tested against the current promoted `knn` default on `decision`'s already-promoted
12-feature set (10 base + `psa_kinetics`, exp_16).

## What would change the promoted config

Only if `KDMRegressorForDecision` beats the current `knn` default on `ranking_score`, bootstrap-
confirmed before promotion.
