# exp_14 Design: Genuine Ordinal Classifiers (`mord`) for confidence / variable_weights

## Background

exp_13 tested a dependency-free ordinal *proxy* (regression on the integer code, rounded and
clipped) and found it didn't help — worse, it regressed rare-class recall (`confidence`'s
"borderline", `variable_weights`' "decisive") by pulling predictions toward the mean/majority class,
a mechanism problem specific to squared-error regression, not necessarily true of ordinal
classification in general. `mord` (the standard ordinal-classifier library, proportional-odds /
cumulative-link models) was unavailable when exp_13 ran; confirmed installable this session
(`pip install mord`, v0.7).

## Hypothesis

A genuine discriminative ordinal classifier — one that fits threshold/cutpoint boundaries directly
(`P(y > k)` for each ordinal threshold `k`) rather than regressing toward a continuous mean — should
not share exp_13's specific failure mode (majority-class collapse), since it isn't pulled toward the
center of the distribution the way OLS/kNN regression is. Whether it beats nominal classification on
`ranking_score` is the open question; whether it avoids exp_13's rare-class regression is the more
important one given what exp_13 found.

## Method

Three of `mord`'s ordinal classifiers, sklearn-compatible out of the box (no wrapper needed, unlike
exp_13's `RoundedRegressor`): `LogisticIT` (immediate-threshold), `LogisticAT` (all-threshold),
`LogisticSE` (squared-error link). Swept independently for `confidence` and `variable_weights` via
`task2_variant_tester.py`, same protocol as exp_12/exp_13 — everything else held at the promoted
default. Classification report pulled for the best candidate on each sub-task, specifically checking
`borderline`/`decisive` recall against exp_13's regression proxy, not just the aggregate score.

## What would change the promoted config

Only a variant that beats the current `knn` default on `ranking_score` by a bootstrap-significant
margin AND doesn't trade away rare-class recall the way exp_13's proxy did — both bars, not just one,
given what the last experiment already taught about the aggregate score hiding per-class harm.
