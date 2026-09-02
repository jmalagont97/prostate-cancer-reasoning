# exp_15 Design: Is confidence's `borderline`-Recall Fix Worth Its ranking_score Cost?

## Background

`confidence`'s classification report (exp_11/12) exposed the composite `ranking_score`/
`confidence_score` hiding a real failure: `knn` predicts "clear" for 71/72 cases, 0/14 `borderline`
recall, macro-F1 0.29 — a majority-class default dressed up as a good score by the ordinal-distance
formula's tolerance for near-misses. Every ordinal-aware alternative tried since (exp_13's
regression proxy, exp_14's `mord` classifiers) made this the same or worse, never better.

A quick check of the `class_weight="balanced"` models already in exp_12's roster (never inspected
per-class before, only by aggregate `ranking_score`) found real signal: `extra_trees` catches 8/14
`borderline` cases (57% recall, best of anything tried) and has the best macro-F1 (0.35) of any
`confidence` variant — but at a cost of ~0.009 on `ranking_score` (0.7166→0.7080).

## Hypothesis

Is that ~0.009 point-estimate cost statistically real, or is `extra_trees` actually indistinguishable
from `knn` on the composite metric once resampling noise is accounted for — in which case promoting
it would fix a real clinical blind spot at no measurable cost?

## Method

Paired bootstrap (2,000 resamples) of `confidence=extra_trees` vs. `confidence=knn`, holding
`decision=knn` and `variable_weights=mord.LogisticIT` at their currently-promoted defaults (matches
exp_11/12/14's isolation discipline — one sub-task varies, the rest held fixed).

## What would change the promoted config

- If the CI shows the cost is **not significant** (crosses zero): promote `extra_trees` for
  `confidence` — a real recall fix at no measurable cost to the metric that's actually promotion-gated.
- If the CI shows a **significant loss**: report honestly, leave `knn` as the default, and document
  the trade-off as a named, deliberate choice (composite score over recall) rather than an
  unexamined gap.
