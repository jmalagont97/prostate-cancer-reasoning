# exp_13 Report: Ordinal-Aware Models for confidence / variable_weights

**Status**: Complete. **Verdict: ❌ Refuted — a well-motivated hypothesis (confirmed straight from
the evaluator's own formula) that didn't survive contact with the data. Not promoted, and the
per-class report shows *why* not to trust the aggregate number alone.**

## What was tested

`confidence_score`/`variable_weight_score` are confirmed-from-source L1 ordinal-distance metrics
(`1 - |true-pred|/max_dist`), yet every model in exp_12's roster treats those targets as nominal
classification. Tested 3 ordinal-regression variants (kNN, Ridge, RandomForest regressors, rounded
+ clipped to the valid integer range) against the current `knn`-classifier default.

## Results — aggregate score

| Sub-task | Variant | ranking_score | component score |
|---|---|---|---|
| — | **baseline (knn, nominal)** | **0.7166** | confidence=0.914 |
| confidence | knn_ordinal | 0.7166 | confidence=0.914 (identical) |
| confidence | ridge_ordinal | 0.7166 | confidence=0.914 (identical) |
| confidence | rf_ordinal | 0.7174 | confidence=0.922 |
| variable_weights | knn_ordinal | **0.7185** | var_weight=0.679, factor_f1=0.448 |
| variable_weights | ridge_ordinal | 0.7173 | var_weight=0.674, factor_f1=0.439 |
| variable_weights | rf_ordinal | 0.7180 | var_weight=0.675, factor_f1=0.448 |

Every gain is tiny — the best (`variable_weights=knn_ordinal`, +0.0019) is an order of magnitude
smaller than exp_12's `random_forest` swap (+0.0149), which itself was **not** bootstrap-significant.
Not formally bootstrapped here — a gain this small, next to a 8x-larger one that already failed to
clear the bar at this sample size (n=72), isn't worth the compute to confirm it's noise.

## Why the classification report says more than the aggregate score

**confidence = rf_ordinal** — looked like the best confidence variant on paper (0.914→0.922). The
classification report says otherwise:

```
              precision    recall  f1-score   support
  borderline       0.00      0.00      0.00        14
       clear       0.81      1.00      0.89        58
```

It now predicts **"clear" for literally 100% of cases** — *more* collapsed than the nominal kNN
classifier, which at least got 1/58 "clear" cases wrong by predicting borderline. With only 72
training points and 58/72 already "clear," a regressor's mean prediction rarely crosses the 1.5
rounding threshold — regression-toward-the-mean, not ordinal awareness, is what nudged the score up.

**variable_weights = knn_ordinal** — the aggregate winner, but the per-class picture is mixed-to-worse
on exactly the class we most wanted to fix:

```
              precision    recall  f1-score   support
    not_used       0.68      0.35      0.46       163   (recall 0.47 -> 0.35, worse)
   important       0.53      0.61      0.56       248   (recall 0.54 -> 0.61, better)
    decisive       0.58      0.10      0.17        71   (recall 0.18 -> 0.10, WORSE)
```

`decisive` recall — the rarest, most clinically consequential label — **dropped from 18% to 10%**.
Precision improved (0.32→0.58: fewer false "decisive" calls) but at the cost of catching barely any
real ones. The tiny aggregate gain comes from `noted`/`important` (the two middle, most common
classes) getting nudged closer under the L1 metric, not from the model getting genuinely better at
the ordinal structure.

## Verdict and action

**Not promoted.** Neither ordinal variant is worth the (tiny, unconfirmed) aggregate gain given it
comes paired with a real regression on rare-class recall — the opposite of what the hypothesis
predicted. `knn` (nominal classification) stays the default for `confidence` and `variable_weights`.

This is a useful negative result nonetheless: it shows the evaluator's ordinal-distance formula is
*forgiving enough* toward majority-class predictions that "regress toward the mean" and "be
genuinely ordinal-aware" produce similar aggregate scores at n=72 — the metric's tolerance for
near-misses, not model sophistication, is doing most of the work in both cases. A real fix for
`confidence`'s borderline-blindness or `variable_weights`' decisive-recall problem likely needs
something that directly targets recall on the rare class (e.g. class-weighting, oversampling, a
cost-sensitive loss) rather than a different way of encoding the ordinal structure.

## Not done

- No `mord`-based true ordinal classifier (proportional-odds / cumulative-link model) — the library
  isn't installed in this environment; only the dependency-free regression-then-round proxy was
  tested.
- No class-weighting or oversampling variant for the rare classes (`borderline`, `decisive`) — a
  more direct fix for the specific failure mode surfaced here, left as a follow-up.
- No formal bootstrap on the small gains found — reasoned, not measured, to be smaller than
  exp_12's already-non-significant +0.0149.
