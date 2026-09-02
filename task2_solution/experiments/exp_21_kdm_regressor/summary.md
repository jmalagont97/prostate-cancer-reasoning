# exp_21 Report: KDMRegressor for `decision`

**Status**: Complete. **Verdict: ❌ Decisively refuted — the honesty-flag raised before running was
correct. Worse than exp_20's already-refuted `KDMClassifier`. No change to the promoted config.**

## What was tested

`KDMRegressor` (built in exp_20, smoke-tested but never run against a real target) applied to
`decision` by encoding the 4 treatment categories on a clinically-motivated, least-to-most-
interventional integer scale (0=`watchful_waiting`, 1=`active_surveillance`,
2=`continued_surveillance`, 3=`active_treatment` — a modeling choice made explicit, not derived
from the evaluator), predicting a continuous value and rounding/clipping to `[0,3]` at inference
(same pattern as exp_13's `RoundedRegressor`). Flagged honestly before running: `decision_score`/
`decision_weighted_f1` are both confirmed nominal (exact match, no partial credit for a
clinically-close wrong category) — so this is a genuine mismatch with how the sub-task is scored.
Tested anyway, per the explicit request, rather than assumed to fail.

## Result

```
baseline (decision=knn): ranking_score=0.7429
decision=KDMRegressor:   ranking_score=0.5865  (−0.1564)  decision_f1=0.6498
```

A massive drop — over 3x the size of exp_20's already-decisive `KDMClassifier` loss (−0.0485). No
bootstrap needed.

## Classification report — shows exactly why it broke

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.76      0.64      0.70        25
continued_surveillance       0.50      0.93      0.65        14
      active_treatment       0.75      0.58      0.65        31
              accuracy                           0.65        72
```

| True \ Pred | ww | as | cs | at |
|---|---|---|---|---|
| ww (2) | 0 | 0 | 1 | 1 |
| as (25) | 1 | 16 | 3 | 5 |
| cs (14) | 0 | 1 | 13 | 0 |
| at (31) | 0 | 4 | 9 | 18 |

`continued_surveillance` (code=2, the *middle* of the imposed 0–3 scale) gets an unusually high
recall (0.93 — 13/14 caught) but its precision collapses to 0.50: the regression is pulling
ambiguous predictions toward the **middle of the arbitrary numeric encoding**, not toward the
clinically correct answer. This is the same "regress toward the center" failure mode exp_13 already
found for `confidence`/`variable_weights` — here it lands on whichever class sits in the middle of
the ordering imposed on it, rather than the majority class, because `decision` genuinely has no
majority-adjacent structure the way a 3-level confidence scale does.

## Verdict and action

**Not promoted.** `decision` stays plain `knn`. This closes the KDM-on-`decision` thread started in
exp_20/exp_21 with a clean, consistent story: `decision` is nominal, both the classifier and
regressor variants of KDM underperformed `knn`, and the regressor's failure mode is directly
traceable to forcing an unordered target onto a numeric scale — confirming, not just assuming, that
`decision`'s nominal framing (already used by every promoted model in this project) is the right
one.

## Not done

- No test of a different integer ordering (e.g. grouping `watchful_waiting` and
  `continued_surveillance` together as "passive management") — the clinical case for the imposed
  ordering was reasonable but not the only possible one; a different encoding might shift which
  class absorbs the "regress to the middle" effect without fixing the underlying mismatch.
- `KDMRegressor` still untested against a genuine regression target (task3's
  `months_to_recurrence`) — the natural next place to try it, flagged again from exp_20.
