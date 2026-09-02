# exp_15 Report: Is confidence's `borderline`-Recall Fix Worth Its ranking_score Cost?

**Status**: Complete. **Verdict: ❌ No — the cost is real, confirmed by bootstrap, not noise. Not
promoted. `knn` stays the default for `confidence`; the recall gap is now a named, examined
trade-off rather than an unexamined blind spot.**

## What was checked

`extra_trees` (`class_weight="balanced"`) catches 8/14 `borderline` cases (57% recall, vs. `knn`'s
0%) and has the best macro-F1 (0.35) of anything tried on `confidence` across exp_11–14 — at an
apparent cost of ~0.009 on `ranking_score`. Bootstrapped (2,000 resamples, `decision=knn` and
`variable_weights=mord.LogisticIT` held at their promoted defaults) to check whether that cost is
real or resampling noise before deciding.

## Result

```
Point estimates: knn=0.7186  extra_trees=0.7100  diff=-0.0086
Bootstrap (2000 resamples): mean diff=-0.0087, 95% CI=[-0.0156, -0.0016], frac favoring extra_trees=0.4%
-> SIGNIFICANT LOSS
```

The 95% CI sits entirely below zero — the opposite pattern from exp_14's `mord.LogisticIT` promotion
(CI barely touching zero, 93.7% favoring). Here only 0.4% of resamples favor `extra_trees`. This is
a confirmed, real cost, not a coin-flip or a borderline case worth a second diagnostic's tie-break.

## Verdict and action

**Not promoted.** `task2_subtask_config.py`'s `confidence` stays on `knn`. Per this project's
standing rule, only a measured improvement to the real metric gets promoted — a confirmed loss,
however clinically motivated the trade might feel, doesn't qualify on its own.

This is not the same situation exp_6/exp_14 faced (a borderline composite-score result rescued by a
second, independently-verified diagnostic moving decisively) — there, the primary metric's CI merely
touched zero and a second diagnostic tipped the balance. Here the primary metric's CI is
unambiguously negative; a second diagnostic (recall) looking good doesn't override a confirmed loss
on the metric the project is actually promotion-gated on.

## What this leaves on the table, named not hidden

`confidence`'s `borderline`-recall problem (0/14 under every promoted-quality model tried so far)
remains open. It is a deliberate, examined trade-off now, not an unexamined gap: the fix exists
(`extra_trees`, `class_weight="balanced"`), its cost is quantified and bootstrap-confirmed, and the
project's standing discipline says not to pay it. If a future context values `borderline`-detection
recall enough to accept a confirmed ~0.009 `ranking_score` cost (e.g. a clinical-safety argument that
this project's discipline doesn't currently weigh), that would be an explicit, out-of-band decision
to override the promotion rule — not something to do quietly inside this experiment.

## Not done

- No further attempt to close the gap without cost (e.g. milder class-weighting, oversampling,
  cost-sensitive thresholds) — exp_15 only tested the one strong candidate exp_12's re-inspection
  surfaced.
- No test of whether a *milder* balance (e.g. `class_weight={0:1, 1:2, 2:1}` instead of full
  `"balanced"`) finds a smaller-cost middle ground — a reasonable follow-up if this is revisited.
