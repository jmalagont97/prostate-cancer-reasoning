# exp_22 Report: Generative Co-Training for KDMClassifier

**Status**: Complete. **Verdict: ❌ Catastrophic collapse, not a graceful regularization effect —
a real, decisive failure of this specific implementation, not just a smaller loss than exp_20's
plain `KDMClassifier`. No change to the promoted config.**

## What was tested

`KDMClassModel`'s own documented "generative co-training" recipe: freeze the encoder partway
through training (`encoder.requires_grad_(False)`) and add
`-generative * kdm.log_marginal(pure2dm(encoder(x))).mean()` to the loss for the remainder, so the
KDM layer's own components get an extra density-estimation signal on top of the discriminative NLL
loss. Implemented in `agent/src/kdm_models.py`'s `KDMClassifier` (`generative`/`freeze_epoch_frac`
parameters, default `generative=0.0` reproducing exp_20's exact prior behaviour). Motivated by
exp_20's likely diagnosis (overfitting on ~70 training rows) — generative co-training is
specifically a regularizer for that regime, not an arbitrary knob.

## Result — identical across every weight tried, which is itself the finding

```
knn baseline:                          ranking_score=0.7429
KDMClassifier(generative=0.0, exp_20): ranking_score=0.6944
KDMClassifier(generative=0.5):         ranking_score=0.2907  (−0.4522)
KDMClassifier(generative=1.0):         ranking_score=0.2907  (−0.4522)
KDMClassifier(generative=2.0):         ranking_score=0.2907  (−0.4522)
```

All three weights land on **exactly** the same score to 4 decimal places — not a smooth
degradation with weight, a collapse to the same fixed point regardless of how strongly the
generative term is applied. Confirmed directly: the predicted-label distribution across all 72
LOOCV folds is **`active_treatment` — 72/72 (100%)**. The model isn't discriminating at all; it's
predicting the majority class for every single case. (`decision_f1=0.2592` matches the weighted-F1
a constant-`active_treatment` predictor would score exactly, confirming this arithmetically, not
just by inspection.)

## Why, as far as this was diagnosed

Once the encoder freezes and the generative loss term is added, the KDM layer's own trainable
components (`c_x`, `c_y`, `c_w`) are the only thing left to optimize, against a combined
loss with no balancing between the NLL term's scale and `log_marginal`'s scale. The most likely
mechanism: the generative term dominates the combined gradient once the encoder can no longer move,
pulling all `n_comp=71` components toward maximizing density fit at the expense of the class
boundary encoded in `c_y` — collapsing the discriminative structure entirely rather than
regularizing it. This is a property of *this specific implementation* (no loss-scale balancing, an
untuned `freeze_epoch_frac=0.5`, an untuned `generative` range) — not evidence the documented
recipe can't work with a more careful implementation, just that this first attempt at it broke
outright rather than helping.

## Verdict and action

**Not promoted, and no further tuning attempted on this recipe in this pass.** `decision` stays
plain `knn`. Given exp_20's non-generative `KDMClassifier` already lost decisively, and this
attempt at the documented fix made things dramatically worse rather than better, KDM is now 0-for-3
on `decision` across exp_20/21/22 — a consistent, reinforcing signal that the package (at least in
the configurations tried so far) isn't a good fit for this specific sub-task at this sample size,
rather than three unrelated near-misses.

## Not done

- No loss-scale balancing between the NLL and generative terms (e.g. normalizing `log_marginal` by
  its own running scale before weighting) — the standard fix for exactly this kind of dominance
  problem, not attempted here.
- No sweep of `freeze_epoch_frac` (e.g. freezing later, after the encoder has more fully converged
  on the discriminative task first) — only the default 0.5 was tried.
- No test of much smaller `generative` weights (e.g. 0.01, 0.05) — the three values tried (0.5, 1.0,
  2.0) may all already be well past the point of collapse; a finer sweep near zero wasn't run.
