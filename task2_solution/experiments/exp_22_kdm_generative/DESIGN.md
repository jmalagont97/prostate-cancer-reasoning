# exp_22 Design: Generative Co-Training for KDMClassifier

## Background

exp_20 found `KDMClassifier` lost decisively to `knn` on `decision` (0.7429→0.6944), plausibly
because the encoder + 71 free KDM components overfit on ~70 training rows per fold — no
regularization was applied. `KDMClassModel`'s own docstring documents a "generative co-training"
recipe for exactly this kind of regime: freeze the encoder partway through training and add
`-generative * kdm.log_marginal(pure2dm(encoder(x))).mean()` to the loss, so the KDM layer's own
components get an extra density-estimation signal (make the components a good generative model of
the encoded inputs, not just a good discriminator) rather than being free to overfit purely to the
NLL classification loss.

## Method

`agent/src/kdm_models.py`'s `KDMClassifier` extended with `generative` (co-training weight,
default 0.0 = exp_20's exact prior behaviour, unchanged) and `freeze_epoch_frac` (default 0.5 — the
back half of training runs with the encoder frozen and the extra generative loss term added).
Smoke-tested before running the real experiment. Swept `generative` ∈ {0.5, 1.0, 2.0} on
`decision`'s promoted 12-feature set, compared against exp_20's own baseline numbers (`knn`:
0.7429, `KDMClassifier` generative=0.0: 0.6944 — not re-run, reused directly).

## What would change the promoted config

Only if some `generative` weight beats plain `knn` (0.7429) on `ranking_score`, bootstrap-confirmed
before promotion — same bar as every other experiment. Beating exp_20's non-generative KDM
(0.6944) alone would be progress but not sufficient for promotion on its own.
