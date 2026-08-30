# exp_9 Design: Bootstrap-Testing the Event Classifier

## Background

exp_7 added `predict_event(horizon, threshold)`, deriving `event` (0/1 BCR status) from the already-
fitted survival curve. A horizon×threshold grid was swept under LOOCV and the winner (horizon=60mo,
threshold=0.7) scored balanced accuracy 0.779 against a trivial "always predict 0" baseline's
useless 0.500. That gain was explicitly flagged, everywhere it's documented, as **not
bootstrap-tested** — unlike every `ranking_score` gain elsewhere in this project (exp_4, exp_5,
exp_6), which all went through a paired bootstrap before being called significant or not. This
closes that gap.

## Hypothesis

1. **Primary**: the winner's balanced-accuracy gain over the trivial baseline (0.779 vs. 0.500) is
   distinguishable from resampling noise at n=75/19-events, the same statistical bar every other
   result in this project is held to.
2. **Secondary** (same discipline as exp_8's fine-grid check): is the winning cell a genuine
   standout in the horizon×threshold grid, or statistically indistinguishable from its closest
   competitors? From `results/summary.json`'s grid, the nearest cells by balanced accuracy are
   (h=24, thr=0.7)=0.727, (h=36, thr=0.7)=0.727, (h=60, thr=0.6)=0.727, (h=110, thr=0.6)=0.744.

## Method

- Reuse `event_horizon_sweep.py`'s exact LOOCV survival-at-horizon computation — same
  `cox_p03_prostatectomy_pca5` model, same protocol, no new model fit.
- Compute `S(h|x)` under LOOCV for h ∈ {24, 36, 60, 110}; derive predictions for the winner
  (h=60, thr=0.7) and the four named competitor cells.
- Paired bootstrap (2,000 resamples): for each resample, recompute balanced accuracy for both the
  winner and the comparison classifier on the *same* resampled case indices, take the difference.
  95% CI and fraction-of-resamples-favoring the winner, same convention as every prior bootstrap in
  this project (exp_5, exp_6, exp_8).

## What would change

If the primary comparison is not significant, that's a real finding to report honestly — it would
mean the event classifier's apparent skill could be noise at this sample size, worth flagging more
prominently than a passive "not yet tested" bullet. If a competitor cell is statistically
indistinguishable from the winner, note it as a caveat (same as exp_8's p=0.92 finding) but the
promoted (horizon, threshold) doesn't change based on point estimates alone without a competitor
actually beating it.
