# exp_9 Report: Bootstrap-Testing the Event Classifier

**Status**: Complete. **Verdict: mixed, and honestly so — the classifier clearly beats the trivial
baseline, but the specific (horizon, threshold) choice does not clearly beat its nearest neighbors.**

## What was checked

exp_7's `predict_event(horizon=60, threshold=0.7)` scored balanced accuracy 0.779 against a trivial
"always predict 0" baseline's 0.500 — a real gap on paper, but explicitly flagged everywhere as
never bootstrap-tested, unlike every `ranking_score` gain elsewhere in this project. Two questions,
both via paired bootstrap (2,000 resamples), reusing the exact LOOCV `S(h|x)` computation from the
original sweep — no new model fit:

1. **Primary**: is the winner's gain over the trivial baseline distinguishable from noise?
2. **Secondary** (same discipline as exp_8): is the winning cell a genuine standout among its
   closest competitors in the original horizon×threshold grid, or statistically indistinguishable?

## Primary result — winner vs. trivial baseline

| | Balanced accuracy |
|---|---|
| Winner (h=60, thr=0.7) | 0.779 |
| Trivial "always 0" | 0.500 |

Paired bootstrap: mean diff **+0.278**, 95% CI **[+0.160, +0.383]**, 100.0% of resamples favor the
winner. **Significant, decisively** — this is not a fragile result. The event classifier's overall
skill over a useless baseline is real, not an artifact of one lucky LOOCV split.

## Secondary result — winner vs. closest competing cells

| Competitor (h, thr) | Balanced accuracy | vs. winner: mean diff | 95% CI | Frac favoring winner | Verdict |
|---|---|---|---|---|---|
| (24, 0.7) | 0.727 | +0.053 | [−0.040, +0.166] | 83.9% | not significant |
| (36, 0.7) | 0.727 | +0.052 | [−0.046, +0.163] | 85.4% | not significant |
| (60, 0.6) | 0.727 | +0.050 | [−0.053, +0.158] | 82.5% | not significant |
| (110, 0.6) | 0.744 | +0.035 | [−0.028, +0.117] | 81.8% | not significant |

None of the four nearest competitors are statistically distinguishable from the winner — every CI
crosses zero. Read honestly: **the choice of exactly (horizon=60, threshold=0.7) over these four
close alternatives is not itself statistically justified** at n=75/19-events; it's the best point
estimate on the original grid, but several nearby cells are consistent with being just as good. This
echoes exp_4's finding for the Cox penalizer sweep (a real point-estimate winner, not a
statistically distinguishable one) — the same honest pattern shows up again here, at a different
sample size regime (n=75, only 19 positive events) where fine-grained hyperparameter distinctions
are genuinely hard to resolve.

## Verdict and action

**No change to the promoted config.** `EVENT_HORIZON_MONTHS=60`, `EVENT_THRESHOLD=0.7` stand — there
is no competitor that beats the winner, only several that aren't statistically separated from it,
and 60 months carries an independent clinical justification (the standard 5-year BCR benchmark) the
alternatives don't.

This replaces the passive "not bootstrap-tested" limitation with two precise, checked claims:
1. The classifier's overall skill over baseline is bootstrap-confirmed, decisively.
2. The specific (horizon, threshold) choice is a best point estimate, not a statistically proven
   optimum — an honest downgrade from an implied-but-untested claim to a stated, sourced one.

## Not done

- Did not re-sweep the full 5×5 horizon×threshold grid under bootstrap — only the winner against
  its four nearest point-estimate competitors, mirroring exp_8's scope (nearest neighbors, not
  exhaustive pairwise testing).
- Did not test alternative model choices (e.g., a directly-fit binary classifier instead of
  thresholding the survival curve) — out of scope for this experiment, which only tests the
  significance of the already-shipped derivation.
