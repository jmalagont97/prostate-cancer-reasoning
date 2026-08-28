# Experiment Design: Task3 Point-Estimate Calibration Check
**Experiment**: experiments/exp_6/
**Project**: challenge_chimera_v3
**Date**: 2026-08-28
**Status**: Complete

---

## 1. Context and hypothesis

A production-simulation dry run (2 spot-checked cases, T3-032 and T3-047) found the promoted model
substantially underestimated urgency on fast recurrences. **Hypothesis, narrow form**: recurrence
speed itself predicts prediction error — faster true recurrences get worse (more optimistic)
predictions.

**Method**: pulled the honest LOOCV predictions already on disk (`agent/output/eval_results_task3.json`,
the exp_5-promoted model, `cox_p03_prostatectomy_pca5`) and checked all 19 event=1 (known-truth)
cases, not just the 2 spot-checked ones. Correlation between true recurrence time and residual
(predicted − true): Pearson r=−0.19 (p=0.45), Spearman ρ=+0.09 (p=0.73) — **narrow hypothesis
refuted**. Splitting at the median true time, "fast" and "slow" halves have near-identical mean
residuals (+40.9mo vs +39.3mo, Mann-Whitney p=1.00).

**What was actually there**: a large, near-universal positive bias across *all* 19 event=1 cases,
not specifically the fast ones — mean residual +40.1 months, MAE 42.2 months, an 11× overestimate
on the 12 cases that recurred within a year. Mechanism: `predict_expectation()` integrates the
entire fitted survival curve; with 75% of this cohort censored, that mean is pulled far right by a
long, poorly-supported tail beyond the last real observation — a known property of "mean" vs.
"median" under heavy censoring, not a bug specific to any case.

## 2. Follow-up test

Does `lifelines.CoxPHFitter.predict_percentile(p)` — the time by which `(1-p)` of the population
would have recurred — produce a better-calibrated point estimate than `predict_expectation()`, and
at what cost (if any) to `ranking_score`? `predict_median()` (p=0.5) was already known to fail
(undefined for 62/75 cases, tried and rejected in exp_4) — swept the wider range
p &isin; {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95} under full LOOCV, same real-`evaluate.py`-function
scoring used throughout.

## 3. Results

| p | C-index | n_capped | event=1 bias | event=1 MAE | ≤12mo bias |
|---|---|---|---|---|---|
| 0.50 (median) | 0.7389 | 61/75 | +153.3mo | 157.0mo | +117.4mo |
| 0.60 | 0.7752 | 55/75 | +122.8mo | 127.1mo | +70.3mo |
| 0.70 | 0.8243 | 42/75 | +73.2mo | 78.0mo | +17.7mo |
| 0.75 | 0.8367 | 39/75 | +37.8mo | 43.8mo | +8.3mo |
| 0.80 | 0.8642 | 31/75 | +2.7mo | 12.7mo | +3.4mo |
| 0.85 | 0.8566 | 19/75 | −2.3mo | 12.9mo | +1.7mo |
| **0.90** | **0.8765** | 7/75 | **−9.7mo** | **10.0mo** | **−1.8mo** |
| 0.95 | 0.8491 | 0/75 | −12.2mo | 12.2mo | −2.3mo |

Reference (`predict_expectation()`, the pre-exp_6 default): C-index=0.8575, bias=+40.1mo,
MAE=42.2mo, ≤12mo bias=+38.7mo.

**p=0.90 wins on both axes simultaneously**: better ranking *and* far better calibration, not a
tradeoff between them.

**Bootstrap significance** (2,000 resamples, paired) on the C-index gain: point estimate +0.0190,
95% CI **[−0.0019, +0.0456]** — borderline, just touches zero (95.7% of resamples favor p=0.90).
Not as clean a significance result as exp_5's embedding-fusion gain, but critically: never shown
worse, only possibly no different.

**A diagnostic paradox, resolved rather than ignored**: the (non-scored) `mean_time_score` metric
*drops* under p=0.90 (0.7723→0.5346) despite the large calibration improvement. Traced to
`recurrence_time_score`'s own formula: for event=1 cases the score is
`1 − |error|/scale` with `scale = true_time` — for this cohort's mostly-short true recurrence times,
even the reduced MAE (10mo) still exceeds `scale` for most cases, clamping the score to 0 either
way; the metric barely moves on the cases it's supposed to reward getting right. What *does* move it
is the 56 censored cases, where predicting anything ≥ the true follow-up time scores a free 1.0 —
`predict_expectation()`'s inflated predictions were incidentally exploiting that rule, not being
more honest. This diagnostic is not a reliable calibration signal on a heavily-censored cohort;
noted, not silently worked around.

T3.7 cross-check (IPCW time-dependent AUC) improved at every horizon under p=0.90 too (mean
0.877→0.904) — independent confirmation this isn't a C-index-specific artifact.

## 4. Promotion

`task3_config.DEFAULT_MODEL_KEY = "cox_p03_prostatectomy_pca5_p90"`. Promoted on the basis that
p=0.90 is never worse on `ranking_score` (the only metric that determines the actual competition
score) while being dramatically more truthful on the cases where ground truth is actually known —
even though the ranking gain itself is borderline rather than cleanly significant. Verified
end-to-end: `models_bridge_task3.py` → `run_task3_agent_loocv.py` → `local_eval_task3.py`
reproduces 0.8765 exactly.

## 5. Not done

- No test of intermediate p values between 0.85 and 0.95 to nail down whether 0.90 is the true local
  optimum or if the curve has more structure there (0.85 dipped slightly below 0.80, suggesting some
  LOOCV noise rather than a perfectly smooth curve).
- Percentile-based prediction was validated only for `cox_p03_prostatectomy_pca5` specifically, not
  extended to the other `MODEL_FACTORIES` entries.
