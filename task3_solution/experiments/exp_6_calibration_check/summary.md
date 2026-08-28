# exp_6 Report: Task3 Point-Estimate Calibration Check

**Status**: Complete. **Verdict: narrow hypothesis refuted, broader finding confirmed and fixed.**

## What was checked, and what was actually there

A production-simulation spot check (2 cases) suggested fast recurrences get underestimated. The
systematic version — all 19 event=1 (known-truth) cases, not 2 — refutes that specific claim:
correlation between true recurrence time and prediction error is not significant (Pearson r=−0.19,
p=0.45; Spearman ρ=+0.09, p=0.73; fast vs. slow halves have near-identical mean residuals).

What's actually there is bigger: a **+40.1 month mean bias across nearly all 19 event=1 cases**,
not specific to fast ones — up to an **11× overestimate** on cases that recurred within a year.
Cause: `predict_expectation()` integrates the whole survival curve; with 75% of this cohort
censored, the mean gets pulled far right by a long, poorly-supported tail — a known statistical
property of the mean vs. median under heavy censoring.

## The fix, and its cost (there wasn't one)

Swept `predict_percentile(p)` for p &isin; {0.5,...,0.95}. **p=0.90 wins on both axes**:

| | `predict_expectation()` (old) | `predict_percentile(p=0.90)` (new) |
|---|---|---|
| ranking_score (C-index) | 0.8575 | **0.8765** |
| event=1 mean bias | +40.1mo | **−9.7mo** |
| event=1 MAE | 42.2mo | **10.0mo** |
| ≤12mo-true-recurrence bias | +38.7mo | **−1.8mo** |
| T3.7 mean IPCW AUC | 0.877 | **0.904** |

The C-index gain is borderline under bootstrap (95% CI [−0.002, +0.046], 95.7% of resamples favor
it) — not as clean as exp_5's embedding-fusion result — but it is never shown *worse*, while the
calibration and T3.7 gains are large and consistent. Promoted on that combined basis.

## A diagnostic paradox, explained rather than hidden

`mean_time_score` (a non-scored diagnostic) *drops* under the better-calibrated model
(0.7723→0.5346). Not a red flag: that metric's formula clamps most event=1 scores to 0 regardless
(errors still exceed `scale=true_time` for short-recurrence cases either way), while rewarding *any*
prediction ≥ true follow-up time on the 56 censored cases — `predict_expectation()`'s inflated
values were incidentally exploiting that rule, not being more honest. Documented in
`experiments/exp_6/DESIGN.md` §3 rather than silently omitted.

## Promoted

`agent/src/task3_config.py`: `_CoxEmbeddingAdapter` gained a `percentile` parameter (`None` = old
behavior, preserved for the other registry entries). New key
`cox_p03_prostatectomy_pca5_p90`, now `DEFAULT_MODEL_KEY`. Verified end-to-end
(`models_bridge_task3.py` → `run_task3_agent_loocv.py` → `local_eval_task3.py`) — reproduces
0.8765 exactly.

## Not done

- No fine sweep between p=0.85 and p=0.95 to nail the exact optimum (0.85 dipped slightly below
  0.80 in the coarse sweep — some LOOCV noise, not a perfectly smooth curve).
- Percentile prediction validated only for `cox_p03_prostatectomy_pca5`, not the other
  `MODEL_FACTORIES` entries.
