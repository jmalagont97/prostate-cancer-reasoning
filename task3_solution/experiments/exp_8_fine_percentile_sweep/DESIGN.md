# exp_8 Design: Fine Percentile Sweep (p ∈ [0.85, 0.95])

## Background

exp_6 swept `predict_percentile(p)` over a coarse grid {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}
and promoted p=0.90 (LOOCV C-index 0.8765, event=1 bias −9.7mo). The three points nearest the peak:

| p | C-index |
|---|---|
| 0.85 | 0.8566 |
| 0.90 | 0.8765 |
| 0.95 | 0.8491 |

This is a dip-then-peak-then-dip shape on a grid spaced 0.05 apart — plausible as a real local
optimum, but also exactly the shape LOOCV noise on n=75 could produce around a flat region. Flagged
as an explicit, named limitation in `experiments/exp_6/reports/summary.md`,
`task3_solution/MODEL_CARD.md` §7, and `task3_solution/results/summary.json`. Never re-examined
until now.

## Hypothesis

A finer grid (step 0.01 instead of 0.05) across the same [0.85, 0.95] window will show whether
p=0.90 is a genuine local maximum (neighbors on both sides score lower, and the point estimate for
0.90 is not just the luckiest resample) or whether the coarse grid's peak was an artifact of
under-sampling a fairly flat region.

## Method

- Reuse exp_6's exact adapter, model, and LOOCV loop (`task3_percentile_sweep.py`'s
  `run_loocv_for_p`) — same `cox_p03_prostatectomy_pca5` model, same protocol, nothing re-derived.
- Sweep p = 0.85, 0.86, …, 0.95 (11 points) under full LOOCV (75 refits per point).
- For each point, record C-index, event=1 bias, event=1 MAE, and count of numerically-capped
  predictions (same convention as exp_6).
- Identify the best point by C-index, then run a **paired bootstrap (2,000 resamples)** of the best
  point against every other grid point — not just against the previous champion (as exp_6 did against
  `predict_expectation()`), to check whether *any* neighboring cell is statistically indistinguishable
  from the winner, which would mean the exact value within that neighborhood is noise-level detail
  rather than a real difference.

## What would change the promoted config

Only if a different p scores measurably and significantly better than 0.90 under this finer grid —
same promotion-gated discipline as every other experiment in this project. If p=0.90 (or a
statistically indistinguishable neighbor) remains the best point, the promoted
`cox_p03_prostatectomy_pca5_p90` config is unchanged; the outcome is downgrading an open limitation
to a checked one, not necessarily a new promotion.
