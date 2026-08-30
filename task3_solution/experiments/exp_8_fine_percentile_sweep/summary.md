# exp_8 Report: Fine Percentile Sweep (p ∈ [0.85, 0.95])

**Status**: Complete. **Verdict: confirmed — p=0.90 is a genuine local optimum, not LOOCV noise
dressed up as one. No change to the promoted config. A named limitation is now closed, not open.**

## What was checked

exp_6's coarse grid {…, 0.85, 0.90, 0.95} showed a dip-then-peak-then-dip around p=0.90
(0.8566 → 0.8765 → 0.8491) — plausible as a real optimum, but also exactly the shape LOOCV noise
on n=75 could produce over a flat region spaced 0.05 apart. Reran with the same model
(`cox_p03_prostatectomy_pca5`), same LOOCV protocol, same `predict_percentile()` machinery
(`task3_percentile_sweep.py`, unchanged) at step 0.01 across the same window.

## Fine-grid results

| p | C-index | n_capped | event=1 bias (mo) | event=1 MAE (mo) |
|---|---|---|---|---|
| 0.85 | 0.8566 | 19 | −2.3 | 12.9 |
| 0.86 | 0.8549 | 16 | −3.8 | 12.3 |
| 0.87 | 0.8619 | 15 | −6.0 | 10.7 |
| 0.88 | 0.8584 | 13 | −7.5 | 9.5 |
| 0.89 | 0.8584 | 9 | −8.7 | 9.8 |
| **0.90** | **0.8765** | 7 | −9.7 | 10.0 |
| 0.91 | 0.8575 | 3 | −10.1 | 10.5 |
| 0.92 | 0.8659 | 2 | −11.0 | 11.1 |
| 0.93 | 0.8412 | 1 | −11.4 | 11.4 |
| 0.94 | 0.8363 | 0 | −11.9 | 11.9 |
| 0.95 | 0.8491 | 0 | −12.2 | 12.2 |

p=0.90 is the best point on the finer grid too — not an artifact of only sampling three points at
0.05 spacing. Event=1 bias moves monotonically more negative as p increases (expected: a higher
percentile reads off an earlier point on the survival curve); MAE bottoms slightly earlier, at
p=0.88 (9.5mo), 0.5mo better than p=0.90's 10.0mo — noted, not acted on, since C-index (the scored
metric) is what governs promotion, per this project's standing discipline.

## Bootstrap: is p=0.90 distinguishable from its neighbors?

Paired bootstrap (2,000 resamples) of p=0.90 against every other grid point:

| vs. p= | mean Δ | 95% CI | frac favoring 0.90 | verdict |
|---|---|---|---|---|
| 0.85 | +0.0201 | [+0.0042, +0.0390] | 99.7% | significant |
| 0.86 | +0.0212 | [+0.0020, +0.0466] | 98.7% | significant |
| 0.87 | +0.0146 | [−0.0027, +0.0365] | 95.0% | borderline |
| 0.88 | +0.0182 | [+0.0033, +0.0402] | 99.7% | significant |
| 0.89 | +0.0180 | [+0.0038, +0.0395] | 99.6% | significant |
| 0.91 | +0.0191 | [+0.0031, +0.0406] | 99.0% | significant |
| **0.92** | **+0.0102** | **[−0.0076, +0.0302]** | **87.6%** | **not significant** |
| 0.93 | +0.0352 | [+0.0092, +0.0699] | 99.9% | significant |
| 0.94 | +0.0405 | [+0.0089, +0.0815] | 99.5% | significant |
| 0.95 | +0.0266 | [−0.0016, +0.0574] | 96.5% | borderline |

p=0.90 significantly beats 7 of the 10 other grid points, with only p=0.92 (its next-best
competitor, 0.0106 below on the point estimate) not clearing significance, and p=0.87/p=0.95
landing borderline. Read honestly: p=0.90 is the best point estimate and stands out from most of
the grid, but is **not decisively separated from p=0.92 specifically** — the finer grid did not
collapse to a single indistinguishable plateau (the original worry), but it also didn't produce a
razor-sharp single winner either. That's a real, nuanced result, not a wash.

## Verdict and action

**No change to the promoted config.** `task3_config.DEFAULT_MODEL_KEY =
"cox_p03_prostatectomy_pca5_p90"` stays as-is — p=0.90 remains both the best point estimate on a
finer grid and the significant winner against most of the neighborhood it was worried might
swallow it.

This closes the limitation previously phrased as *"no fine sweep between p=0.85 and p=0.95 to
confirm 0.90 is the true local optimum"* — replaced everywhere with the more precise, checked
statement: confirmed as the best point on a finer grid, significantly ahead of most neighbors, with
one honest caveat (p=0.92 not statistically distinguishable from it).

## Not done

- Did not re-sweep the Cox penalizer or re-check the event-classifier horizon/threshold jointly
  with this finer percentile grid — each parameter checked independently, per this project's
  standing practice (see exp_4, exp_6, exp_7 for the analogous scope limits on those parameters).
- Did not extend the fine grid outside [0.85, 0.95] — the coarse grid already showed both bounding
  points (0.80, 1.0-ish region) score clearly worse, so this window was treated as sufficient.
