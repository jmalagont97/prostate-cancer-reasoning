# exp_10 Report: The Remaining Gated Text Sections

**Status**: Complete. **Verdict: ❌ Refuted, cleanly — the skeptical hypothesis held. Extraction code
kept (real, documented, additive columns); nothing promoted into the model.**

## What was tried

Extracted the three previously-untouched gated text sections into 8 new candidate features (see
`experiments/exp_10/DESIGN.md` for the full corpus scan and closed-vocabulary confirmation),
added to `scripts/extract_features_task3.py` and `data/processed/task3/engineered_features.csv`:

| Group | Features | Coverage |
|---|---|---|
| Radiology | `prostate_volume_cc`, `psa_density`, `pirads`, `ai_prob_cspca` | 75/75 |
| Prior notes | `has_comorbidities`, `cci` | 30/75, 19/75 |
| Family history | `family_history_present` | 35/75 |

`on_5ari` was extracted but **excluded from testing**: 43 of 44 non-null cases are "not on" — only
one case (T3-057) is positive. Holding that single case out during LOOCV collapses the column to
zero variance in the training fold, which crashes lifelines' Newton-Raphson step
(`ConvergenceError: delta contains nan`) — confirmed by isolating it (full-data fit succeeds; only
LOOCV folds excluding T3-057 fail). Same near-zero-variance problem exp_3 hit with
`active_treatment_flag`, same resolution: kept in the CSV as documented output, not fed to the model.

## Results (LOOCV, `cox_p03_prostatectomy_pca5_p90` pipeline unchanged except feature set)

| Variant | # features | LOOCV C-index | vs. baseline |
|---|---|---|---|
| **Baseline (promoted)** | 13 | **0.8765** | — |
| + radiology | 17 | 0.8646 | −0.0119 |
| + prior notes | 15 | 0.8686 | −0.0079 |
| + family history | 14 | 0.8743 | −0.0022 |
| + all three combined | 20 | 0.8642 | −0.0123 |

The baseline wins outright on the point estimate against every variant — no bootstrap needed (this
project's standing rule: only test significance when something beats the champion; nothing did
here).

## Why, diagnosed rather than left a mystery

`psa_density` correlates **0.969** with the already-included `psa` field — almost entirely
redundant information (arithmetically expected: density is PSA divided by a volume this model
already has no direct read on). The radiology group's four features add one near-duplicate signal
plus three genuinely new but individually weak ones (`pirads`, `ai_prob_cspca`,
`prostate_volume_cc`) — at n=75/19-events, that's added ridge-penalized noise, not resolution.
Family history came closest to breaking even (−0.0022, the smallest gap) — plausible given prostate
cancer's real but modest heritable component — but still didn't clear the baseline.

## Verdict and action

**No change to the promoted config.** `task3_config.NUMERIC_FEATURES` stays at its current 13
fields. This matches this project's stated, skeptical prior — naive feature expansion has now lost
in task1 (exp_2, for decision/variable-weights), and 3 of 4 embedding-fusion variants (exp_5) — and
extends that track record to task3's text features.

The extraction code (`scripts/extract_features_task3.py`'s new parse functions,
`engineered_features.csv`'s 8 new columns) is kept, not deleted — documented, additive, unused by
the promoted model, available if a future experiment wants to build on it (e.g., testing
`family_history_present` alone against a different embedding/penalizer combination, or as an
interaction term).

This closes the limitation previously phrased as *"radiology_report, previous_notes, and
family_history text — never extracted at all — untried, not refuted"* — replaced with the more
precise, checked statement: **tried, and refuted** on the promoted pipeline.

## Not done

- Did not re-sweep the Cox penalizer jointly with any new feature group (each parameter checked
  independently, per this project's standing scope limits — see exp_4, exp_6, exp_8, exp_9 for the
  same pattern on other parameters).
- Did not test feature-selection subsets within each group (e.g., `pirads` alone, without
  `psa_density`) — tested as designed groups, not an exhaustive per-feature search.
- Did not test interaction terms (e.g., family history × ISUP grade) — out of scope, a materially
  different kind of experiment.
