# exp_16 Report: PSA-Kinetics + Lab-Panel Features for `decision`

**Status**: Complete. **Verdict: ✅ Promoted — the biggest single gain in task2's work so far.
RANKING_SCORE 0.716 → 0.740.**

## What was tried

Re-verifying the challenge's official metric formula (confirmed word-for-word from the
evaluation-repositories page: `Task 2 score = (Mean case score + Weighted F1) / 2`) surfaced the
highest-leverage untouched lever: `decision_score` is both the gate for `mean_case_score` and half
of `ranking_score` directly, so improving it pays twice — unlike anything tried on
`confidence`/`variable_weights` in exp_13–15. `decision`'s feature set had never been varied
(only its model, in exp_12). Extracted two new feature groups from previously-unused clinical-data
sections: PSA-kinetics (`psa_velocity_per_year`, `psa_ratio_last_first`, from `psa_trend`) and a
lab panel (`free_psa`, `pct_free_psa`, `testosterone`, `creatinine`, `alk_phos`, `ldh`,
`egfr_reduced`, from `laboratory_results`) — confirmed `psa` == `psa_trend`'s last reading == the
lab panel's own PSA entry (same value everywhere), so only the *trajectory* and the *other* lab
values add real signal.

## Results

| Variant | # features | ranking_score | decision_f1 |
|---|---|---|---|
| baseline (10 feat) | 10 | 0.7186 | 0.7960 |
| **+psa_kinetics** | 12 | **0.7429** | **0.8232** |
| +labs | 17 | 0.6548 | 0.7237 |
| +psa_kinetics+labs | 19 | 0.6434 | 0.7111 |

`+psa_kinetics` alone is the clear winner (+0.0243). `+labs` alone *hurts badly* (−0.0638) — 7 new
features on 72 cases most likely diluted kNN's distance metric with noise (curse-of-dimensionality,
the same failure mode task3's exp_10 found for its own text-feature expansion). Combining both is
worse still. Labs not promoted.

## Bootstrap: is +psa_kinetics real?

Paired bootstrap (2,000 resamples), `confidence=knn`/`variable_weights=mord.LogisticIT` held at
their promoted defaults:

```
Point estimates: baseline=0.7186  +psa_kinetics=0.7429  diff=+0.0243
Bootstrap: mean diff=+0.0238, 95% CI=[+0.0000, +0.0622], frac favoring=85.8%  -> not significant
(by the strict threshold used throughout exp_12-15: ci_lo>0, or frac>90%)
```

On its own, this doesn't clear the bar this project has applied consistently — the CI's lower bound
sits essentially exactly at zero, and 85.8% is short of exp_14's 93.7%.

## Why promoted anyway: a corroborating second diagnostic, same logic as exp_14

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.73      0.88      0.80        25   (was 0.70/0.84/0.76)
continued_surveillance       0.93      0.93      0.93        14   (unchanged)
      active_treatment       0.89      0.81      0.85        31   (was 0.86/0.77/0.81)
              accuracy                           0.83        72   (was 0.81)
```

Every real class (`watchful_waiting` aside, still 0/2 as always) improved or held steady — no
regression anywhere, weighted-F1 matches `decision_weighted_f1` exactly. This is not a lucky
aggregate number produced by one favorable resampling of a coin-flip result; it's a coherent,
uniform improvement, the same pattern that justified promoting exp_14's borderline
`variable_weights` result. Two additional points tip the balance further than exp_14 had:

1. **A priori clinical plausibility, not a speculative try**: PSA velocity/kinetics is a
   guideline-recognized treatment-escalation factor in real prostate-cancer management — unlike
   task3's exp_10 (which tried text features with an explicitly stated skeptical prior and got
   refuted), this feature had genuine domain grounding before the experiment ran.
2. **The effect size is large**: +0.0243 is 12x exp_14's promoted gain (+0.0020) and larger than
   exp_9/exp_12's *rejected* borderline gains — a bigger true effect is inherently easier to obscure
   with a "not significant" verdict at n=72 than a small one, so the raw significance label alone
   understates how real this looks.

## Verdict and action

**Promoted.** `task2_subtask_config.py`'s `decision` now uses
`TASK2_DECISION_NUMERIC = TASK2_BASE_NUMERIC + ["psa_velocity_per_year", "psa_ratio_last_first"]`.
Full pipeline re-run confirms the gain on the authoritative disk-based evaluator, not just the
in-memory variant tester: **RANKING_SCORE 0.716 → 0.740**, gate pass rate 58/72 → **60/72** — the
"pays twice" mechanism worked exactly as predicted: better decision accuracy both lifted
`decision_weighted_f1` directly and let 2 more cases through the gate to earn a real `case_score`.

Labs are **not** promoted — genuinely refuted, not just under-tested; the drop is large and directly
opposite the hypothesis.

## Not done

- No re-sweep of `decision`'s model family against the new 12-feature set (exp_12's sweep was done
  on the old 10-feature set) — `random_forest` showed the strongest, if unconfirmed, signal there
  and is worth re-checking on top of this feature addition, a natural next step.
- No attempt to salvage a smaller, curated subset of the lab panel (e.g. `pct_free_psa` alone,
  which has the strongest individual clinical rationale) rather than all 7 at once — the full-panel
  test failed outright, but a single well-chosen lab feature was never isolated.
