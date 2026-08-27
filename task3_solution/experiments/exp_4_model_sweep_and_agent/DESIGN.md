# Experiment Design: Task3 Model-Family Sweep (T3.4/T3.5) + Agent Wiring
**Experiment**: experiments/exp_4/
**Project**: challenge_chimera_v3
**Date**: 2026-08-24
**Status**: Complete

---

## 1. Context and hypothesis

`exp_3` (T3.6) established the winning feature set (age/psa/cleaned-dre + 8 surgical-pathology
fields) and used a single Cox model (penalizer=0.1) to validate it, scoring LOOCV C-index=0.772 —
but that number ranked on the model's raw `-partial_hazard`, not on a literal `months_to_recurrence`
value, because no agent existed yet to produce the actual submission-shaped output. This experiment:

1. **Builds the missing task3 agent** (`agent/src/{task3_schema,task3_free_text,run_task3_agent,
   run_task3_agent_loocv}.py`, `agent/local_eval_task3.py`) so task3 has the same production +
   honest-LOOCV + real-evaluator-scoring parity task1 already has.
2. **Re-measures under the real contract**: `evaluate.py`'s `concordance_index` takes literal
   predicted months (not hazard) as `preds` — converting a Cox model's hazard into an actual time
   (`predict_expectation()`) is a real step with its own failure modes, not a formality, and the
   resulting number is what actually determines the leaderboard score.
3. **Sweeps model family + Cox regularization strength** (`agent/src/task3_config.py`'s
   `MODEL_FACTORIES`, tested via `agent/src/task3_variant_tester.py`), per the user's explicit ask
   to be able to swap the model easily and to keep pushing on it, the same way task1's `exp_2` swept
   model families for its sub-tasks. Covers the plan's T3.4 (alternate model class) and T3.5
   (parametric AFT models) rows, within what's actually installable here (see §2).

**Hypothesis**: some combination of {model family, regularization strength} beats the exp_3-default
Cox (penalizer=0.1) under this stricter, real-contract LOOCV C-index.

---

## 2. Models tested

`scikit-survival` (Random Survival Forest / Gradient-Boosted Cox — the plan's T3.4 candidates) was
evaluated for install and is **not available**: `pip install scikit-survival` fails building its
`ecos` dependency from source, requiring a C++ compiler ("Microsoft Visual C++ 14.0 or greater")
not present in this environment. Not chased further (would require installing a full build
toolchain, out of scope without explicit ask).

All models below are from `lifelines` (already a dependency, T3.5's family plus Cox at multiple
regularization strengths):

| Model | Notes |
|---|---|
| Cox, penalizer ∈ {0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0} | `predict_expectation()` → months (see §3, why not `predict_median()`) |
| WeibullAFTFitter | direct time prediction, no hazard→time conversion needed |
| LogNormalAFTFitter | ″ |
| LogLogisticAFTFitter | ″ |

## 3. A real bug caught mid-experiment

First pass used `CoxPHFitter.predict_median()` to get a literal months value (seemed the more
"robust" choice over `predict_expectation()`'s open-ended integral). Result: LOOCV C-index=0.652,
a full 0.12 below exp_3's 0.772. Root cause, confirmed directly: `predict_median()` requires the
fitted survival curve to cross 50% survival probability, which it never does *within the observed
follow-up window* for 62/75 cases — all 62 silently collapse to the same capped fallback value,
destroying nearly all of the model's actual ranking resolution. Switched to `predict_expectation()`
(0/75 capped, smooth 0.6–104.6 month range) — this alone recovered most of the gap (0.749 vs.
0.652 for the same penalizer=0.1 model). A reminder that "predicted median survival" is not always
well-defined for real cohorts with heavy right-censoring, even though it sounds like the safer
choice.

## 4. Results (LOOCV, `evaluate.py`'s own `concordance_index` on literal predicted months)

| Model | ranking_score (C-index) |
|---|---|
| cox_p001 | 0.7035 |
| cox_p01 (exp_3's original default) | 0.7487 |
| **cox_p03** | **0.7655** |
| cox_p05 | 0.7584 |
| cox_p07 | 0.7540 |
| cox_p10 | 0.7513 |
| cox_p20 | 0.7363 |
| weibull_aft | 0.7553 |
| lognormal_aft | 0.7398 |
| loglogistic_aft | 0.5628 |

Clean unimodal curve over the penalizer sweep, peaking at **0.3**. `cox_p03` beats the original
default by **+0.0168** and every AFT model tried. `loglogistic_aft` is a clear outlier-bad fit
(0.563, near-random) — not investigated further since it's dominated by every other candidate, not
worth debugging in isolation.

**Promoted**: `task3_config.DEFAULT_MODEL_KEY = "cox_p03"`. Cross-validated two ways: the in-memory
sweep (`task3_variant_tester.py`, 0.7655) and the full on-disk path
(`run_task3_agent_loocv.py` → `agent/local_eval_task3.py`, also 0.7655) agree exactly.

## 5. Honest reconciliation with exp_3's headline number

exp_3 reported LOOCV C-index=0.772 for the same feature set, ranking on `-partial_hazard` via
`lifelines.utils.concordance_index`. This experiment's `cox_p03` (same feature set, better
regularization) gets 0.7655 via `evaluate.py`'s `concordance_index` on `predict_expectation()`-derived
months. The two numbers are close but not identical because they're not measuring quite the same
thing: partial-hazard ranking is the model's *purest* internal risk ordering, while converting
through `predict_expectation()`'s survival-curve integral introduces some numerical
approximation (extrapolation past the fitted horizon, etc.). **This experiment's number (0.7655) is
the more trustworthy one** — it's produced by the actual code path that would generate a real
submission and scored with the evaluator's own function on the literal value it reads, not a proxy.

## 6. Model-swap infrastructure (per explicit request)

`agent/src/task3_config.py::MODEL_FACTORIES` is a plain dict of `{key: factory}`; every entry
exposes the same `.fit(df)` / `.predict_months(df)` interface. Swapping the deployed model is:
change `DEFAULT_MODEL_KEY`, rerun `models_bridge_task3.py`. Testing a candidate before promoting is
`task3_variant_tester.test_variant("model_key")` — no disk writes needed, mirrors task1's
`variant_tester.py`. Adding a new model family (e.g., once/if `scikit-survival` becomes installable)
is one new dict entry wrapped in the same adapter pattern.
