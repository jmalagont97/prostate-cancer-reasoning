# exp_4 Report: Task3 Model-Family Sweep + Agent Wiring

**Status**: Complete. **Verdict: confirmed — `cox_p03` promoted as the new default.**

## What was built (parity with task1)

| File | Role |
|---|---|
| `agent/src/task3_schema.py` | `Task3Output` (ported from source: `case_id`, `task=3`, `months_to_recurrence`≥0, `reasoning`≥40 chars) |
| `agent/src/task3_free_text.py` | Template reasoning generation, grounded in the case's actual pathology fields |
| `agent/src/task3_config.py` | Model registry (`MODEL_FACTORIES`) + winning feature set + `DEFAULT_MODEL_KEY` — the swap point |
| `agent/src/task3_variant_tester.py` | LOOCV swap-test harness, scores with `evaluate.py`'s real `concordance_index` |
| `agent/src/models_bridge_task3.py` | Fits+saves the deployable model (`agent/models/task3_agent_model.joblib`) |
| `agent/src/run_task3_agent.py` | Production (in-sample fit) — for scoring genuinely new cases |
| `agent/src/run_task3_agent_loocv.py` | Honest, leakage-free LOOCV — for evaluation |
| `agent/local_eval_task3.py` | Scores `agent/output/task3/` against ground truth with real evaluator functions |

## Model sweep result

| Model | LOOCV ranking_score |
|---|---|
| cox_p001 | 0.7035 |
| cox_p01 (old default) | 0.7487 |
| **cox_p03 (new default)** | **0.7655** |
| cox_p05 | 0.7584 |
| cox_p07 | 0.7540 |
| cox_p10 | 0.7513 |
| cox_p20 | 0.7363 |
| weibull_aft | 0.7553 |
| lognormal_aft | 0.7398 |
| loglogistic_aft | 0.5628 |

Clean unimodal curve over the Cox penalizer, peak at 0.3. Promoted. Cross-validated two independent
ways (in-memory sweep vs. full on-disk LOOCV-outputs-then-real-evaluator path) — both give 0.7655
exactly.

## Two real issues caught during this experiment

1. **`predict_median()` silently destroys ranking resolution**: 62/75 cases never cross 50%
   survival probability within the observed follow-up window, so `predict_median()` collapses all
   of them to the same capped fallback value. This alone accounted for a 0.65→0.75 score jump once
   switched to `predict_expectation()` (0/75 capped) — not a modeling improvement, a bug fix.
2. **exp_3's headline number (0.772) used a proxy, not the real contract.** It ranked on
   `-partial_hazard` directly; `evaluate.py`'s actual `concordance_index` ranks on the literal
   `months_to_recurrence` value a real submission would contain. The two are close (0.772 vs.
   0.7655) but not identical — this experiment's number is the one to trust going forward, since it
   comes from the actual code path that would generate a real submission.

## Model-swap workflow, going forward

```python
# agent/src/task3_config.py
DEFAULT_MODEL_KEY = "cox_p03"   # change this line...
```
```bash
python agent/src/models_bridge_task3.py   # ...then refit+save
```
To test a candidate before promoting: `task3_variant_tester.test_variant("model_key")` — no disk
writes, real evaluator scoring, same discipline as task1's `variant_tester.py`.

## Follow-up (2026-08-25): production bug fix, T3.7 cross-check, significance check

1. **Bug fix**: `run_task3_agent.py`'s data loader inner-joined `labels.csv`, meaning the
   "production" script could only run on cases that already had a known outcome -- backwards for
   its stated purpose. Fixed (dropped the labels merge; the model doesn't need it to predict).
   Re-verified LOOCV still scores exactly 0.7655 after the fix (unaffected, as expected -- this was
   a robustness fix, not a modeling change).
2. **T3.7 (IPCW time-dependent AUC) done**: 12m=0.854, 24m=0.760, 36m=0.753, 60m=0.758, mean=0.781
   -- all comfortably above 0.5 and above the overall C-index, no sign the model is overfit to
   concordance specifically. Added to `agent/local_eval_task3.py` as a permanent cross-check.
3. **Significance check on the cox_p01→cox_p03 gain**: paired bootstrap (2000 resamples, n=75).
   Point estimate +0.0168, but the 95% CI is **[-0.0068, +0.0416]** — crosses zero, so **not
   formally significant** at conventional thresholds, despite 90.8% of resamples favoring cox_p03.
   Honest read: directionally real and worth keeping (clean unimodal sweep, no reason to expect
   cox_p03 is actually worse), but should not be oversold as a settled, decisive win the way
   exp_3's pathology-feature gain (+0.07-0.09, consistent under both LOOCV and 5-fold) was. Same
   discipline this project has applied to every marginal gain (e.g. task1's earlier +0.001
   confidence-swap case).

## Not done / next candidates

- `scikit-survival` (Random Survival Forest / Gradient-Boosted Cox) blocked on a missing C++
  compiler toolchain — would need an explicit ask to install Visual C++ Build Tools.
- T3.7 (IPCW time-dependent AUC cross-check) still not attempted.
- T3.2/T3.3 (MRI/slide-embedding fusion) still not attempted — now has a much stronger baseline
  (0.7655) to try to beat.
