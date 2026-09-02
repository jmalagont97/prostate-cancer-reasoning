# exp_17 Report: Feature-Subset Sweep, All 3 Sub-Tasks, Evaluator-Vocabulary Floor

**Status**: Complete. **Verdict: ❌ No promotion anywhere. `family_history_present` (the one gap
this experiment closed) actively hurts `decision`, and is noise-level neutral for
`confidence`/`variable_weights`. Current promoted configs (exp_11/12/14/16) are confirmed as still
the best found, not superseded.**

## What was tested

Closed a real gap first: `fh`/`family_history` (part of the user-chosen 11-variable evaluator-scored
floor, `task2_schema.TASK2_VARIABLES`) had never been extracted as a usable numeric feature — only
consumed for tool-gating logic. Added `parse_family_history()` to `scripts/extract_features_task2.py`
(Yes/No/Unknown → 1.0/0.0/None, same convention as task3's own family-history feature, 94.8%
coverage). Swept the 11-variable floor against `psa_kinetics` (exp_16) and `labs` (exp_16, already
refuted for `decision`) independently for all three sub-tasks, each held at its own currently-
promoted model.

## Results, corrected against the right baseline

An early read of these numbers mistakenly compared `confidence`'s results against a stale baseline
(from before `decision`'s exp_16 kinetics promotion changed the shared `decision_f1` term shared
across all sub-task variants) — caught and corrected before reporting. The one clean baseline for
all three (current full promoted pipeline) is **0.7429**.

| Sub-task | Variant | ranking_score | vs. baseline |
|---|---|---|---|
| decision | floor_11 (+family_history, drops kinetics) | 0.7067 | **−0.0362** |
| decision | floor+kinetics (+family_history) | 0.6818 | **−0.0611** |
| decision | floor+labs | 0.6401 | −0.1028 |
| decision | floor+kinetics+labs | 0.6288 | −0.1141 |
| confidence | floor_11 (+family_history) | 0.7437 | +0.0008 |
| confidence | floor+psa_kinetics | 0.7437 | +0.0008 |
| confidence | floor+labs | 0.7452 | +0.0023 |
| confidence | floor+psa_kinetics+labs | 0.7452 | +0.0023 |
| variable_weights | floor_11 (+family_history) | 0.7425 | −0.0004 |
| variable_weights | floor+psa_kinetics | 0.7415 | −0.0014 |
| variable_weights | floor+labs | 0.7405 | −0.0024 |
| variable_weights | floor+psa_kinetics+labs | 0.7408 | −0.0021 |

## Two distinct, honest findings

**`decision`**: adding `family_history_present` is clearly, decisively bad — confirmed even in
isolation (floor_11 alone, no kinetics: −0.0362), and it actively fights the already-promoted
kinetics feature rather than combining neutrally (floor+kinetics: −0.0611, *worse* than
family_history alone). At n=72 with 94.8% coverage and real missingness on the rest, this one
categorical addition is enough to meaningfully degrade kNN's distance metric. No bootstrap needed —
every variant lost outright on the point estimate, all four, unambiguously.

**`confidence` / `variable_weights`**: every variant landed within noise of baseline (largest
magnitude: +0.0023 for confidence+labs, −0.0024 for variable_weights+labs) — an order of magnitude
smaller than anything bootstrap-tested so far in this project (exp_9's smallest confirmed-noise gain
was already larger than these). Not bootstrapped — not worth the compute to confirm what's already
almost certainly nothing.

## Verdict and action

**No change to any promoted config.** `decision` stays on its exp_16 config (10 base + kinetics, no
family history); `confidence` stays `knn`/10-feature (exp_11); `variable_weights` stays
`mord.LogisticIT`/10-feature (exp_14). `family_history_present` is now a real, available engineered
column (kept, documented, unused) — closing the gap this experiment set out to close, even though
the feature itself didn't pan out anywhere it was tried.

This directly answers the question that motivated the experiment: for `confidence` and
`variable_weights`, the 11-variable floor is essentially *sufficient* — nothing tried on top of it
moves the needle either way. For `decision`, the floor is not enough on its own (kinetics still
matters, per exp_16) but `family_history` specifically should stay excluded, not just untested.

## Not done

- No test of `family_history_present` combined with a tool-call declaration (`get_family_history` in
  `required_tools`) — given exp_11's decisive, bootstrap-confirmed finding that *any* tool call costs
  the whole case's `tool_score`, and this experiment already found the raw feature itself doesn't
  help, that combination was not worth testing.
- No narrower subset search below the 11-variable floor (e.g. dropping individual low-signal
  variables) — the user's explicit choice fixed the floor as a lower bound, not something to shrink
  further.
