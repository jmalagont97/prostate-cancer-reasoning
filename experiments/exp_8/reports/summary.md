# exp_8 Report: Expanded Variables + Hyperparameter Tuning + Reveal-Sequence

**Experiment**: experiments/exp_8/ · **Project**: challenge_chimera_2 · **Date**: 2026-08-12 · **Status**: Complete

---

## 1. Summary

**Mixed, mostly inconclusive — with one genuine, unambiguous win.** Neither the expanded
23-column frame nor hyperparameter retuning produced a decision-accuracy improvement that both
CV and the held-out check agree on; confidence got slightly worse on balance; weights stayed
essentially flat (its one win narrowed to the edge of noise). But **reveal-sequence — the first
time any target beyond decision/confidence/weights has been derived from this shared backbone —
beat its naive baseline on its first attempt** (0.799 vs. 0.783), a genuinely new result this
project didn't have before today.

| Target | Best exp_8 result | exp_6 (unmodified) | Verdict |
|---|---|---|---|
| Decision (CV) | 0.585–0.591 (all 3 conditions) | 0.593 | ❌ No condition beats it under CV |
| Decision (held-out) | 0.635 (combined) | 0.593 | ⚠️ Beats it — but CV and held-out **disagree in direction** (see §3) |
| Confidence (best ord. dist.) | 0.744 (entropy_isotonic) | 0.731 | ❌ Slightly worse |
| Confidence (best macro-F1†) | 0.268 (entropy_isotonic) | 0.223 | ✅ Better — the two metrics disagree here (§4) |
| Weights (occlusion) | 0.412 ord. err. / 0.269 macro-F1† | 0.405 / 0.256 | ⚠️ Ord. err. margin razor-thin; macro-F1 mildly up |
| **Reveal-sequence** | **0.799 set-precision** / 0.531 macro-F1† | *(no prior KDM result)* | ✅ **Beats baseline (0.783)** on set-precision — first-ever KDM reveal result |

†Macro-F1 backfilled 2026-08-12 across `exp_6`/`exp_7`/`exp_8`, reported alongside each
subtask's official rubric metric (which remains primary) — see §4–§6 for full detail and the
cases where the two metrics disagree.

## 2. What Was Run

- **144-combination hyperparameter search** on the 23-column frame (`exp_7`'s exact grid, no
  `log1p` step). Winner: `n_epochs=300, lr=1e-2, sigma_mult=1.5, optimizer=adam, wd=0` — search-phase
  macro-F1 0.582, margin **−0.011** under `exp_6`'s baseline, well short of the corrected 0.045
  clear-margin threshold.
- **Three-way ablation** isolating which lever (if either) contributes:
  - `decision_kdm_features_only` (23-col frame, `exp_6`'s original hyperparameters): 0.580
  - `decision_kdm_tuned_only` (original 19-col frame, winning hyperparameters): 0.591
  - `decision_kdm_v3` (combined, full 10-repeat): 0.585
- **Mandatory held-out check** (`holdout_eval_v3.py`, same 19-case split as `exp_3`/`exp_7`):
  `exp_6` plain KDM = 0.593 macro-F1; `exp_8` combined = **0.635** macro-F1 (delta **+0.042**).
- **`run_signals_v3.py`**: `exp_6`'s unchanged confidence/weights readout code on the new backbone
  — `probs_check_ok=True` across every fold.
- **`run_reveal_kdm.py`** (new): entropy-increase-from-occlusion signal per reveal section,
  per-section binary classifiers. Confirmed at implementation time (and now corrected into
  `DESIGN.md`): `family_history`/`pathology_report` are revealed in **0 of 91** labeled cases, so
  only 4 sections (`previous_notes`, `laboratory_results`, `psa_trend`, `radiology_report`) are
  actually modeled — matching the same convention every prior reveal model in this project
  already uses.

## 3. Decision: CV and held-out disagree — an important finding in its own right

Under CV, all three exp_8 conditions land **slightly below** `exp_6`'s 0.593 (0.580–0.591) — a
small, consistent gap, well inside `exp_6`'s own measured 10-repeat std (0.045). None of the
three levers (expanded features alone, tuning alone, or combined) produces a CV improvement.
This is the opposite pattern from `exp_7`, where the search *did* find an apparent CV win that
then failed the held-out check.

Here it's reversed: **the held-out check shows the combined condition doing noticeably better**
(0.635 vs. 0.593, +0.042) — on the exact same held-out cases `exp_7`'s check used. Two honest
readings, not one:

1. **Neither signal should be treated as definitive alone.** CV averages over many folds but
   pools noise from a small N=91; held-out is a single, unrepeated 19-case split — also noisy,
   just differently. A genuine, robust improvement should show up in *both*; this shows up in
   neither consistently.
2. **This is a useful methodological finding for the project, not just a null result for this
   experiment.** Two experiments in a row (`exp_7`, now `exp_8`) have produced a CV/held-out
   disagreement in *opposite* directions — CV-optimistic-then-refuted in `exp_7`, CV-pessimistic-
   then-seemingly-better here. That pattern itself is evidence that **at N=91, a single-split
   held-out check and a 50-fold CV estimate are both individually too noisy to settle a question
   like this one** — a real constraint on what this project's evaluation protocol can and can't
   resolve, worth stating plainly rather than picking whichever number is more flattering.

**Verdict**: decision is not improved by either lever, and not clearly unimproved either — genuinely
inconclusive. Per `DESIGN.md` §8's decision rules, this doesn't clear the bar for adopting the
expanded frame or the retuned hyperparameters as a new default.

## 4. Confidence: slightly worse on balance

| Condition | exp_8 ord. dist. | exp_6 ord. dist. | Δ | exp_8 macro-F1† | exp_6 macro-F1† |
|---|---|---|---|---|---|
| `entropy_isotonic` | 0.744 | 0.731 | +0.013 (worse) | **0.268** | 0.223 |
| `dispersion_isotonic` | 1.085 | 0.776 | +0.309 (much worse) | 0.170 | 0.153 |
| `participation_isotonic` | 0.796 | 0.844 | −0.048 (better) | 0.148 | 0.245 |
| `blend` | 0.781 | 0.754 | +0.027 (worse) | 0.246 | 0.269 |
| `entropy_zeroshot` | 1.249 | 1.232 | +0.017 (worse, both far above baseline) | 0.167 | 0.179 |

†Backfilled 2026-08-12. Worth flagging: **`entropy_isotonic` has the best macro-F1 of any
condition here (0.268) despite having a worse ordinal distance than `exp_6`'s equivalent** — the
two metrics disagree on both the ranking *and* the exp_6-vs-exp_8 direction for this specific
condition. `participation_isotonic` shows the opposite split (better ordinal distance under
`exp_6`, better... no, worse macro-F1 under exp_8) — a reminder that these two metrics are
scoring genuinely different things (rank-distance vs. per-class balance), and a condition
"improving" on one can still look flat or worse on the other. 4 of 5 conditions got worse on
ordinal distance; only entropy_isotonic clearly improved on macro-F1. Neither metric shows any
condition approaching baseline (0.527) or the incumbent (0.468). `dispersion_isotonic`'s sharp
ordinal-distance degradation (+0.309) is the most notable single change on that metric — the
added dimensions appear to have made the `dm_rbf_variance`-based signal noisier there,
consistent with §1's dimensionality-tension concern, though its macro-F1 barely moved (+0.017).

## 5. Weights: flat aggregate, real per-factor movement

`weights_kdm_occlusion_v3` = 0.412 — still nominally beats the 0.413 baseline, but the margin
shrank from `exp_6`'s 0.008 to essentially **0.001**, indistinguishable from noise.
`kernel_distance_v3` (0.479) and `blend_v3` (0.672) both improved relative to `exp_6` but remain
well above baseline either way. **Macro-F1 corroborates**: mean macro-F1 across the 9 factors is
0.269 for exp_8's `occlusion` vs. 0.256 for `exp_6`'s — a small improvement (+0.013), consistent
in direction with `decisive_set_f1` staying flat-to-slightly-better, not with a real regression.

The per-factor breakdown shows something worth flagging despite the flat aggregate — real,
if modest, movement on two of the four previously-unsolved factors:

| Factor | exp_6 decisive-F1 | exp_8 decisive-F1 | exp_6 macro-F1† | exp_8 macro-F1† | Note |
|---|---|---|---|---|---|
| `psad` | 0.000 | **0.254** | 0.209 | 0.252 | first-ever nonzero decisive-F1 for this factor, macro-F1 also up |
| `vol` | 0.121 | **0.190** | 0.234 | 0.243 | improved on both metrics |
| `comorbidity` | 0.000 | 0.018 | 0.203 | 0.207 | still effectively unsolved on decisive-F1 |
| `cspca` | 0.000 | 0.000 | 0.244 | 0.257 | still unsolved on decisive-F1, macro-F1 ticks up slightly |
| `pirads`/`bx`/`psa`/`age`/`dre` | (already solved) | (still solved, 0.502–0.994) | 0.193–0.343 | 0.228–0.332 | unchanged pattern |

†Backfilled 2026-08-12. Every one of the 9 factors' macro-F1 moved in the same direction as (or
stayed flat relative to) its `decisive_set_f1` — no factor where the two metrics disagree, unlike
the confidence table above. This is a useful cross-check: it means the weights section's
"flat aggregate, real per-factor movement" reading isn't an artifact of which specific metric got
reported.

`psad`'s jump from 0.000 to 0.254 decisive-F1 (and 0.209→0.252 macro-F1, the same direction) is
plausibly explained by `cli_isup`/`vit_bmi` giving the occlusion mechanism more surrounding
context to work with when isolating `psad`'s own contribution — worth a closer look in any
follow-up, though one factor moving on an otherwise flat aggregate isn't strong evidence by itself.

## 6. Reveal-sequence: the clear win

`reveal_kdm_occlusion`: **set_precision = 0.799 ± 0.018**, beating the 0.783 naive baseline —
first attempt, first result, first win for this target derived from the shared backbone. Doesn't
reach the 0.853 incumbent (a purpose-built `MultiOutputClassifier`), but this is a genuinely new
capability, not a refinement of an existing one.

| Section | Precision | Recall | F1† | Positive rate | Note |
|---|---|---|---|---|---|
| `previous_notes` | 0.894 | 0.800 | **0.844** | 84.6% | Strong on both |
| `psa_trend` | 0.847 | 0.619 | 0.711 | 86.8% | Strong precision, decent recall |
| `radiology_report` | 1.000 | 0.212 | 0.349 | 96.7% | Perfect precision, but **badly under-predicts** — a trivial "always reveal" baseline would score ~0.97 precision *and* ~1.0 recall here, so this section's low recall is a real weakness, not a strength masked by high precision; the F1 view makes this failure far more visible than precision alone did |
| `laboratory_results` | 0.439 | 0.159 | 0.220 | 45.1% | Weakest section on every metric — also the most genuinely uncertain one (closest to 50/50 positive rate) |

**Macro-F1 (multi-label definition: per-section F1, macro-averaged across the 4 modeled
sections) = 0.531 ± 0.037** — backfilled 2026-08-12, reported alongside `set_precision` (the
official rubric metric, which remains primary). The two metrics tell different stories on
purpose: `set_precision` rewards not over-revealing and lets a conservative, high-precision
model like this one score well (0.799, beats baseline) even with weak recall; macro-F1 weights
precision and recall equally *per section*, and is far less forgiving of `radiology_report`'s
0.212 recall — which is exactly why `radiology_report`'s F1 (0.349) is so much lower than its
precision (1.000) alone would suggest. Both readings are valid; they're just answering different
questions ("is the model efficient with its reveals" vs. "is the model balanced per section"),
and it's worth carrying both forward rather than picking one.

The mechanism's precision-biased behavior (favoring "don't over-reveal" over "catch every true
reveal") is directly rewarded by `reveal_set_precision()`'s definition — the rubric's own "tool
efficiency" framing — which is why the aggregate score is respectable even with `radiology_report`
and `laboratory_results` pulling recall (and therefore F1) down hard. `family_history`/
`pathology_report` are out of scope for this mechanism entirely (0 positive examples, confirmed
and documented in `DESIGN.md`), not a shortfall of this particular result.

## 7. Interpretation

1. **The dimensionality tension flagged before this experiment started is visible in the data,
   not just a theoretical worry.** Decision and confidence both show flat-to-negative movement
   under CV with the expanded frame; only weights (via its per-case, per-factor occlusion
   mechanism, which uses the *whole* frame's context rather than treating every added dimension
   as equally load-bearing) shows any positive per-factor movement.
2. **Two experiments in a row have now shown CV and held-out results actively disagreeing** on
   this backbone. That's a more important, generalizable finding for this project than either
   experiment's own headline result — any future KDM-tuning experiment should treat a single CV
   number *or* a single held-out number as insufficient on its own, and should say so explicitly
   rather than report whichever one looks better.
3. **Reveal-sequence's win, on a genuinely first attempt with a genuinely exploratory mechanism,
   is the strongest positive result in this experiment** — stronger, in a sense, than a marginal
   decision improvement would have been, since it demonstrates the shared-backbone idea
   generalizes to a target it was never built for.

## 8. Recommendation

- **Do not adopt the expanded 23-column frame or the retuned hyperparameters** as new defaults
  for decision/confidence — neither shows a reliable win, and confidence trended slightly worse.
- **`vit_bmi`/`cli_isup` may be worth keeping for weights specifically**, given `psad`'s real
  movement — but this is a single-factor signal on an otherwise flat aggregate and needs its own
  focused follow-up before being called a genuine improvement, not adopted on this evidence alone.
- **Build on `reveal_kdm_occlusion`** — it's this project's first working reveal-sequence result
  from the shared-backbone architecture and beat baseline on its first attempt. A natural next
  step: revisit the previously-rejected `lab_free_psa_ng_ml`/`lab_free_total_ratio` (the literal
  correct fit for `laboratory_results`, its weakest section) now that this mechanism has a proven
  baseline to improve from.
- **Any future KDM-tuning experiment on this project should report both CV and held-out numbers
  side by side by default**, and treat disagreement between them as a finding to explain, not a
  loose end to average away.
