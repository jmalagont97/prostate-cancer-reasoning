# exp_9 Report: ARD (Per-Dimension) Kernel Bandwidth for the KDM Backbone

## 1. Summary

Replacing `exp_6`'s single shared kernel bandwidth `σ` with one trained `σⱼ` per input dimension
(ARD) **does not survive held-out verification on the 19-column frame** (same failure mode as
`exp_7`'s hyperparameter search), but **does show a genuine, held-out-confirmed improvement on
the 23-column frame** — and, most cleanly, the frame comparison run *within* ARD itself (same
architecture, same untuned hyperparameters, only the frame differs) shows the 23-column frame
decisively beating the 19-column one on held-out (+0.128 macro-F1), the first frame-only
comparison this project has been able to run without confounding it with a hyperparameter change.

The experiment's central, most literal test — does ARD rescue `confidence_kdm_dispersion_isotonic`
from the +0.309 ordinal-distance regression `exp_8` measured when the frame grew from 19 to 23
columns — **is directly confirmed**: that regression collapses from `0.776→1.085` (scalar `σ`,
`exp_6`→`exp_8`) to `0.797→0.804` (ARD, this experiment), a difference of essentially nothing.
That is the cleanest single result in this report. It does not generalize to every signal,
however — `entropy_isotonic` got *worse* on the 23-column frame under ARD, and the one weights
mechanism that had ever beaten baseline (`weights_kdm_occlusion`) loses that narrow win under ARD
on both frames. Reveal-sequence improved on both metrics for the 23-column frame and produced a
first-ever, standalone 19-column result. The secondary hypothesis — that trained `σⱼ` values
would corroborate `exp_5`'s SVM-based solvable/unsolvable factor split — was **not** corroborated:
only 2/5 of ARD's top-relevance factors match `exp_5`'s solved set, on both frames.

| Target | This experiment's best | Baseline | Incumbent | Beats baseline? | Beats incumbent? |
|---|---|---|---|---|---|
| Decision, 19-col | 0.552 macro-F1 (held-out) | 0.381 | 0.650 (Extra Trees) | ✅ | ❌ |
| Decision, 23-col | 0.680 macro-F1 (held-out) | 0.381 | 0.650 (Extra Trees) | ✅ | ❌ |
| Confidence, 19-col | 0.757 ord. dist. (`entropy_isotonic`) | 0.527 | 0.468 (`confidence_svm`) | ❌ | ❌ |
| Confidence, 23-col | 0.779 ord. dist. (`blend`) | 0.527 | 0.468 (`confidence_svm`) | ❌ | ❌ |
| Variable weights, 19-col | 0.457 ord. error (`occlusion`) | 0.413 | 0.382/0.392 (`weights_svm`) | ❌ (no longer) | ❌ |
| Variable weights, 23-col | 0.459 ord. error (`occlusion`) | 0.413 | 0.382/0.392 (`weights_svm`) | ❌ (no longer) | ❌ |
| Reveal, 23-col | 0.823 set-precision (`occlusion`) | 0.783 | 0.853 (`reveal_flags`) | ✅ | ❌ |
| Reveal, 19-col | 0.796 set-precision (`occlusion`) | 0.783 | 0.853 (`reveal_flags`) | ✅ (narrowly) | ❌ |

## 2. What Was Run

21 conditions across two feature frames (`exp_3`'s 19-column frame, `exp_8`'s 23-column frame),
fixed hyperparameters throughout (`n_epochs=300, lr=1e-2, sigma_mult=1.0`, `exp_6`'s original
defaults — no search this round, per `DESIGN.md` §1's explicit guardrail):

- `decision_kdm_ard_{19,23}col` — the ARD backbone's own decision output.
- 5 confidence conditions × 2 frames — same signal family as `exp_6`/`exp_8` (entropy zero-shot,
  entropy/dispersion/participation isotonic, A+B+C blend), computed via `compute_signals_ard()`.
- 3 weights conditions × 2 frames — occlusion, kernel-distance, D+E blend, via `occlusion_delta_ard()`.
- `holdout_eval_ard` — 3-way comparison on the fixed 19-case held-out split used since `exp_3`:
  (a) `exp_6`'s original scalar-`σ` model on the 19-column frame, (b) ARD on the 19-column frame,
  (c) ARD on the 23-column frame.
- `reveal_kdm_ard_{19,23}col` — per-section occlusion-entropy classifiers, same 4 modeled sections
  as `exp_8` (`family_history`/`pathology_report` still 0/91 positive, unmodeled on both frames).
- `importance_comparison` — trained `σⱼ` (full 91-case fit, no CV) vs. `exp_5`'s solved/unsolved split.

Macro-F1 is reported natively for every condition alongside the official rubric metric, per this
session's cross-experiment reporting convention (backfilled into `exp_6`–`exp_8` beforehand).

One implementation bug was caught and fixed before any scored run: `kdm_backbone.occlusion_delta()`
looked sigma-shape-agnostic but internally calls the *full* scalar `compute_signals()`, which
crashes on an ARD model's vector-valued `σ` inside the library's `dm_rbf_variance`. Fixed by adding
`occlusion_delta_ard()` (built on `compute_signals_ard()`) to `ard_kernel.py`. `kernel_distance_contribution()`
was confirmed genuinely sigma-agnostic (never references `σ` at all) and reused unchanged.

## 3. Decision: 19-column CV win doesn't survive held-out; 23-column frame does — and beats 19-column cleanly under ARD

| Comparison | CV macro-F1 | Held-out macro-F1 (n=19) |
|---|---|---|
| `exp_6` scalar, 19-col | 0.593 | 0.593 |
| `exp_8` scalar (tuned), 23-col combined | 0.585 | 0.635 |
| ARD, 19-col | **0.642** (Δ+0.049 vs. exp_6) | **0.552** (Δ**-0.041** vs. exp_6) |
| ARD, 23-col | **0.608** (Δ+0.023 vs. exp_8 CV) | **0.680** (Δ**+0.045** vs. exp_8 held-out, Δ+0.087 vs. exp_6) |

Two results, read separately:

1. **On the 19-column frame, ARD repeats `exp_7`'s exact failure pattern.** A real CV
   improvement (+0.049, well outside noise) evaporates and reverses on the fixed held-out split
   (-0.041). At N=91 with 18 extra trainable `σⱼ` parameters instead of 1, this is the overfitting
   failure mode `DESIGN.md` §9 flagged as the main risk of ARD specifically — individual `σⱼ`
   fitting fold-specific noise rather than a real signal. The held-out check exists to catch
   exactly this, and it did.
2. **On the 23-column frame, ARD's CV improvement is smaller (+0.023) but the held-out result
   holds up and even grows (+0.045 vs. exp_8's own held-out number, +0.087 vs. exp_6's).** This is
   the opposite failure mode from (1) — CV understated the improvement rather than manufacturing
   a fake one.

**The cleanest single comparison in this report** is not either of the above, but the frame
comparison *within* ARD itself: holding architecture and hyperparameters fixed, 23-column
held-out (0.680) beats 19-column held-out (0.552) by **+0.128**. Every prior frame comparison in
this project (`exp_7`'s tuned-19-col vs. `exp_8`'s tuned-and-expanded-23-col) confounded the frame
change with a hyperparameter change; this is the first time the frame is the *only* thing that
differs. Under CV, though, the direction is mild and reversed (19-col 0.642 > 23-col 0.608 by
0.034) — so even this cleanest comparison reproduces the same CV/held-out disagreement pattern
`exp_8`'s §3 first named as a standing limitation of this project's evaluation protocol at N=91.
This is now the **third consecutive KDM architecture experiment** (`exp_7`, `exp_8`, `exp_9`) to
show CV and held-out disagreeing, in different directions each time — strong, repeated evidence
that single-split held-out and 50-fold CV are each individually too noisy at this sample size to
settle a decision question on their own, not a property specific to any one lever tried.

**Verdict**: no clean, unconditional decision win. The 19-column ARD result is a clear negative
(reproduces `exp_7`'s failure). The 23-column ARD result is the most credible decision improvement
this project has produced to date (a real CV gain that grows rather than shrinks on held-out), but
given the standing CV/held-out reliability concern, it should be treated as a promising signal, not
adopted as a new default without a second, independent held-out replication.

**Update (added during `exp_10`'s session, 2026-08-13): that second, independent replication has
since happened**, and AUROC/Brier score were added alongside macro-F1. Both fully confirm the
23-column ARD result:

| Metric | Method | 19-col ARD | 23-col ARD |
|---|---|---|---|
| AUROC (↑ better) | CV (10-repeat mean±std) | 0.699 ± 0.023 | 0.676 ± 0.018 |
| AUROC | Held-out (seed=0) | 0.821 | **0.857** |
| AUROC | Held-out (10-seed mean±std) | — *(19-col not re-verified)* | 0.739 ± 0.062 |
| AUROC | Leave-one-out (91-fold, pooled) | — | 0.694 |
| Brier score (↓ better) | CV (10-repeat mean±std) | 0.257 ± 0.015 | 0.269 ± 0.013 |
| Brier score | Held-out (seed=0) | 0.195 | **0.170** |
| Brier score | Held-out (10-seed mean±std) | — | 0.240 ± 0.044 |
| Brier score | Leave-one-out (91-fold, pooled) | — | 0.264 |

23-column ARD's held-out AUROC/Brier both beat 19-column's on the same split, and its own repeated
LOO/held-out numbers cluster tightly with its original CV/held-out figures (AUROC 0.676–0.739,
Brier 0.240–0.269 across all four independent methods) — the tightest agreement any KDM decision
result has shown in this project. See `experiments/exp_10/reports/summary.md` §3b–3c for the full
verification (`experiments/exp_9/scripts/backfill_decision_auroc_brier.py`,
`experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py`) — it was run against `exp_10`'s
competing full-schema frame, and 23-column ARD won on every metric and method tried, no exceptions.
This upgrades the earlier "promising signal, not yet a new default" verdict: **23-column ARD-KDM is
now this project's best-validated decision configuration.**

**Update (added 2026-08-18): confusion matrix + classification_report, previously only printed to
stdout by `holdout_eval_ard.py`/`verify_decision_loo_repeated_holdout.py`, now persisted** via
`experiments/exp_9/scripts/decision_ard_23col_report.py` (re-fits the identical pipeline/config;
both macro-F1 values matched the already-reported 0.680/0.639 exactly before trusting the new
output) — saved to `experiments/exp_9/results/decision_kdm_ard_23col_report/metrics.json`:

Held-out (seed=0, n=19), rows=true/cols=pred, order `[no, yes]`:
```
           pred_no  pred_yes
true_no        3        4
true_yes       1       11
```
|  | precision | recall | f1-score | support |
|---|---|---|---|---|
| no | 0.750 | 0.429 | 0.545 | 7 |
| yes | 0.733 | 0.917 | 0.815 | 12 |
| macro avg | 0.742 | 0.673 | 0.680 | 19 |

LOO (91 folds, pooled predictions):
```
           pred_no  pred_yes
true_no       17       18
true_yes      12       44
```
|  | precision | recall | f1-score | support |
|---|---|---|---|---|
| no | 0.586 | 0.486 | 0.531 | 35 |
| yes | 0.710 | 0.786 | 0.746 | 56 |
| macro avg | 0.648 | 0.636 | 0.639 | 91 |

Both protocols show the same asymmetry: recall on "yes" (0.917 held-out / 0.786 LOO) comfortably
beats recall on "no" (0.429 held-out / 0.486 LOO) — a persistent bias toward the majority class
(61.5% "yes") rather than a collapse to predicting only one class. LOO's larger, less noisy pooled
sample (91 vs. 19) gives the more trustworthy per-class picture of the two.

## 4. Confidence: the motivating hypothesis is directly confirmed for `dispersion_isotonic` — but not for every signal

| Signal | exp_6 (19-col scalar) ord.dist / macro-F1 | ARD 19-col ord.dist / macro-F1 | exp_8 (23-col scalar) ord.dist / macro-F1 | ARD 23-col ord.dist / macro-F1 |
|---|---|---|---|---|
| `entropy_zeroshot` | 1.232 / 0.179 | 1.363 / 0.151 | 1.249 / 0.167 | 1.415 / 0.119 |
| `entropy_isotonic` | 0.731 / 0.223 | 0.757 / 0.186 | 0.744 / 0.268 | **0.945** / 0.177 |
| `dispersion_isotonic` | 0.776 / 0.153 | 0.797 / 0.143 | **1.085** / 0.170 | **0.804** / 0.132 |
| `participation_isotonic` | 0.844 / 0.245 | 0.957 / 0.186 | 0.796 / 0.148 | 0.787 / 0.131 |
| `blend` | 0.754 / **0.269** | 0.836 / **0.283** | 0.781 / 0.246 | 0.779 / 0.224 |

**The one number this experiment was designed around**: `dispersion_isotonic`'s ordinal distance
degraded from `0.776` (exp_6, 19-col scalar) to `1.085` (exp_8, 23-col scalar) — a +0.309
regression directly attributed in `exp_8`'s report to the shared bandwidth diluting the kernel's
sensitivity as weaker dimensions were added. Under ARD, the same 19→23-column expansion produces
`0.797→0.804`, a difference of 0.007 — the regression is **gone**. This is a direct, isolated
confirmation of `DESIGN.md` §1's central mechanism: giving the kernel a separate `σⱼ` per
dimension lets it down-weight the added, weaker columns instead of being forced to dilate its
whole distance scale for all of them at once. Both `dm_rbf_variance_ard()`'s formula (§4 of the
design doc) and this empirical outcome now agree with each other.

That confirmation does not extend to the other four signals uniformly:

- `entropy_isotonic` on the 23-column frame gets **worse** under ARD (0.744→0.945) than it was
  under scalar `σ` — the opposite of the dispersion result. Entropy is computed purely from
  `probs`, not from `σ` directly, so this is a second-order effect of the ARD-fitted decision
  boundary shifting, not something the dispersion story explains.
- `participation_isotonic` and `entropy_zeroshot` are worse under ARD on both frames.
- `blend` is the one signal that improves in macro-F1 terms on the 19-column frame (0.269→0.283,
  the highest macro-F1 any confidence condition has reached in this project) despite its ordinal
  distance getting worse (0.754→0.836) — another instance, now recurring across `exp_6`–`exp_9`,
  of macro-F1 and ordinal distance disagreeing on ranking for this signal specifically.

No confidence condition beats the `confidence_svm` incumbent (0.468) or even the naive baseline
(0.527) under ARD, same as every KDM confidence attempt since `exp_6`.

## 5. Weights: ARD loses the one narrow win the project had

`weights_kdm_occlusion` was the sole mechanism, across `exp_6`–`exp_8`, that ever beat the 0.413
weights baseline — narrowly (`exp_6`: 0.405; `exp_8`: 0.412, margin shrinking already). Under ARD,
that win is gone on both frames:

| Condition | Mean ord. error | Mean macro-F1 | vs. baseline (0.413) |
|---|---|---|---|
| `occlusion`, 19-col (`exp_6` scalar, for reference) | 0.405 | 0.256 | ✅ narrowly beats |
| `occlusion`, 19-col (ARD) | **0.457** | 0.258 | ❌ no longer beats |
| `occlusion`, 23-col (`exp_8` scalar, for reference) | 0.412 | 0.269 | ✅ narrowly beats |
| `occlusion`, 23-col (ARD) | **0.459** | 0.259 | ❌ no longer beats |
| `kernel_distance`, 19-col (ARD) | 0.503 | 0.216 | ❌ |
| `kernel_distance`, 23-col (ARD) | 0.491 | 0.208 | ❌ |
| `blend`, 19-col (ARD) | 0.661 | 0.244 | ❌ |
| `blend`, 23-col (ARD) | 0.637 | 0.248 | ❌ |

`kernel_distance_contribution()`'s Signal E formula was deliberately left unweighted by `σⱼ` this
round (per `DESIGN.md` finding #4, to avoid confounding "does ARD help the backbone" with "does an
ARD-aware attribution formula help weights") — so its degradation here is expected and not
informative about ARD's backbone effect specifically. `occlusion`'s degradation is more notable:
it re-runs inference on the *same* trained model with one factor's columns occluded, so its
regression reflects the ARD-fitted model's own decision surface becoming less locally
well-behaved for this particular probing method, not a formula mismatch.

**Per-factor** (occlusion, macro-F1 in the last column):

| Factor | 19-col ord.err (ARD) | 19-col macro-F1 | 23-col ord.err (ARD) | 23-col macro-F1 | `exp_6`/`exp_8` scalar ord.err (19/23-col) |
|---|---|---|---|---|---|
| dre | 0.258 | 0.343 | 0.274 | 0.336 | 0.263 / 0.264 |
| comorbidity | 0.308 | 0.205 | 0.315 | 0.204 | 0.316 / 0.316 |
| age | 0.423 | 0.231 | 0.426 | 0.216 | 0.414 / 0.398 |
| cspca | 0.473 | 0.222 | 0.449 | 0.231 | 0.470 / 0.444 |
| pirads | 0.455 | 0.274 | 0.404 | 0.306 | 0.490 / 0.455 |
| bx | 0.479 | 0.337 | 0.468 | 0.332 | 0.451 / 0.477 |
| vol | 0.463 | 0.241 | 0.578 | 0.223 | 0.288 / 0.341 |
| psa | 0.584 | 0.241 | 0.586 | 0.235 | 0.426 / 0.427 |
| psad | **0.669** | 0.229 | **0.632** | 0.252 | 0.526 / 0.584 |

`pirads` on the 23-column frame is the one factor that clearly improves (0.455→0.404, near
`dre`/`comorbidity`'s already-strong level) — plausibly related to `mri_pca_1` picking up the
smallest `σⱼ` of any column on that frame (see §7), giving the model a more sensitive read on the
radiology-adjacent signal that also informs `pirads`. `vol` and `psa` both get noticeably worse,
particularly `vol` on the 23-column frame (0.288→0.578, now the second-worst factor after
`psad`) — consistent with `vol`'s and `psa`'s relatively large trained `σⱼ` (low relevance) on
both frames.

## 6. Reveal-sequence: 23-column frame improves on both metrics; first standalone 19-column result

| Condition | Set-precision | Macro-F1 |
|---|---|---|
| `exp_8` scalar, 23-col (`occlusion`) | 0.799 | 0.531 |
| ARD, 23-col | **0.823** (Δ+0.024) | **0.599** (Δ+0.068) |
| ARD, 19-col | 0.796 | 0.576 |

The 23-column frame improves on *both* metrics under ARD — a consistent win, not a metric-specific
one. Per-section detail shows the improvement is concentrated in exactly the section that was
weakest under the scalar backbone:

| Section | `exp_8` scalar 23-col P/R/F1 | ARD 23-col P/R/F1 | ARD 19-col P/R/F1 |
|---|---|---|---|
| `previous_notes` | 0.894 / 0.800 / 0.844 | 0.865 / 0.699 / 0.772 | 0.846 / 0.708 / 0.770 |
| `psa_trend` | 0.847 / 0.619 / 0.711 | 0.846 / 0.709 / 0.769 | 0.852 / 0.749 / 0.795 |
| `radiology_report` | 1.000 / 0.212 / 0.349 | 0.972 / **0.359** / **0.521** | 0.984 / 0.282 / 0.435 |
| `laboratory_results` | 0.439 / 0.159 / 0.220 | 0.518 / 0.246 / 0.334 | 0.473 / 0.232 / 0.306 |

`radiology_report` badly under-predicted under the scalar backbone despite 96.7% prevalence
(precision 1.000 but recall only 0.212, F1 0.349) — the occlusion-entropy classifier was
essentially only catching the most obvious cases. Under ARD on the 23-column frame, recall nearly
doubles (0.212→0.359, F1 0.349→0.521) while precision stays almost perfectly intact
(1.000→0.972). This is consistent with `mri_pca_1` (the column most associated with radiology
signal on the 23-column frame) receiving the single smallest `σⱼ` of any column in the entire
frame (§7) — the ARD kernel is more sensitive to exactly the dimension this section's occlusion
probe depends on. `laboratory_results` stays the weakest section under both backbones, on both
frames — this looks like a property of that section's underlying features (`cspca`/`vol`, both
consistently low-relevance per §7 and §5), not something either backbone variant fixes.

The 19-column frame produces the project's first reveal-sequence result on that frame at all
(`exp_6`/`exp_7` never ran reveal), giving no direct prior comparison — but its numbers sit close
to, and for `psa_trend` above, the 23-column results, suggesting reveal-sequence performance here
is not strongly frame-size-dependent the way decision and confidence are.

## 7. Importance comparison: weak agreement with `exp_5`'s solved/unsolved split

Fitting the ARD backbone once per frame on the full 91-case set and ranking factors by mean
`1/σⱼ` across each factor's column group:

| Frame | Top-5 ARD relevance (highest to lowest) | Agreement with `exp_5`'s solved set `{age, bx, dre, pirads, psa}` |
|---|---|---|
| 19-col | age, vol, psad, pirads, cspca | **2/5** (age, pirads) |
| 23-col | age, vol, pirads, cspca, psad | **2/5** (age, pirads) |

Both frames rank identically at the top: `age` is the single most relevant factor by ARD's own
learned `σⱼ` on both frames — the one clean, frame-independent agreement with `exp_5`. But `psa`,
one of `exp_5`'s most confidently solved factors, ranks **last or near-last** on both frames
(19-col: rank 7/9, `1/σⱼ`=0.7066; 23-col: rank 9/9 — dead last, `1/σⱼ`=0.5079) — a genuine,
repeated disagreement, not noise. `bx` and `dre`, also `exp_5`-solved, land in the middle of the
ranking on both frames rather than the top 5.

This does **not** corroborate `DESIGN.md` §6's third ablation hypothesis. The most defensible
reading is that ARD's `σⱼ` reflects **decision-relevance under this specific KDM's kernel
geometry** — how much the model's own distance-based classifier needs to attend to a dimension to
separate `yes`/`no` cases — which is a different quantity from `exp_5`'s SVM-based per-*factor*
weight, itself trained against a different target (the human-annotated importance label, not the
decision itself). `psa` may simply be more useful for a linear/SVM decision boundary than for a
distance-based kernel method on this particular 19–23-dimensional embedding, or its information
may already be substantially redundant with `psad`/`psav`/`psap`/`cli_isup` in a way that a kernel
model can exploit but a per-factor SVM (trained one factor at a time) cannot see. Either way, this
is worth stating plainly as a non-corroboration rather than reading the one point of agreement
(`age`) as validating the whole comparison.

## 8. Interpretation

Mapping onto `DESIGN.md` §8's decision-rule branches: the outcome doesn't cleanly match any single
branch as written. It is closest to the first ("ARD clearly beats scalar on 23-col") for the
*decision* and *reveal* subtasks specifically, where the 23-column improvement is real,
held-out-confirmed (decision) or dual-metric-confirmed (reveal). But it does not clear that bar
project-wide: confidence is a mixed bag (one signal decisively rescued, one made worse), and
weights get uniformly worse under ARD, losing the one narrow win the project had. The 19-column
frame's ARD result is close to the "no clear win" branch for decision (CV gain reversed on
held-out) even though its `blend` confidence macro-F1 is the best seen in the project so far.

Two findings matter beyond this experiment's own scorecard:

1. **The motivating mechanism is real and precisely located.** `dispersion_isotonic`'s
   regression was specifically caused by the shared-bandwidth dilution effect `DESIGN.md` §1
   described, and ARD fixes specifically that — not a general "ARD makes everything better"
   result, but a targeted confirmation of a targeted hypothesis, which is the more useful kind of
   evidence.
2. **CV/held-out disagreement is now a load-bearing finding of its own**, observed in three
   consecutive architecture experiments (`exp_7`, `exp_8`, `exp_9`) in different directions each
   time. Any future experiment claiming a decision improvement from CV alone should be treated
   with real skepticism until this project either grows its held-out set past 19 cases or adopts
   a nested/repeated held-out protocol — a concrete `exp_10`+-level methodology question, separate
   from any specific architecture choice.

## 9. Recommendation

- **Do not adopt ARD as the new default backbone project-wide.** It has a real, held-out-confirmed
  win on the 23-column frame for decision and reveal, but a real held-out *loss* on the
  19-column frame for decision, and a net loss for weights on both frames. "ARD fixes everything"
  is not the finding; "ARD fixes the specific dilution mechanism it targeted, on the frame that
  motivated it, at some cost elsewhere" is.
- **The 23-column-frame decision result is worth a second independent replication** (a different
  held-out split, or a nested CV protocol) before treating it as settled, given this project's now
  three-for-three record of CV/held-out disagreement on architecture changes.
- **`weights_kdm_occlusion`'s narrow baseline win should be considered fragile, not lost to ARD
  specifically** — it was already shrinking from `exp_6` (0.405) to `exp_8` (0.412) before ARD's
  larger jump to ~0.458; worth checking whether it survives a repeat of `exp_6`'s exact original
  scalar-`σ` 19-column condition run again today, to rule out an unrelated environment/seed drift
  before attributing the loss entirely to the architecture change.
- **Do not replace `confidence_svm` or `weights_svm` with any KDM mechanism** — unchanged from
  every prior KDM experiment's recommendation.
- **A natural `exp_10`**: an ARD-aware Signal E (`kernel_distance_contribution`'s per-dimension
  formula rescaled by the trained `σⱼ`, deliberately deferred here per `DESIGN.md` finding #4) and
  a small hyperparameter search *around* ARD specifically (deliberately out of scope this round
  per §1's guardrail), now that a stable ARD baseline exists to search around.
