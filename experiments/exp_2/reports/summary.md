# Experiment Report: Official-Schema Feature Scope + Comorbidity Grouping + KDM Decision Model for Task 1
**Experiment**: experiments/exp_2/
**Project**: challenge_chimera_2
**Report date**: 2026-08-08
**Plan date**: 2026-08-08
**Author**: TBD
**Status**: Complete

---

## ⚠️ Erratum (2026-08-10)

All **decision** numbers in this report were originally computed on a corrupted target vector —
see `experiments/exp_1/reports/summary.md`'s erratum for the root cause
(`target_biopsy_decision` is `NaN`, not "no", for 104/195 cases; the old code silently coded
those as `y=0`). **The corrected decision numbers substantially change this report's headline
finding**: `decision_logistic_count` no longer beats the naive baseline (the corrected baseline,
computed on the correct N=91, is much higher — 0.762, not 0.446), and **KDM is now the best
decision model in this experiment**, reversing the original "KDM underperforms logistic" verdict
and retiring the "`decision_kdm_flags` looks unstable" caveat, which was itself a downstream
artifact of the same bug. Confidence and weights results were unaffected (already correctly
filtered to N=91). §§1–2, 5.1–5.3, 7–9 below are corrected; §§3–4, 6, 10 are unchanged from the
original run.

---

## 1. Summary

Three changes were tested against `exp_1`'s 47-column baseline, each isolated via a fully-crossed
16-condition ablation: restricting features to the 11 officially documented Task-1 input
variables, replacing the comorbidity count with 6 grouped binary flags, and adding a KDM
classifier for decision and confidence. **The results are genuinely mixed, not a clean win**:
*(numbers corrected 2026-08-10 — see erratum above)* none of the decision conditions beat the
corrected naive baseline (0.762), though `decision_kdm_flags` (F1 0.723) comes closest and is a
real improvement over `exp_1`'s comparable no-MRI result (0.658) — **KDM turns out to be the best
decision model tested in this experiment**, not the worst as originally reported. Schema-
restriction made confidence prediction *worse* across the board (0.657–0.673 vs. `exp_1`'s
0.576) — that finding is unaffected by the correction. Per-factor feature restriction for the
variable-weight models backfired — the opposite of what was hypothesized — and reveal-sequence
remained solidly good regardless of any change tested.

---

## 2. Hypothesis & Verdict

**Hypothesis 1 (primary — weights):** restricting to the 11 official variables + grouped
comorbidity flags improves the variable-weight models enough to beat `exp_1`'s naive baseline.

**Verdict:** ❌ Refuted. Best weights result (`weights_official_flags`, mean ordinal error 0.585)
is worse than `exp_1`'s own 47-column result (0.574), and both remain well short of the naive
per-factor baseline (0.401–0.413 depending on factor count). Per-factor restriction made it
markedly worse still (0.711–0.720).

**Hypothesis 2 (secondary — schema-fidelity risk for decision):** dropping non-official columns
(`vit_*`/`lab_*`/`psa_tr_*`/`path_hist_*`) should not hurt, and may help, decision F1/AUC.

**Verdict (⚠️ corrected 2026-08-10):** ⚠️ Partially supported. `decision_logistic_count` (F1
0.678, ROC-AUC 0.69) does beat `exp_1`'s comparable no-MRI logistic result (0.658) — a modest,
genuine +0.020 improvement, same direction as originally reported — but **it no longer beats the
naive baseline**, which corrected to 0.762 (N=91), far above the originally-reported 0.446
(N=195, itself an artifact of the bug). No decision condition in this experiment beats baseline.

**Hypothesis 3 (KDM for decision):** a memory-based KDM classifier gives a better-calibrated
decision boundary than logistic/HGB, following the "real but modest" improvement direction KDM
showed for confidence in `exp_1`.

**Verdict (⚠️ reversed 2026-08-10):** ✅ Supported — the opposite of the original finding.
`decision_kdm_flags` (F1 0.723) and `decision_kdm_count` (F1 0.710) both clearly beat their
matching-comorbidity-treatment logistic (0.665/0.678) and HGB (0.652/0.655) conditions. **KDM is
the best decision model tested in this experiment.** The original "`decision_kdm_flags` looks
unstable, below-chance ROC-AUC" caveat is retired — that was itself a downstream artifact of the
same data bug, not a real KDM/comorbidity-flags interaction.

**Unplanned finding (confidence, schema-fidelity):** unlike decision, restricting confidence's
features to the 11 official variables made it *worse* on every condition (logistic 0.657–0.673,
KDM 0.578–0.598) versus `exp_1`'s 47-column numbers (logistic 0.576, KDM 0.564) — confidence
prediction appears to depend on some of the excluded vitals/labs/PSA-trend signal that decision
doesn't need.

---

## 3. Experimental Setup (as run)

Built exactly as `experiments/exp_2/IMPLEMENTATION.md` describes — no deviations. `dre` encoded
ordinal by clinical severity with `Not done` as a separate flag; `bx` encoded as two binary flags
(`bx_positive`, `bx_missing`); `pmhx`/comorbidity handled via the pre-existing `comorbidity_flags()`
for the "flags" treatment or `cli_comorbidity_count` for "count"; `fh` excluded from the
11-variable restriction, kept on its own separate source per the design's explicit decision.
`ct` excluded (confirmed absent from the data in an earlier investigation this session).

- **Dataset**: same as `exp_1` — `data/inputs.csv` + `data/ground_truth.csv`, 91 annotated cases
  for confidence/weights/reveal, 195 for decision.
- **Models**: logistic regression (`class_weight="balanced", C=0.5`) and HistGradientBoosting
  (regularized, same hyperparameters as `exp_1`) for decision; `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`
  for confidence/weights/reveal (unchanged from `exp_1`); KDM (memory-based, `x_train=y_train=w_train=False`,
  only kernel bandwidth `sigma` trained) for decision and confidence.
- **Hardware**: CPU only (`torch==2.13.0+cpu`), same as `exp_1`.
- **Deviations from plan**: none. All four runner scripts (`run_decision.py`, `run_confidence.py`,
  `run_reveal.py`, `run_weights.py`) implement exactly what `IMPLEMENTATION.md` specified, reusing
  `chimera_task1.{train_decision,train_reasoning,train_confidence_kdm}` without modification.

---

## 4. Code Version

| Condition | Git commit | Commit message |
|-----------|-----------|-----------------|
| all | _N/A_ | This project is not a git repository — same caveat as `exp_1`. |

---

## 5. Results

### 5.1 Primary Metric

**Decision (F1, higher=better) — ⚠️ table corrected 2026-08-10, N=91 not 195:**

| Condition | F1 | ROC-AUC | vs. exp_1 comparable (0.658) |
|---|---|---|---|
| Naive "always yes" baseline (N=91) | 0.762 | — | — |
| `exp_1` logistic, no MRI (47 cols) | 0.658 | 0.673 | — |
| **`decision_kdm_flags`** | **0.723** | **0.688** | **+0.065** (best, still < baseline) |
| `decision_kdm_count` | 0.710 | 0.708 | +0.052 |
| `decision_logistic_count` | 0.678 | 0.690 | +0.020 |
| `decision_logistic_flags` | 0.665 | 0.678 | +0.007 |
| `decision_hgb_count` | 0.655 | 0.668 | −0.003 |
| `decision_hgb_flags` | 0.652 | 0.664 | −0.006 |

Every condition here is below the corrected naive baseline (0.762) — none of the ✅/❌ per-row
calls from the original report apply anymore; see the erratum and §2 for the corrected verdicts.

**Confidence (ordinal distance, lower=better):**

| Condition | Ordinal distance | vs. exp_1 best (0.564) |
|---|---|---|
| Naive "always clear" baseline | 0.527 | — |
| `exp_1` KDM (47 cols) | 0.564 | — |
| `exp_1` logistic (47 cols) | 0.576 | — |
| `confidence_kdm_count` | 0.578 | +0.014 (worse) |
| `confidence_kdm_flags` | 0.598 | +0.034 (worse) |
| `confidence_logistic_flags` | 0.657 | +0.081 (worse) |
| `confidence_logistic_count` | 0.673 | +0.097 (worse) |

**Variable-weights (mean ordinal error, lower=better; 9 in-scope factors, `fh` excluded):**

| Condition | Ordinal error | Decisive-set F1 | vs. exp_1 (0.574 / 0.451, 10-factor) |
|---|---|---|---|
| Naive baseline (9-factor) | 0.413 | 0.379 | — |
| `exp_1` logistic (47 cols, 10-factor) | 0.574 | 0.451 | — |
| `weights_official_flags` | 0.585 | 0.546 | +0.011 error, +0.095 F1 |
| `weights_official_count` | 0.600 | 0.540 | +0.026 error, +0.089 F1 |
| `weights_restricted_count` | 0.711 | 0.512 | +0.137 error (much worse) |
| `weights_restricted_flags` | 0.720 | 0.519 | +0.146 error (much worse) |

**Reveal-sequence (set precision, higher=better):**

| Condition | Set precision | vs. exp_1 (0.840) |
|---|---|---|
| Naive baseline | 0.783 | — |
| `reveal_flags` | 0.853 | +0.013 |
| `reveal_count` | 0.852 | +0.012 |

### 5.2 Secondary Metrics

- **Decision KDM predictive entropy** ⚠️ *corrected*: 0.25 (`count`) / 0.28 (`flags`), against a
  max of 0.693 for 2 classes — not degenerate, and notably lower (more confident) than
  confidence's KDM entropy (0.371–0.411 of a max 1.099), consistent with a cleaner 2-class
  problem. This entropy pattern was actually correct pre-fix too (0.255/0.329) — it was the F1/
  ROC-AUC numbers built on top of the corrupted labels that were wrong, not the entropy
  diagnostic itself.
- ⚠️ *Retired caveat*: the original report flagged `decision_kdm_flags`'s ROC-AUC (then 0.483,
  below chance) as a likely instability. Post-correction, `decision_kdm_flags`'s ROC-AUC is 0.688
  and it's the best decision condition in this experiment — that instability was a downstream
  artifact of the corrupted labels, not a real KDM/comorbidity-flags interaction.

### 5.3 Ablation Results

**Comorbidity treatment (count vs. flags), by target:**

| Target | Winner | Margin |
|---|---|---|
| Decision ⚠️ *corrected* | mixed | logistic +0.013 F1 (count), HGB +0.003 F1 (count), KDM +0.013 F1 (**flags**) — no longer a clean count sweep |
| Confidence (both classifiers) | flags | logistic +0.016, KDM −0.020 (mixed: better ordinal distance for logistic, worse for KDM) |
| Weights, official scope | flags | +0.015 ordinal error, +0.006 decisive-F1 |
| Weights, restricted scope | count | +0.009 ordinal error (flags worse here) |
| Reveal | flags (barely) | +0.001 set precision — negligible |

No consistent winner across targets, and (post-correction) not even a consistent winner *within*
decision across its three classifiers — flags mostly help confidence and official-scope weights,
count edges out flags for logistic/HGB decision, but flags wins for KDM decision specifically.
This directly answers the question that motivated expanding to 16 conditions: the
comorbidity-treatment choice is target- *and model-* dependent, not universal.

**Feature scope (official vs. restricted), weights only:**

Restricted lost to official on both comorbidity treatments (0.711 vs. 0.600 for count; 0.720 vs.
0.585 for flags) — the opposite of the hypothesized direction. Restricting each factor's
classifier to only its own column(s) removed cross-factor information that the models were
apparently using (e.g., a factor's rated importance may correlate with the *overall* clinical
picture, not just its own raw value).

### 5.4 Learning Curves

Not applicable — cross-validated classical/KDM models, no iterative learning curves. No figures
generated (`reports/figures/` empty), consistent with `exp_1`.

---

## 6. Statistical Analysis

- **Test used**: none — same limitation as `exp_1`. Repeated CV gives mean ± std per condition,
  but per-repeat values weren't persisted to `results/`, so no formal significance test.
- **Given that**, the clearest way to read these results is the *consistency of direction* across
  related conditions rather than any single point estimate: schema-restriction hurting confidence
  holds across both comorbidity treatments, which is stronger evidence than any one
  ordinal-distance number in isolation. Decision's story post-correction is less uniform — KDM
  beats logistic/HGB in both comorbidity treatments (consistent), but which comorbidity
  treatment wins now differs *by classifier* (§5.3), which is itself a less consistent, harder-
  to-trust-in-isolation pattern given no significance test backs it up.
- ⚠️ *Corrected*: decision F1 stds are now 0.10–0.11 for logistic/HGB (up from the
  pre-correction 0.024–0.112 range, similar order of magnitude) but much tighter for KDM
  (0.021–0.030) — KDM's predictions are notably more stable across CV repeats than the other two
  model families on this target, on top of also having the best mean.

---

## 7. Comparison to Expected Results

| Expected (from DESIGN.md §8) | Observed | Match? |
|---|---|---|
| Weights condition clearly beats naive baseline → resume toward steps 5-8 | No weights condition beat baseline; restriction made it worse | ❌ |
| `decision_official` matches/beats exp_1 → schema-fidelity is free | ⚠️ *corrected*: `decision_logistic_count` beats exp_1's comparable result (+0.020) but not the (corrected, much higher) naive baseline | ⚠️ Partial |
| KDM beats logistic/HGB and baseline on decision → adopt KDM | ⚠️ *corrected*: KDM beats logistic/HGB in both comorbidity treatments; still doesn't beat the corrected baseline | ⚠️ Partial (reversed from ❌ pre-correction) |
| All conditions ≤ baseline everywhere → prioritize free-text/LLM-agent path over more tabular work | ⚠️ *corrected*: now true for **all four** targets, including decision (post-correction, no decision condition beats baseline either) | ✅ (stronger match than originally reported) |

---

## 8. Missing Data & Caveats

- All 16 planned conditions ran to completion — no missing runs.
- No `IMPLEMENTATION.md` deviations to report (§3) — built exactly as planned.
- **See the erratum at the top of this report** — all decision numbers were corrected on
  2026-08-10 after discovering `target_biopsy_decision` was NaN (not "no") for 104/195 cases.
  The original "`decision_kdm_flags` looks unstable" caveat is retired: that ROC-AUC was
  below-chance specifically *because* of the corrupted labels, not a real KDM/comorbidity-flags
  interaction — post-correction it's the best decision condition in the experiment.
- No formal significance testing (§6) — same structural limitation as `exp_1`, not fixed here.
- Comorbidity-treatment "winners" in §5.3 are small margins relative to CV noise across every
  target now, decision included post-correction (no longer a large, consistent margin there).

---

## 9. Conclusions & Next Steps

- **What this experiment established** (⚠️ revised 2026-08-10): schema-fidelity (restricting to
  the 11 officially documented variables) gives a modest genuine lift to decision (+0.020 vs.
  `exp_1`'s comparable result) but hurts confidence and is neutral-to-slightly-negative for
  weights — not the clean "helped decision, hurt confidence" story originally reported, since
  decision no longer beats baseline either. Per-factor feature restriction for weight models
  remains actively counterproductive, unaffected by the correction. **KDM *does* generalize its
  confidence-target benefit to decision** — the opposite of the original conclusion — and is now
  the best-performing decision model in this experiment, with the tightest CV variance of any
  classifier tried on that target.
- **What remains uncertain**: *why* confidence specifically needs the excluded vitals/labs/
  PSA-trend columns while decision doesn't — worth investigating which specific dropped column(s)
  matter for confidence before concluding the schema-restriction trade-off is fixed. Also open:
  *why* KDM specifically outperforms logistic/HGB on decision but not on confidence relative to
  its own naive baseline (KDM comes close to confidence's baseline but not decision's, even
  though it's the best *relative* performer on decision) — not explained by this experiment.
- **Recommended follow-up experiments** (⚠️ revised 2026-08-10; became `exp_3` in practice):
  1. Isolate which non-official column(s) confidence depends on (add them back one group at a
     time — vitals, labs, PSA-trend — to the 11-variable frame and re-test confidence only).
  2. ~~Investigate the `decision_kdm_flags` instability~~ — resolved by the bugfix; no longer
     applicable.
  3. ⚠️ *No longer applicable as stated* — decision does not have a baseline-beating result
     post-correction after all. The broader point stands in a weaker form: KDM is now the
     strongest decision model found across `exp_1`/`exp_2`, worth carrying forward as a
     candidate even though it hasn't yet cleared baseline.
  4. The free-text/LLM-agent alternatives remain on record as options for confidence/weights —
     and now decision too, given no tabular approach has beaten its (corrected, much higher)
     baseline across either experiment.

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Seeds logged | ✅ (`RANDOM_STATE = 0`, matching `exp_1`) |
| Configs versioned | ⚠️ inline constants, not separate config files (same as `exp_1`) |
| Git commits recorded | ❌ not a git repository |
| Checkpoints saved | ❌ N/A, no persisted model artifacts |
| Environment frozen | ⚠️ recorded in prose (`exp_1`'s DESIGN.md §10), no `requirements.txt`/`environment.yml` committed |
| Experiment tracker linked | ❌ not used |
