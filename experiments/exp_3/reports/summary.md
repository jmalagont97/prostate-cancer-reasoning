# Experiment Report: Broader Model Family Comparison + MRI-PCA + Decorrelated PSA Family for Task 1
**Experiment**: experiments/exp_3/
**Project**: challenge_chimera_2
**Report date**: 2026-08-09
**Plan date**: 2026-08-09
**Author**: TBD
**Status**: Complete

---

## ⚠️ Erratum (2026-08-10)

All **decision** numbers in this report were originally computed on a corrupted target vector —
see `experiments/exp_1/reports/summary.md`'s erratum for the root cause
(`target_biopsy_decision` is `NaN`, not "no", for 104/195 cases; the old code silently coded
those as `y=0`, and additionally used the wrong 195-case pool for MRI-PCA alignment purposes).
**This retires the original headline finding that `decision_extratrees` was this experiment's
best result and that tree ensembles beat margin/distance-based methods on decision while the
opposite held for confidence.** Post-correction, **kNN and SVM are now the best decision models**
(not Extra Trees), Extra Trees drops to mid-pack, Naive Bayes collapses to the worst result in
the experiment, and the "opposite pattern by target" story is gone — margin/distance-based
methods (SVM, KDM, kNN) now do comparatively well on *both* decision and confidence, a more
coherent finding than what was originally reported. Confidence, weights, and reveal were
unaffected (already correctly filtered to N=91). §§1–2, 5.1–5.3, 7–9 below are corrected.

---

## 1. Summary

An 8-model search (SVM, Random Forest, XGBoost, Extra Trees, MLP, Gaussian Naive Bayes, kNN,
KDM) on a refined 19-column feature set — PSA family reduced to `psa`+`psad`, comorbidity fixed
to grouped flags, and a 2-component MRI-embedding PCA added — produced **the best confidence
result across all three experiments**, and a decision result that, while much closer to baseline
than originally reported, still doesn't clear it. *(Numbers corrected 2026-08-10 — see erratum
above.)* **`confidence_svm` (ordinal distance 0.468) is the first confidence result in any
experiment to clearly beat the naive baseline (0.527)** — this finding is unaffected by the
correction. For decision, `decision_knn` (F1 0.733) and `decision_svm` (F1 0.730) are now the
top two models, with `decision_kdm` close behind (0.709); none beat the corrected baseline
(0.762). Weights remained stuck below baseline regardless of feature scope (replicating `exp_2`'s
finding that per-factor restriction hurts). Model choice still mattered enormously across
targets, but the post-correction picture is that margin/distance-based methods (SVM, kNN, KDM) do
comparatively well on *both* decision and confidence, while tree ensembles and Naive Bayes lag on
both — not the "opposite pattern per target" originally reported.

---

## 2. Hypothesis & Verdict

**Hypothesis**: a broader model search + refined features finds at least one model/target
combination clearly beating both the naive baseline and the prior best for that target,
particularly for confidence and weights (neither had beaten baseline in `exp_1`/`exp_2`).

**Verdict (⚠️ decision corrected 2026-08-10):** ⚠️ Partially supported for decision / ✅ Supported
for confidence / ❌ Refuted for weights.

**Evidence:**
- **Decision** ⚠️ *corrected*: `decision_knn` (F1 0.733, ROC-AUC 0.689) and `decision_svm`
  (F1 0.730) both beat `exp_2`'s incumbent best (`decision_kdm_flags`, 0.723) by a small margin,
  but **none of the 8 models beat the corrected naive baseline (0.762)** — the original claim
  that decision had a baseline-beating result no longer holds.
- **Confidence** (unaffected by the correction): `confidence_svm` ordinal distance 0.468 clearly
  beats both the naive baseline (0.527) and every prior result (`exp_1`'s best was 0.564) — the
  first time this target has beaten baseline at all, not just improved on the prior best.
- **Weights** (unaffected): `weights_official` (0.614 mean ordinal error) is *worse* than
  `exp_2`'s best (0.585), and `weights_restricted` (0.720) again underperforms official scope,
  replicating `exp_2`'s finding rather than reversing it. Neither condition approaches the naive
  baseline (0.413).
- **Reveal** (unaffected) wasn't part of the model-search hypothesis (fixed to logistic
  regression); its single new-feature-set result (0.833) is roughly flat vs. `exp_2` (0.853),
  slightly down but still clearly ahead of baseline (0.783).

---

## 3. Experimental Setup (as run)

Built exactly as `experiments/exp_3/IMPLEMENTATION.md` describes — no deviations. Feature
scaling (`StandardScaler`) applied uniformly across all 8 models per the plan's resolution of
that open question. Class-imbalance handling varied by model capability as planned:
`class_weight="balanced"` for SVM/RF/Extra Trees, `sample_weight` (via
`compute_sample_weight("balanced", ...)`) for XGBoost/Naive Bayes, and no rebalancing for
MLP/kNN/KDM — recorded explicitly per condition's `metrics.json` rather than left implicit.

- **Dataset**: same as `exp_1`/`exp_2` — 91 annotated cases (confidence/weights/reveal), 195
  (decision).
- **Feature frame**: 19 columns — `select_exp3_feature_frame()`, i.e. `exp_2`'s official-flags
  frame minus `cli_psap`/`cli_psav`, plus `mri_pca_0`/`mri_pca_1`/`mri_missing`.
- **MRI-PCA correctness note**: for confidence/weights/reveal (91-case subset), the PCA was
  fit on the *full* 195-case embedding population and then aligned to the 91 annotated cases by
  `case_id` — not fit on the 91-case subset directly, and not joined by raw row position (an
  early draft of `run_confidence.py` would have silently misaligned this; caught and fixed during
  implementation, before any results were generated).
- **Hardware**: CPU only, same as prior experiments; `xgboost==3.4.0` newly installed.
- **Deviations from plan**: none substantive. The MRI-alignment fix above was corrected during
  implementation, before the reported run — not a deviation from the final `IMPLEMENTATION.md`,
  which already reflected the corrected approach by the time results were generated.

---

## 4. Code Version

| Condition | Git commit | Commit message |
|-----------|-----------|-----------------|
| all | _N/A_ | Not a git repository — same caveat as `exp_1`/`exp_2`. |

---

## 5. Results

### 5.1 Primary Metric

**Decision (F1, higher=better; naive baseline = 0.762, N=91) — ⚠️ table corrected 2026-08-10:**

| Condition | F1 | ROC-AUC | vs. exp_2 best (0.723) |
|---|---|---|---|
| **`decision_knn`** | **0.733** | 0.689 | **+0.010** (best, still < baseline) |
| `decision_svm` | 0.730 | 0.675 | +0.007 |
| `decision_kdm` | 0.709 | 0.639 | −0.014 |
| `decision_extratrees` | 0.665 | 0.667 | −0.058 |
| `decision_xgb` | 0.644 | 0.614 | −0.079 |
| `decision_rf` | 0.618 | 0.603 | −0.105 |
| `decision_mlp` | 0.383 | 0.486 | −0.340 |
| `decision_nb` | 0.078 | 0.636 | −0.645 (worst result in the experiment) |

**Confidence (ordinal distance, lower=better; naive baseline = 0.527):**

| Condition | Ordinal distance | vs. exp_1 best (0.564) |
|---|---|---|
| **`confidence_svm`** | **0.468** | **−0.096 (better)** ✅ |
| `confidence_kdm` | 0.530 | −0.034 (better) |
| `confidence_knn` | 0.558 | −0.006 (better) |
| `confidence_xgb` | 0.678 | +0.114 (worse) |
| `confidence_rf` | 0.748 | +0.184 (worse) |
| `confidence_nb` | 0.831 | +0.267 (worse) |
| `confidence_extratrees` | 0.778 | +0.214 (worse) |
| `confidence_mlp` | 0.856 | +0.292 (worse) |

**Variable-weights (mean ordinal error / mean decisive-set F1; naive baseline = 0.413 / 0.379):**

| Condition | Ordinal error | Decisive-set F1 | vs. exp_2 best (0.585 / 0.546) |
|---|---|---|---|
| `weights_official` | 0.614 | 0.526 | +0.029 error (worse), −0.020 F1 (worse) |
| `weights_restricted` | 0.720 | 0.519 | +0.135 error (much worse) |

**Reveal-sequence (set precision, higher=better; naive baseline = 0.783):**

| Condition | Set precision | vs. exp_2 best (0.853) |
|---|---|---|
| `reveal` | 0.833 | −0.020 (slightly worse, still clearly beats baseline) |

### 5.2 Secondary Metrics

- **`decision_kdm` predictive entropy** ⚠️ *corrected*: 0.317 of max 0.693 — similar range to
  `exp_2`'s KDM decision conditions (0.25–0.28), not degenerate.
- **`confidence_kdm` predictive entropy**: 0.402 of max 1.099 — comparable to `exp_1`/`exp_2`'s
  KDM confidence entropy (0.371–0.411), not degenerate. (Unaffected by the correction.)
- ⚠️ *Retired*: the original report flagged `decision_svm`'s F1 (then 0.101, despite decent
  ROC-AUC) as a likely threshold/calibration artifact needing investigation. Post-correction,
  `decision_svm`'s F1 is 0.730 — the apparent collapse was entirely a symptom of the corrupted
  labels, not a genuine SVM calibration problem.

### 5.3 Ablation Results

**Model family, decision (8-way)** ⚠️ *corrected — this finding reversed*: margin/distance-based
methods now win: kNN (0.733), SVM (0.730), and KDM (0.709) are the top three, with the tree
ensembles (Extra Trees 0.665, XGBoost 0.644, Random Forest 0.618) trailing behind, and Naive
Bayes collapsing entirely (0.078 — likely a Gaussian-assumption mismatch on the mixed
continuous/binary 19-column frame, not investigated further here).

**Model family, confidence (8-way, unaffected by the correction):** SVM and KDM are again the two
best models, same as originally reported.

**Combined, the corrected picture is materially different from what was originally reported**:
margin/distance-based methods (SVM, KDM, and now kNN too) do comparatively well on *both*
decision and confidence, while tree ensembles and MLP lag on both. This is a more coherent story
than the original "opposite pattern by target" finding, which was itself an artifact of the
decision-side corruption — there was never a genuine target-dependent inductive-bias asymmetry to
explain, just a data bug that happened to scramble decision's ranking in a way that looked like a
meaningful reversal.

**Feature scope, weights (official vs. restricted):** restricted again loses to official
(0.720 vs. 0.614 ordinal error) — the same direction as `exp_2` (0.711–0.720 vs. 0.585–0.600),
now confirmed with a different feature set. This looks like a settled finding, not an artifact
of `exp_2`'s particular feature choices.

### 5.4 Learning Curves

Not applicable — cross-validated classical/kernel/tree models, no iterative learning curves to
plot for most; MLP/KDM use gradient descent internally but per-epoch loss wasn't logged. No
figures generated (`reports/figures/` empty), consistent with prior experiments.

---

## 6. Statistical Analysis

- **Test used**: none — same structural limitation as `exp_1`/`exp_2`. Per-repeat values weren't
  persisted, only aggregates.
- **The 8-way comparison specifically increases false-winner risk** (flagged in `DESIGN.md` §9
  before running) — mitigated partially by reporting the full ranked table rather than only the
  argmax (§5.1), so a close second isn't hidden. For decision ⚠️ *corrected*, kNN's margin over
  SVM (0.733 vs. 0.730, a 0.003 gap) and over KDM (0.733 vs. 0.709, a 0.024 gap) are both small
  relative to those models' fold stds (0.027, 0.015, 0.033 respectively) — the top three should
  be read as "kNN, SVM, and KDM are all genuinely competitive here," not "kNN is definitively the
  winner." For confidence, SVM's margin over KDM (0.468 vs. 0.530, a 0.062 gap) is more modest
  relative to those two models' individual stds (0.022 and 0.043) and should be read the same way
  — both good, SVM somewhat ahead, not a settled single winner. (This caution applies with even
  more force now than originally written, since decision's apparent clear winner pre-correction
  turned out to be substantially a data artifact.)

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md §8) | Observed | Match? |
|---|---|---|
| Any decision/confidence model clearly beats baseline + prior best → candidate for steps 5-8 | ⚠️ *corrected*: true for confidence (`confidence_svm`) only; no decision model beats the corrected baseline | ⚠️ Partial (was ✅ for both, pre-correction) |
| `weights_restricted` again underperforms `weights_official`, replicating exp_2 → settled negative | Confirmed again (0.720 vs. 0.614) | ✅ (as a "settled negative" finding) |
| Confidence + weights remain below baseline everywhere → LLM-agent path more justified for those two | ⚠️ *corrected*: true for weights (unaffected); no longer true for confidence (`confidence_svm` beat baseline); **now also true for decision again** post-correction | ⚠️ Partial — confidence is the one target that's genuinely broken through |

---

## 8. Missing Data & Caveats

- All 19 planned conditions ran to completion — no missing runs.
- **See the erratum at the top of this report** — all decision numbers were corrected on
  2026-08-10. The original "`decision_svm`'s F1 looks like a threshold/calibration artifact"
  caveat is retired: SVM's F1 is 0.730 post-correction, essentially tied for best — there was no
  real calibration problem, only corrupted labels.
- **No formal significance testing** (§6) — same limitation as `exp_1`/`exp_2`, not fixed here.
- **Hyperparameters for all 8 models were hand-chosen for N=91–195, not tuned via a validation
  search** — explicitly out of scope per `IMPLEMENTATION.md`, but means e.g. XGBoost's
  middling result could reflect suboptimal defaults rather than a genuine model-family ceiling.
- **`decision_nb`'s collapse to F1=0.078** is a new, unexplained result post-correction (Naive
  Bayes was a middling performer, F1 0.415, before the fix) — plausibly a Gaussian-assumption
  mismatch with the mixed continuous/binary 19-column frame, but not investigated further here.

---

## 9. Conclusions & Next Steps

- **What this experiment established** (⚠️ revised 2026-08-10): model family choice matters as
  much as, or more than, feature engineering did in `exp_2` — but margin/distance-based methods
  (SVM, kNN, KDM) now look like the consistently strong family across *both* decision and
  confidence, not opposite winners per target as originally concluded. Confidence has, for the
  first time across three experiments, a model that beats naive baseline
  (`confidence_svm`) — this is the one genuinely new, unaffected result from this experiment.
  Decision's best result (`decision_knn`, 0.733) is only a small improvement over `exp_2`'s KDM
  result (0.723), not the "substantial jump" originally reported, and still falls short of
  baseline. Weights remains resistant to every approach tried across all three experiments
  (feature scope, comorbidity treatment, model family) — a well-replicated negative result.
- **What remains uncertain**: why margin/distance-based methods do comparatively well on both
  targets while tree ensembles lag on both — not explained by this experiment's design (which
  tested *whether* a model-family pattern exists, not *why*). Also unclear whether tree ensembles
  or MLP would improve with different (tuned) hyperparameters — this experiment tested one
  hyperparameter setting per model, not a search. `decision_nb`'s collapse (§8) is unexplained.
- **Recommended follow-up experiments** (`exp_4` via `ml-experiment-planner`):
  1. Confidence is the one target with a real, correction-surviving win — this is the strongest
     candidate to revisit the paused steps 5-8 (rubric scorer, MCP+LLM wiring, Docker,
     submission), at least starting with confidence alone rather than assuming decision is ready
     too.
  2. Decision remains close to but short of baseline across all three experiments now
     (`exp_2`'s KDM: 0.723, `exp_3`'s kNN: 0.733, baseline: 0.762) — worth a focused push (e.g.
     ensembling the top 2-3 decision models, or a modest hyperparameter search on kNN/SVM/KDM
     specifically) before concluding tabular ML has hit its ceiling there too.
  3. A small hyperparameter search (not full tuning, but a modest grid) for the strongest models
     per target (kNN/SVM/KDM for both decision and confidence) before treating these numbers as
     final, given §6's caution about reading an 8-way comparison's winner too confidently.
  4. This experiment is a reminder to build a data-integrity check (e.g. assert expected N and
     label non-null counts at load time) into the shared loading code, given how far a silent
     `NaN == "yes" → False` bug propagated across two full experiments before being caught.

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Seeds logged | ✅ (`RANDOM_STATE = 0`, consistent across all scripts) |
| Configs versioned | ⚠️ inline constants (`models.py`), not separate config files |
| Git commits recorded | ❌ not a git repository |
| Checkpoints saved | ❌ N/A, no persisted model artifacts |
| Environment frozen | ⚠️ recorded in prose only; `xgboost==3.4.0` added this experiment |
| Experiment tracker linked | ❌ not used |
