# Experiment Report: Bayesian Evidence-Combination Fusion — Log-Linear Pooling and a Non-Parametric KDM Reliability Gate (exp_31)
**Experiment**: experiments/fagonzalezo/exp_31/
**Project**: pathology-reasoning
**Report date**: 2026-08-20
**Plan date**: 2026-08-20
**Author**: Fabio A. González
**Status**: Complete

---

## 1. Summary

Holding exp_29's frozen per-modality experts fixed, this experiment tested two alternatives to
arithmetic soft-voting fusion: log-linear (geometric-mean) pooling (C2) and a non-parametric,
per-patient KDM reliability gate (C3). Neither beat the matched-input arithmetic-pooling reference
(C1, LOOCV Macro-F1 0.7068): C2 scored 0.6914 (Δ −0.0154, McNemar p=1.0) and C3 scored 0.6111
(Δ −0.0957, McNemar p=0.057) — both changes moved performance in the *wrong* direction, and neither
paired difference was statistically significant. C3's per-patient gate weights collapsed to
near-uniform (0.34/0.32/0.34 ± ~0.05) rather than reproducing anything resembling exp_29's fixed
`(0.4, 0.1, 0.5)` triple, suggesting the reliability gate did not learn a useful differential signal
from the 3-D evidence space. Both primary hypotheses (H1, H2) are refuted.

---

## 2. Hypothesis & Verdict

**Hypothesis (from plan):**
- **H1 (primary):** `fusion_loglinear__hard` (C2) exceeds `fusion_arithmetic_smoothed__hard` (C1) in
  LOOCV Macro-F1, with `mcnemar_exact` significance (p < 0.05) between their paired LOOCV predictions.
- **H2 (primary):** `fusion_kdm_gate__hard` (C3) exceeds `fusion_arithmetic_smoothed__hard` (C1) in
  LOOCV Macro-F1, with `mcnemar_exact` significance (p < 0.05) between their paired LOOCV predictions.

**Verdict:** ❌ **H1 Refuted** · ❌ **H2 Refuted**

**Evidence:** C2's Macro-F1 (0.6914) is *lower* than C1's (0.7068), Δ = −0.0154, and the paired McNemar
test is not significant (p = 1.0, b=1, c=2 — nearly symmetric disagreement). C3's Macro-F1 (0.6111) is
also lower than C1's, Δ = −0.0957, and while its McNemar p-value (0.057) is the closest to the 0.05
threshold of the three comparisons, the direction is wrong — C3 loses to C1 more often than it wins
(b=3 vs c=11), so even a significant result would have refuted, not supported, H2. Neither hypothesis's
success condition (exceed C1 *and* reach significance) was met.

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md` and `IMPLEMENTATION.md` — no deviations found. `scripts/train.py` reloads
exp_29's frozen hard-arm per-modality winners (`tab`: `knn_k=3, sigma=2.0873`; `mri`: `knn_k=40,
sigma=0.0601`; `txt`: `rep=(500, 0.90), knn_k=3, sigma=0.3374`) verbatim from
`experiments/fagonzalezo/exp_29/results/stage1_best_hparams.json`, applies `label_smoothing=0.05`
uniformly to all three modalities' `BrentMemKDM` experts (fixed a priori, not searched — DESIGN.md
§2.2), and varies only the fusion layer across three conditions:

- **C1** `fusion_arithmetic_smoothed` — weighted arithmetic mean (`soft_vote`), weights from a
  `simplex_grid(3, 0.05)` search over 100 MCCV splits' Stage-1 validation probabilities (same
  construction as exp_29 S1).
- **C2** `fusion_loglinear` — weighted geometric mean / log-odds pooling (`logit_pool`), same
  weight-search protocol as C1, `logit_pool` substituted for `soft_vote` as the scoring function.
- **C3** `fusion_kdm_gate` — arithmetic pooling (same rule as C1), but with a per-patient weight
  produced by three independent `BrentMemKDM` Nadaraya-Watson estimators over the 3-D evidence space
  `e(x) = [p_tab(x), p_mri(x), p_txt(x)]`, bandwidth fixed at the data-driven default
  (`_sigma_from_knn`, `sigma_mult=1.0`) computed on a deduplicated 88-row (one-per-patient) evidence
  table, memory built leak-free per LOOCV fold (query patient's own rows excluded).

- **Dataset**: `Data/preprocessed_old/task1/` cohort, N=88 complete-case patients (same cohort as
  exp_1–exp_30).
- **Model**: `BrentMemKDM` per modality (unchanged from exp_29), plus experiment-local fusion logic
  (`logit_pool`, `modality_reliability`) implemented inline in `scripts/train.py` — no changes to
  `src/methods/*` or `src/evaluation/*` (matches scope discipline in DESIGN.md §8).
- **Training/evaluation**: Two-phase MCCV (Phase A, 100 splits, weight/gate construction) →
  LOOCV (Phase B, 88 folds, frozen fusion mechanism, per-modality experts refit per fold).
- **Hardware**: Not recorded in results (CPU-only KNN-style method; no GPU dependency for this
  experiment).
- **Deviations from plan**: None.

---

## 4. Code Version

| Condition | Git commit | Commit message |
|-----------|-----------|-----------------|
| All (C1/C2/C3, unimodal) | `2837b9b8c3e10a867b9ea6d4604768eb0e676727` | Re-run exp_29/exp_30 on fagonzalezo_dev; port their transitive result dependencies |

A single `git_commit.txt` covers the whole run (one script, `scripts/train.py`, produces all
conditions in one invocation).

---

## 5. Results

### 5.1 Primary Metric — LOOCV Macro-F1 (Phase B, 88 folds)

| Condition | Macro-F1 | vs. C1 (Δ) | McNemar p (vs. C1) |
|---|---|---|---|
| **C1** `fusion_arithmetic_smoothed` (reference) | 0.7068 | — | — |
| C2 `fusion_loglinear` | 0.6914 | −0.0154 | 1.000 (H1: ❌ not significant) |
| C3 `fusion_kdm_gate` | 0.6111 | −0.0957 | 0.057 (H2: ❌ not significant, and wrong direction) |

> Success threshold from plan (§7): exceed C1 **and** reach McNemar p < 0.05. *Threshold not met for
> either C2 or C3.*

Context (not scored against H1/H2, DESIGN.md §7): C2-vs-C3 pairwise McNemar p = 0.093 (b=3, c=10).

### 5.2 Secondary Metrics

| Condition | Accuracy | Sensitivity | Specificity | AUROC | Brier | TP/TN/FP/FN |
|---|---|---|---|---|---|---|
| C1 `fusion_arithmetic_smoothed` | 0.7386 | 0.8704 | 0.5294 | 0.7650 | 0.1915 | 47/18/16/7 |
| C2 `fusion_loglinear` | 0.7273 | 0.8704 | 0.5000 | 0.7557 | 0.1987 | 47/17/17/7 |
| C3 `fusion_kdm_gate` | 0.6477 | 0.7778 | 0.4412 | 0.7108 | 0.2100 | 42/15/19/12 |
| `unimodal_tab_smoothed` | 0.6818 | 0.8519 | 0.4118 | 0.7053 | 0.2354 | 46/14/20/8 |
| `unimodal_mri_smoothed` | 0.5909 | 0.7778 | 0.2941 | 0.5202 | 0.3165 | 42/10/24/12 |
| `unimodal_txt_smoothed` | 0.7273 | 0.7963 | 0.6176 | 0.7146 | 0.2275 | 43/21/13/11 |

All three fusion conditions still beat every individual unimodal expert on Macro-F1 (C3's 0.6111 is
the weakest fusion result but still above `unimodal_mri_smoothed`'s 0.5286 and `unimodal_tab_smoothed`'s
0.6333, though *below* `unimodal_txt_smoothed`'s 0.7091 alone) — C3 is the only fusion condition in
this lineage's history to underperform its single best unimodal input.

### 5.3 Weight-source comparison (C1/C2 vs. C3, descriptive — DESIGN.md §3.3, not hypothesis-tested)

| | tab | mri | txt |
|---|---|---|---|
| C1/C2 global simplex weights (`fusion_weights_arithmetic.json`, `fusion_weights_loglinear.json`) | 0.400 | 0.100 | 0.500 |
| exp_29 `fusion_weights_leakfree` (context) | 0.400 | 0.100 | 0.500 |
| **C3 gate weights, mean ± std over 88 LOOCV patients** (`gate_weights_summary.json`) | 0.342 ± 0.052 | 0.320 ± 0.052 | 0.338 ± 0.051 |

C1 and C2's independently-run simplex searches converged to the *identical* global weight triple
`(0.4, 0.1, 0.5)` — the same optimum exp_29 found under `soft_vote`, now also optimal under
`logit_pool`. C3's per-patient gate weights, in contrast, are nearly uniform across all three
modalities and vary little from patient to patient (std ≈ 0.05 on each), despite `mri`'s per-modality
weakness (Macro-F1 0.5286, the worst of the three) — the gate does not down-weight `mri` the way the
global searches do. See `reports/figures/gate_weights_boxplot.png` for the full per-patient
distribution against exp_29's fixed triple.

### 5.4 Figures

- `reports/figures/confusion_matrix.png` — C3 (`fusion_kdm_gate`) confusion matrix, LOOCV.
- `reports/figures/roc_curve.png` — ROC overlay, C1 vs. C2 vs. C3.
- `reports/figures/gate_weights_boxplot.png` — per-patient C3 gate-weight distribution (boxplot per
  modality) with exp_29's fixed `(0.4, 0.1, 0.5)` triple marked for reference.

---

## 6. Statistical Analysis

- **Test used**: `mcnemar_exact` (exact McNemar test on paired LOOCV predictions), as specified in
  DESIGN.md §7 — appropriate here since Phase B is a single 88-fold LOOCV run (not multi-seed), so
  paired-prediction agreement/disagreement is the correct unit of comparison, not a between-seed test.
- **H1 (C2 vs. C1)**: b=1, c=2, statistic=1, p=1.000 → not significant.
- **H2 (C3 vs. C1)**: b=3, c=11, statistic=3, p=0.0574 → not significant at the α=0.05 threshold (and
  the discordant-pair direction, c ≫ b, means C3 disagrees with C1 mostly by being *wrong* where C1 is
  right, not the reverse).
- **Context (C3 vs. C2)**: b=3, c=10, p=0.0923 → not significant (not scored against a hypothesis).
- Given `std_macro_f1 ≈ 0.11` across MCCV splits in this cohort (exp_28/29's own estimate, unchanged
  cohort — DESIGN.md §7), neither result is reported as a win, consistent with this lineage's standard.

---

## 7. Comparison to Expected Results

Refer to DESIGN.md §7–8.

| Expected / checked | Observed | Match? |
|---|---|---|
| G0: reload exp_29's frozen per-modality winners, refit at `label_smoothing=0`, exact match to exp_29's stored unimodal hard-arm Macro-F1 (`abs(diff) < 1e-9`) | `tab`, `mri`, `txt` all matched exactly (`reproduction_gates.json`) | ✅ |
| G1: C1's simplex-grid machinery at `label_smoothing=0` reproduces exp_29 S1's weights `(0.4, 0.1, 0.5)` and `fusion_optimal_leakfree__hard = 0.7067941474721136` bit-for-bit | Weights and Phase-B Macro-F1 both matched exactly | ✅ |
| C1 (`label_smoothing=0.05`) expected "close but not identical" to exp_29's stored `0.7068` (DESIGN.md §2.2 predicted a possible small divergence since `label_smoothing` changes the raw probabilities even though the frozen `sigma`s were tuned at `label_smoothing=0`) | C1's LOOCV Macro-F1 = 0.7067941474721136 — **identical** to exp_29's stored value, not merely close | ✅ (stronger than expected) — Macro-F1 depends only on the `≥0.5` thresholded hard label, and for this cohort/weight triple, smoothing shifted no patient's fused probability across the 0.5 boundary, so the hard classification (and thus Macro-F1) came out bit-identical even though the underlying continuous probabilities differ. Worth flagging explicitly since DESIGN.md's own framing ("no condition ... is numerically comparable ... on a bit-for-bit basis") predicted otherwise for the *metric*, though it was correct about the raw probabilities. |
| H1: C2 > C1 with McNemar p<0.05 | C2 < C1 (Δ −0.0154), p=1.0 | ❌ |
| H2: C3 > C1 with McNemar p<0.05 | C3 < C1 (Δ −0.0957), p=0.057 | ❌ |
| exp_8's 0.7171 explicitly **not** the success bar (§1) | Noted for context only: none of C1/C2/C3 reached it (best = C1 at 0.7068) | — (not a decision criterion, as designed) |

---

## 8. Missing Data & Caveats

All planned conditions (C1, C2, C3) and both reproduction gates (G0, G1) completed and passed. No
missing runs.

Caveats carried over from DESIGN.md, still applicable to interpreting these results:

- **§2.4 leakage-asymmetry caveat**: C1/C2's global weights carry the same accepted aggregate-MCCV
  selection bias as every prior weight search in this lineage; C3's gate is leak-free at the
  per-fold memory level and uses an un-tuned bandwidth default. This makes C3 the *stricter*-protocol
  condition — and it still underperformed C1/C2, which (per DESIGN.md's own framing) is a clean result
  in the "protocol worked against C3 and it still lost" direction, not a confounded one. This
  strengthens, rather than undermines, the conclusion that the reliability-gate mechanism itself
  did not help here.
- **§2.5 evidence-table imbalance**: the pooled MCCV evidence table over-represents patients who land
  in more validation folds; not audited or corrected here. Given how flat C3's gate weights turned out
  (§5.3), this imbalance seems unlikely to be masking a strong per-patient signal, but it was not
  formally ruled out.
- **§2.2 label_smoothing not re-tuned**: `eps=0.05` was fixed a priori and the frozen `sigma`s were
  Brent-searched under `label_smoothing=0` in exp_29; not re-searched here by design.

---

## 9. Conclusions & Next Steps

**What this experiment established:**
- Neither reframing of the fusion mechanism tested here beats exp_29's plain arithmetic soft-voting
  under this cohort's evaluation protocol — this is now five straight conditions (exp_23, exp_25–29's
  successors here) in the exp_23-lineage that have failed to significantly beat the arithmetic-pooling
  baseline, this time isolating the fusion mechanism itself as the tested variable rather than the
  per-modality architecture.
- The reproduction gates (G0/G1) confirm exp_29's frozen winners and weight-search construction were
  reloaded and reused faithfully — the null results are not an artifact of a broken reload.
- C3's near-uniform, low-variance gate weights (0.34/0.32/0.34 ± ~0.05) are a substantive negative
  finding in their own right: a genuinely instance-adaptive reliability signal, if it existed in this
  evidence space at this cohort size (88 patients, ~1800 pooled evidence-table rows), was not recovered
  by this Nadaraya-Watson construction.

**What remains uncertain:**
- Whether C3's flat gate weights reflect a true absence of per-patient reliability structure in this
  cohort, or an estimation-power limitation (88 patients is a small memory for a 3-D non-parametric
  regression, even pooled across MCCV splits) — not distinguishable from this experiment alone.
- Whether combining C2's log-linear pooling with a *working* per-patient weight source (untested here,
  explicitly out of scope per DESIGN.md §8) would behave differently; nothing here tests that
  interaction, since neither component beat the arithmetic/global baseline on its own.

**Recommended follow-up:**
- Per DESIGN.md §1's citation of exp_29 §9: continued search within the sigma/representation/knn_k
  family of the per-modality experts is already discouraged. This result extends that caution to the
  fusion layer — log-linear pooling and this particular non-parametric gate construction are also now
  ruled out as easy wins.
- If a per-patient reliability signal is still worth pursuing, a follow-up would need either a larger
  evidence pool (relaxing the deduplication-for-bandwidth-only compromise in §3.3) or a different
  reliability feature space, not a re-tuning of this gate's bandwidth (already at scope per §8).
  To set up that experiment, use the `ml-experiment-planner` skill.
- exp_8's 0.7171 (soft-vote over the non-KDM baseline architecture) remains the lineage's unbeaten
  reference; no condition in exp_23–31 has matched it with significance.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | — (deterministic KNN-style method; LOOCV is exhaustive, not seed-dependent) |
| Configs versioned | ✅ (frozen hyperparameters loaded from `exp_29/results/stage1_best_hparams.json`, committed) |
| Git commits recorded | ✅ (`results/git_commit.txt`, §4) |
| Checkpoints saved | — (not applicable; no trained weights beyond the KDM memory itself) |
| Environment frozen | ⚠️ No `requirements.txt`/`environment.yml` committed (per `CLAUDE.md`); run under local `pytorch` conda env |
| Reproduction gates (G0/G1) | ✅ Both passed exactly (`results/reproduction_gates.json`) |
| Experiment tracker linked | — (no external tracker; this repo's own `DESIGN.md`/`results/` convention) |
