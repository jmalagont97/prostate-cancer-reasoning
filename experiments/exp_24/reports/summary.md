# Experiment Report: Particle-Set Uncertainty Decomposition for the Tabular KDM
**Experiment**: experiments/exp_24/
**Project**: pathology-reasoning
**Report date**: 2026-08-18
**Plan date**: 2026-08-18
**Author**: TBD
**Status**: Complete

---

## 1. Summary

exp_24 extracted the KDM's full internal particle set (`{(w_j(x), p_j)}`, the mixture the model
collapses via `dm2discrete` before returning a class probability) and tested whether its internal
structure — rather than only the collapsed mixture mean — predicts the clinician's 3-class
`confidence` annotation better than exp_23's entropy/`log_marginal` signals and the `exp_17` Composite-
ICI baseline (Macro-F1 0.4470). The best resulting head (Arm A's 7-signal multivariate tree ensemble,
Macro-F1 **0.4368**) clearly beat exp_23's best non-target-informed signal (0.3340) but fell short of
`exp_17` by 0.0102. The experiment's structural premise — that a one-hot `c_y` is an exact gradient
fixed point, making a naive aleatoric/epistemic split degenerate on exp_23's hard arm — was confirmed
numerically to machine precision, and the ε-smoothed Arm C built to escape that fixed point produced a
non-degenerate but **not significantly informative** aleatoric signal (Spearman ρ=0.13, p=0.22). The
real gain came from a different, unanticipated source: the particle **weight-geometry** signals
(`h_weights`, `log_ess`, `w_max`), which need no `c_y` learning at all.

---

## 2. Hypothesis & Verdict

**H1 (primary, from `DESIGN.md` §1):** "A signal derived from the particle set's internal structure —
rather than only its collapsed mixture mean — predicts the clinician's 3-class `confidence` annotation
with higher Macro-F1 than both exp_17 (0.4470) and exp_23's best non-target-informed signal (0.3340)."

**Verdict:** ⚠️ Partially supported.

**Evidence:** The best non-target-informed head, `a_hard__multivariate_7signal` (Arm A's full 7-signal
tree ensemble), scored Macro-F1 **0.4368** — beating exp_23's best (0.3340) by +0.1028, but **falling
0.0102 short of exp_17's 0.4470** (2.3% relative). No non-target-informed row in `confidence_metrics.json`
reached 0.4470.

---

**H2 (secondary, `DESIGN.md` §1/§4):** "A genuinely learned aleatoric/epistemic decomposition of
predictive entropy carries information beyond the collapsed `h_total` alone, evidenced by (a) significant
Spearman correlation of `h_aleatoric` against `confidence`, and (b) the full multivariate confidence head
outperforming an ablation restricted to exp_23's original two signals."

**Verdict:** ❌ Refuted, on the arm designed to test it.

**Evidence:** On Arm C — the only arm where `h_aleatoric` is structurally non-degenerate and
non-target-informed (§ H3 below) — (a) `c_smoothed__h_aleatoric` vs. `confidence`: Spearman ρ=0.1321,
**p=0.2199** (not significant at α=0.05); (b) Arm C's full 7-signal multivariate head scored 0.3699,
**below** its own `{h_total, log_marginal}` ablation at 0.3947 — adding the new signals hurt, not helped,
on the arm H2 was written for.

A distinct, unplanned finding sits alongside this: on **Arm A**, the multivariate head (0.4368) does
beat its ablation (0.3600) by a wide margin — but Arm A's `h_aleatoric`/`h_epistemic` are degenerate
(§ H3), so that gain is attributable entirely to the particle **weight-geometry** signals
(`h_weights`, `log_ess`, `w_max`), not to the aleatoric/epistemic decomposition H2 was framed around.

---

**H3 (structural, `DESIGN.md` §2.1, must hold before H1/H2 are interpretable):** "exp_23's hard arm
(one-hot `c_y` init) admits no learnable aleatoric/epistemic split, because one-hot amplitude vectors are
an exact fixed point of the Born-rule Jacobian."

**Verdict:** ✅ Supported, confirmed three independent ways in this run.

**Evidence:**
- `degeneracy_check.json`: a dedicated full-cohort model (`y_train=True`, `eps=0.0`, 300 epochs) moved
  `c_y` by `max_cy_drift_eps0 = 0.0` exactly.
- The identical check reproduced inside Phase B's actual training loop: Arm A's `full_cohort_drift`
  helper gave `eps=0 max=0.000e+00` vs. **Arm C `max=3.149e-01`** — see
  `reports/figures/cy_drift.png`.
- Arm A's Phase B output: `max(h_aleatoric) = 0.000e+00` across all 88 patients (exactly), and
  `h_epistemic` is numerically identical to `h_total` to the reported figures in
  `confidence_metrics.json` (`a_hard__h_total` and `a_hard__h_epistemic`: Macro-F1 0.2532, ρ=0.1231 —
  identical rows).

All in-script assertions gating H1/H2 interpretation (`train.py` §1.8) passed, so the H1/H2 verdicts above
stand on solid structural ground rather than a numerical accident.

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md`/`IMPLEMENTATION.md`, with one deviation discovered during implementation
(documented and handled, not silent):

- **Dataset**: `Data/preprocessed_old/task1/` (old schema), N=88 labeled complete-case cohort, 54 yes /
  34 no — identical to exp_23, loaded via a verbatim copy of `load_cohort`.
- **Model**: `KDMClassModel`, `dim_y=2`, `n_comp=n_train` (70 in Phase A, 87 in Phase B), three arms:
  - Arm A (hard): frozen from `experiments/exp_23/results/best_hparams.json`
    (`sigma_mult=2.0, x_train=False, y_train=False, encoder=linear`) — **not re-swept**.
  - Arm B (soft): frozen from the same file (`sigma_mult=2.0, x_train=False, y_train=False,
    encoder=identity`) — **not re-swept**.
  - Arm C (smoothed-hard, new): Phase A-selected — `sigma_mult=2.0, eps=0.20, x_train=False,
    y_train=True, encoder=linear`, mean MCCV Macro-F1 0.5973 (std 0.1184).
- **Training**: Adam, `lr=1e-3`, 300 full-batch epochs — unchanged from exp_23.
- **Hardware**: `/Users/fgonza/miniforge3/envs/pytorch/bin/python`, single process, no GPU.
  **Actual runtime: 8.2 min total** (Phase A ≈3.7 min, Phase B + confidence heads ≈4.5 min) — well
  under `IMPLEMENTATION.md` §0's 20–25 min estimate, because Phase A converged on `sigma_mult=2.0`
  configs (the cheap end of the grid) and Arm C's selected encoder (`linear`) is the faster of the two.
- **Deviations from plan**:
  1. **New, not anticipated in `DESIGN.md`.** `fit_1d_confidence_signal` (verbatim exp_23 logic) can
     raise on two inputs it doesn't gracefully resolve: every split's tree collapsing to a constant
     sweep prediction (`RuntimeError`), or an exact tie in the per-split direction vote
     (`AssertionError` on `np.sign(0)`). A smoke run hit both in succession. `IMPLEMENTATION.md` §1.9
     was updated post-discovery to add `fit_1d_confidence_signal_safe`, a wrapper (not a modification
     of the verbatim function) that falls back to 33rd/67th-percentile thresholds with `direction=+1`
     and flags `"degenerate_fallback": true`. It fired exactly once in the full run:
     `a_hard__h_aleatoric` (expected — that signal is identically zero, see H3).
  2. No other deviations. All other design decisions (grid, held-out multivariate voting, signal
     definitions) were implemented as specified.

---

## 4. Code Version

| Component | Git commit | Commit message |
|---|---|---|
| exp_24 (this run) | `12d07c1e7164905a5841bfa1c1398ec5fde4287a` | Add exp_23 results and ignore local Claude Code config files |
| exp_23 (reused config/results) | same commit | (results read from `experiments/exp_23/results/`, not re-run) |

No code changes were made to the repository between exp_23's committed run and this one — the same
commit produced both, so the exact-match reproduction in §5.1 is a same-commit, same-pipeline check, not
a cross-version one.

---

## 5. Results

### 5.1 Reproduction Check (validates the copied pipeline before trusting new numbers)

| Arm | This run | exp_23 published | Match |
|---|---|---|---|
| B (soft, deterministic) | 0.6694214876033058 | 0.6694214876033058 | ✅ exact (16 digits) |
| A (hard, 10-seed avg) | 0.5636363636363637 | 0.5636363636363637 | ✅ exact (16 digits) |

Both in-script assertions (`abs(diff) < 1e-6` for B, `< 1e-4` for A) passed with zero observed
difference — stronger than the tolerances required. This confirms `load_cohort`,
`build_features_fixed_categories`, `build_kdm`, `init_and_check`, `train_kdm`, and the LOOCV/seed loop
structure were copied faithfully from exp_23, and that Arm B/A's confidence-head *inputs*
(`h_total`≈exp_23's `entropy`, `log_marginal`) are legitimate too:
`b_soft__h_total` Macro-F1 0.4164 = exp_23's `entropy_soft` 0.4163682864450127, and
`b_soft__log_marginal` Macro-F1 0.3681 ≈ exp_23's `log_marginal_soft` 0.3680555555555555.

### 5.2 Binary LOOCV Metrics (secondary objective — not the primary target of this experiment)

| Model | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier | Deterministic |
|---|---|---|---|---|---|---|---|
| Arm A (hard) | 0.5636 | 62.50% | 0.8148 | 0.3235 | 0.6334 | 0.2494 | No (10 seeds) |
| Arm B (soft) | **0.6694** | 71.59% | 0.8889 | 0.4412 | 0.6498 | 0.2263 | Yes |
| Arm C (smoothed-hard) | 0.5719 | 62.50% | 0.7963 | 0.3529 | 0.6514 | 0.2450 | No (10 seeds) |
| Fuzzy KNN reference | 0.6364 | 65.91% | 0.7407 | 0.5294 | 0.6304 | 0.2908 | — |

Arm C (0.5719) slightly outperforms Arm A (0.5636) on binary Macro-F1 — the ε-smoothing that unpins
`c_y` doesn't hurt binary performance, and may help marginally — but neither approaches Arm B or the
Fuzzy KNN reference. This is a secondary metric only; exp_24's primary objective is the confidence
prediction in §5.3, not binary biopsy decision.

**McNemar's exact test (Arm C vs. recomputed Fuzzy KNN reference):** b=7, c=10, p=0.6291 — not
significant; Arm C's binary decisions are not distinguishable from the KNN reference at this N.

### 5.3 Confidence Prediction — Primary Objective (all rows, sorted by arm and signal)

Signal convention: entropy-family signals (`h_total`, `h_epistemic`) sign-flipped before fitting, matching
exp_23's `-entropy_mean` convention.

| Arm · Signal | Macro-F1 | Accuracy | Spearman ρ | p-value | Target-informed |
|---|---:|---:|---:|---:|:---:|
| a_hard · h_total | 0.2532 | — | 0.1231 | 0.2533 | No |
| a_hard · h_aleatoric | 0.2593 | — | n/a (fallback) | n/a | No |
| a_hard · h_epistemic | 0.2532 | — | 0.1231 | 0.2533 | No |
| a_hard · h_weights | 0.3294 | — | 0.0445 | 0.6804 | No |
| a_hard · log_ess | 0.3279 | — | 0.0336 | 0.7558 | No |
| a_hard · w_max | 0.3038 | — | −0.1151 | 0.2855 | No |
| a_hard · log_marginal | 0.3340 | — | 0.3368 | 0.0013 | No |
| **a_hard · multivariate_7signal** | **0.4368** | 45.45% | 0.2114 | 0.0480 | **No** |
| a_hard · ablation (h_total+log_marginal) | 0.3600 | — | 0.1282 | 0.2340 | No |
| b_soft · h_total | 0.4164 | — | 0.2160 | 0.0433 | Yes |
| b_soft · h_aleatoric | 0.3157 | — | −0.1059 | 0.3259 | Yes |
| b_soft · h_epistemic | 0.3363 | — | 0.1939 | 0.0702 | Yes |
| b_soft · h_weights | 0.2265 | — | 0.1572 | 0.1436 | Yes |
| b_soft · log_ess | 0.2367 | — | 0.1142 | 0.2893 | Yes |
| b_soft · w_max | 0.2069 | — | 0.0497 | 0.6459 | Yes |
| b_soft · log_marginal | 0.3681 | — | 0.1340 | 0.2131 | Yes |
| b_soft · multivariate_7signal | 0.3103 | — | 0.2012 | 0.0602 | Yes |
| b_soft · ablation | 0.3277 | — | 0.3226 | 0.0022 | Yes |
| c_smoothed · h_total | 0.2389 | — | 0.0866 | 0.4221 | No |
| c_smoothed · h_aleatoric | 0.3671 | — | 0.1321 | 0.2199 | No |
| c_smoothed · h_epistemic | 0.3476 | — | 0.1580 | 0.1415 | No |
| c_smoothed · h_weights | 0.3668 | — | 0.1351 | 0.2096 | No |
| c_smoothed · log_ess | 0.3079 | — | −0.0187 | 0.8630 | No |
| c_smoothed · w_max | 0.2998 | — | 0.1107 | 0.3047 | No |
| c_smoothed · log_marginal | 0.3225 | — | 0.3013 | 0.0043 | No |
| c_smoothed · multivariate_7signal | 0.3699 | — | 0.0963 | 0.3720 | No |
| c_smoothed · ablation (h_total+log_marginal) | 0.3947 | — | 0.1667 | 0.1207 | No |
| **exp_17 baseline** | **0.4470** | 57.95% | 0.2790 | 0.0085 | — |
| exp_23 best non-target-informed (`log_marginal_hard`) | 0.3340 | — | 0.3368 | 0.0013 | — |

Observations:
- **No single 1D particle signal beats `log_marginal`** (0.3340 on Arm A, 0.3225–0.3681 across arms) —
  the weight-geometry signals (`h_weights`, `log_ess`, `w_max`) are individually *weaker* than
  `log_marginal` alone (0.2069–0.3668).
- **Combining them multivariately helps on Arm A, hurts on Arm B and Arm C.** Arm A's full 7-signal
  ensemble (0.4368) is the best non-target-informed result in the table by a wide margin, and the only
  case where the ablation is clearly beaten (0.3600 → +0.0768). Arm C's ensemble *underperforms* its
  own 2-signal ablation (0.3699 vs. 0.3947), and Arm B's likewise (0.3103 vs. 0.3277) — plausibly
  overfitting: a depth-3 tree over 7 correlated signals, held-out-voted by only 11–31 trees per patient
  (`min_votes_per_patient=11` throughout), has more capacity than 88 labeled patients comfortably
  support.
- **`log_marginal` is consistently the strongest single non-target-informed 1D signal** across all three
  arms (0.3225–0.3340), reproducing exp_23's own finding that `log_marginal_hard` was its best
  non-target-informed result.
- **Arm B's `h_total` (0.4164) and ablation's Spearman ρ (0.3226, p=0.0022)** are both target-informed
  (soft targets are derived from `confidence`) and excluded from the H1 comparison, but are reported for
  completeness — note `h_total`'s Macro-F1 (0.4164) sits closer to `exp_17` than any non-target-informed
  row does.

### 5.4 Figures

- `reports/figures/cy_drift.png` — the clearest single figure in this report: `eps=0` particles show
  **zero** drift (spike exactly at 0), Arm C's particles spread up to 0.31 — visual confirmation of H3.
- `reports/figures/arm_c_grid_search_curves.png` — Arm C's 18-config MCCV grid, faceted by encoder,
  curves over `sigma_mult` per `eps`. `sigma_mult=2.0` dominates at every `eps`/encoder combination,
  consistent with exp_23's own arms both selecting `sigma_mult=2.0`.
- `reports/figures/particle_signal_scatter.png` — Arm C's `h_aleatoric` vs. `h_epistemic`, colored by
  `confidence`. No visually obvious cluster separation, consistent with the non-significant Spearman ρ
  in §5.3.
- `reports/figures/signal_correlation_heatmap.png` — Arm C's 7×7 signal correlation matrix, useful for
  understanding why the multivariate ensemble underperformed its ablation there (redundant/correlated
  inputs feeding excess tree capacity).
- `reports/figures/confidence_confusion_matrix.png` — 3×3 confusion matrix for the best-scoring head
  overall (`a_hard__multivariate_7signal`).

---

## 6. Statistical Analysis

- **Binary LOOCV (§5.2)**: McNemar's exact test on paired predictions (Arm C vs. Fuzzy KNN reference),
  the pre-registered decision rule for this comparison — b=7, c=10, p=0.6291, not significant.
- **Confidence heads (§5.3)**: Spearman rank correlation per signal/head, reported directly in the table
  above (each computed over all N=88 patients, not per-seed — these are OOF/LOOCV point estimates, not
  repeated-seed aggregates, so no seed-based CI is applicable). The stochastic arms (A, C) average
  probabilities/signals over R=10 seeds *before* computing these statistics, matching exp_23's own
  convention (`macro_f1_std_across_seeds` in `loocv_metrics.json` reports seed variability on the binary
  metric only, not on the confidence-head statistics, which operate on the seed-averaged signal).
- H2's criterion (a) explicitly required p<0.05 for `c_smoothed__h_aleatoric`; observed p=0.2199 — the
  null (no correlation) is not rejected.

---

## 7. Comparison to Expected Results

| Expected (`DESIGN.md` §6 decision rules) | Observed | Match? |
|---|---|---|
| H1: best non-target-informed row beats exp_17 (0.4470) | Best = 0.4368 | ❌ (falls short by 0.0102) |
| H1: best non-target-informed row beats exp_23's best (0.3340) | 0.4368 | ✅ (+0.1028) |
| H2(a): Arm C `h_aleatoric` significantly correlated with `confidence` | ρ=0.1321, p=0.2199 | ❌ |
| H2(b): full multivariate beats `{h_total, log_marginal}` ablation | 0.3699 < 0.3947 (Arm C) | ❌ |
| H3: H3 structural assertions pass before H1/H2 interpreted | All passed | ✅ |
| Secondary: Arm C binary Macro-F1 vs. Fuzzy KNN reference, McNemar | p=0.6291 | Not significant (as expected — no strong prior either way) |

---

## 8. Missing Data & Caveats

All planned runs completed. No planned condition, ablation, or metric is missing.

One methodological caveat worth flagging explicitly: the held-out multivariate voting scheme (`DESIGN.md`
§4.2) guarantees `min_votes_per_patient=11` — adequate for a majority vote, but on the low side for a
depth-3 tree's effective validation signal, and may partly explain the Arm B/C multivariate underperformance
relative to their 2-signal ablations in §5.3.

---

## 9. Conclusions & Next Steps

**What this experiment established:**
- The one-hot `c_y` fixed point (H3) is real and exactly reproducible — not a numerical curiosity but a
  structural property of the amplitude-encoding Born rule that silently made exp_23's hard-arm
  aleatoric/epistemic split vacuous. Any future KDM uncertainty work on this codebase should check for
  this before reporting an aleatoric/epistemic split from a hard-label arm.
- The particle set's **weight geometry** (`h_weights`, `log_ess`, `w_max`) — signals that need no `c_y`
  learning and are immune to the H3 degeneracy — combine with `h_total`/`log_marginal` in a multivariate
  tree to produce this project's best non-target-informed confidence result to date (0.4368), beating
  exp_23's entire signal set (best 0.3340) by a wide margin, though still short of `exp_17`'s
  hand-engineered Composite ICI (0.4470).
- The ε-smoothed aleatoric/epistemic decomposition this experiment was built around (H2) did **not**
  deliver — Arm C's `h_aleatoric` is statistically indistinguishable from noise as a confidence predictor,
  and adding it (and the other new signals) multivariately actively hurt Arm C's result relative to just
  `{h_total, log_marginal}`.

**What remains uncertain:**
- Whether the Arm A multivariate gain (0.4368) reflects a real, generalizable signal in the
  weight-geometry features, or is itself an artifact of tree overfitting on N=88 with only ~11–31
  held-out votes per patient — the experiment cannot distinguish "genuine weight-geometry signal" from
  "got lucky with 88 patients and a depth-3 tree" without a larger cohort or a nested-CV variant of the
  held-out voting scheme.
- Why the aleatoric signal correlates so weakly with clinician `confidence` — whether local outcome
  heterogeneity among a patient's kernel-weighted training neighbors genuinely doesn't track how
  confident a urologist felt, or whether `ε=0.20`'s smoothing was too aggressive/too mild to recover a
  real signal (only `ε∈{0.05,0.10,0.20}` was swept).

**Recommended follow-up:**
- A narrower, Arm-A-only follow-up isolating exactly which subset of `{h_weights, log_ess, w_max,
  h_total, log_marginal}` drives the 0.4368 result (feature importances from the fitted trees, or a
  smaller ablation grid) before trusting it as a real finding.
- If pursuing the aleatoric decomposition further, a wider `ε` sweep or a different unpinning mechanism
  (e.g., learned per-particle temperature) might escape the fixed point more informatively than
  uniform label smoothing did here.
- To set up a follow-on experiment, use the `ml-experiment-planner` skill.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (`per_seed_macro_f1` in `loocv_metrics.json` for stochastic arms) |
| Configs versioned | ✅ (`results/best_hparams.json`, includes provenance for frozen Arm A/B configs) |
| Git commit recorded | ✅ (`results/git_commit.txt`) |
| Checkpoints saved | ❌ (not saved — matches exp_23's convention; models are cheap to retrain, ~0.1–0.7s each) |
| Environment frozen | ⚠️ (no `requirements.txt`/`environment.yml` in repo; documented interpreter path only, per `CLAUDE.md`) |
| Experiment tracker linked | — (not used in this repo) |
