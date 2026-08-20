# Experiment Design: KDM Regression for Weights — Ordinal-Aware Training
**Experiment**: experiments/exp_14/
**Project**: challenge_chimera_2
**Date**: 2026-08-18
**Author**: TBD
**Status**: Complete

## Verdict (2026-08-18)

**Genuinely mixed result — the two official metrics diverged for the first time in this project.**
Neither condition (per-factor `dim_y=1`, joint `dim_y=9`) beats the primary bar
(`weights_kdm_occlusion`'s 0.405 ordinal error) — both land at 0.46–0.49 across CV/held-out/
repeated-holdout, confirmed consistently, no LOO warranted per the staged criterion. But
**decisive-set F1 reaches 0.51–0.52, the best result this project has ever produced for that
metric by any model** (beating the SVM incumbent's 0.457), confirmed real (not a lucky held-out
split) via a targeted repeated-holdout check. The joint multi-factor architecture performed
statistically identically to 9 independent regressors — the hoped-for information-sharing benefit
for data-scarce factors didn't materialize. Full detail: `reports/summary.md`.

---

## 1. Hypothesis

Every KDM attempt at **weights** so far — `exp_6`/`exp_9`'s derived-signal approach and `exp_13`'s
direct-training revival (4 conditions: scalar/ARD × official/restricted) — trained `KDMClassModel`,
a *classifier*. Its NLL loss treats every misclassification identically: predicting `not_used` when
the truth is `decisive` (rank distance 3) costs the same as predicting `noted` when the truth is
`important` (rank distance 1). But this subtask's **primary official metric is mean ordinal error**
— the rank distance itself. No KDM condition tried anywhere in this project has ever trained on a
loss that respects that structure; every variation so far (bandwidth: scalar vs. ARD; supervision:
direct vs. derived) kept the same rank-blind classification objective.

`kdm-torch` ships a second model class, `KDMRegressModel`, trained via `dm_rbf_loglik` — a genuine
regression loss over a continuous target. Training directly on the weight *rank* (0–3) as a
continuous value, instead of a 4-way class label, means the loss itself is ordinal-distance-shaped:
a prediction 3 ranks off is penalized far more than a prediction 1 rank off, during training, not
just at evaluation time. This is a mechanically different lever than anything tried in `exp_6`–
`exp_13`, staying entirely within the KDM family per this project's standing scope.

**Two conditions, both newly enabled by the same architecture change:**
- **Per-factor regression**: 9 independent `KDMRegressModel`s (`dim_y=1`), the direct ordinal-aware
  analogue of `exp_13`'s per-factor classification KDM — apples-to-apples comparable.
- **Joint multi-factor regression**: *one* `KDMRegressModel` (`dim_y=9`) predicting all 9 factors'
  ranks at once from a single shared prototype pool. Untried anywhere in this project — every prior
  weights condition (SVM included) fit 9 independent per-factor models. A joint fit lets factors
  that share statistical structure (e.g. `psa`/`psad`/`vol` are physically related) borrow strength
  from each other, which may specifically help the 4 data-scarce factors (`cspca`, `comorbidity`,
  `psad`, `vol`) that have sat near decisive-set F1 ≈ 0 in every prior experiment.

## 2. Experimental Setup

### 2a. Frame and backbone

23-column frame (`select_exp8_feature_frame`) — this project's consistently stronger frame for
direct KDM training (`exp_9` decision, `exp_11`/`exp_12` confidence). Scalar bandwidth only for the
primary run: `exp_13` already showed backbone choice (scalar vs. ARD) makes almost no difference for
weights (0.454 vs. 0.454, 0.483 vs. 0.484), so this experiment spends its budget on the *training
objective* change instead of re-litigating bandwidth. An ARD regression follow-up is a cheap,
optional stretch goal, not primary scope.

### 2b. Memory-based configuration, matching every prior KDM condition

`KDMRegressModel(encoded_size=dim_x, dim_y=1 or 9, encoder=nn.Identity(), n_comp=n_train, x_train=False, y_train=False, w_train=False)` — frozen prototypes, only `sigma_x` (input kernel bandwidth) and `sigma_y` (output kernel bandwidth) trained via Adam, 300 epochs, matching `fit_kdm_backbone`'s existing defaults as closely as the regression model's extra `sigma_y` parameter allows. Verified directly against the installed `kdm-torch==2.0.0` API before committing to this design (`dim_y=9` confirmed to fit and predict without error).

### 2c. Turning a continuous prediction back into an ordinal class

`model.predict_reg(X)` returns `(mean, variance)`. Round `mean` to the nearest integer and clip to
`[0, 3]` for the predicted rank — used for accuracy, macro-F1, ordinal error, and decisive-set F1,
identically to every classification-KDM condition's reporting.

**AUROC and Brier score need class probabilities, which a regression model doesn't natively
produce.** These are constructed as *regression-derived pseudo-probabilities*: treat
`(mean, variance)` as a Normal distribution over the continuous rank, and assign each of the 4
discrete ranks the probability mass in `[k−0.5, k+0.5)` under that Normal (renormalized to sum to
1 over the 4 valid ranks, via the CDF). This is a standard technique for scoring ordinal regression
with classification metrics, but it is an approximation, not a calibrated classifier output — every
report from this experiment labels these two metrics accordingly, so they're never confused with a
`KDMClassModel`'s native `probs`.

**Known limitation for the joint condition**: `predict_reg`'s variance is a single scalar per case,
not one per output dimension — confirmed by direct test against the installed library. The joint
model's pseudo-probabilities therefore share one variance across all 9 factors within a case, only
the per-factor mean differs. This is weaker than the per-factor condition's fully independent
variances, and is called out explicitly wherever the joint condition's AUROC/Brier are reported.

### 2d. Evaluation protocol

Same full metric suite + staged evaluation as `exp_13`: CV (5×10) and held-out mandatory for both
conditions; LOO only for whichever condition clears/approaches baseline on that first pass (cost:
per-factor regression is exactly as expensive as `exp_13`'s per-factor conditions — 9×50 CV fits;
joint regression is far cheaper, needing only 1×50 CV fits total instead of 9×50, since one fit
covers all 9 factors at once).

## 3. File Layout

- `experiments/exp_14/scripts/kdm_regress_backbone.py` — shared `fit_kdm_regress`/`compute_signals_regress` helpers (per-factor and joint, both call the same underlying fit function with `dim_y=1` or `dim_y=9`), plus the Normal-CDF pseudo-probability discretizer.
- `experiments/exp_14/scripts/run_weights_regress_per_factor.py` — CV, 23-col frame, 9 independent regressors.
- `experiments/exp_14/scripts/run_weights_regress_joint.py` — CV, 23-col frame, 1 joint regressor.
- `experiments/exp_14/scripts/holdout_eval_weights_regress_{per_factor,joint}.py` — held-out, same fixed split used since `exp_3`.
- `experiments/exp_14/scripts/loo_weights_regress_{per_factor,joint}.py` — LOO, staged per §2d.
- Reuses `experiments/exp_11/scripts/metrics_multiclass.py` unchanged (`n_classes=4`).

## 4. Baselines

| Comparison | Mean ordinal error | Mean decisive-set F1 |
|---|---|---|
| Baseline (majority class, per factor) | 0.413 | 0.379 |
| **Incumbent (`weights_svm`, restricted, non-KDM)** | **0.382** | 0.457 |
| Best KDM ever (`weights_kdm_occlusion`, `exp_6`) | 0.405 | 0.442 |
| `exp_13` direct classification KDM (all 4 conditions) | 0.454–0.484 | 0.374–0.523 |

## 5. Decision Rules

Per this project's KDM-only scope for this line of work (confirmed by the user): the primary bar is
**beating `weights_kdm_occlusion`'s 0.405** — the best KDM result to date, not yet beaten by any
KDM condition since. Beating the SVM incumbent (0.382) would be the strongest possible outcome and
is reported regardless, in keeping with this project's standing discipline of never hiding an
unflattering comparison — but it is not the primary success bar for this specific experiment.

- If either condition beats 0.405 → new best-ever KDM weights result, worth a follow-up (ARD
  variant, frame search).
- If the joint condition specifically beats the per-factor condition → evidence that sharing
  statistical strength across factors is the actual lever, motivating a joint-model follow-up for
  confidence/decision too.
- If neither beats 0.405 but both beat baseline (0.413) → real, if modest, progress — matches the
  shape of `exp_6`'s original result.
- If neither beats baseline → a third independent confirmation that weights' 4-class, 9-factor,
  small-N structure doesn't respond to architecture changes the way confidence did — genuinely
  narrows where further KDM effort on this subtask is worth spending.

## 6. Risks & Mitigations

- **Pseudo-probability approximation** (§2c) could make AUROC/Brier look better or worse than a
  genuine classifier's calibrated output would — mitigated by explicit labeling in every report and
  by treating ordinal error / decisive-set F1 (both computed from the same rounded rank every
  classification condition uses) as the metrics that actually decide the verdict.
- **The joint model's shared per-case variance** (§2c) could distort AUROC/Brier specifically for
  that condition — flagged, and ordinal error / decisive-set F1 remain the primary comparison there
  too.
- **Rare per-factor classes** (known since `exp_5`) — same `ValueError`-catch / skip-and-log
  discipline as every prior weights script.

## 7. Next Steps

Implement directly, following `exp_11`'s smoke-test discipline before any scored run: confirm
`fit_kdm_regress` with `dim_y=1` and `dim_y=9` both produce valid `(mean, variance)` shapes and that
the Normal-CDF discretizer's 4 pseudo-probabilities sum to 1 per row, before committing to the full
CV/held-out runs.
