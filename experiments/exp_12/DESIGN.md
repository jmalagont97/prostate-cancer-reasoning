# Experiment Design: Direct Scalar-KDM for Confidence (ARD Ablation of exp_11)
**Experiment**: experiments/exp_12/
**Project**: challenge_chimera_2
**Date**: 2026-08-14
**Author**: TBD
**Status**: Complete — see `experiments/exp_12/reports/summary.md`. Verdict: scalar clearly beats
ARD for this target (decision rule branch 3, not branch 1) — the 23-column frame under the plain
scalar backbone converges to ordinal distance 0.44–0.49 across CV/LOO/repeated-holdout (vs. `exp_11`
ARD's 0.47–0.55), with LOO (0.440) and the repeated-holdout mean (0.447) both numerically beating
the `confidence_svm` incumbent (0.468) for the first time in this project — CV alone (0.491) does
not quite. Macro-F1 reaches a new project-best (0.589, LOO). A single held-out split again showed
an identical, suspicious value across both frames (0.368) — this time the repeated-holdout follow-up
*confirmed* the promising signal rather than refuting it, underscoring that the check has to run
every time, not just when a result looks too good.

---

## 1. Hypothesis

`exp_11` found that training a KDM directly on the confidence label (rather than deriving signals
from a decision-trained backbone) is a large, four-independent-methods-confirmed improvement over
every derived-signal condition tried since `exp_6` — but that ARD specifically did not improve on
`exp_3`'s original *scalar*-backbone direct training (0.530 ordinal distance / 0.508 macro-F1):
`exp_11`'s ARD numbers (CV/LOO/repeated-holdout, both frames) landed in the same 0.47–0.55 /
0.47–0.51 band as `exp_3`'s single number, not clearly better.

This experiment isolates the one variable `exp_11` couldn't: `exp_3`'s original result used both a
different backbone (scalar `σ`) *and* a different, worse frame. `exp_12` reruns `exp_11`'s exact
protocol — same frames (19-col, 23-col), same direct-training setup, same evaluation methods (CV,
held-out, LOO) — with the **scalar** backbone (`experiments/exp_6/scripts/kdm_backbone.py`'s
`fit_kdm_backbone`/`compute_signals`) instead of ARD's. If scalar performs the same as `exp_11`'s
ARD numbers, the 23-column frame alone (not ARD) explains any residual gap from `exp_3`. If scalar
performs *worse* than `exp_11`, ARD was doing something after all, just not enough to show up
against `exp_3`'s specific number.

This is a direct, low-risk ablation of already-implemented, already-validated code — no new
patterns, isolated to 3 files structurally identical to `exp_11`'s. Given the small, well-specified
scope building directly on `exp_11`'s just-completed implementation, this experiment skips a
separate plan-mode cycle per the user's direct "run it" instruction — `experiments/exp_11/scripts/`
is copied and adapted, not redesigned.

## 2. Experimental Setup

Identical to `exp_11` in every respect except the backbone:
- **Frames**: `select_exp3_feature_frame` (19-col), `select_exp8_feature_frame` (23-col) — unchanged.
- **Backbone**: `experiments/exp_6/scripts/kdm_backbone.py`'s `fit_kdm_backbone`/`compute_signals`
  (scalar `σ`, `n_epochs=300, lr=1e-2` hardcoded internally — already matches `exp_9`'s ARD defaults
  exactly, so no config mismatch to control for). No new code — `fit_kdm_backbone` is already
  generic over `n_classes` (confirmed by inspection, same as its ARD counterpart), so `n_classes=3`
  needs zero changes there either.
- **No recalibration step** — same as `exp_11`, the model's own `argmax(probs)` is the prediction.
- **Full metric suite + protocols**: CV (5×10), held-out (same fixed split), LOO (91-fold) — all
  three mandatory, reusing `experiments/exp_11/scripts/metrics_multiclass.py` unchanged. Repeated
  held-out is run only if a result looks suspicious, same discipline `exp_11` established.

## 3. File Layout

- `experiments/exp_12/scripts/run_confidence_direct_scalar.py`
- `experiments/exp_12/scripts/holdout_eval_confidence_direct_scalar.py`
- `experiments/exp_12/scripts/loo_confidence_direct_scalar.py`

Each is `exp_11`'s equivalent script with `ard_kernel.fit_kdm_backbone_ard`/`compute_signals_ard`
swapped for `kdm_backbone.fit_kdm_backbone`/`compute_signals`, and the `**ARD_CONFIG` kwargs
removed (scalar's `fit_kdm_backbone` takes no hyperparameter arguments — everything is a module
constant already matching ARD's values). No changes to `exp_1`–`exp_11`'s scripts, the `kdm`
library, or `src/chimera_task1/*.py`.

## 4. Baselines

| Comparison | Ordinal distance | Macro-F1 |
|---|---|---|
| Baseline | 0.527 | 0.260 |
| Incumbent (`confidence_svm`) | 0.468 | 0.404 |
| `exp_3` direct scalar-KDM (worse frame) | 0.530 | 0.508 |
| `exp_11` direct ARD-KDM, converged (CV/LOO/repeated-holdout) | 0.47–0.55 | 0.47–0.51 |

## 5. Decision Rules

- If `exp_12`'s scalar numbers land in the same band as `exp_11`'s ARD numbers → the 23-column
  frame (not ARD) explains `exp_11`'s improvement over `exp_3`; ARD adds nothing for this target.
- If `exp_12` is clearly worse than `exp_11` → ARD does help direct confidence training after all,
  just not enough to separate from `exp_3`'s number given the frame change was confounded with it.
- If `exp_12` is clearly better than both `exp_11` and `exp_3` → an unexpected finding (frame helps
  more without ARD's added parameters at this N) worth its own follow-up.
- Any single held-out result that looks suspiciously clean (echoing `exp_11`'s identical-0.316
  finding) gets the same repeated-holdout treatment before being reported.

## 6. Next Steps

Implement directly (3 files, structural copies of `exp_11`'s equivalents) — no `IMPLEMENTATION.md`
this round, per the scope note in §1.
