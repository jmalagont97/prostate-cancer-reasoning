# Experiment Design: Direct KDM for Weights — Backbone and Scope Ablation
**Experiment**: experiments/exp_13/
**Project**: challenge_chimera_2
**Date**: 2026-08-15
**Author**: TBD
**Status**: Complete

## Verdict (2026-08-15)

**Clean negative result.** All 4 conditions (2 backbones × 2 scopes) converge tightly on almost
exactly `exp_5`'s original pre-modernization numbers (restricted-scope CV reproduces `exp_5`'s
0.454 to three decimal places; official-scope CV sits within 0.006 of `exp_5`'s 0.478). Backbone
choice (scalar vs. ARD) makes no meaningful difference — the two land within 0.001–0.005 of each
other on every metric, unlike confidence (`exp_11`/`exp_12`), where scalar vs. ARD was the entire
finding. None of the 4 conditions beat baseline (0.413) or the `weights_svm` incumbent
(0.382/0.392). A suspiciously clean single held-out split (restricted scope tying baseline exactly)
was checked via a targeted 10-seed repeated-holdout follow-up and confirmed as noise — 9/10 seeds
land above baseline, mean 0.438–0.441. Per §2d's staged criterion, full LOO was skipped for all 4
conditions since none approached the bar that would have warranted it. `weights_kdm_occlusion`
(`exp_6`) remains this project's best KDM-based weights approach; `weights_svm` remains the
unmatched incumbent. Full detail: `reports/summary.md`.

---

## 1. Hypothesis

`exp_11`/`exp_12` found that for **confidence**, deriving signals from the decision-trained
backbone (`exp_6`–`exp_10`, 25 conditions, 5 experiments) was a dead end, while training a KDM
**directly** on the target label — revisiting a pre-`exp_6` approach — closed most of the gap to
the incumbent, once the right backbone (scalar, not ARD) and frame (23-col) were found.

**Weights** has its own pre-`exp_6` direct-training precedent, never revisited: `exp_5` trained a
KDM directly on each of the 9 in-scope factors' weight labels (`weights_official_kdm`/
`weights_restricted_kdm`), and it **scored worse than baseline** in both scopes (0.478/0.454 mean
ordinal error vs. baseline 0.413) — the opposite of confidence's near-baseline starting point.
Like `exp_3`'s confidence number, `exp_5`'s weights number predates every backbone improvement
since (`exp_9`'s ARD, the 23-column frame). Nothing has retried it.

Unlike confidence, weights has no prior pointing to which backbone or scope might work — decision
favored ARD (`exp_9`), confidence favored scalar (`exp_12`), and `exp_5`'s original scope
comparison was ambiguous (the incumbent SVM does *better* on official scope, 0.382 vs. 0.392, the
opposite of what "restricted = more focused" might suggest). Per this session's decision, `exp_13`
tests **both backbones × both scopes** as four parallel primary conditions, on the 23-column frame
(this project's consistently stronger frame for both decision and confidence's direct-training
results) — settling the question in one experiment rather than repeating confidence's two-round
trip.

## 2. Experimental Setup

### 2a. Direct training, per factor — same simplification as exp_11/exp_12

No recalibration step. For each of the 9 in-scope factors, a KDM is trained **directly** on that
factor's 4-class weight label (`not_used < noted < important < decisive`) — the model's own
`argmax(probs)` is the prediction, exactly like confidence's direct-training conditions. This
differs from every `exp_6`–`exp_10` weights condition, which derived a signal (occlusion delta,
kernel-distance contribution) from the *decision*-trained backbone and only then fit a small
recalibration model on top.

### 2b. Four conditions: 2 backbones × 2 scopes

- **Backbone**: scalar (`experiments/exp_6/scripts/kdm_backbone.py`) and ARD
  (`experiments/exp_9/scripts/ard_kernel.py`) — both already confirmed generic over `n_classes`
  (used at `n_classes=3` for confidence in `exp_11`/`exp_12`; `n_classes=4` here, no new code
  needed in either module).
- **Scope**: **official** (every factor's classifier sees the full 23-column frame — the
  incumbent's stronger config, 0.382) and **restricted** (each factor's classifier sees only its
  own columns via `restricted_feature_group`, already used unchanged by `exp_2`–`exp_10`'s weights
  conditions).

### 2c. Frame: 23-column only

`select_exp8_feature_frame` — this project's consistently better-performing frame for direct
training (`exp_9` decision, `exp_11`/`exp_12` confidence). Not testing 19-column this round to keep
the already-large 4-condition × 9-factor scope from doubling again; a natural follow-up if any
23-column result looks promising enough to isolate frame effects the way `exp_9` did for decision.

### 2d. Staged execution — CV/held-out first, LOO only where warranted

Full cost if every protocol runs for every condition: CV alone is `4 conditions × 9 factors × 50
folds` = 1,800 fits; LOO is `4 × 9 × 91` = 3,276 fits. Per this session's cost discussion, **CV and
held-out run for all 4 conditions first** (cheaper, ~1,836 fits total); **LOO then runs only for
conditions that beat or approach baseline** on that first pass — no point spending 91×9 fits per
condition confirming a result CV and held-out already show is clearly non-competitive. This is a
deliberate scope-management decision, not a silent shortcut around the project's LOO-mandatory
convention — every condition still gets a real multi-protocol check, just not all three protocols
on conditions CV/held-out already rule out.

## 3. File Layout

- `experiments/exp_13/scripts/run_weights_direct_scalar.py`, `run_weights_direct_ard.py` — CV, both
  scopes, all 9 factors, in one script per backbone (mirrors `exp_11`/`exp_12`'s one-script-per-
  backbone pattern).
- `experiments/exp_13/scripts/holdout_eval_weights_direct_scalar.py`, `_ard.py` — held-out, same
  fixed split, both scopes.
- `experiments/exp_13/scripts/loo_weights_direct_scalar.py`, `_ard.py` — LOO, run only for the
  scope(s)/backbone(s) that clear §2d's bar.
- Reuses `experiments/exp_11/scripts/metrics_multiclass.py` unchanged (`multiclass_brier_score`,
  `safe_multiclass_auroc` are already generic over `n_classes`, used here at 4 instead of 3) and
  `restricted_feature_group`/`decisive_set_f1` unchanged from `src/chimera_task1/`. No changes to
  `exp_1`–`exp_12`'s scripts, the `kdm` library, or `src/chimera_task1/*.py`.

## 4. Baselines

| Comparison | Mean ordinal error | Mean decisive-set F1 | Mean macro-F1 |
|---|---|---|---|
| Baseline (majority class, per factor) | 0.413 | 0.379 | *(not yet computed for this exact 9-factor set — will compute alongside)* |
| Incumbent, official (`weights_svm`) | **0.382** | — | 0.235 |
| Incumbent, restricted (`weights_svm`) | 0.392 | — | — |
| `exp_5` direct scalar-KDM, official (worse frame) | 0.478 | 0.521 | 0.295 |
| `exp_5` direct scalar-KDM, restricted (worse frame) | 0.454 | 0.379 | 0.275 |
| Best derived-signal ever (`weights_kdm_occlusion`, scalar, `exp_6`) | 0.405 | 0.442 | 0.256 |

## 5. Evaluation Protocol

Full metric suite per factor, then averaged across the 9 in-scope factors (matching every prior
weights condition's aggregation): mean ordinal error (official metric), mean decisive-set F1
(official rubric component), macro-F1, one-vs-rest AUROC (guarded for missing classes — some
factors have rare weight levels with very few examples, already a known issue since `exp_5`), and
multiclass Brier score. CV (5×10) and held-out (fixed split) mandatory for all 4 conditions; LOO
per §2d's staged criterion.

## 6. Decision Rules

- If any condition's mean ordinal error clearly beats the incumbent (0.382/0.392, scope-matched) →
  the first-ever KDM win over `weights_svm` — the strongest possible outcome, worth pursuing further
  (frame search, the way `exp_9`/`exp_11` did after their own wins).
- If a condition beats baseline (0.413) but not incumbent → matches confidence's `exp_12` shape —
  real progress, not yet a leaderboard win.
- If direct training doesn't beat `exp_5`'s own original numbers at all → evidence that weights'
  4-class, 9-factor structure doesn't benefit from direct supervision the way confidence's single
  3-class target did — a genuine negative result, narrowing where this project's "revisit direct
  training" idea actually applies.
- Whichever backbone/scope wins, report the loser's numbers too — this project's discipline has
  never hidden an unflattering comparison, and "ARD helps decision, hurts confidence" is already a
  finding worth a third data point either way.

## 7. Risks & Mitigations

- **Scale**: the largest single-experiment fit count this project has attempted (§2d addresses this
  directly with staged LOO).
- **Rare per-factor classes**: some factors have very few examples of a given weight level (already
  known since `exp_5`'s `ValueError`-catch precedent for sklearn models) — `safe_multiclass_auroc`
  already handles this gracefully (returns `None`, logged, not crashed) for the KDM path too, reused
  unchanged from `exp_11`.
- **9 separate small-N fits per condition** compounds this project's standing N=91 noise concern —
  per-factor results should be read with the same caution `exp_6`'s per-factor weights breakdown
  already established, not just the aggregate.

## 8. Next Steps

Implement directly — file layout mirrors `exp_11`/`exp_12` closely enough that a separate plan-mode
cycle is optional; will use it if the per-factor loop structure raises anything genuinely new during
implementation.
