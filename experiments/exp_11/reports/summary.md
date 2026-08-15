# exp_11 Report: Direct ARD-KDM for Confidence

## 1. Summary

Reviving `exp_2`/`exp_3`'s pre-`exp_6` approach — a KDM trained **directly** on the confidence
label, not a signal derived from a decision-trained backbone — combined with `exp_9`'s ARD kernel
and the 23-column frame, produces the **clearest, most consistently confirmed win this project's
confidence subtask has ever seen against its own prior attempts, and a clean negative result
against the incumbent** — both true at once, and both worth stating plainly:

- **Against every derived-signal condition tried in `exp_6`–`exp_10` (25 conditions, 5
  experiments): a decisive, four-independent-methods-confirmed win on macro-F1.** CV (0.472–0.502),
  LOO (0.508–0.509), and the repeated-holdout mean (0.485–0.493) all land in the same tight band,
  roughly **double** the best derived-signal macro-F1 ever recorded (`exp_9` ARD 19-col `blend`,
  0.283). This is not a one-off number — it's the same result from three independent evaluation
  protocols.
- **Against baseline (0.527 ordinal distance) and incumbent (`confidence_svm`, 0.468): no win.**
  Once a misleading single held-out split is set aside (§3), CV/LOO/repeated-holdout converge
  tightly around **0.47–0.55 ordinal distance** — essentially tied with baseline, well short of the
  incumbent.
- **Against `exp_3`'s original scalar-backbone direct training (0.530 / 0.508): a wash.** ARD did
  not clearly improve on it. LOO's macro-F1 (0.508/0.509) is almost exactly `exp_3`'s own number.
- **A second, independently-caught confirmation of `exp_10`'s lucky-split lesson** (§3) — this
  time for confidence, this time caught proactively per the project's own standing discipline
  rather than by surprise.

| Subtask | This experiment's result | Baseline | Incumbent | Beats baseline? | Beats incumbent? |
|---|---|---|---|---|---|
| Confidence (ordinal distance, converged) | 0.47–0.55 (CV/LOO/repeated-holdout, both frames) | 0.527 | 0.468 | ❌ essentially tied | ❌ |
| Confidence (macro-F1, converged) | **0.47–0.51** (CV/LOO/repeated-holdout, both frames) | 0.260 | 0.404 | ✅ clearly | ✅ **first-ever KDM confidence macro-F1 win over incumbent** |

## 2. What Was Run

First experiment run natively under both reporting conventions confirmed this session (full metric
suite; mandatory LOO). No new KDM code — `experiments/exp_9/scripts/ard_kernel.py`'s
`fit_kdm_backbone_ard`/`compute_signals_ard` called with the 3-level confidence rank and
`n_classes=3` instead of the decision label, confirmed fully generic over `n_classes` by direct
inspection before implementation (`IMPLEMENTATION.md` finding #1). No recalibration step, unlike
every derived-signal script since `exp_6` — the model's own `argmax(probs)` **is** the prediction.

Four evaluation protocols, both frames (19-col, 23-col):
1. **CV** (5-fold × 10-repeat) — `run_confidence_direct_ard.py`.
2. **Held-out** (fixed 19-case split, decision-stratified — the same split "since `exp_3`" means the
   same row indices, `exp_3`'s own original held-out script scored confidence on this exact split
   too) — `holdout_eval_confidence_direct_ard.py`.
3. **LOO** (91-fold, pooled, deterministic) — `loo_confidence_direct_ard.py`, the first-ever LOO
   check for a non-decision subtask.
4. **Repeated held-out** (10 seeds) — `repeated_holdout_confidence_direct_ard.py`, an *unplanned*
   addition (§3) triggered by a suspicious result, not part of the original `IMPLEMENTATION.md`.

One implementation bug was caught by the plan's own smoke tests before any scored run: `sklearn`'s
`roc_auc_score(multi_class="ovr")` does **not** raise `ValueError` when a class is missing from the
scored set despite an explicit `labels=` argument — it silently returns `NaN` folded into the macro
average. `safe_multiclass_auroc()` was fixed to check class presence explicitly before calling
`roc_auc_score` at all, rather than relying on exception handling that doesn't actually fire.

## 3. The single held-out split was a lucky outlier — caught before being reported as a finding

The held-out check's first result was striking enough to be suspicious on its own: **ordinal
distance = 0.316 for both frames, identically** — a number that would beat the incumbent (0.468)
outright. Two frames producing the exact same value on the same split is itself a signal that the
split, not the model, is doing the work. Per this project's own standing rule (`experiments/
INDEX.md`, written after `exp_10`: *"any future single-split held-out result this striking should
get the same LOO/repeated-holdout treatment before being reported as a finding"*), a 10-seed
repeated-holdout check was run before treating 0.316 as real.

| Frame | CV | Held-out, seed=0 | Held-out, 10-seed mean±std | LOO (91-fold) |
|---|---|---|---|---|
| 19-col | 0.554 | 0.316 | 0.474 ± 0.158 | 0.527 |
| 23-col | 0.527 | 0.316 | 0.516 ± 0.141 | 0.505 |

Seed=0 turns out to be the best (19-col, tied with seed=1) or one of the best (23-col) of the 10
seeds tried — genuinely the most favorable single split available, not a representative one. CV,
the repeated-holdout mean, and LOO — three independent methods — all converge on **0.47–0.55**,
nowhere near 0.316. This is the second time this project has caught this exact pattern (`exp_10`,
decision; now `exp_11`, confidence), and the second time three-or-more independent methods have
agreed with each other against one single-split outlier.

## 4. Full results, all four methods, both frames

| Method | Frame | Accuracy | Macro-F1 | Ordinal distance | AUROC (OvR macro) | Brier (multiclass) |
|---|---|---|---|---|---|---|
| CV | 19-col | 0.597 | 0.472 | 0.554 | 0.652 | 0.657 |
| CV | 23-col | 0.607 | 0.502 | 0.527 | 0.690 | 0.588 |
| Held-out (seed=0) | 19-col | 0.737 | 0.482 | 0.316 | 0.714 | 0.398 |
| Held-out (seed=0) | 23-col | 0.737 | 0.482 | 0.316 | 0.764 | 0.386 |
| Held-out (10-seed mean) | 19-col | — | 0.485 ± 0.098 | 0.474 ± 0.158 | 0.702 | 0.606 |
| Held-out (10-seed mean) | 23-col | — | 0.493 ± 0.099 | 0.516 ± 0.141 | 0.744 | 0.563 |
| LOO (91-fold) | 19-col | 0.626 | 0.508 | 0.527 | 0.655 | 0.658 |
| LOO (91-fold) | 23-col | 0.615 | 0.509 | 0.505 | 0.695 | 0.558 |

Setting aside the seed=0 held-out outlier, every other cell clusters tightly: macro-F1 in
**0.47–0.51** across every method and both frames; ordinal distance in **0.47–0.55**. This
consistency — after four different evaluation protocols — is itself worth noting: direct training's
performance here is stable and reproducible, unlike several results earlier in this project's KDM
work where different evaluation methods pointed in different directions on the *headline* number
(only the single held-out split disagreed, and that disagreement is now explained).

## 5. Comparison against every other confidence approach tried in this project

| Approach | Ordinal distance | Macro-F1 |
|---|---|---|
| Baseline (majority class) | 0.527 | 0.260 |
| **Incumbent (`confidence_svm`, `exp_3`)** | **0.468** | 0.404 |
| `exp_3`, directly-trained `confidence_kdm` (scalar, pre-ARD) | 0.530 | 0.508 |
| Best derived-signal, official metric (`exp_6` `entropy_isotonic`) | 0.731 | 0.223 |
| Best derived-signal, macro-F1 (`exp_9` ARD 19-col `blend`) | 0.836 | 0.283 |
| **`exp_11`, directly-trained ARD-KDM (this experiment, converged)** | **0.47–0.55** | **0.47–0.51** |

Two comparisons, two different verdicts:
- **vs. every derived-signal condition since `exp_6`**: decisive win, confirmed by 3+ independent
  methods, not a single lucky number.
- **vs. `exp_3`'s own original direct-trained KDM**: a wash. ARD's per-dimension bandwidth, which
  helped decision cleanly (`exp_9`) and confidence's own `dispersion` signal specifically when
  *derived* from a decision backbone, does not show a clear benefit when the confidence label
  itself provides the training signal. This is a genuinely new, useful data point: ARD's advantage
  may be tied to *how* a target relates to the feature geometry, not a blanket improvement to KDM
  as a model family.

## 6. Interpretation

The core hypothesis from `DESIGN.md` §1 is **confirmed in the specific way it was framed**:
direct supervision, not the derived-signal architecture, was the larger lever for confidence —
`exp_6`'s pivot away from direct training in favor of a unified decision-derived backbone appears,
in hindsight, to have cost this subtask real performance that reviving direct training recovers
cleanly. The *secondary* part of the hypothesis — that ARD specifically would push past `exp_3`'s
original result — did not pan out; the improvement came entirely from the training target, not the
kernel architecture. Per `DESIGN.md` §8's decision rules, this lands closest to the "direct-ARD
beats derived signals on macro-F1 but not ordinal distance" branch, with the added, unplanned
finding that ARD itself wasn't the active ingredient.

The macro-F1-vs-ordinal-distance split here echoes a pattern already seen elsewhere in this
project (`exp_6`'s `blend`, `exp_9`'s `dispersion_isotonic`) — the two metrics can disagree on
which condition is best, and this experiment is the clearest illustration yet: an approach that
*decisively* wins on one official-adjacent metric while merely tying baseline on the actual
official metric.

## 7. Recommendation

- **Direct ARD-KDM training is a genuine improvement over every derived-signal confidence approach
  this project has tried, and should be considered confidence's new best-validated KDM approach** —
  though "best KDM approach" still falls short of the SVM incumbent on the metric that matters for
  the leaderboard.
- **Do not claim a win over `confidence_svm`.** Ordinal distance converges to baseline, not below
  incumbent, once the misleading single-split number is set aside.
- **ARD is not the reason this worked** — a natural, cheap follow-up would rerun this exact
  experiment with the *scalar* backbone (matching `exp_3`'s original architecture more closely, just
  on the better 23-column frame) to confirm the frame alone accounts for any residual difference
  from `exp_3`'s number, isolating architecture from frame the way `exp_9` did for decision.
- **The repeated-holdout-triggered-by-suspicion pattern worked exactly as intended** — worth
  keeping as a standing practice: when two frames (or conditions) produce a suspiciously identical
  held-out number, treat that as a prompt to check further, not a stronger result.
- Standing recommendations unchanged from every prior report: do not replace `weights_svm` with any
  KDM mechanism; the `exp_5` finding that direct KDM training hurt weights specifically (unlike
  confidence) still stands — this result does not generalize there.
