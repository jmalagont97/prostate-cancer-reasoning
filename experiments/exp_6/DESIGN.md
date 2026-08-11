# Experiment Design: KDM as a Unified Probabilistic Backbone for Decision + Confidence + Weights
**Experiment**: experiments/exp_6/
**Project**: challenge_chimera_2
**Date**: 2026-08-10
**Author**: TBD
**Status**: Complete — see `reports/summary.md`

---

## 1. Hypothesis

A single memory-based KDM trained on the biopsy-decision target can serve as a **shared
probabilistic backbone** for all three model-searched Task-1 targets, instead of three
unrelated model families (`decision_extratrees`, `confidence_svm`, `weights_svm`) trained
independently. Two signals derived directly from that one trained model's internal
density-matrix structure — with no extra training beyond a small recalibration step — can
match or beat the current per-target bests:

- **Confidence**, from the model's own predictive uncertainty (entropy of the output
  class-probabilities, and/or the dispersion of the training prototypes actually driving
  the vote), against `confidence_svm`'s 0.468 ordinal distance (`exp_3`).
- **Variable weights**, from per-case local sensitivity of that same prediction to each
  factor's feature block, against `weights_svm`'s 0.382 (restricted) / 0.392 (official)
  ordinal error (`exp_5`).

**Explicit caution built into this hypothesis**: this is a genuine trade, not a strict
improvement. KDM's own decision macro-F1 (0.588 `exp_3` with MRI / 0.584 `exp_4` without)
trails Extra Trees' 0.642/0.650 by roughly 0.05–0.06. Adopting KDM as the shared backbone
means accepting that decision-accuracy cost in exchange for confidence/weights signals no
other model family in this project produces for free. Both outcomes — the trade paying off
or not — are reportable results; this experiment is not designed to only succeed one way.

**What this experiment deliberately does NOT claim**: this project's KDM configuration
(`x_train=y_train=w_train=False`, prototypes frozen at the training data with exact one-hot
label vectors — see `src/chimera_task1/train_confidence_kdm.py`) makes the model
mathematically a kernel-weighted-vote (Nadaraya-Watson) classifier. Traced through
`kdm/layers/kdm_layer.py`, `kdm/init.py`, and `kdm/utils.py` this session: because `c_y` is
copied verbatim from one-hot labels and never trained, the predictive state is diagonal in
the label basis, so **output-level Shannon entropy already is the full "quantum" uncertainty
available at that level** — there is no exploitable off-diagonal coherence at the output.
The genuinely new signal this experiment adds lives one level upstream: the *dispersion of
the training prototypes that a given query's kernel weights actually select*, computed via
`kdm.utils.dm_rbf_variance()` — a function already shipped with the library but never called
anywhere in this project to date.

## 2. Experimental Setup

- **Dataset**: same 91-case labeled set as `exp_3`–`exp_5`.
- **Feature frame**: `exp_3`'s 19-column with-MRI frame (`select_exp3_feature_frame`), not
  `exp_4`'s 16-column no-MRI frame. Chosen specifically because `confidence_kdm` is
  substantially better with MRI (0.530 vs. 0.623 ordinal distance, `exp_3` vs `exp_4`) while
  `decision_kdm` is roughly indifferent to it (0.588 vs. 0.584 macro-F1) — since this
  experiment's confidence signal rides on the same backbone, the frame is chosen for the
  backbone's *weakest link*, not its strongest.
- **Backbone model**: one `KDMClassModel` per CV fold, identical hyperparameters/training
  loop to `fit_predict_kdm()` in `train_confidence_kdm.py` (memory-based, `sigma` the only
  trained parameter, 300 epochs, Adam). No architecture changes to KDM itself this round
  (see §9, scope guardrail).

### Confidence signals (all computed from the already-fitted backbone, zero extra training)

| Signal | Definition | What it measures |
|---|---|---|
| A. Output entropy | `H(x*) = -Σ_c p_c log p_c` over `dm2discrete()`'s output | how evenly split the class vote is (already computed in `train_confidence_kdm.py`, never routed to a confidence prediction) |
| B. Neighborhood dispersion | `dm_rbf_variance(comp2dm(out_w, c_x), σ)`, where `out_w` is the per-prototype posterior weight from `KDMLayer._compute_mixture()` and `c_x` is the frozen training-prototype matrix | how spread apart, in feature space, the training cases actually casting the vote are |
| C. Participation ratio | `1 / Σ_i out_w_i²` | effective number of prototypes driving the vote |

A and B can diverge: a case can have low output entropy (confidently "yes") while still
sitting in a sparse, high-dispersion neighborhood — few reliable precedents propping up a
clean-looking probability. Whether that divergence tracks the schema's "uncertain" cases is
a direct, testable question for this experiment, not assumed.

### Weights signals (per-case, per-factor, off the same backbone)

| Signal | Definition | Cost |
|---|---|---|
| D. Local occlusion | replace factor *f*'s feature block with that fold's training-median (or mode, for binary flags), re-run the fitted backbone's forward pass, measure `Δp(yes)` (or `ΔSignal B`) vs. the unperturbed prediction | 9 extra forward passes per test case — cheap, no retraining (unlike `exp_5`'s "144 fits" concern) |
| E. Kernel-distance contribution | `Σ_i out_w_i(x*) · (x*_f - c_{x,i,f})²` summed over factor *f*'s feature columns — a per-case attribution read directly off quantities the forward pass already computes | free, no re-inference needed |

## 3. File Layout for This Experiment

```
experiments/exp_6/
├── DESIGN.md
├── IMPLEMENTATION.md            ← written after this design is accepted, in plan mode
├── scripts/
├── results/
│   ├── decision_kdm_backbone/                    (1 — re-verify the shared model's own decision cost)
│   ├── confidence_kdm_entropy_zeroshot/          (1 — no-training tercile binning, Signal A)
│   ├── confidence_kdm_entropy_isotonic/          (1 — isotonic-calibrated Signal A)
│   ├── confidence_kdm_dispersion_isotonic/       (1 — isotonic-calibrated Signal B)
│   ├── confidence_kdm_participation_isotonic/    (1 — isotonic-calibrated Signal C)
│   ├── confidence_kdm_blend/                     (1 — ordinal-logistic blend of A+B+C)
│   ├── weights_kdm_occlusion/                    (1 — Signal D, recalibrated per factor)
│   ├── weights_kdm_kernel_distance/               (1 — Signal E, recalibrated per factor)
│   └── weights_kdm_blend/                        (1 — D+E combined)
└── reports/
    └── summary.md
```
(9 condition folders total.)

## 4. Baselines

- **Decision**: naive baseline macro-F1 = 0.381 (`always predict yes`); incumbent best =
  Extra Trees, 0.650 macro-F1 (`exp_4`).
- **Confidence**: naive baseline ordinal distance = 0.527; incumbent best = `confidence_svm`,
  0.468 (`exp_3`).
- **Weights**: naive per-factor baseline ordinal error ≈ 0.413; incumbent best =
  `weights_svm`, 0.382 restricted / 0.392 official (`exp_5`).

## 5. Proposed Conditions

| Condition | Target | Mechanism |
|---|---|---|
| `decision_kdm_backbone` | decision | re-verification only — same as `decision_kdm` (`exp_3`), re-run so the exact fitted-model artifact used for confidence/weights is the one reported |
| `confidence_kdm_entropy_zeroshot` | confidence | Signal A, fixed-tercile binning against the training distribution, **no supervised training at all** — directly tests whether "confidence" is just decision-uncertainty |
| `confidence_kdm_entropy_isotonic` | confidence | Signal A, isotonic regression → confidence rank |
| `confidence_kdm_dispersion_isotonic` | confidence | Signal B, isotonic regression → confidence rank |
| `confidence_kdm_participation_isotonic` | confidence | Signal C, isotonic regression → confidence rank |
| `confidence_kdm_blend` | confidence | small ordinal-logistic regression on [A, B, C] → confidence rank |
| `weights_kdm_occlusion` | variable-weights | Signal D, per-factor isotonic/ordinal-logistic recalibration |
| `weights_kdm_kernel_distance` | variable-weights | Signal E, per-factor isotonic/ordinal-logistic recalibration |
| `weights_kdm_blend` | variable-weights | Signals D+E combined, per factor |

## 6. Ablation Studies

- **Signal choice for confidence** (A vs. B vs. C vs. blend) — the primary comparison this
  experiment adds. Tests whether feature-space dispersion (B) contributes information beyond
  plain output entropy (A), which is the closest thing to a novel empirical claim here.
- **Recalibration strategy for confidence's Signal A** (zero-shot tercile binning vs.
  isotonic) — isolates how much of any win is "the signal is informative" vs. "the
  recalibration step is doing real work."
- **Attribution mechanism for weights** (occlusion D vs. kernel-distance E vs. blend) — tests
  whether the cheaper, no-re-inference signal (E) is competitive with the more expensive,
  more standard one (D).

## 7. Evaluation Protocol

- Same 5-fold × 10-repeat CV shape as `exp_3`–`exp_5` (`N_SPLITS=5`, `RANDOM_STATE=0`), reused
  from `cv_utils.py` where the fold-splitting logic applies unchanged.
- **Nested calibration, no leakage**: within each fold, the backbone KDM is fit on that
  fold's training rows only (as today). Raw signals (A/B/C/D/E) are then computed for both
  train and test rows using that same fitted backbone. Any recalibrator (isotonic regression,
  ordinal-logistic blend) is fit on the **training rows' signals + labels only**, then applied
  to the held-out test rows' signals — mirrors the discipline `exp_5`'s per-factor conditions
  already follow, extended to a second, calibration-specific step.
- **Occlusion's training-marginal values** (Signal D) are computed from that fold's training
  split only, not the full 91-case pool — same leakage discipline.
- **Decision**: macro-F1 (per the project-wide metric switch), compared against 0.381
  baseline and 0.650 incumbent (Extra Trees).
- **Confidence**: ordinal distance (primary), compared against 0.527 baseline and 0.468
  incumbent (`confidence_svm`). Report all 5 confidence conditions, not just the winner —
  the zero-shot vs. isotonic comparison and the A-vs-B-vs-C comparison are both
  reportable findings independent of whether either beats baseline.
- **Weights**: mean ordinal error + mean decisive-set F1 across the 9 in-scope factors
  (`fh` excluded, unchanged from `exp_1`–`exp_5`), compared against 0.413 baseline and
  0.382/0.392 incumbent. Per-factor breakdown reported explicitly — `exp_5` found the
  win concentrated in 5 of 9 factors (`pirads`, `bx`, `dre`, `age`, `psa`); this experiment
  should report the same breakdown rather than only the 9-factor aggregate, so a mechanism
  that wins on aggregate but only by improving the already-solved factors isn't
  mistaken for closing the data-scarcity ceiling on the other 4.

## 8. Expected Results & Decision Rules

- If any confidence condition clearly beats 0.468 → KDM's own uncertainty is a genuinely
  competitive confidence signal, and — since it requires no separate model family — a
  strong candidate for the eventual submission pipeline specifically because decision and
  confidence would then share one trained artifact instead of two.
- If confidence_kdm_dispersion (Signal B) beats confidence_kdm_entropy (Signal A) by a
  real margin → evidence that feature-space dispersion carries information beyond output
  entropy, worth carrying into any future KDM work on this project. If it doesn't → Signal B
  was a reasonable idea that didn't pay off empirically, and future confidence work should
  default to the cheaper Signal A.
- If confidence_kdm_entropy_zeroshot (no training at all) comes within a small margin of the
  isotonic-calibrated version → answers a real scientific question raised in this project's
  prior discussion: the schema's "confidence" label may largely *be* decision-uncertainty,
  not an orthogonal clinical judgment.
- If any weights condition clearly beats 0.382 → re-run `holdout_eval.py`-style verification
  before treating it as settled, given `exp_5`'s per-factor structure has already produced
  one metric-definition surprise this project (macro vs. binary F1) and a second one for
  weights specifically would be worth confirming out-of-sample.
- If weights conditions replicate `exp_5`'s 5-of-9 pattern (winning on `pirads`/`bx`/`dre`/
  `age`/`psa`, flat on the other 4) → treat the 4-factor ceiling as confirmed from a second,
  structurally different mechanism, strengthening the "data-scarcity, not modeling gap"
  conclusion already on record.
- If nothing beats baseline anywhere → still a useful negative result: it would mean KDM's
  internal uncertainty structure, however principled, doesn't carry more signal than the
  discriminative SVM/kNN models already found independently — worth stating plainly rather
  than quietly dropping the experiment.

## 9. Risks & Mitigations

- **Scope guardrail**: no KDM architecture changes this round — no per-dimension (ARD)
  `sigma`, no `y_train=True`. Confirmed this session that `RBFKernelLayer.sigma` is a single
  global scalar (`.venv/Lib/site-packages/kdm/layers/rbf_kernel_layer.py`), so there is no
  free, model-native *global* variable-importance signal available without such a change —
  a legitimate future extension, deliberately out of scope here to keep this experiment to
  read-outs off the existing, already-validated "lowest-variance KDM configuration"
  (`train_confidence_kdm.py`'s own stated design rationale).
- **Occlusion's fill value choice (median/mode) is itself a design decision**, not the only
  option — mean, a random resample from the training marginal, or zeroing after
  standardization are alternatives. Median/mode chosen for robustness to the small N and
  outliers (e.g. `vol`, `psa`); worth a footnote in the report rather than treated as
  self-evidently correct.
- **Per-factor recalibration (weights) needs its own small held-out check**: each factor's
  raw signal (D or E) gets its own isotonic/ordinal-logistic fit, meaning up to 9 separate
  small calibrators per condition, each trained on ≈73 rows/fold — genuinely tight. If any
  factor's calibrator behaves degenerately (e.g. collapses to a constant prediction), that's
  a reportable outcome for that factor, not a bug to force past — consistent with how
  `exp_5` treated KDM's restricted-scope sigma failures.
- **This experiment is more diagnostic than most prior ones** — several of its conditions
  (zero-shot entropy, the A-vs-B ablation) are designed to answer "why," not just "which
  number is highest." The report should preserve that distinction rather than collapsing
  everything into a single leaderboard, which would bury the more interesting negative or
  mechanistic findings.
- **Runtime**: cheap relative to `exp_3`/`exp_5` — no new model families, no retraining beyond
  what `confidence_kdm`/`decision_kdm` already require; signals D/E and all recalibration
  steps are closed-form or small (≤3-feature) fits, not iterative training.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, `torch.manual_seed(RANDOM_STATE)`, unchanged
      from `exp_1`–`exp_5`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_5`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — **N/A: project is not a git repository** (same caveat as prior experiments)

## 11. Next Steps

1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (Claude Code plan mode) covering: exactly
   where the `dm_rbf_variance`/`_compute_mixture` access points live relative to
   `KDMClassModel`'s public interface (likely a small helper that reaches into
   `model.kdm.kdm` rather than a library change), the occlusion loop's fill-value logic per
   feature type (continuous vs. binary vs. one-hot block), and which script(s) under
   `experiments/exp_6/scripts/` house the signal-extraction step vs. the recalibration step.
   Save as `experiments/exp_6/IMPLEMENTATION.md` before editing any files.
