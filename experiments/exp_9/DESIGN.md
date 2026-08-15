# Experiment Design: ARD (Per-Dimension) Kernel Bandwidth for the KDM Backbone
**Experiment**: experiments/exp_9/
**Project**: challenge_chimera_2
**Date**: 2026-08-12
**Author**: TBD
**Status**: Complete — see `experiments/exp_9/reports/summary.md`. Verdict: the motivating
mechanism (shared-`σ` dilution behind `dispersion_isotonic`'s +0.309 regression in `exp_8`) is
directly confirmed fixed (0.776→1.085 under scalar `σ`, 0.797→0.804 under ARD). Decision improves,
held-out-confirmed, on the 23-column frame (+0.087 vs. `exp_6`) but the same CV win on the
19-column frame reverses on held-out (`exp_7`'s failure mode repeated). Weights lose the one
narrow baseline win the project had (`occlusion`, both frames). Reveal improves on both metrics
for the 23-column frame and produced the first 19-column result. Importance comparison: only 2/5
agreement with `exp_5`'s solved set on both frames — not corroborated. Not adopted as the new
default backbone project-wide.

---

## 1. Hypothesis

`exp_6`'s KDM backbone uses a single, shared kernel bandwidth `σ` across every input dimension.
Two experiments since have exhausted the "don't touch the architecture" options without a
verified win: `exp_7` (hyperparameter tuning + skew preprocessing) found a CV improvement that
failed the held-out check; `exp_8` (feature expansion + retuning) found the opposite — no CV
improvement, an ambiguous held-out signal, and, concretely, **`confidence_kdm_dispersion_isotonic`
degrading by +0.309 ordinal distance when the frame grew from 19 to 23 columns**. That last number
is the direct motivation for this experiment: a shared bandwidth forces every dimension into the
same distance scale, so adding weaker dimensions dilutes the kernel's discriminative power for
stronger ones. Nothing in the current architecture lets the model learn to down-weight a
dimension — it can only be excluded by hand, which is exactly what `exp_8`'s manual EDA-based
feature selection was already trying to do less directly.

**ARD (Automatic Relevance Determination)** replaces the single `σ` with one `σⱼ` per input
dimension, trained the same way (gradient descent on the same NLL loss). Dimensions the model
doesn't need get pushed toward large `σⱼ` (effectively ignored); dimensions that matter get small
`σⱼ` (the kernel stays sensitive to them). This directly targets the failure mode `exp_8` measured,
rather than another round of tuning a scalar or hand-picking columns.

**This experiment tests ARD on both feature frames deliberately** — not just to see if ARD helps
in general, but specifically to test whether it **rescues** the 23-column frame `exp_8` showed was
actively harmful under a scalar bandwidth. If ARD closes that gap, it validates the mechanism
directly against the exact evidence that motivated it, rather than a generic "does this help"
question.

**Secondary hypothesis**: the trained `σⱼ` vector is a free, model-native global variable-
importance signal (small `σⱼ` = the kernel needs to be sensitive to that dimension = important).
Compare it against `exp_5`'s SVM-based per-factor weights results as an independent check on
whether KDM's own learned structure agrees with what a dedicated classifier found important.

**Explicit scope guardrail**: no hyperparameter search this round. `exp_7`/`exp_8` both showed
that searching a large grid on top of an architecture change reintroduces real CV-noise risk, and
there is no stable ARD baseline yet to search *against*. Hyperparameters stay fixed at `exp_6`'s
original values (`n_epochs=300, lr=1e-2, sigma_mult=1.0` as the shared initial value for every
`σⱼ`, `optimizer=adam`) — isolating ARD's own effect is this experiment's entire job. A
hyperparameter search *for* the ARD architecture specifically is a legitimate `exp_10`, once this
experiment establishes whether ARD is worth searching around at all.

## 2. Experimental Setup

### 2a. Implementation

Subclass `RBFKernelLayer` (not a library patch — additive, in `exp_9`'s own scripts, per this
project's established "small per-experiment adaptations live in the experiment layer"
convention) as `ARDRBFKernelLayer`: replace the scalar `raw_sigma` parameter with a
per-dimension vector (`shape=(dim,)`), and pre-scale the input features by `1/σⱼ` before the
existing uniform-bandwidth squared-distance computation — a mathematically equivalent
reformulation that reuses the parent class's distance trick unchanged rather than rewriting the
kernel math. `min_sigma` and the softplus positivity reparameterization carry over unchanged,
just applied per-dimension.

**Initialization**: every `σⱼ` starts at the same scalar value `exp_6`'s existing KNN-based
`_sigma_from_knn()` heuristic already computes for the whole frame — training then differentiates
them. This avoids engineering a per-dimension init heuristic from scratch, which would be its own
source of uncertainty about what's actually being tested.

### 2b. Two frames, same architecture, same hyperparameters

| Frame | Columns | Why included |
|---|---|---|
| `exp_3`'s original | 19 | Clean baseline — matches `exp_6`'s own reference configuration |
| `exp_8`'s expanded | 23 | The frame ARD is specifically meant to rescue (§1) |

### 2c. Macro-F1 reported for all four subtasks, from the start

Per this project's cross-experiment macro-F1 reporting initiative (backfilled into `exp_6`–`exp_8`
this same session — see those reports' updated tables), `exp_9` reports macro-F1 natively rather
than as an afterthought:

| Subtask | Macro-F1 definition | Reported alongside |
|---|---|---|
| Decision | Standard binary macro-F1 (already the primary metric) | — |
| Confidence | Standard 3-class macro-F1 (`uncertain`/`borderline`/`clear`) | `ordinal_distance` (stays primary) |
| Weights | Per-factor 4-class macro-F1 (`labels=[0,1,2,3]` explicit, so an absent class counts at F1=0), averaged across the 9 factors | `ordinal_error` + `decisive_set_f1` (stay primary) |
| Reveal | Per-section binary F1, macro-averaged across the 4 modeled sections | `reveal_set_precision` (stays primary) |

### 2d. Reveal-sequence included on both frames

`exp_8`'s `run_reveal_kdm.py` mechanism (entropy-increase-from-occlusion, per-section binary
classifiers) is reused unchanged, applied to the ARD backbone on both frames. The only frame-
dependent detail: `psa_trend`'s feature group is `cli_psa`/`cli_psad` on the 19-column frame
(no `psav`/`psap` there) vs. the full 4-column PSA family on the 23-column frame — otherwise
identical. `pathology_report`/`family_history` remain unmodeled on both frames (0 positive
examples among the 91 labeled cases, confirmed in `exp_8`'s planning — this is a property of the
labels, not the frame).

## 3. File Layout for This Experiment

```
experiments/exp_9/
├── DESIGN.md
├── IMPLEMENTATION.md              ← written after this design is accepted
├── scripts/
│   ├── ard_kernel.py               (ARDRBFKernelLayer + fit_kdm_backbone_ard())
│   ├── run_signals_19col.py        (decision+confidence+weights, exp_3's frame, ARD backbone)
│   ├── run_signals_23col.py        (decision+confidence+weights, exp_8's frame, ARD backbone)
│   ├── run_reveal_19col.py
│   ├── run_reveal_23col.py
│   ├── holdout_eval_ard.py         (mandatory held-out check, both frames)
│   └── importance_comparison.py    (trained sigma_j vs. exp_5's SVM weights results)
├── results/
│   ├── decision_kdm_ard_19col/, decision_kdm_ard_23col/
│   ├── confidence_kdm_*_ard_{19,23}col/    (5 conditions x 2 frames)
│   ├── weights_kdm_*_ard_{19,23}col/       (3 conditions x 2 frames)
│   ├── reveal_kdm_ard_{19,23}col/
│   ├── holdout_eval_ard/
│   └── importance_comparison/
└── reports/
    └── summary.md
```
(21 result folders total.)

## 4. Baselines

All figures below now include macro-F1, backfilled 2026-08-12:

| Subtask | Frame | Best prior result | Official metric | Macro-F1 |
|---|---|---|---|---|
| Decision | 19-col | `exp_6` (0.593 CV/held-out) | macro-F1 | *(same)* |
| Decision | 23-col | `exp_8` combined (0.585 CV / 0.635 held-out — disagree, §3 of that report) | macro-F1 | *(same)* |
| Decision | — | Extra Trees incumbent | 0.650 macro-F1 | *(same)* |
| Confidence | 19-col | `exp_6` `entropy_isotonic` (0.731 ord.dist.) | 0.731 | 0.223 |
| Confidence | 19-col | `exp_6` `blend` (best macro-F1) | 0.754 | **0.269** |
| Confidence | 23-col | `exp_8` `entropy_isotonic` (best macro-F1, 0.268) | 0.744 | 0.268 |
| Confidence | — | `confidence_svm` incumbent | 0.468 ord.dist. | *(not computed)* |
| Weights | 19-col | `exp_6` `occlusion` (0.405 ord.err.) | 0.405 | 0.256 |
| Weights | 23-col | `exp_8` `occlusion` (0.412 ord.err.) | 0.412 | 0.269 |
| Weights | — | `weights_svm` incumbent | 0.382/0.392 ord.err. | *(not computed)* |
| Reveal | 23-col only | `exp_8` `reveal_kdm_occlusion` (0.799 set-precision) | 0.799 | 0.531 |
| Reveal | — | `reveal_flags` incumbent | 0.853 set-precision | *(not computed)* |
| Reveal | — | naive baseline | 0.783 set-precision | *(not computed)* |

Note: `exp_9` is the first experiment to run reveal-sequence on the 19-column frame — no prior
19-column reveal baseline exists to compare against, only the 23-column one from `exp_8`.

## 5. Proposed Conditions

| Condition | Target | Frame |
|---|---|---|
| `decision_kdm_ard_19col`, `decision_kdm_ard_23col` | decision | both |
| `confidence_kdm_{5 signals}_ard_{19,23}col` | confidence | both (10 conditions) |
| `weights_kdm_{occlusion,kernel_distance,blend}_ard_{19,23}col` | variable-weights | both (6 conditions) |
| `reveal_kdm_ard_19col`, `reveal_kdm_ard_23col` | reveal-sequence | both |
| `holdout_eval_ard` | decision | both, side by side |
| `importance_comparison` | — | trained `σⱼ` vs. `exp_5`'s SVM weights, both frames |

## 6. Ablation Studies

- **ARD vs. scalar, same frame** — the core comparison: `decision_kdm_ard_19col` vs. `exp_6`'s
  `decision_kdm_backbone`; `decision_kdm_ard_23col` vs. `exp_8`'s `decision_kdm_v3`. Isolates
  ARD's own effect, holding the frame fixed.
- **Does ARD close the 19-vs-23-column gap `exp_8` found?** Compare `decision_kdm_ard_19col` vs.
  `decision_kdm_ard_23col` directly — if the confidence/decision degradation `exp_8` measured
  going from 19→23 columns *shrinks* under ARD relative to how large it was under the scalar
  backbone, that's direct evidence for §1's central hypothesis.
- **`σⱼ` vs. `exp_5`'s SVM weights**: does the trained per-dimension bandwidth agree with which
  factors `exp_5`'s independent per-factor model search found solvable (`pirads`, `bx`, `dre`,
  `age`, `psa`) vs. not (`cspca`, `comorbidity`, `psad`, `vol`)? A small `σⱼ` on a solved factor's
  columns and large `σⱼ` on an unsolved factor's columns would be a genuine, independent
  corroboration of that project-wide finding via a completely different mechanism.

## 7. Evaluation Protocol

- Same 5-fold × 10-repeat CV (`RANDOM_STATE=0`) as every KDM condition since `exp_3`.
- **Held-out check is mandatory for decision on both frames**, per `exp_7`/`exp_8`'s established
  discipline — no CV-measured decision improvement gets reported as genuine without it.
- **Clear-margin threshold relative to the relevant baseline's own measured CV std** — 0.045 for
  the 19-column comparison (`exp_6`'s measured std), and `exp_8`'s own measured std for the
  23-column comparison — not a fixed constant, per `exp_7`'s stated lesson.
- Confidence/weights/reveal: report both the official rubric metric and macro-F1 side by side for
  every condition (§2c) — neither displaces the other.
- Weights and reveal: per-factor / per-section breakdowns required, not aggregate-only, per this
  project's established convention since `exp_6`.

## 8. Expected Results & Decision Rules

- If ARD clearly beats the scalar backbone on the 23-column frame **and** the gap to the
  19-column frame's own ARD result shrinks relative to `exp_8`'s scalar-backbone gap → direct
  confirmation of §1's central hypothesis; ARD becomes the new default backbone architecture for
  any future experiment.
- If ARD helps the 19-column frame but not the 23-column one → the dimensionality story is more
  complicated than "ARD fixes it" — worth understanding which specific added columns remain
  harmful even with per-dimension weighting before concluding anything further.
- If ARD doesn't clearly beat the scalar backbone on either frame (by both CV and held-out) →
  a real, informative negative result — the shared-bandwidth limitation wasn't actually the
  bottleneck, and the remaining deferred levers (`y_train=True`, reduced-set prototypes,
  alternate kernels) become the next candidates, in that order of increasing risk.
- If `σⱼ` agrees with `exp_5`'s solvable/unsolvable factor split → strengthens confidence that
  split reflects a real data property (a third independent mechanism corroborating it, after
  `exp_5`'s SVM search and `exp_6`/`exp_8`'s occlusion-based weights). If it disagrees → worth
  understanding why before trusting either signal blindly.
- Regardless of decision's outcome, report reveal's 19-column result on its own terms — this is
  new data with no prior comparison point, not something that needs to "win" against anything.

## 9. Risks & Mitigations

- **+18 or +22 trainable parameters instead of 1** — still tiny relative to N=91, but a real
  increase from KDM's deliberately minimal "lowest-variance configuration" design philosophy
  (`exp_1`'s original rationale). Worth watching for the failure mode ARD is classically prone to
  at small N: overfitting individual `σⱼ` to noise in specific folds. The held-out check exists
  specifically to catch this, same as it caught `exp_7`'s spurious CV win.
- **21 result folders is a large single experiment** — if implementation runs long, decision/
  confidence/weights on both frames (the core ARD-vs-scalar test) should be prioritized over
  reveal-on-both-frames and the importance-comparison script, which are valuable but secondary to
  this experiment's central question.
- **No hyperparameter search this round is a deliberate scope cut, not an oversight** — see §1's
  explicit guardrail. Don't be tempted to add one mid-implementation; that's `exp_10`'s job once
  ARD's own untuned behavior is understood.
- **`pathology_report`/`family_history` still unmodeled on both frames** — a property of the
  labels (0 positive examples), not something this experiment's frame choice affects either way.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, unchanged from `exp_1`–`exp_8`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_8`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — record the commit this experiment builds from (the project has had git
      history since the session that pushed `exp_1`–`exp_7` to GitHub)

## 11. Next Steps

1. Review this plan — the "no hyperparameter search this round" scope cut (§1/§9) and the
   two-frame comparison design (§2b/§6) are the two things most worth pushback before
   implementation.
2. Once accepted, an implementation plan (Claude Code plan mode) covering `ARDRBFKernelLayer`'s
   exact reformulation of the parent class's distance computation, the per-dimension
   initialization logic, and confirming `compute_signals()`/`occlusion_delta()`/
   `kernel_distance_contribution()` (all already generic over the fitted model, per `exp_6`'s
   original design) need zero changes to work with an ARD-backed model. Save as
   `experiments/exp_9/IMPLEMENTATION.md` before editing any files.
