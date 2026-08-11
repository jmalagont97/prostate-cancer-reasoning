# Experiment Design: Improving exp_6's KDM Backbone (Hyperparameter Tuning + Skew-Aware Preprocessing)
**Experiment**: experiments/exp_7/
**Project**: challenge_chimera_2
**Date**: 2026-08-11
**Author**: TBD
**Status**: Complete — see `reports/summary.md`

---

## 1. Hypothesis

`exp_6`'s shared KDM backbone (0.588–0.593 decision macro-F1) has never had its hyperparameters
tuned — every prior use of KDM in this project (`exp_1`–`exp_6`) fixed `N_EPOCHS=300`, Adam
`lr=1e-2`, and the KNN-based initial-sigma scale (`sigma_mult=1.0`) by convention, not by search
— and its input features have never been preprocessed with KDM's own sensitivities in mind (its
single global RBF bandwidth makes it more sensitive to skew/outlier scale than the tree ensembles
and margin classifiers this project otherwise favors). A modest, bounded hyperparameter search
plus a skew-aware log-transform of the three continuous columns confirmed this session to be
meaningfully right-skewed (`cli_psa` skew=4.28, `cli_psad` skew=4.24, `cli_vol` skew=1.29) can
close some of the gap between KDM's own decision macro-F1 (0.588–0.593) and the project incumbent
(Extra Trees, 0.650) — and, since `exp_6`'s confidence/weights signals are derived from this same
backbone with no other code changes, an improved backbone may also close some of `exp_6`'s
confidence (0.731 best vs. 0.468 incumbent) and weights (0.405 vs. 0.382/0.392 incumbent) gaps,
testing whether those results reflected a weak backbone or a weak readout mechanism.

**Explicit scope decision (from this session's discussion)**: this experiment is deliberately
bounded to the **low-risk** end of the lever menu discussed — hyperparameter tuning and feature
preprocessing only. Architecture changes (per-dimension/ARD sigma, reduced-set/clustered
prototypes, alternate kernels, trainable label vectors) were explicitly considered and set aside
for a possible future experiment, not folded in here, per this session's explicit choice to keep
`exp_7` low-risk-only.

## 2. Experimental Setup

- **Dataset**: same 91-case set as `exp_3`–`exp_6`.
- **Feature frame**: `exp_3`'s 19-column with-MRI frame (`select_exp3_feature_frame`), unchanged
  in composition — this experiment changes how 3 of its columns are *transformed*, not which
  columns are included.
- **New preprocessing step (KDM-specific only, not applied to any other model/experiment)**:
  `np.log1p()` applied to `cli_psa`, `cli_psad`, `cli_vol` after median-imputation and before
  `StandardScaler`, inside the KDM backbone's own fit path only. `log1p` (not plain `log`) chosen
  because none of the three columns are guaranteed strictly positive after imputation edge cases
  (`cli_psad`'s min is 0.01, close to zero) — `log1p` is defined at 0 and behaves near-identically
  to `log` away from it. `cli_cspca` (skew=-2.06) is explicitly **excluded** — it's left-skewed
  and bounded near 1, not the same failure mode a log-transform fixes, and forcing one on it would
  be an unjustified transform with no grounding in the data's actual shape.
- **Hyperparameter search space** (KDM backbone fit only):

  | Hyperparameter | Current fixed value | Search values |
  |---|---|---|
  | `N_EPOCHS` | 300 | {150, 300, 600} |
  | learning rate | 1e-2 | {3e-3, 1e-2, 3e-2} |
  | `sigma_mult` (scales the KNN-distance-based initial sigma) | 1.0 (implicit default) | {0.5, 1.0, 1.5, 2.0} |
  | optimizer | Adam (implicit, only option used `exp_1`–`exp_6`) | {Adam, AdamW} |
  | `weight_decay` | N/A (Adam always used at PyTorch's default `weight_decay=0`) | Adam: fixed at 0. AdamW: {0, 1e-4, 1e-3} |

  Only `sigma` is ever trained (the memory-based config's only free parameter — `x_train=
  y_train=w_train=False`), so Adam vs. AdamW's usual distinction (decoupled weight decay vs.
  L2-via-gradient) reduces to a single, interpretable question here: does directly decaying the
  scalar `sigma` toward `min_sigma` (a smaller, more peaked kernel) over training help or hurt,
  independent of the loss gradient's own pull on it. Adam is kept at `weight_decay=0` throughout
  (its default, and its L2-via-gradient behavior would conflate regularization with the loss
  gradient in a way this experiment isn't trying to test) rather than also grid-searching Adam's
  weight decay — that would double-count the same question AdamW's decoupled version already
  answers more cleanly.

  3 × 3 × 4 × (1 Adam config + 3 AdamW configs) = 3 × 3 × 4 × 4 = **144 combinations**. `min_sigma`,
  the KNN neighbor count (`k=3`, hardcoded inside the installed `kdm` library's `init.py`), and
  `n_comp` (fixed at `len(X_train)`, i.e. one prototype per training row) are **not** searched
  this round — changing `n_comp` is architecture-adjacent (the "reduced-set prototypes" lever
  explicitly deferred, §1), and the KNN neighbor count would require patching the installed
  library rather than an additive change, which this project avoids.
- **Selection protocol**: the 144-combination grid is evaluated via the same 5-fold CV used
  throughout this project (`RANDOM_STATE=0`), but with a **reduced repeat count (3, not 10)**
  for the search itself — chosen for tractability (144 × 5 × 3 = 2,160 individual KDM fits;
  each fit is a single scalar parameter trained for ≤600 epochs on N≤73 rows, so still fast
  in aggregate, but background this run regardless) — then the winning combination is
  **re-evaluated at the full 10-repeat protocol** for the number actually reported and compared
  against `exp_6`. This is not a formally nested inner/outer CV split (matching this project's
  existing precedent — `exp_3`'s 8-model comparison wasn't nested either); the explicit
  mitigation, also following `exp_3`'s precedent, is requiring a **clear margin** over `exp_6`'s
  existing 0.588–0.593 before treating any result as a genuine improvement rather than CV noise
  from searching 144 configurations (a noticeably larger multi-way comparison than `exp_3`'s
  original 8-model one, so this caution matters more here, not less), and confirming the winner
  against `exp_3/scripts/holdout_eval.py`'s genuine held-out split (n=19, never used for any
  model selection in this project) as an out-of-sample sanity check.
- **Confidence/weights re-evaluation**: once a winning backbone configuration is selected and
  verified, `exp_6`'s exact readout code (`kdm_backbone.py`'s `compute_signals`/
  `occlusion_delta`/`kernel_distance_contribution`, `run_signals.py`'s 8 non-backbone conditions)
  is re-run **unchanged** against the new backbone — isolates whether any confidence/weights
  improvement comes from the backbone being better, not from a new readout mechanism.

## 3. File Layout for This Experiment

```
experiments/exp_7/
├── DESIGN.md
├── IMPLEMENTATION.md                    ← written after this design is accepted
├── scripts/
│   ├── kdm_backbone_v2.py               (extends exp_6's kdm_backbone.py: configurable
│   │                                      hyperparameters + log1p preprocessing; exp_6's own
│   │                                      kdm_backbone.py stays untouched so exp_6's results
│   │                                      remain reproducible from the same code)
│   ├── search_hyperparameters.py        (144-combination grid, 3-repeat search)
│   └── run_signals_v2.py                (exp_6's run_signals.py, backbone swapped for the
│                                          winning kdm_backbone_v2 configuration)
├── results/
│   ├── hyperparameter_search/           (all 144 combinations' 3-repeat macro-F1, for the record)
│   ├── decision_kdm_v2/                 (winning config, full 10-repeat + holdout_eval check)
│   ├── confidence_kdm_*_v2/             (5 conditions, same names as exp_6 + _v2 suffix)
│   └── weights_kdm_*_v2/                (3 conditions, same names as exp_6 + _v2 suffix)
└── reports/
    └── summary.md
```

## 4. Baselines

- **Decision**: `exp_6`'s `decision_kdm_backbone` (0.593 macro-F1) is the number this experiment
  must beat with a clear margin; Extra Trees (0.650) remains the project incumbent this
  experiment is not expected to surpass, only to move closer to.
- **Confidence**: `exp_6`'s best (`confidence_kdm_entropy_isotonic`, 0.731 ordinal distance);
  project incumbent `confidence_svm` (0.468); naive baseline (0.527).
- **Weights**: `exp_6`'s best (`weights_kdm_occlusion`, 0.405 mean ordinal error); project
  incumbent `weights_svm` (0.382/0.392); naive baseline (0.413).

## 5. Proposed Conditions

| Condition | Purpose |
|---|---|
| `hyperparameter_search` (144 sub-conditions) | find the best (epochs, lr, sigma_mult, optimizer, weight_decay) combination, 3-repeat CV |
| `decision_kdm_v2` | winning config, full 10-repeat + holdout_eval verification |
| `confidence_kdm_{entropy_zeroshot,entropy_isotonic,dispersion_isotonic,participation_isotonic,blend}_v2` | exp_6's unchanged readout code on the new backbone |
| `weights_kdm_{occlusion,kernel_distance,blend}_v2` | exp_6's unchanged readout code on the new backbone |

## 6. Ablation Studies

- **Preprocessing alone vs. hyperparameters alone vs. both combined** — worth isolating in the
  report which lever (if either) is doing the work, not just reporting the combined winner. A
  cheap addition: also evaluate (a) log1p-only with the *original* fixed hyperparameters, and
  (b) tuned-hyperparameters-only with the *original* untransformed features, alongside the full
  grid — 2 extra conditions, not 2 extra grids.

## 7. Evaluation Protocol

- Decision: macro-F1 (project-wide metric convention since the confusion-matrix bugfix
  correction), compared against 0.593 (exp_6), 0.381 (baseline), 0.650 (Extra Trees incumbent).
- Confidence: ordinal distance, compared against 0.731 (exp_6 best), 0.527 (baseline), 0.468
  (incumbent).
- Weights: mean ordinal error + mean decisive-set F1 across the 9 factors, with the same
  per-factor breakdown discipline `exp_6`'s report followed (not aggregate-only) — compared
  against 0.405 (exp_6 best), 0.413 (baseline), 0.382/0.392 (incumbent).
- **A "clear margin" bar for calling anything a genuine win**: given 144+ configurations are
  being compared, a nominal improvement of a few thousandths is not treated as a real result —
  consistent with `exp_3`'s precedent for multi-way comparisons at this N.

## 8. Expected Results & Decision Rules

- If the tuned/preprocessed backbone clearly beats 0.593 by a real margin **and** confirms on
  `holdout_eval.py`'s held-out 19 cases → genuine improvement, worth carrying forward as the new
  KDM backbone default for any future experiment building on `exp_6`'s design.
- If decision improves but confidence/weights don't → the backbone's *accuracy* was not the
  bottleneck in `exp_6`; the readout mechanism (entropy/dispersion/occlusion, or the isotonic
  recalibration layer) is the more promising place to keep investigating, not the backbone.
- If decision improves and confidence/weights improve proportionally → supports the "shared
  backbone" idea from `exp_6`'s original hypothesis more broadly — a better backbone genuinely
  helps every derived signal, strengthening the case for eventually trying `exp_6`'s deferred
  higher-risk levers (ARD sigma, `y_train=True`) in a future experiment.
- If nothing beats `exp_6`'s numbers by a clear margin → hyperparameters/preprocessing were not
  the bottleneck either; the deferred architecture-level levers (or accepting KDM's current
  ceiling for this task) become the more clearly justified next questions.

## 9. Risks & Mitigations

- **144-way (or 144+2) grid search at N=91 risks a false winner from CV noise**, same risk `exp_3`
  flagged for its 8-model comparison, considerably amplified by a grid this size — mitigated by
  the reduced 3-repeat search pass being explicitly provisional, a full 10-repeat re-evaluation of
  only the winner, a required clear margin (§7), and a genuine held-out check via `holdout_eval.py`
  before calling anything settled. Worth watching in particular: with 144 candidates, some will
  beat 0.593 by chance alone even if no real improvement exists — the clear-margin bar and the
  held-out check both exist specifically to catch that, not as boilerplate caution.
- **`log1p` transform is KDM-specific, not project-wide** — deliberately scoped this way so
  `exp_3`–`exp_6`'s other model conditions (SVM, Extra Trees, etc.) remain exactly reproducible;
  this is *not* proposing to change the shared `features.py` frame used elsewhere.
- **Runtime**: the 3-repeat search (2,160 fits) is the most expensive part of this experiment;
  cheap individually (~2s/fit observed in `exp_6`) but background it regardless, same pattern as
  every prior multi-condition experiment in this project.
- **`sigma_mult` search range (0.5–2.0) is centered on the current default (1.0), not fully
  open-ended** — a narrow, defensible range given `_sigma_from_knn`'s own data-driven KNN-distance
  basis is already a sensible starting point; a search across orders of magnitude wasn't judged
  worth the added grid size for this bounded, low-risk experiment.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, `torch.manual_seed(RANDOM_STATE)`, unchanged from
      `exp_1`–`exp_6`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_6`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — **N/A: project is not a git repository** (same caveat as prior experiments)

## 11. Next Steps

1. Review and accept this experiment plan.
2. Given how much of this experiment reuses `exp_6`'s existing, already-verified code
   (`kdm_backbone.py`'s `compute_signals`/`occlusion_delta`/`kernel_distance_contribution`,
   `run_signals.py`'s recalibration logic — both unchanged) — this is closer to `exp_4`/`exp_5`'s
   "Lean" implementation tier than `exp_6`'s from-scratch build. A short implementation plan
   (Claude Code plan mode) covering just `kdm_backbone_v2.py`'s configurable-hyperparameter
   signature and the log1p preprocessing hook is enough; save as `experiments/exp_7/
   IMPLEMENTATION.md` before editing any files.
