# exp_20 Report: KDM (Kernel Density Matrices) for `decision`

**Status**: Complete. **Verdict: ❌ Refuted, a real and sizable loss. No change to the promoted
config — `decision` stays on plain `knn`.**

## What was built and tested

`agent/src/kdm_models.py`: sklearn-style wrappers (`KDMClassifier`, `KDMRegressor`) around
`fagonzalezo/kdm`'s `KDMClassModel`/`KDMRegressModel`, matching the package's README training-loop
example (encoder → KDM layer → NLL loss → Adam), encapsulated so either drops into the existing
`model_factory` slot unchanged. `KDMClassifier(encoded_size=16, sigma=0.5, epochs=300, lr=1e-2)`
tested on `decision`'s already-promoted 12-feature set (10 base + `psa_kinetics`, exp_16).
`KDMRegressor` built and smoke-tested independently but not yet run against a real task (task2 has
no regression sub-task; task3's `months_to_recurrence` is the natural next target).

## Environment blocker found and fixed (real fix, unlike task3's still-open `scikit-survival` gap)

`pip install torch` pulled the latest release (2.13.0), which failed to import with a DLL-init
error (`WinError 1114`, `c10.dll`). Diagnosed properly before working around it: confirmed the
standard VC++-redistributable DLLs (`msvcp140.dll`, `vcruntime140.dll`) are present, ruling out the
usual missing-redistributable cause; the CPU (AMD Ryzen 7 7735HS) is modern enough to rule out an
instruction-set incompatibility; the standard `KMP_DUPLICATE_LIB_OK=TRUE` OpenMP-conflict fix didn't
help either. `torch==2.2.0+cpu` imports and runs cleanly — pinned as the working version for this
environment.

## Result

```
baseline (decision=knn): ranking_score=0.7429
decision=KDMClassifier:  ranking_score=0.6944  (vs baseline −0.0485)  decision_f1=0.7724
```

A real, sizable loss — an order of magnitude bigger than any of the "not significant" gains found
elsewhere in this project, so no bootstrap was needed to conclude it isn't a promotion candidate.
Also markedly slower: 172 seconds for one 72-fold LOOCV pass (each fold trains a real PyTorch model
for 300 Adam steps) versus effectively instant for every sklearn-based model tried so far — a real
practical cost on top of the accuracy loss.

## Classification report — a coherent, not catastrophic, underperformance

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.75      0.72      0.73        25   (knn: 0.73/0.88/0.80)
continued_surveillance       0.92      0.86      0.89        14   (knn: 0.93/0.93/0.93)
      active_treatment       0.76      0.84      0.80        31   (knn: 0.89/0.81/0.85)
              accuracy                           0.78        72   (knn: 0.83)
```

| True \ Pred | ww | as | cs | at |
|---|---|---|---|---|
| ww (2) | 0 | 0 | 0 | 2 |
| as (25) | 1 | 18 | 1 | 5 |
| cs (14) | 0 | 1 | 12 | 1 |
| at (31) | 0 | 5 | 0 | 26 |

Not a collapse into a majority-class default (unlike `confidence`'s persistent failure pattern) —
KDM still distinguishes all four classes reasonably. It's simply worse than `knn` across the two
largest classes (`active_surveillance` recall 0.88→0.72, `continued_surveillance` 0.93→0.86), with
a small offsetting gain on `active_treatment` (0.81→0.84) that doesn't make up the difference.

## Why, plausibly

Not formally diagnosed (would need a proper capacity/regularization sweep to confirm), but
consistent with this project's repeated finding that higher-capacity models tend to lose at this
sample size: `KDMClassifier` fits a real neural encoder (2 linear layers) plus `n_comp=71` learned
KDM components — substantially more free parameters than `knn`'s zero-training, purely non-
parametric nearest-neighbor lookup — on only 71 training rows per fold, with no hyperparameter
tuning done here (default-ish settings from the package's own README, not adapted for n≈70). This
echoes `mlp`'s consistent mediocrity across every model sweep in this project (task1's exp_2,
task2's exp_12) and the low/medium-capacity model preference already documented in
`experiments/exp_2/scripts/models.py`.

## Verdict and action

**Not promoted.** `decision` stays plain `knn`. `agent/src/kdm_models.py` is kept — the environment
blocker is now genuinely fixed (unlike `scikit-survival`'s still-open one), and the wrapper
infrastructure is real, tested, reusable plumbing for a future, better-tuned attempt or a different
target (task3's `months_to_recurrence` regression is the more natural fit for `KDMRegressor`, not
yet tried).

## Not done

- No hyperparameter sweep (`encoded_size`, `n_comp` below the full training-set size, `sigma`,
  `epochs`, `lr`) — only one default-ish configuration was tried, following the README's own
  example rather than tuning for this specific small-n regime.
- `KDMRegressor` built and smoke-tested but never run against a real regression target.
- No test on `confidence`/`variable_weights` — scoped to `decision` only, per the explicit request.
