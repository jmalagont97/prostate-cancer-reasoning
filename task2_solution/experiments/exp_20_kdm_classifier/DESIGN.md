# exp_20 Design: KDM (Kernel Density Matrices) for `decision`

## Background

Requested directly: bring in `fagonzalezo/kdm` (Kernel Density Matrices — a quantum-measurement-
inspired probabilistic model, PyTorch-native as of its v2.0 port), using its classification and
regression model classes, tested first on `decision`. This was flagged as deferred since early in
the project (`experiments/exp_2/scripts/models.py`'s docstring: "KDM deferred" from task1's own
model-family sweep) — never actually tried until now.

## Environment blocker found and fixed

`pip install torch` pulled the latest release (2.13.0), which fails to import on this machine with
a DLL-init error (`WinError 1114`, `c10.dll`). Diagnosed, not assumed: confirmed the standard
Visual-C++-redistributable DLLs (`msvcp140.dll`, `vcruntime140.dll`) are present in `System32`, so
it isn't the usual missing-redistributable cause; the CPU (AMD Ryzen 7 7735HS, a modern Zen-based
chip) rules out an instruction-set incompatibility; the `KMP_DUPLICATE_LIB_OK=TRUE` OpenMP-conflict
workaround didn't help either. `torch==2.2.0+cpu` (`pip install torch==2.2.0 --index-url
https://download.pytorch.org/whl/cpu`) imports and runs cleanly — pinned as the working version.
Unlike `scikit-survival`'s still-unresolved C++-compiler blocker (task3), this one had a real fix.

## What was built

`agent/src/kdm_models.py`: sklearn-style `.fit(X,y)/.predict(X)` wrappers around KDM's
`KDMClassModel` and `KDMRegressModel`, matching the README's manual training-loop example
(encoder → KDM layer → NLL loss → Adam) but encapsulated so either model drops straight into the
existing `model_factory` slot (`task2_variant_tester.py`, `subtask_config.py`) unchanged — same
pattern `mord.LogisticIT` used in exp_14. `n_comp` (the number of KDM components) is set to the
training-fold size, matching `init_kdm_layer`'s requirement that its initialization data have one
row per component — the standard "centers = training points" scheme for kernel methods at this
sample size. Both wrapper classes smoke-tested independently (classification + regression) before
being wired into the real pipeline.

## Method

`KDMClassifier(encoded_size=16, sigma=0.5, epochs=300, lr=1e-2)` tested on `decision`'s already-
promoted 12-feature set (10 base + `psa_kinetics`, exp_16) via `task2_variant_tester.py`,
`confidence`/`variable_weights` held at their own promoted defaults. `KDMRegressor` built and
smoke-tested but not yet run against a real task (task2 has no regression sub-task; a natural
follow-up is testing it against task3's `months_to_recurrence`, a genuine regression target).

## What would change the promoted config

Only if `KDMClassifier` beats the current `knn` default on `ranking_score`, bootstrap-confirmed
before promotion — same bar as every other experiment.
