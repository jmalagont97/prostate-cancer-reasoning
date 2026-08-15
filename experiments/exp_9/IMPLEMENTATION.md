# exp_9 Implementation Plan — ARD (Per-Dimension) Kernel Bandwidth

## Context

This implements `experiments/exp_9/DESIGN.md` (status: Proposed, reviewed this session). `exp_6`
built a shared KDM backbone; `exp_7` (hyperparameter tuning) and `exp_8` (feature expansion)
both failed to improve it, and `exp_8` produced concrete evidence of *why* — a single shared
kernel bandwidth `σ` can't down-weight weaker added dimensions, causing measurable confidence
degradation (+0.309 ordinal distance) when the frame grew from 19 to 23 columns. `exp_9` replaces
the scalar `σ` with one trained `σⱼ` per dimension (ARD), tested on both frames specifically to
see whether it rescues the regression `exp_8` measured. Macro-F1 is now reported natively across
all four subtasks (the reporting convention just backfilled into `exp_6`–`exp_8` this session).

**This plan, once approved, gets saved as `experiments/exp_9/IMPLEMENTATION.md`.**

## Key technical findings from this planning session (these determine the implementation)

1. **The ARD reformulation is a rescale, not a rewrite.** `RBFKernelLayer.forward()`
   (`kdm/layers/rbf_kernel_layer.py`) computes `‖A−B‖²` via the norm trick
   (`A_norm + B_norm − 2·A·Bᵀ`) then divides by `2σ²`. For per-dimension weighting,
   `Σⱼ(aⱼ−bⱼ)²/σⱼ² = ‖A/σ − B/σ‖²` — so `ARDRBFKernelLayer.forward()` just rescales `A` and `B`
   by `1/σ` (broadcasting a `(dim,)` vector over the last axis) **before** calling the exact same
   norm-trick distance computation, then divides by `2` (not `2σ²` again — that's already baked
   into the rescaled coordinates). This reuses the parent's distance logic unchanged; only the
   rescale-then-call wrapping is new.
2. **`kdm.init.init_kdm_layer()` already works unchanged for ARD initialization, once `raw_sigma`
   has the right shape.** The `sigma` property setter does `self.raw_sigma.copy_(...)`, which
   broadcasts a scalar source into a `(dim,)`-shaped destination correctly (`copy_` performs a
   real per-element copy, not an aliased view) — so `init_kdm_layer(model.kdm, ..., init_sigma=True,
   sigma_mult=sigma_mult)` needs zero changes to initialize every `σⱼ` at the same KNN-based
   scalar value. **Must build `raw_sigma` as `torch.full((dim,), value)` in `__init__`, never
   `.expand()`** — `.expand()` returns a view where all "copies" alias the same memory, which
   would silently make every `σⱼ` move in lockstep during training (defeating ARD's entire point
   without raising any error). This is the single highest-risk implementation detail — verified
   explicitly in step 2 of Verification below, not assumed.
3. **`compute_signals()`'s dispersion signal (Signal B) needs an ARD-aware correction — everything
   else in `kdm_backbone.py` is already sigma-shape-agnostic.** Entropy (uses only `probs`),
   participation ratio (uses only `out_w`), `occlusion_delta()`, and `kernel_distance_contribution()`
   (neither references `sigma` at all — they use raw, unweighted squared distance) all work
   unchanged with an ARD-fitted model. But `dm_rbf_variance(dm, sigma)`
   (`kdm/utils.py:110`) computes its within-component term as `d · σ²`, which assumes an
   *isotropic* per-component covariance (`σ²·I`) — wrong for ARD's anisotropic
   `diag(σ₁², ..., σ_d²)`. The correct generalization is `Σⱼ σⱼ²` (sum of per-dimension
   variances), not `d·σ²`. This needs a small new function, `dm_rbf_variance_ard()`, in `exp_9`'s
   own scripts (not a library patch) — silently reusing the unmodified library function with a
   sigma *vector* would produce a shape-mismatched or silently-wrong dispersion value.
4. **`kernel_distance_contribution()`'s Signal E formula (`Σⱼ(xⱼ−cⱼ)²`, unweighted) stays
   unchanged for this experiment, deliberately** — even though the model's own kernel now
   *does* weight dimensions via `σⱼ`, changing Signal E's own formula to match
   (`Σⱼ(xⱼ−cⱼ)²/σⱼ²`) would confound "does ARD help the backbone" with "does an ARD-aware
   attribution formula help weights," which isn't this experiment's question. Noted as a natural
   `exp_10` idea in the report, not implemented here.

## Files to Add

### 1. `experiments/exp_9/scripts/ard_kernel.py` — the core new module

- `ARDRBFKernelLayer(RBFKernelLayer)`: overrides `__init__` (builds `raw_sigma` as
  `torch.full((dim,), softplus_inv(sigma - min_sigma))`, not `.expand()`), overrides `forward()`
  per finding #1, overrides `log_weight()` to sum `log(σⱼ)` over dimensions instead of
  `dim · log(σ)` (for API completeness — confirmed this project's training loop never actually
  calls `log_weight()`/`log_marginal()`, only the discriminative `forward()` + NLL path, so this
  is unexercised but should still be correct, not left silently wrong).
- `ARDKDMClassModel(KDMClassModel)`: overrides only `__init__`'s kernel construction line to
  build `self.kernel = ARDRBFKernelLayer(...)` instead of the hardcoded `RBFKernelLayer` — `self.kdm
  = KDMLayer(kernel=self.kernel, ...)` and `forward()` are inherited unchanged, since `KDMLayer`
  accepts any kernel object satisfying `forward(A,B)`.
- `fit_kdm_backbone_ard(X_train, y_train, n_classes=2, n_epochs=300, lr=1e-2, sigma_mult=1.0) ->
  ARDKDMClassModel`: mirrors `kdm_backbone.fit_kdm_backbone()`'s body exactly (same
  `torch.manual_seed`, same `init_kdm_layer(..., init_sigma=True, sigma_mult=sigma_mult)` call
  per finding #2, same Adam-only training loop per `DESIGN.md`'s no-search guardrail — no
  optimizer/weight_decay branching this round), constructing `ARDKDMClassModel` instead.
- `dm_rbf_variance_ard(dm, sigma_vector)`: per finding #3.
- `compute_signals_ard(model, X) -> dict`: copy of `kdm_backbone.compute_signals()` with only the
  dispersion line changed to call `dm_rbf_variance_ard(..., model.kernel.sigma)` instead of the
  library's `dm_rbf_variance`. Same `probs_check_ok` cross-check retained unchanged (still valid —
  it doesn't touch sigma at all, only `out_w`/`c_y`).
- Re-export `occlusion_delta`, `kernel_distance_contribution` from `exp_6/scripts/kdm_backbone.py`
  unchanged (finding #3 confirms they need no ARD-specific version).

### 2. `experiments/exp_9/scripts/run_signals_19col.py` and `run_signals_23col.py`

Each is `exp_8/scripts/run_signals_v3.py`'s structure (decision + 5 confidence + 3 weights
conditions, same isotonic/blend recalibration logic unchanged) with: `fit_kdm_backbone_ard`/
`compute_signals_ard` from `ard_kernel.py` instead of `kdm_backbone_v2`'s versions; feature frame
is `select_exp3_feature_frame` (19-col) or `select_exp8_feature_frame` (23-col) respectively; no
`winner.json` dependency (fixed hyperparameters per the no-search guardrail — `n_epochs=300,
lr=1e-2, sigma_mult=1.0`, matching `exp_6`'s original defaults exactly, so any difference from
`exp_6`'s numbers is attributable to ARD alone, not a confounded hyperparameter change).

### 3. `experiments/exp_9/scripts/run_reveal_19col.py` and `run_reveal_23col.py`

Copies of `exp_8/scripts/run_reveal_kdm.py` using `fit_kdm_backbone_ard`, the two feature frames,
and (19-column version only) a trimmed `SECTION_FEATURE_GROUPS["psa_trend"] = ["cli_psa",
"cli_psad"]` (no `psav`/`psap` — absent from that frame). Both confirm the same 4 modeled
sections dynamically (`[s for s in REVEAL_SECTIONS if any(...)]`), not hardcoded.

### 4. `experiments/exp_9/scripts/holdout_eval_ard.py`

Adapted from `exp_8/scripts/holdout_eval_v3.py` — same held-out split, compares `exp_6`'s
original scalar-sigma KDM vs. ARD on both frames (3-way comparison on the same 19 held-out cases:
scalar/19-col, ARD/19-col, ARD/23-col) rather than a 2-way one, since this experiment's central
question is specifically about the frame-crossed-with-architecture interaction.

### 5. `experiments/exp_9/scripts/importance_comparison.py`

Fits the ARD backbone once per frame (full data, no CV — this is a diagnostic script, not a
scored condition), extracts trained `σⱼ` per column, inverts to a relevance score (`1/σⱼ`),
groups columns by `TASK1_VARIABLE_TO_FEATURE_GROUP`/`restricted_feature_group()` (reuse
unchanged) to get one aggregate relevance score per weight factor, and prints/writes a comparison
table against `exp_5`'s per-factor SVM results (`results/weights_official_svm/metrics.json`'s
solvable/unsolvable pattern) and this project's `exp_5`-established factor split
(`pirads`/`bx`/`dre`/`age`/`psa` solved vs. `cspca`/`comorbidity`/`psad`/`vol` not).

### 6. No changes to `exp_6`/`exp_7`/`exp_8`'s scripts, the `kdm` library, or any `src/chimera_task1/*.py`

Same rule as every prior experiment.

## Execution Order (priority per DESIGN.md §9 if time-constrained)

**Priority 1** (the core ARD-vs-scalar test): `run_signals_19col.py`, `run_signals_23col.py`,
`holdout_eval_ard.py`.
**Priority 2**: `run_reveal_19col.py`, `run_reveal_23col.py`, `importance_comparison.py`.

## Verification

1. **Smoke-test `ARDRBFKernelLayer` against `RBFKernelLayer` at matched-identical `σⱼ`** — if
   every `σⱼ` is initialized to the exact same scalar value, `ARDRBFKernelLayer.forward()` must
   produce numerically identical output to the parent `RBFKernelLayer.forward()` on the same
   inputs (a degenerate ARD model *is* a scalar model). This is the cheapest, most direct
   correctness check on finding #1's reformulation, before trusting anything built on top of it.
2. **Smoke-test that `σⱼ` values actually diverge after training** (finding #2's risk) — fit on
   one fold, confirm `model.kernel.sigma`'s per-dimension values are *not* all equal after
   training (they start equal from `init_kdm_layer`, so any divergence confirms independent
   gradients; if they stay locked together, the `.expand()` vs. `torch.full()` bug is present).
3. **Smoke-test `compute_signals_ard`'s `probs_check_ok`** on one fold, same as every prior
   experiment's verification discipline.
4. **Smoke-test `dm_rbf_variance_ard` against the library's `dm_rbf_variance`** at matched-equal
   `σⱼ` (should agree: `Σⱼσⱼ²` with all `σⱼ` equal to `σ` reduces to `d·σ²`) — confirms finding
   #3's correction is a true generalization, not an unrelated formula.
5. Run Priority 1 scripts (background each — 2 frames × 10 conditions × 5×10 CV is comparable in
   scale to `exp_8`'s full run); run the mandatory held-out check.
6. Run Priority 2 scripts.
7. Compare every result against `DESIGN.md` §4's baseline table (now including backfilled
   macro-F1) and §8's decision-rule branches before writing `experiments/exp_9/reports/summary.md`.
