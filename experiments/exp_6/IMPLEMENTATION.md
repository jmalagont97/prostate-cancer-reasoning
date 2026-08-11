# exp_6 Implementation Plan — KDM as a Unified Backbone for Decision + Confidence + Weights

## Context

This implements `experiments/exp_6/DESIGN.md` (status: Proposed, written and reviewed this
session). `exp_3`–`exp_5` treat decision, confidence, and variable-weights as three independent
model-search problems, each won by a different model family (Extra Trees / SVM / SVM). `exp_6`
tests whether **one trained KDM**, on the decision target, can instead serve as a shared
backbone — deriving confidence from the model's own predictive uncertainty and per-case variable
importance from its own kernel-attention structure, with no separate model family per target.

This session traced the actual `kdm` package source (`kdm/layers/kdm_layer.py`, `kdm/utils.py`,
`kdm/init.py`, `kdm/layers/rbf_kernel_layer.py`) to ground the design in what's really there:
- This project's KDM config (`x_train=y_train=w_train=False`) freezes `c_y` at exact one-hot
  labels (`init_kdm_layer` copies `y_onehot` straight in, `y_train=False` means it's never
  updated) — so the model is a Nadaraya-Watson kernel-vote classifier, and output-level Shannon
  entropy is already the full output-side uncertainty; no exploitable label-space coherence.
- The genuinely new, currently-unused signal is one level upstream: `KDMLayer._compute_mixture()`
  computes a per-query posterior weight over the frozen training prototypes (`out_w`, shape
  `(bs, n_comp)`), and `kdm.utils.dm_rbf_variance(dm, sigma)` — shipped in the library, never
  called anywhere in this project — computes the RKHS-space dispersion of a weighted point
  mixture. Feeding it `(out_w, c_x)` gives a genuine epistemic-dispersion signal distinct from
  entropy.
- `RBFKernelLayer.sigma` is a single global scalar (confirmed in `rbf_kernel_layer.py`), not
  per-dimension — ruling out a free, model-native *global* importance signal. Per-case
  importance (what the `variable_weights` target actually needs) comes from a different
  mechanism: local occlusion, or a per-feature decomposition of the kernel distance term.

**This plan, once approved, gets saved as `experiments/exp_6/IMPLEMENTATION.md`** before any
other files are touched, per this project's established convention.

## Design decisions locked by this plan

1. **One shared fold loop, not 8 independent condition scripts.** Unlike `exp_3`–`exp_5` (each
   condition retrains its own model), `exp_6`'s whole hypothesis is "one trained artifact, several
   readouts" — so the backbone KDM is fit **once per fold per repeat**, and every signal (A–E) is
   computed from that single fit. Recalibration (the only per-condition-specific step) happens
   after, on cached signal arrays. This is both truer to the hypothesis and ~8x cheaper than
   independent refits.
2. **CV folds are decision-stratified... actually plain `KFold`**, matching the exact
   fold-splitting already used by `decision_kdm`/`confidence_kdm` in `exp_3` (`KFold(n_splits=5,
   shuffle=True, random_state=RANDOM_STATE+repeat)`, not `StratifiedKFold`) — so
   `decision_kdm_backbone`'s re-verification number should reproduce `exp_3`'s existing
   `decision_kdm` macro-F1 (0.588) as a correctness check on the refactor, before trusting any
   downstream signal.
3. **`N_REPEATS=10`**, matching `exp_3`/`exp_4`'s decision/confidence KDM conditions (not `exp_5`'s
   reduced 5) — exp_6 is cheaper per-repeat than exp_5 was, no reason to cut repeats.
4. **Signal D (occlusion) uses `Δp(yes)`** as the primary occlusion metric (simplest, most
   interpretable — "how much does hiding this factor change the decision probability"), with
   fill values = that fold's **training-set median** (continuous columns) or **mode** (binary/
   one-hot columns) for the occluded factor's column group.
5. **Factor → column-group mapping reuses `features.restricted_feature_group(factor, "flags")`
   unchanged** — the exact same 9-factor mapping already used by `weights_restricted_*` in
   `exp_2`/`exp_3`/`exp_5`. No new mapping invented.
6. **"Blend" conditions reuse `train_reasoning.make_classifier()`** (the project's existing
   `OneVsRestClassifier(LogisticRegression)` pattern) fit on 2–3 raw signals as features, rather
   than inventing a new small-model type.
7. **No changes to `src/chimera_task1/{train_confidence_kdm,train_reasoning,train_decision,features,reasoning_labels}.py`** — same rule as every prior experiment. `fit_predict_kdm()` in
   particular is left untouched (still used verbatim by `exp_2`–`exp_5`); exp_6 adds a **new**
   function alongside it that returns the fitted model object instead of just probabilities.

## Files to Add

### 1. `experiments/exp_6/scripts/kdm_backbone.py` — new, the core diagnostic module

- `fit_kdm_backbone(X_train, y_train, n_classes=2) -> KDMClassModel`: a copy of
  `fit_predict_kdm`'s fit loop from `train_confidence_kdm.py` (identical hyperparameters: 300
  epochs, Adam lr=1e-2, sigma-only trainable, `n_comp=len(X_train)`), but returns the trained
  `model` object in eval mode instead of calling `.numpy()` on a prediction. `torch.manual_seed`
  kept identical for reproducibility with the existing `decision_kdm` numbers.
- `compute_signals(model, X) -> dict` — given a fitted model and a batch of (already-scaled)
  query rows, returns:
  - `probs`: `dm2discrete(model.kdm(pure2dm(X_t)))`, shape `(n, 2)` — Signal-A input.
  - `entropy`: `-Σ probs·log(probs)`, shape `(n,)` — **Signal A**.
  - `out_w`: the normalized per-prototype posterior weight, replicating `KDMLayer.forward`'s
    `_compute_mixture` → clamp → normalize steps *exactly* (up to but not including the final
    `c_y` matmul), shape `(n, n_comp)`.
  - `dispersion`: `dm_rbf_variance(comp2dm(out_w, c_x_expanded), model.kernel.sigma)` — **Signal
    B**. `c_x_expanded` = `model.kdm.c_x` broadcast to `(n, n_comp, dim_x)`.
  - `participation`: `1 / (out_w**2).sum(dim=1)` — **Signal C**.
  - **Verification built into this function**: assert `probs` reconstructed manually from
    `out_w` and `model.kdm.c_y` (`Σ_i out_w_i · normalize(c_y_i)²`) matches `model.kdm.forward`'s
    own output to floating-point tolerance — catches any subtle mismatch in replicating the
    internal normalization before any signal is trusted downstream (see Verification §2 below).
- `occlusion_delta(model, X, factor_cols, fill_values) -> np.ndarray` — **Signal D**. For each
  row, build a copy with `factor_cols` set to `fill_values` (a dict of column → fold-training
  median/mode, computed by the caller), re-run `compute_signals(model, X_occluded)`, return
  `probs_occluded[:,1] - probs_original[:,1]` (signed; caller takes `abs()` for a magnitude
  score if needed).
- `kernel_distance_contribution(model, X, factor_cols) -> np.ndarray` — **Signal E**. Direct
  computation, no re-inference: for each row, `Σ_i out_w_i · Σ_{j∈factor_cols} (X_j - c_x[i,j])²`
  using the same `out_w`/`c_x` already computed by `compute_signals`.

### 2. `experiments/exp_6/scripts/run_signals.py` — the single driver script

- Loads data via `train_reasoning.load_annotated()` (gives `ann`/`inp_ann`, the same 91-case set
  decision/confidence/weights all already share — confirmed identical across the project).
- Builds the 19-column frame via `select_exp3_feature_frame` + `mri_pca_features(full_inp,
  n_components=2)` aligned by `case_id`, exactly as `exp_3`'s `run_confidence.py` does.
- Extracts `y_decision` (`ann["target_biopsy_decision"]`), `y_confidence`
  (`ann["target_confidence"]`), and per-factor `ann[weight_col(f)]` for `f in IN_SCOPE_FACTORS`
  (`reasoning_labels.TASK1_FACTORS` minus `fh`, same as `exp_5`).
- **Main loop** — for `repeat in range(10)`, `KFold(5, shuffle=True, random_state=repeat)` over
  the 91 rows:
  - Fit `preprocessor` + `StandardScaler` on the fold's train rows (same pattern as every prior
    KDM condition), fit `fit_kdm_backbone(X_train, y_decision_train)`.
  - `compute_signals()` on both train and test rows (train needed for recalibration fitting).
  - `occlusion_delta` / `kernel_distance_contribution` for each of the 9 factors, train + test.
  - For each of the 8 non-backbone conditions (§ table below), fit that condition's recalibrator
    on **train rows' signals + train rows' target** only, predict on test rows' signals,
    accumulate into that condition's out-of-fold prediction array.
  - Accumulate `probs` for the `decision_kdm_backbone` condition directly (no recalibration
    needed — it's just the backbone's own decision output).
- After the full loop, score every condition with the existing metric functions
  (`ordinal_distance`, `decisive_set_f1` from `reasoning_labels.py`; `f1_score(average="macro")`
  for decision) and write `results/<condition>/metrics.json`, same schema/`write_metrics()`
  pattern as `exp_3`/`exp_5`.

### 3. Recalibrators (small, inline in `run_signals.py`, no new shared module needed)

| Condition | Recalibrator |
|---|---|
| `confidence_kdm_entropy_zeroshot` | tercile boundaries of **train** entropy → bin test entropy into 3 confidence levels. No fit params beyond 2 percentile thresholds. |
| `confidence_kdm_entropy_isotonic` | `sklearn.isotonic.IsotonicRegression()` on (train entropy → train `CONFIDENCE_RANK`), predict test, round+clip to `{0,1,2}`. |
| `confidence_kdm_dispersion_isotonic` | same, on Signal B. |
| `confidence_kdm_participation_isotonic` | same, on Signal C. |
| `confidence_kdm_blend` | `train_reasoning.make_classifier()` fit on `[A,B,C]` (train) → predict on test. |
| `weights_kdm_occlusion` | per-factor `IsotonicRegression()` on (train `|Δp(yes)|` → train `WEIGHT_RANK`), round+clip to `{0,1,2,3}`. |
| `weights_kdm_kernel_distance` | same, on Signal E per factor. |
| `weights_kdm_blend` | per-factor `make_classifier()` fit on `[D,E]`. |

Weights conditions wrap each factor's fit in `try/except ValueError` exactly as `exp_5`'s
`run_weights.py` does, recording `skipped`/`n_factors_included` — same data-scarcity classes
(e.g. `psa`'s `not_used` has 1 example total) apply here too.

## Verification

1. **Reproduce `decision_kdm_backbone` against `exp_3`'s existing `decision_kdm`** (macro-F1
   0.588) before trusting anything else — same fold seed, same hyperparameters, refactored fit
   function only. A mismatch means the fit-vs-predict split broke something.
2. **Assert the manual `probs` reconstruction from `out_w`/`c_y` matches `model.kdm.forward()`'s
   own output** (built into `compute_signals`, per §1 above) — confirms the hand-replicated
   normalization logic is correct before Signals B/C/D/E are trusted at all.
3. **Smoke-test on a small subset first** (one fold, one repeat) — print signal ranges (entropy
   ∈ [0, log 2], dispersion ≥ 0, participation ∈ [1, n_comp]) and sanity-check occlusion deltas
   are near-zero for an obviously-irrelevant factor vs. larger for `pirads`/`bx` before committing
   to the full 10×5 loop, consistent with this project's established smoke-test-before-full-run
   discipline (caught the KDM `liblinear` issue in `exp_1`, the rare-class shape mismatch in
   `exp_5`).
4. Run the full script (background it — 10×5×(1 backbone fit + 9-factor occlusion/kernel-distance
   passes) is more fits than `exp_3`'s decision/confidence KDM but far fewer than `exp_5`'s
   restricted-scope weights); confirm all 9 `results/<condition>/metrics.json` files are written
   and valid JSON.
5. Compare against the incumbents from `experiments/exp_6/DESIGN.md` §4 (confidence baseline
   0.527 / incumbent 0.468; weights baseline 0.413 / incumbent 0.382–0.392; decision baseline
   0.381 / incumbent 0.650) before writing `experiments/exp_6/reports/summary.md`, per this
   project's established review-before-report pattern. Report the per-factor weights breakdown
   explicitly (not just the 9-factor aggregate), per `DESIGN.md` §7.
