# exp_14 Implementation Plan — KDM Regression for Weights (Ordinal-Aware Training)

## Context

`experiments/exp_14/DESIGN.md` (status: Proposed, written this session) proposes the first
genuinely new KDM model class this project has used for **weights**: `kdm-torch`'s
`KDMRegressModel`, trained with a regression loss (`dm_rbf_loglik`) directly on the ordinal weight
rank (0–3), instead of `KDMClassModel`'s classification loss (which treats every misclassification
identically regardless of rank distance). Every prior KDM weights attempt — `exp_6`/`exp_9`'s
derived-signal approach and `exp_13`'s direct-training revival (scalar/ARD × official/restricted) —
kept the classification objective and only varied bandwidth structure or supervision source. This
experiment changes the training objective itself to one that is shaped like the subtask's own
primary metric (mean ordinal error), confirmed as in-scope by the user ("this project is completely
based on KDM, then all the attempts here are in that way").

Two conditions: **per-factor regression** (9 independent regressors, `dim_y=1`, directly comparable
to `exp_13`'s per-factor classification KDM) and **joint multi-factor regression** (1 regressor,
`dim_y=9`, predicting all 9 factors from one shared prototype pool — untried by any prior weights
condition, including the SVM incumbent).

**This plan, once approved, gets saved as `experiments/exp_14/IMPLEMENTATION.md`.**

## Verified before writing this plan

Ran three direct smoke tests against the installed `kdm-torch==2.0.0` this session (not assumed
from documentation):

1. **`KDMRegressModel(dim_y=9, x_train=False, y_train=False, w_train=False)` fits and predicts
   cleanly** — memory-based config (matching every other KDM condition in this project) works
   for the joint condition exactly as it does for `dim_y=1`.
2. **`predict_reg(x)` returns `(mean, variance)` with `mean.shape == (n, dim_y)` but
   `variance.shape == (n,)`** — a single scalar variance per case, not one per output dimension,
   confirmed by reading `KDMRegressModel.predict_reg`'s source (`dm_rbf_variance(rho_y, sigma_y)`
   is inherently scalar-valued, the same between/within-component formula documented in the
   "One Backbone, Four Signals" artifact). **This is a real limitation of the joint condition**:
   its regression-derived pseudo-probabilities (see below) will share one variance across all 9
   factors within a case, only the per-factor mean differs — call this out explicitly in the
   report, don't quietly average over it.
3. **A full 300-epoch training run on 80 rows × 23 dims × random 0–3 integer targets converges
   cleanly** — loss decreases monotonically (−0.52 → −4.45), `sigma_x` shrinks from its
   nearest-neighbor init (4.99 → 2.14), `sigma_y` shrinks from its constructor default
   (0.10 → 0.010), no NaNs, in-sample rounded accuracy reaches 1.0 (expected memorization,
   the same sanity signature every classification-KDM smoke test in this project has shown).
   **`init_kdm_layer(..., init_sigma=True)` only sets `kernel.sigma` (the input-side bandwidth)
   — `sigma_y` (the output-side kernel width) is left at its constructor default (0.1) and trained
   from there.** This is a real asymmetry vs. classification KDM (which has no `sigma_y` at all)
   worth a one-line comment in the fit function, not a bug to fix.

## Key design decisions

1. **Frame**: 23-column only (`select_exp8_feature_frame`), scalar bandwidth only for the primary
   run — `exp_13` already showed backbone choice barely matters for weights (0.454 vs. 0.454 CV
   across scalar/ARD), so this experiment's budget goes to the training-objective change, not a
   second backbone sweep. ARD-regression is an optional stretch goal only if the scalar results
   are promising enough to warrant it.
2. **Continuous → ordinal**: round `predict_reg`'s mean to the nearest integer, clip to `[0, 3]`,
   map through `WEIGHT_LEVELS` — identical downstream handling to every classification-KDM
   condition (`ordinal_distance`, `decisive_set_f1`, `accuracy_score`, `f1_score` all reused
   unchanged from `chimera_task1.reasoning_labels`/`sklearn.metrics`).
3. **AUROC/Brier via regression-derived pseudo-probabilities**: treat `(mean, variance)` as a
   Normal distribution over the continuous rank; assign each of the 4 discrete ranks the
   probability mass in `[k−0.5, k+0.5)` via `scipy.stats.norm.cdf` (or the `torch` equivalent),
   renormalized to sum to 1 over the 4 valid ranks. Labeled explicitly as an approximation in every
   metrics payload (`"auroc_brier_note"` field), never presented as a native classifier probability.
4. **Staged LOO** (same cost-management pattern as `exp_13` §2d): CV + held-out first for both
   conditions; LOO only for whichever clears/approaches the primary bar (`weights_kdm_occlusion`'s
   0.405 ordinal error — the best-ever KDM result, per the KDM-only scope confirmed by the user;
   the SVM incumbent 0.382 is still reported for transparency but is not the primary bar).

## Files to Add

### 1. `experiments/exp_14/scripts/kdm_regress_backbone.py` — shared regression-KDM helpers

- `fit_kdm_regress(X_train, y_train_rank, dim_y, n_epochs=300, lr=1e-2) -> KDMRegressModel`:
  mirrors `experiments/exp_6/scripts/kdm_backbone.py`'s `fit_kdm_backbone` shape exactly
  (`torch.manual_seed(0)`, `nn.Identity()` encoder, `n_comp=len(X_train)`, memory-based
  `x_train=y_train=w_train=False`, `init_kdm_layer(..., init_sigma=True)`, Adam over
  `[p for p in model.parameters() if p.requires_grad]` at `lr=1e-2`, 300 epochs) but builds
  `KDMRegressModel(dim_y=dim_y, ...)`, reshapes `y_train_rank` to `(n, dim_y)` (a `(n,1)` column
  for per-factor, a `(n,9)` matrix of every factor's rank at once for joint), and trains with
  `loss = -dm_rbf_loglik(yt, model(Xt), model.sigma_y).mean()` instead of `F.nll_loss`.
- `compute_signals_regress(model, X, n_levels=4) -> dict`: calls `predict_reg(Xt)` for
  `(mean, variance)`; rounds+clips `mean` to `pred_rank` (shape `(n, dim_y)`); builds the 4-bin
  Normal-CDF pseudo-probabilities per output dimension from `(mean[:, j], variance)` (note:
  `variance` is broadcast across every `j` for the joint condition, per finding #2 above); returns
  `{"mean": ..., "variance": ..., "pred_rank": ..., "pseudo_probs": ...}` — shape `(n, dim_y)` for
  `mean`/`pred_rank`, `(n, dim_y, 4)` for `pseudo_probs`.
- Both functions take `dim_y` explicitly rather than inferring it, so the exact same pair of
  functions serves both the per-factor (`dim_y=1`, called 9 times) and joint (`dim_y=9`, called
  once) conditions — no duplicated fit/predict logic between the two run scripts.

### 2. `experiments/exp_14/scripts/run_weights_regress_per_factor.py`

CV loop (5×10, `RANDOM_STATE + repeat` seeding, matching `exp_13`'s exact shape), 23-col frame,
looping the 9 in-scope factors, calling `fit_kdm_regress(..., dim_y=1)` per factor per fold. Same
`try/except ValueError` degenerate-fit skip discipline as every prior weights script. Reuses
`experiments/exp_11/scripts/metrics_multiclass.py`'s `multiclass_brier_score`/`safe_multiclass_auroc`
unchanged, fed the pseudo-probabilities instead of a classifier's native `probs`. Writes
`results/weights_kdm_regress_per_factor/metrics.json` in the same per-factor-aggregate shape every
prior weights condition uses.

### 3. `experiments/exp_14/scripts/run_weights_regress_joint.py`

Same CV shape, but one `fit_kdm_regress(..., dim_y=9)` call per fold covering every factor at once
— `y_train_rank` built by stacking all 9 factors' ranks into an `(n_train, 9)` array before the
fold loop. Per-factor metrics are then sliced out of the joint model's `(n, 9)` predictions/
pseudo-probs for reporting in the same shape as condition 2, so the two conditions are directly
comparable factor-by-factor, not just in aggregate.

### 4. `experiments/exp_14/scripts/holdout_eval_weights_regress_{per_factor,joint}.py`

Same fixed decision-stratified split used since `exp_3` (`holdout_eval.py`'s
`mri_pca_train_only`/`fit_transform_features`, reused unchanged, same import pattern as
`experiments/exp_13/scripts/holdout_eval_weights_direct_scalar.py`).

### 5. `experiments/exp_14/scripts/loo_weights_regress_{per_factor,joint}.py`

Only written/run for whichever condition(s) clear the CV/held-out bar (§ staged LOO above) —
`LeaveOneOut()`, pooled predictions, scored once, following `exp_13`'s LOO-skip precedent (which
this experiment may also end up exercising if neither condition clears the bar).

### 6. No changes to `exp_1`–`exp_13`'s scripts, the `kdm` library, or `src/chimera_task1/*.py`

Same rule as every prior experiment.

## Execution Order

1. **Smoke test** (`experiments/exp_14/scripts/smoke_test.py`, new): confirm
   `fit_kdm_regress`/`compute_signals_regress` on one real factor (`dre`, historically easiest,
   same choice `exp_13`'s smoke test made) for both `dim_y=1` and `dim_y=9` on the real 23-col
   frame — check `pseudo_probs` rows sum to 1 per output dimension, `pred_rank` stays in `[0,3]`,
   no NaNs — before any scored run (mirrors every prior experiment's pre-flight discipline).
2. Run `run_weights_regress_per_factor.py` and `run_weights_regress_joint.py` (CV, background).
3. Run both held-out scripts.
4. Compare against `DESIGN.md` §4's baseline table and §5's decision rules; decide LOO scope.
5. Run LOO for qualifying condition(s) only.
6. Write `experiments/exp_14/reports/summary.md`, update `DESIGN.md` status to Complete,
   `experiments/INDEX.md`'s row, and project memory — same closing sequence as `exp_13`.

## Verification

1. Smoke test passes (step 1 above) before any CV run is trusted.
2. Sanity-check the CV result isn't degenerate (e.g., all predictions collapsing to one rank) —
   print per-factor prediction distributions alongside the metrics, the way `exp_13`'s scripts
   already print per-factor `ordinal_error`/`macro_f1` as they go.
3. Confirm the joint condition's per-factor breakdown is derived from the *same* single fit per
   fold (not accidentally re-fit per factor) — verify by checking the joint script's fit-call count
   equals `n_folds`, not `n_folds × 9`.
4. Every metrics payload's AUROC/Brier fields carry the `"auroc_brier_note"` approximation flag
   before being trusted or quoted in the report.
