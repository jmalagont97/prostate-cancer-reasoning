# exp_7 Implementation Plan — Tuning + Skew-Aware Preprocessing for exp_6's KDM Backbone

## Context

This implements `experiments/exp_7/DESIGN.md` (status: Proposed, written and refined this
session — hyperparameter grid, AdamW addition, and scope guardrails are already locked, not open
questions). `exp_6` built a KDM backbone for decision that also derives confidence/weights
signals, but never tuned its hyperparameters (`N_EPOCHS=300`, `lr=1e-2`, implicit `sigma_mult=1.0`,
Adam-only) or preprocessed its input with the model's own sensitivities in mind. `exp_7` runs a
bounded 144-combination search (adding AdamW/`weight_decay` as a lever per this session's
explicit request) plus a skew-aware `log1p` transform on `cli_psa`/`cli_psad`/`cli_vol`, then
re-runs `exp_6`'s exact, unchanged confidence/weights readout code on the resulting backbone —
isolating whether `exp_6`'s weak confidence/weights results were a backbone problem or a readout
problem.

**This plan, once approved, gets saved as `experiments/exp_7/IMPLEMENTATION.md`** before any
other files are touched, per this project's established convention.

## Key reuse decision (found during exploration, simplifies the build significantly)

`exp_6/scripts/kdm_backbone.py`'s `compute_signals()`, `occlusion_delta()`, and
`kernel_distance_contribution()` all operate purely on an already-fitted model object — none of
them care *how* that model was trained. So `exp_7` does **not** need to duplicate them; it
imports them directly from `exp_6/scripts/kdm_backbone.py` (same cross-experiment reuse pattern
`exp_4`/`exp_5` already established for `exp_3`'s `cv_utils.py`/`models.py`). Only the **fit**
function needs a new, configurable version.

Also confirmed: `kdm.init.init_kdm_layer()` already accepts `sigma_mult` as an explicit parameter
— no need to reimplement the KNN-based sigma-init logic, just thread the value through.

## Files to Add

### 1. `experiments/exp_7/scripts/kdm_backbone_v2.py`

- Imports `compute_signals`, `occlusion_delta`, `kernel_distance_contribution` from
  `exp_6/scripts/kdm_backbone.py` unchanged (re-exported for downstream scripts' convenience).
- `LOG1P_COLUMNS = ["cli_psa", "cli_psad", "cli_vol"]` and `apply_log1p_transform(X_pre: np.ndarray,
  col_idx: list[int]) -> np.ndarray` — copies the array, applies `np.log1p` to the given columns.
  Called once on the post-imputation, pre-scaling array (no fitted parameters, so safe to apply
  before the CV split — leakage discipline only applies to *fitted* statistics like the scaler's
  mean/std or the imputer's median, which still get fit per-fold downstream as today).
- `fit_kdm_backbone(X_train, y_train, n_classes=2, n_epochs=300, lr=1e-2, sigma_mult=1.0,
  optimizer="adam", weight_decay=0.0) -> KDMClassModel` — a copy of `exp_6`'s fit loop with these
  five values threaded through: `sigma_mult` into the existing `init_kdm_layer(..., sigma_mult=
  sigma_mult)` call, `n_epochs` as the training-loop range, and an `if optimizer == "adamw":
  torch.optim.AdamW(..., lr=lr, weight_decay=weight_decay) else torch.optim.Adam(..., lr=lr,
  weight_decay=weight_decay)` branch. `torch.manual_seed(RANDOM_STATE)` kept identical.

### 2. `experiments/exp_7/scripts/search_hyperparameters.py`

- Same data/frame setup as `exp_6/scripts/run_signals.py` (`load_annotated`, `mri_pca_features`,
  `select_exp3_feature_frame`, `build_preprocessor`), plus one call to `apply_log1p_transform`
  on the resulting `X_pre` before the CV loop.
- `itertools.product` over the 144-combination grid from `DESIGN.md` §2 (epochs × lr × sigma_mult
  × {(Adam, wd=0), (AdamW, wd=0), (AdamW, wd=1e-4), (AdamW, wd=1e-3)}).
- For each combination: 5-fold × 3-repeat CV (`RANDOM_STATE=0`, same splitting convention as
  every prior experiment), `StandardScaler` fit per fold as today, `fit_kdm_backbone(...)` with
  that combination's hyperparameters, decision macro-F1 via `compute_signals(model, X_test)
  ["probs"].argmax(axis=1)` and `f1_score(..., average="macro")`.
- Writes **all 144 results** to `results/hyperparameter_search/grid.csv` (one row per
  combination, for the record — a table, not 144 JSON folders) and the single best-by-mean-macro-F1
  combination to `results/hyperparameter_search/winner.json` — downstream scripts read this file
  programmatically rather than a hardcoded/copy-pasted config, avoiding transcription risk.
- Prints the top 10 combinations and explicitly prints the margin over `exp_6`'s 0.593, per
  `DESIGN.md`'s "clear margin" discipline — flags in the printed output (not just silently
  proceeds) if the margin is small enough that CV noise is a plausible explanation.

### 3. `experiments/exp_7/scripts/run_signals_v2.py`

Copy of `exp_6/scripts/run_signals.py` with exactly two changes: (a) `X_pre` gets
`apply_log1p_transform`'d right after `build_preprocessor(...).fit_transform(...)`, before the
per-fold `StandardScaler`; (b) every `fit_kdm_backbone(X_train, y_decision[train_idx],
n_classes=2)` call becomes `fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2,
**winning_config)`, where `winning_config` is loaded from `results/hyperparameter_search/
winner.json` at the top of the script. All 9 conditions, all recalibration logic (isotonic
`increasing="auto"`, the `make_classifier()` blend, per-factor try/except), and the per-factor
weights breakdown are otherwise **identical** to `exp_6` — this script's whole purpose is
isolating the backbone change, not introducing new readout logic. Writes to
`results/{condition}_v2/metrics.json` (9 folders, matching `DESIGN.md` §3/§5 naming) —
`decision_kdm_v2`'s full 10-repeat number comes out of this same unified loop, exactly as
`exp_6`'s `decision_kdm_backbone` did, so no separate decision-only re-verification script is
needed.

### 4. `experiments/exp_7/scripts/run_ablations.py`

The 2 isolation conditions from `DESIGN.md` §6 — decision-only macro-F1 (no confidence/weights
readout needed for these), full 5×10 CV:
- `decision_kdm_log1p_only`: `fit_kdm_backbone` at the **original** fixed hyperparameters
  (`n_epochs=300, lr=1e-2, sigma_mult=1.0, optimizer="adam", weight_decay=0.0`) on the
  **log1p-transformed** frame.
- `decision_kdm_tuned_only`: `fit_kdm_backbone` at the **winning** hyperparameters (loaded from
  `winner.json`) on the **original, untransformed** frame.

Both write to `results/decision_kdm_{log1p_only,tuned_only}/metrics.json`.

### 5. `experiments/exp_7/scripts/holdout_eval_v2.py`

Adapted from `experiments/exp_3/scripts/holdout_eval.py` — imports its `mri_pca_train_only()`
and `fit_transform_features()` directly (same train-only-fit leakage discipline, no need to
duplicate), same `train_test_split(..., test_size=0.2, stratify=y_decision, random_state=0)`
producing the same held-out ~18-case split already used for `exp_3`'s original held-out check.
Fits **two** models on the train portion and scores both on the untouched test portion: (a)
`exp_6`'s plain `fit_kdm_backbone` (import from `exp_6/scripts/kdm_backbone.py`) as the
"before" baseline, (b) `exp_7`'s `fit_kdm_backbone` with the winning config (log1p applied to
both train/test consistently, same train-only-fit discipline) as the "after" comparison — prints
both F1/macro-F1 side by side. This is the out-of-sample check `DESIGN.md` §2/§9 requires before
calling the search result a genuine improvement.

### 6. No changes to `experiments/exp_6/scripts/*.py` or any `src/chimera_task1/*.py`

Same rule as every prior experiment — `exp_6`'s results stay reproducible from the same code;
all `exp_7` logic is additive, living entirely under `experiments/exp_7/scripts/`.

## Execution Order (two phases, since later scripts depend on the search's winner)

**Phase A**: run `search_hyperparameters.py` (background it — 144 × 5 × 3 = 2,160 fits), inspect
`results/hyperparameter_search/{grid.csv,winner.json}` and the printed margin-over-0.593 check.

**Phase B**: run `run_ablations.py`, `holdout_eval_v2.py`, and `run_signals_v2.py` (each reads
`winner.json` directly, no manual step in between) — all can run in this order or in parallel
since none depend on each other, only on `winner.json` existing.

## Verification

1. **Smoke-test `kdm_backbone_v2.fit_kdm_backbone` at the current defaults** (`n_epochs=300,
   lr=1e-2, sigma_mult=1.0, optimizer="adam", weight_decay=0.0`, no log1p) on one fold and
   confirm it reproduces `exp_6`'s `decision_kdm_backbone` macro-F1 (0.593) and
   `probs_check_ok=True` via `compute_signals` — confirms the configurable refactor didn't change
   behavior at the old fixed values before trusting any new configuration.
2. **Smoke-test the AdamW branch specifically** (one fold, `optimizer="adamw", weight_decay=1e-3`)
   — confirm it trains without error and `sigma` actually moves from its KNN-based init (sanity
   check that weight decay is doing something, not silently inert).
3. Run `search_hyperparameters.py` in the background; once complete, read `winner.json` and the
   printed margin check before proceeding to Phase B.
4. Run Phase B's three scripts; confirm all `results/*/metrics.json` files are written and valid,
   and that `holdout_eval_v2.py`'s two-model comparison prints cleanly.
5. Compare every number against `DESIGN.md` §4's baselines and §8's decision rules before writing
   `experiments/exp_7/reports/summary.md`, per this project's established review-before-report
   pattern — explicitly state which of §8's four decision-rule branches applies (backbone-only
   win, backbone-and-readout win, readout-was-never-the-bottleneck, or no improvement at all).
