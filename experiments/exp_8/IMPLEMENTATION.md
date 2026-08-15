# exp_8 Implementation Plan — Expanded Variables + Hyperparameter Tuning + Reveal-Sequence

## Context

This implements `experiments/exp_8/DESIGN.md` (status: Proposed, reviewed and refined across
several rounds this session — the 23-column frame, the corrected 144-combo hyperparameter search
with mandatory held-out check, and the entropy-occlusion reveal mechanism are all locked). `exp_6`
built a shared KDM backbone for decision/confidence/weights; `exp_7` tried tuning it and found
nothing that survived held-out verification. `exp_8` re-tests tuning on a genuinely expanded
feature set (not a blind retry), and extends the shared-backbone idea to reveal-sequence for the
first time.

**Finding from this planning session that corrects part of `DESIGN.md`**: checked how many of the
91 labeled cases actually have each of the 6 reveal sections marked as revealed —
`previous_notes` 77/91, `psa_trend` 79/91, `radiology_report` 88/91, `laboratory_results` 41/91,
**`family_history` 0/91, `pathology_report` 0/91**. Every prior reveal-sequence model in this
project (`exp_1`–`exp_5`) already handles this by dynamically restricting to sections with at
least one positive example (`experiments/exp_3/scripts/run_reveal.py:32`) — `exp_8`'s reveal
mechanism follows that same, already-established convention. Practical effect: only 4 sections
get modeled, not 6. `family_history` staying unmapped was already a disclosed limitation;
`pathology_report`'s new `cli_isup` mapping (§2b) was meant to resolve a shared-feature-group risk
for reveal specifically, but since `pathology_report` never appears as revealed in this dataset,
it won't be part of the modeled reveal conditions either way — `cli_isup`'s justification for the
frame stands on its own (§2a's decision-target correlation), independent of this. **`DESIGN.md`
gets corrected to state this plainly once implementation starts**, not silently worked around.

**This plan, once approved, gets saved as `experiments/exp_8/IMPLEMENTATION.md`.**

## Key reuse decisions

1. **`cli_isup`/`vit_bmi` get NO manual imputation in the feature-frame builder.** `build_preprocessor()` (`src/chimera_task1/features.py`) already applies `SimpleImputer(strategy="median")` to every non-categorical column, refit per CV fold on train rows only — exactly the correct, leak-free behavior. The frame builder just needs to include these two raw columns (with their natural NaNs); no custom imputation code, and no risk of the global-median leakage a naive `.fillna()` in the frame builder would introduce.
2. **`select_official_feature_frame()` already includes `psap`/`psav`** — `exp_3` was the one that dropped them (`EXP3_DROPPED_PSA_COLS`). `exp_8`'s frame builder is `select_official_feature_frame(...) .join(mri_pca)` plus the two new raw columns — no need to reconstruct the PSA family from scratch.
3. **`exp_7`'s search/ablation/holdout scripts are the templates**, not new designs — `search_hyperparameters_v3.py`, `run_ablations_v3.py`, `holdout_eval_v3.py`, `run_signals_v3.py` are each `exp_7`'s equivalent file with the feature frame swapped (no `log1p` step this time — that was `exp_7`-specific) and the corrected margin threshold (§7 below).
4. **`kdm_backbone_v2.py`'s `compute_signals()` already returns `entropy`** — the reveal mechanism's occlusion-on-entropy signal needs one new small helper (not a change to `exp_6`/`exp_7`'s existing files): compute `entropy` on an occluded copy of `X` the same way `occlusion_delta()` already does for `p(yes)`, just reading a different key out of `compute_signals()`'s return dict. Lives in `exp_8`'s own new script, not `kdm_backbone.py`/`kdm_backbone_v2.py` — same "small per-experiment adaptations live in the experiment layer" convention `exp_2`'s `run_reveal.py` already established.

## Files to Add

### 1. `experiments/exp_8/scripts/features_v3.py`

- `select_exp8_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame`:
  ```python
  frame = select_official_feature_frame(inp, comorbidity_treatment="flags").join(mri_pca)
  frame["cli_isup"] = inp["path_hist_bx_isup"].values   # NaN for the 24 no-prior-biopsy cases; imputed per-fold downstream
  frame["vit_bmi"] = inp["vit_bmi"].values               # 0% missing
  return frame
  ```
  23 columns total (verify against `experiments/exp_8/DESIGN.md` §2a's table: 14 clinical + 6
  comorbidity + 3 MRI).

### 2. `experiments/exp_8/scripts/search_hyperparameters_v3.py`

Copy of `exp_7/scripts/search_hyperparameters.py` with: `select_exp8_feature_frame` instead of
`select_exp3_feature_frame` + `apply_log1p_transform`, no `log1p` step at all (not part of this
experiment's hypothesis). `EXP6_DECISION_MACRO_F1 = 0.593` stays the comparison point.
**`CLEAR_MARGIN` changed from `exp_7`'s `0.02` to `0.045`** (`exp_6`'s measured 10-repeat std),
per `DESIGN.md` §7's corrected methodology. Same 144-combination grid, same 5-fold × 3-repeat
search protocol, writes `results/hyperparameter_search/{grid.csv,winner.json}`.

### 3. `experiments/exp_8/scripts/run_ablations_v3.py`

Copy of `exp_7/scripts/run_ablations.py`'s structure, but the two isolated conditions are:
- `decision_kdm_features_only`: 23-column frame (`select_exp8_feature_frame`), `exp_6`'s
  **original** fixed hyperparameters (`kdm_backbone_v2.EXP6_DEFAULTS`-equivalent: `n_epochs=300,
  lr=1e-2, sigma_mult=1.0, optimizer="adam", weight_decay=0.0`).
- `decision_kdm_tuned_only`: `exp_3`'s **original** 19-column frame (`select_exp3_feature_frame`),
  winning hyperparameters from `results/hyperparameter_search/winner.json`.

Decision-only macro-F1, full 5-fold × 10-repeat CV, matching `exp_7`'s ablation protocol exactly.

### 4. `experiments/exp_8/scripts/holdout_eval_v3.py`

Copy of `exp_7/scripts/holdout_eval_v2.py`'s structure — same held-out split
(`train_test_split(..., test_size=0.2, stratify=y_decision, random_state=0)`, n_test=19), same
`mri_pca_train_only()`/`fit_transform_features()` reuse from `exp_3/scripts/holdout_eval.py`.
Compares (a) `exp_6`'s plain `fit_kdm_backbone` on the **original 19-column** frame (the
"before" reference, matching what `exp_7`'s holdout check compared against) vs. (b) `exp_8`'s
**combined** config: 23-column frame + winning hyperparameters. Prints both F1/macro-F1 side by
side; writes `results/holdout_eval_v3/metrics.json`.

### 5. `experiments/exp_8/scripts/run_signals_v3.py`

Copy of `exp_7/scripts/run_signals_v2.py` with: `select_exp8_feature_frame` instead of
`select_exp3_feature_frame`, no `log1p` step, winning hyperparameters loaded from
`results/hyperparameter_search/winner.json` (same file `run_ablations_v3.py`/`holdout_eval_v3.py`
also read — one source of truth for "the winner"). Produces `decision_kdm_v3` (the combined
condition — decision's own number from this same unified loop, no separate script needed, same
pattern `exp_6`/`exp_7` used) plus the 5 confidence `_v3` and 3 weights `_v3` conditions,
byte-identical signal/recalibration logic to `exp_6`/`exp_7`.

### 6. `experiments/exp_8/scripts/run_reveal_kdm.py` — new

- Load `ann, inp_ann = load_annotated()`, build the 23-column frame (winning hyperparameters,
  same backbone config as `run_signals_v3.py` — genuinely the same shared model, not a
  differently-tuned one).
- `seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])`;
  `sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]` — dynamically
  resolves to the same 4 sections every prior reveal model has used
  (`previous_notes`, `laboratory_results`, `psa_trend`, `radiology_report`), per this session's
  finding above.
- `SECTION_FEATURE_GROUPS` dict, only for those 4 sections (drop the `pathology_report`/
  `family_history` entries from `DESIGN.md` §2b's table — moot for a section with zero positive
  examples):
  ```python
  SECTION_FEATURE_GROUPS = {
      "psa_trend": ["cli_psa", "cli_psad", "cli_psav", "cli_psap"],
      "radiology_report": ["cli_pirads", "mri_pca_0", "mri_pca_1", "mri_missing"],
      "laboratory_results": ["cli_cspca", "cli_vol"],
      "previous_notes": ["cli_bx_positive", "cli_bx_missing", "cli_age"],
  }
  ```
- New small helper (local to this script): `occlusion_entropy_delta(model, X, col_idx,
  fill_values) -> np.ndarray` — same shape as `kdm_backbone.occlusion_delta()` but returns
  `compute_signals(model, X_occluded)["entropy"] - compute_signals(model, X)["entropy"]` instead
  of the `p(yes)` delta.
- Per fold/repeat (same 5×10 CV as everything else): fit the shared backbone once, then for each
  of the 4 sections fit **one univariate** `make_classifier()` per `DESIGN.md` §2c (a single
  feature, that section's own `R_S(x)`, not a joint multi-section model — matches what's already
  documented and reviewed) predicting `P(section revealed | R_S(x))` on train rows, apply to test
  rows, threshold at 0.5. Predicted reveal set per case = union of sections predicted positive.
- Score via `reveal_set_precision()` per case, averaged across repeats; **also report each
  section's individual precision/recall** per `DESIGN.md` §7's explicit requirement.
- Compare against the existing incumbent (`reveal_flags`, `exp_2`, 0.853) and naive baseline
  (0.783) — both already established, no need to recompute.

### 7. No changes to `exp_6`/`exp_7`'s scripts or any `src/chimera_task1/*.py`

Same rule as every prior experiment.

## Execution Order

**Phase A**: `search_hyperparameters_v3.py` (background — 144×5×3 = 2,160 fits, expect similar
runtime to `exp_7`'s ~55 minutes). Inspect `winner.json` and the printed margin check.

**Phase B** (each reads `winner.json`, can run in parallel): `run_ablations_v3.py`,
`holdout_eval_v3.py`, `run_signals_v3.py`, `run_reveal_kdm.py`.

## Verification

1. **Smoke-test `select_exp8_feature_frame()`** — confirm 23 columns, confirm `cli_isup`'s NaN
   pattern matches `cli_bx_missing` exactly (24 rows), confirm `build_preprocessor` + per-fold
   `StandardScaler` handles both new columns without error on one fold before any full run.
2. **Smoke-test `occlusion_entropy_delta()`** on one fold — confirm nonzero, differentiated `R_S`
   values across the 4 sections (same sanity-check spirit as `exp_6`'s original occlusion smoke
   test: `pirads`/`bx` should move more than a section with weak backing features).
3. Run Phase A; before proceeding, check the printed margin against `exp_6`'s 0.593 clears the
   corrected 0.045 threshold — proceed to Phase B regardless (per `exp_7`'s "test anyway, verify
   honestly" precedent) but note upfront whether it's expected to survive held-out.
4. Run Phase B; confirm every `results/*/metrics.json` is written and valid, and
   `holdout_eval_v3.py`'s comparison prints cleanly.
5. Compare every number against `DESIGN.md` §4's baselines and §8's decision-rule branches
   (features-only vs. tuned-only vs. combined, each crossed with held-out survival; reveal's
   per-section breakdown) before writing `experiments/exp_8/reports/summary.md`. Correct
   `DESIGN.md` itself (§1c, §2b) to reflect the family_history/pathology_report finding from this
   planning session as part of this same pass, not as an afterthought.
