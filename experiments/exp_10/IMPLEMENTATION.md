# exp_10 Implementation Plan — External Full-Schema Replication (Frame.md) under ARD-KDM

## Context

`experiments/exp_10/DESIGN.md` (status: Design finalized 2026-08-13) adopts the external repo's
complete 37-variable tabular schema (pasted into `Frame.md` this session) + MRI-PCA(2), encoded
with that repo's own stated preprocessing convention (MinMax scaling + one-hot categoricals +
explicit missing-flag columns, **not** this project's `build_preprocessor` median-impute
convention), fit through `exp_9`'s ARD-KDM backbone (chosen over scalar KDM given the frame's
size). The question: does this much wider, externally-sourced frame beat `exp_9`'s already-curated
23-column ARD-KDM reference (0.680 held-out decision macro-F1, the project's best result to date)?

Every §2 EDA judgment call in `DESIGN.md` (duplicate PSA-trend columns dropped, shared
`cli_bx_missing` flag, no redundant `pack_years` flag, keep-all-three Gleason/ISUP) was reviewed
point by point with the user and confirmed. **This plan, once approved, gets saved as
`experiments/exp_10/IMPLEMENTATION.md`.**

## Key technical findings from this planning session (these determine the implementation)

1. **`build_preprocessor()` (`src/chimera_task1/features.py`) only imputes — it does not scale.**
   Every existing script applies `StandardScaler` *externally*, after
   `build_preprocessor().fit_transform()`, both fit fresh per CV fold on train rows only. `exp_10`
   follows the exact same two-stage shape, just swapping in a new imputation/encoding step and
   `MinMaxScaler` instead of `StandardScaler` at the call site — no new architectural pattern
   needed, just new per-fold-fit logic plugged into the same slot.
2. **`restricted_feature_group()`/`TASK1_VARIABLE_TO_FEATURE_GROUP` are hardcoded to the 19/23-col
   frames' engineered column names** (`cli_dre_ordinal`, `cli_bx_positive`, etc.) and cannot
   resolve against `exp_10`'s frame, which keeps `cli_dre`/`cli_bx`/`vit_smoking_status` as
   one-hot-expanded categoricals per `Frame.md`'s convention (a deliberate choice already made —
   not reverting to the ordinal/binary engineered encoding just to reuse the existing function).
   **A new `restricted_feature_group_fullschema()` is required**, mapping the 9 in-scope factors to
   `exp_10`'s actual column names (§4 below).
3. **One-hot encoding does not need `sklearn.OneHotEncoder`/`ColumnTransformer` at all, and
   shouldn't use them.** Every one-hot category `Frame.md` needs (`cli_dre`'s 5 values, `cli_bx`'s
   2, `vit_smoking_status`'s 3) is a small, *fixed, known-in-advance* set — the same pattern
   `features.py`'s own `encode_dre_ordinal`/`encode_bx_binary` already use for the 19/23-col
   frames' engineered columns, just producing more columns per category instead of one ordinal
   column. Building these as plain `(series == "Category").astype(int)` columns **inside frame
   construction** (deterministic, target-independent, computed once — like the existing
   `cli_bx_missing`/`mri_missing` flags already are) means the frame arrives with its *final* 48
   named columns from the start. This sidesteps `ColumnTransformer.get_feature_names_out()`
   entirely (whose exact output-naming behavior is sklearn-version-dependent and would otherwise
   break every `X_frame.columns.get_loc(c)` lookup this project's occlusion/section-grouping code
   relies on) — `exp_10`'s frame behaves exactly like `exp_3`/`exp_8`'s for indexing purposes, only
   wider. `NaN` never reaches a one-hot check (`cli_bx.isna()` is simply `False` for both the
   `"Positive"` and `"Negative"` comparisons), so no NaN-sentinel trick is needed either.
4. **Only 5 of the 48 final columns need genuine per-fold-fit imputation** (`cli_pirads`,
   `path_hist_bx_isup`, `path_hist_bx_gl_prim`, `path_hist_bx_gl_sec`, `cli_fh_binary` — real
   missingness, median-filled from train-fold data only, to avoid leakage). Every other column
   (continuous-complete, the 4 missing-flags, all 10 one-hot columns) is fully finished at
   frame-construction time — same "compute the deterministic part once, fit only the
   train/target-dependent part per fold" split this project already uses for `SimpleImputer` inside
   `build_preprocessor`.
5. **Arithmetic correction to `DESIGN.md` §3**: the original table stated 42 raw fields / 46
   encoded columns — recounted programmatically (not by hand this time, to avoid a repeat error):
   **37 raw source fields, 48 encoded columns** (29 complete-continuous + 5 impute-needed
   continuous + 4 missing-flags + 10 one-hot). No decision changes; `DESIGN.md` §3 already updated
   to match.
6. **`mri_pca_train_only()` and the general held-out pattern in
   `experiments/exp_3/scripts/holdout_eval.py` are frame-agnostic and reused unchanged** — only
   `fit_transform_features()` (which calls `build_preprocessor`) needs an `exp_10`-specific
   equivalent (`fit_transform_fullschema()`, finding #1's pattern).
7. **`ard_kernel.py` (`experiments/exp_9/scripts/`) needs zero changes.** `fit_kdm_backbone_ard`,
   `compute_signals_ard`, `occlusion_delta_ard`, `dm_rbf_variance_ard`, and the re-exported
   `kernel_distance_contribution` all operate on a plain `(n, dim)` numpy/torch array — nothing
   about them depends on which preprocessing produced it. Reused via `sys.path` import exactly like
   `exp_9` reused `exp_6`'s `kernel_distance_contribution`.

## Files to Add

### 1. `experiments/exp_10/scripts/features_fullschema.py` — the new module

- `select_exp10_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame`: builds
  the full 48-column frame per `DESIGN.md` §2/§3 (as corrected by finding #5):
  - Passthrough continuous (28, already complete): `cli_age, cli_psa, cli_psap, cli_psav,
    cli_psad, cli_vol, cli_months, cli_cspca, cli_comorbidity_count, cli_allergies_count,
    cli_ipss_score, vit_weight_kg, vit_height_cm, vit_bmi, vit_bp_systolic, vit_bp_diastolic,
    vit_heart_rate_bpm, vit_smoking_pack_years (filled with 0 here, per §2d), psa_tr_count,
    psa_tr_first_val, psa_tr_min, psa_tr_mean, psa_tr_delta, psa_tr_slope,
    lab_creatinine_mg_dl, lab_free_psa_ng_ml, lab_free_total_ratio, mri_pca_0, mri_pca_1`
    (`psa_tr_last_val`/`psa_tr_max` omitted per §2a — exact duplicates of `cli_psa`).
  - Left-with-NaN continuous, for per-fold imputation (5): `cli_pirads, path_hist_bx_isup,
    path_hist_bx_gl_prim, path_hist_bx_gl_sec, cli_fh_binary`.
  - Missing-flags (4, computed once via `.isna()`, deterministic): `cli_pirads_missing`,
    `cli_bx_missing` (from `inp["cli_bx"].isna()`, shared by `isup`/`gl_prim`/`gl_sec` per §2e),
    `cli_fh_missing`, `mri_missing` (already provided by the `mri_pca` argument).
  - One-hot, built via `(series == "Category").astype(int)` per finding #3 (10):
    `cli_dre_Normal, cli_dre_Nodus, cli_dre_Abnormal, cli_dre_Not done, cli_dre_Suspicious`;
    `cli_bx_Positive, cli_bx_Negative`; `vit_smoking_status_Never,
    vit_smoking_status_Ex-smoker, vit_smoking_status_Current`.
- `restricted_feature_group_fullschema(factor: str) -> list[str]` — the 9 in-scope-factor mapping
  (finding #2), used by both the weights occlusion loop and `importance_comparison_fullschema.py`:
  ```
  age: [cli_age]                      psa: [cli_psa]
  dre: [all 5 cli_dre_* columns]      bx: [cli_bx_Positive, cli_bx_Negative, cli_bx_missing]
  pirads: [cli_pirads, cli_pirads_missing]     psad: [cli_psad]
  vol: [cli_vol]                      cspca: [cli_cspca]
  comorbidity: [cli_comorbidity_count]   # Frame.md has no comorb_* grouped flags, unlike exp_6-9
  ```
  (`fh` excluded, same as `IN_SCOPE_FACTORS` everywhere else in this project — unrelated to
  `cli_fh_binary`/`cli_fh_missing` still being present as raw predictive features.)
- `fit_transform_fullschema(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]`
  (finding #1/#4's pattern, mirrors `holdout_eval.py`'s `fit_transform_features` in spirit): computes
  medians of the 5 NaN-bearing columns from `X_train_raw` only, fills both frames, converts to
  numpy (column order = `X_train_raw.columns`, stable across train/test since both share the same
  48 columns — **no expansion happens here**, so `X_frame.columns.get_loc(c)` keeps working exactly
  like `exp_3`/`exp_8`/`exp_9`'s scripts), fits `MinMaxScaler` on the train matrix, transforms both.

### 2. `experiments/exp_10/scripts/run_signals_fullschema.py`

Direct structural copy of `experiments/exp_9/scripts/run_signals_23col.py` (decision + 5 confidence
signals + 3 weights conditions, same isotonic/blend recalibration logic unchanged), with:
`select_exp10_feature_frame`/`fit_transform_fullschema`/`restricted_feature_group_fullschema` from
file 1 instead of `select_exp8_feature_frame`/`build_preprocessor`+`StandardScaler`/
`chimera_task1.features.restricted_feature_group`; `fit_kdm_backbone_ard`/`compute_signals_ard`/
`occlusion_delta_ard`/`kernel_distance_contribution` imported unchanged from `exp_9/scripts/ard_kernel.py`
(finding #7); `ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}` fixed, no search, per
`DESIGN.md` §1's guardrail. Per-fold: `X_train, X_test =
fit_transform_fullschema(X_frame.iloc[train_idx], X_frame.iloc[test_idx])` replaces the existing
`build_preprocessor(...).fit_transform/transform` + separate `StandardScaler` two-liner. Conditions
produced: `decision_kdm_ard_fullschema`, `confidence_kdm_{5 signals}_ard_fullschema`,
`weights_kdm_{occlusion,kernel_distance,blend}_ard_fullschema`.

### 3. `experiments/exp_10/scripts/run_reveal_fullschema.py`

Structural copy of `run_reveal_23col.py`, same 4 dynamically-confirmed modeled sections
(`family_history`/`pathology_report` still 0/91 positive). `SECTION_FEATURE_GROUPS`, extended from
`exp_8`/`exp_9`'s mapping with `exp_10`'s new columns:
```
psa_trend:          cli_psa, cli_psad, cli_psav, cli_psap, psa_tr_count, psa_tr_first_val,
                     psa_tr_min, psa_tr_mean, psa_tr_delta, psa_tr_slope
radiology_report:    cli_pirads, cli_pirads_missing, mri_pca_0, mri_pca_1, mri_missing
laboratory_results:  cli_cspca, cli_vol, lab_creatinine_mg_dl, lab_free_psa_ng_ml,
                     lab_free_total_ratio
previous_notes:      cli_bx_Positive, cli_bx_Negative, cli_bx_missing, cli_age, cli_months,
                     cli_ipss_score, cli_comorbidity_count, cli_allergies_count, cli_fh_binary,
                     cli_fh_missing, vit_weight_kg, vit_height_cm, vit_bmi, vit_bp_systolic,
                     vit_bp_diastolic, vit_heart_rate_bpm, vit_smoking_pack_years,
                     vit_smoking_status_Never, vit_smoking_status_Ex-smoker,
                     vit_smoking_status_Current, path_hist_bx_isup, path_hist_bx_gl_prim,
                     path_hist_bx_gl_sec, cli_dre_Normal, cli_dre_Nodus, cli_dre_Abnormal,
                     cli_dre_Not done, cli_dre_Suspicious
```
`previous_notes` absorbs every general-clinical-history column not specifically about PSA trend,
radiology, or labs (28 of 48 columns) — an explicit design choice (documented in the script's
docstring, not hidden), consistent with how `exp_8`/`exp_9` already used it as the catch-all for
`bx`/`age`. Local `occlusion_entropy_delta_ard()` helper reused verbatim from `run_reveal_23col.py`
(calls `compute_signals_ard`, unchanged). Condition produced: `reveal_kdm_ard_fullschema`.

### 4. `experiments/exp_10/scripts/holdout_eval_fullschema.py`

Same fixed 19-case held-out split as every prior held-out check
(`train_test_split(..., test_size=0.2, stratify=y_decision, random_state=0)`). Reuses
`mri_pca_train_only` from `experiments/exp_3/scripts/holdout_eval.py` unchanged (finding #6). Fits
the full-schema ARD model on the train portion, scores held-out macro-F1, and reports it alongside
already-established reference constants (not recomputed — cited exactly as `exp_9`'s own report
cited `exp_8`'s numbers): `exp_9` ARD 23-col held-out = 0.680, `exp_6` scalar 19-col held-out =
0.593.

### 5. `experiments/exp_10/scripts/importance_comparison_fullschema.py`

Structural copy of `exp_9/scripts/importance_comparison.py`, single frame this time (no 19-vs-23
loop): fits ARD once on the full 91-case set, prints per-column trained `σⱼ` ranking, aggregates by
`restricted_feature_group_fullschema` into per-factor relevance scores, compares top-5 against
`exp_5`'s solved set `{pirads, bx, dre, age, psa}` — same diagnostic pattern, not a scored CV
condition.

### 6. No changes to `exp_3`/`exp_6`/`exp_8`/`exp_9`'s scripts, the `kdm` library, or
   `src/chimera_task1/*.py`

Same rule as every prior experiment. `restricted_feature_group_fullschema` and
`fit_transform_fullschema` are new, `exp_10`-only functions, not edits to `features.py`.

### 7. `experiments/exp_10/DESIGN.md` — one small correction

~~Fix finding #5's arithmetic in §3's column-count table~~ — **done**: `DESIGN.md` §3 now states
37 raw source fields / 48 encoded columns, matching finding #5.

## Execution Order

**Priority 1** (core result): `run_signals_fullschema.py`, `holdout_eval_fullschema.py` (mandatory
per `DESIGN.md` §7 — the widest frame this project has tried).
**Priority 2**: `run_reveal_fullschema.py`, `importance_comparison_fullschema.py`.

## Verification

1. **Smoke-test `select_exp10_feature_frame`** on the full 91-case data: assert exactly 48 columns,
   assert NaN present *only* in the 5 designated impute-needed columns (everything else fully
   populated), print the column list once for a manual eyeball check against this plan's §1 list.
2. **Smoke-test `fit_transform_fullschema`** on one fold: assert output shapes `(n_train, 48)` /
   `(n_test, 48)`, no NaN, all values in `[0, 1]` (MinMax range).
3. **Smoke-test `restricted_feature_group_fullschema`** for all 9 in-scope factors: assert every
   returned column name actually exists in the frame's column list (catches typos before the full
   run, same discipline as every prior experiment's smoke tests).
4. **Smoke-test one `fit_kdm_backbone_ard` + `compute_signals_ard` + `occlusion_delta_ard` call**
   on the 48-dim full-schema frame (`probs_check_ok` must be `True`) before committing to the full
   10-repeat CV loop — confirms no shape mismatch from the wider frame.
5. Run Priority 1 (background — 48 dims × 5×10 CV × 9 conditions is the largest single KDM run this
   project has attempted; expect it to take longer than `exp_9`'s 23-col run).
6. Run Priority 2.
7. Compare every result against `DESIGN.md` §4's baseline table and §8's decision-rule branches
   before writing `experiments/exp_10/reports/summary.md`.
