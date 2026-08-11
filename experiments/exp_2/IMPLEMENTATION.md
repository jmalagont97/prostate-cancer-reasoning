# exp_2 Implementation Plan — Official-Schema Feature Scope + Comorbidity Grouping + KDM

## Context

This implements `experiments/exp_2/DESIGN.md` (already reviewed and refined with the user across
several turns). `exp_1` found that a hybrid ML approach on a 47-column engineered feature frame
barely beat naive baselines on Task 1's decision/confidence/weights targets, and only clearly won
on reveal-sequence. `exp_2` tests three changes at once, each isolated via a fully-crossed
ablation (16 conditions total): (1) restricting the feature set to the **11 officially documented
Task-1 input variables** instead of the 47-column frame (many of which — vitals, labs, PSA-trend
stats, pathology history — aren't part of the real `structured-prompt.json` schema and may not
exist at actual inference time), (2) replacing the single comorbidity-count feature with 6
clinically-grouped binary flags derived from `pmhx`, and (3) adding a KDM classifier (memory-based,
sigma-only trained — same design as `exp_1`'s `confidence_kdm`) for both the decision and
confidence targets. The goal is 16 `results/<condition>/metrics.json` files comparable to `exp_1`'s
existing numbers, so the report can say clearly what helped and what didn't.

**This plan, once approved, gets saved as `experiments/exp_2/IMPLEMENTATION.md`** (the
ml-experiment-planner/-reporter convention this project follows — see `experiments/exp_1/` for
precedent) before any other files are touched, per `DESIGN.md` §11.

Confirmed encoding/scoping decisions (from prior discussion, not open questions):
- `dre`: ordinal `Normal(0) < Abnormal(1) < Nodus(2) < Suspicious(3)`; `Not done` → separate
  `dre_not_done` flag, ordinal value left `NaN` (imputed normally) for those rows.
- `bx`: two binary flags, `bx_positive` (1 if "Positive") and `bx_missing` (1 if `NaN`) —
  "Negative" is the implicit (0, 0) baseline.
- `fh` stays out of the 11-variable restriction entirely (separate tool-revealed source,
  unchanged from `exp_1`).
- The `psa` factor's **restricted** weight group is `cli_psa` alone — `psap`/`psav` remain
  available only in the "official" (all-11) scope, not in any restricted per-factor group.
- `ct` excluded (confirmed absent from the data).

## Files to Change

### 1. `src/chimera_task1/features.py` — add the official-schema feature path

New additions, alongside the existing `exp_1` code (which stays untouched — its results remain
the 47-column comparison point):

- `OFFICIAL_CONTINUOUS_COLS = ["cli_psa", "cli_psap", "cli_psav", "cli_psad", "cli_vol", "cli_age", "cli_cspca"]`
- `encode_dre_ordinal(series: pd.Series) -> pd.DataFrame`: returns a 2-column frame
  (`cli_dre_ordinal`, `cli_dre_not_done`) per the mapping confirmed above.
- `encode_bx_binary(series: pd.Series) -> pd.DataFrame`: returns (`cli_bx_positive`, `cli_bx_missing`).
- `select_official_feature_frame(inp: pd.DataFrame, *, comorbidity_treatment: Literal["count", "flags"] = "flags") -> pd.DataFrame`:
  assembles `OFFICIAL_CONTINUOUS_COLS` + `cli_pirads` + `encode_dre_ordinal(inp["cli_dre"])` +
  `encode_bx_binary(inp["cli_bx"])` + either `cli_comorbidity_count` (treatment="count") or
  `comorbidity_flags(inp["txt_comorbidities"])` (treatment="flags", reusing the existing function
  as-is). No missingness-indicator pass needed beyond what's already explicit (`dre_not_done`,
  `bx_missing`) — the continuous columns here have ~0% missingness per the original inventory,
  unlike the dropped `lab_*`/`vit_*` columns.
- `TASK1_VARIABLE_TO_FEATURE_GROUP: dict[str, list[str]]` — the 9 in-scope factors (excludes
  `fh`) each mapped to their restricted-scope column list, e.g. `"psa": ["cli_psa"]`,
  `"dre": ["cli_dre_ordinal", "cli_dre_not_done"]`, `"bx": ["cli_bx_positive", "cli_bx_missing"]`,
  `"comorbidity": ["cli_comorbidity_count"]` or the 6 `comorb_*` names depending on the
  comorbidity-treatment condition being run (the weights runner picks the right one, see below).
- `build_preprocessor` is reused unchanged (it already works off whatever columns a frame has —
  no categorical columns remain in the official frame since `dre`/`bx` are now pre-encoded
  numeric, so it degenerates to numeric-only imputation, which is fine).

### 2. `experiments/exp_2/scripts/` — one runner per target, reusing exp_1's CV/eval code

Each script builds the right `X` via `select_official_feature_frame(...)` (or a factor-sliced
subset of it, for restricted weights), then calls **existing, already-generic functions** rather
than re-implementing CV:

- `run_decision.py` — 6 conditions (`decision_{logistic,hgb,kdm}_{count,flags}`). For
  logistic/HGB, reuses `chimera_task1.train_decision.evaluate()` and `naive_baselines` pattern
  (already parameterized by `(feature_frame, y, label)` — no changes needed there). For KDM,
  reuses `chimera_task1.train_confidence_kdm.fit_predict_kdm()` as-is (already generic over
  `n_classes`, so `n_classes=2` works unmodified) inside a manual out-of-fold loop mirroring that
  module's existing pattern, adapted to F1/ROC-AUC/PR-AUC instead of ordinal distance.
- `run_confidence.py` — 4 conditions (`confidence_{logistic,kdm}_{count,flags}`). Reuses
  `train_reasoning.repeated_out_of_fold_predict()` for logistic and
  `train_confidence_kdm.fit_predict_kdm()` for KDM, both already generic over the feature frame
  passed in.
- `run_reveal.py` — 2 conditions (`reveal_{count,flags}`). Thin wrapper around
  `train_reasoning.eval_reveal()`'s logic with the official frame substituted for the 47-column
  one (that function is currently print-only; the runner captures its return values instead —
  minor adaptation, not a rewrite).
- `run_weights.py` — 4 conditions (`weights_{official,restricted}_{count,flags}`). For
  `official`, one shared `X` (all 9 in-scope factors' models trained on the same frame) via
  `repeated_out_of_fold_predict()` exactly as `exp_1` did. For `restricted`, loops per factor
  using `TASK1_VARIABLE_TO_FEATURE_GROUP[factor]` to slice a factor-specific `X` before calling
  the same `repeated_out_of_fold_predict()` — one call per factor instead of one call for all.

Each script writes `experiments/exp_2/results/<condition>/metrics.json`, matching the exact
schema/fields already established in `experiments/exp_1/results/*/metrics.json` (condition,
model, features, n, cv description, primary metric(s), any notes) so the two experiments'
results are directly comparable side by side.

### 3. No changes to `src/chimera_task1/{train_decision,train_reasoning,train_confidence_kdm}.py`

These stay exactly as they are — `exp_1`'s results must remain reproducible from this same code.
All new logic is additive (`features.py`) or lives in `experiments/exp_2/scripts/` per the
project's established shared-code-vs-experiment-script convention.

## Verification

1. Unit-check `select_official_feature_frame()` and the new encoders against known rows (same
   spot-check style used for `comorbidity_flags()` earlier — confirm `dre`/`bx` encodings match
   the real value counts already gathered).
2. Run each of the 4 new scripts from the project root via the `.venv` interpreter; confirm all
   16 `results/<condition>/metrics.json` files are written and parse as valid JSON.
3. Sanity-print a summary table comparing all 16 new numbers against `exp_1`'s corresponding
   baseline/model numbers (already on disk in `experiments/exp_1/results/`), to review together
   before deciding whether/how to write `experiments/exp_2/reports/summary.md`.
