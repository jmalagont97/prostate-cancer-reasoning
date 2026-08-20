# Experiment Design: Retraining the Best Variable-Weights Models to Include `fh`
**Experiment**: experiments/exp_15/
**Project**: challenge_chimera_2
**Date**: 2026-08-19
**Author**: TBD
**Status**: Complete

## Verdict (2026-08-19)

Both models retrained and reported at 10 factors via LOO. `fh`'s own numbers are identical across
both models (accuracy 0.703, macro-F1 0.206, ordinal error 0.297, decisive-set F1 **0.000**) —
confirmed to be an exact reproduction of the naive majority-class baseline (always predicting
`noted`, 64/91=70.3%): `cli_fh_binary` alone (11/91 positive) isn't informative enough for either
architecture to beat guessing the majority class, and neither ever predicts `important`/`decisive`
for it. Aggregate mean ordinal error moved down slightly in both models (SVM: 0.378→0.370; KDM:
0.390→0.385, since majority-guessing on a skewed 4-class factor isn't a bad ordinal-distance
strategy) while mean decisive-set F1 dropped meaningfully in both (SVM: 0.446→0.401; KDM:
0.427→0.390, `fh`'s 0.000 pulling the average down). Full detail: `reports/summary.md`.

---

## 1. Hypothesis

Every weights experiment since `exp_2` has deliberately excluded `fh` (family history) from
`IN_SCOPE_FACTORS`, on the grounds that its underlying value (`cli_fh_binary`) sits behind a
separate MCP tool-reveal action rather than always being visible like the other 9 schema factors
(`age`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`). Checked directly
against the data (2026-08-19): `family_history` is revealed in **0 of the 91 labeled cases** —
never, in the entire training set. `target_weight_fh` nonetheless has real importance labels for
all 91 cases (64 `noted` / 24 `not_used` / 3 `important` / 0 `decisive`).

Per explicit user request, `fh` is now to be included, predicted, and reported the same way the
other 9 factors are. **The user was asked and confirmed the specific methodological choice this
requires**: `cli_fh_binary` will be used directly as an input feature (the same way the other 9
factors' underlying values are), not withheld to simulate a not-yet-revealed value. This is a
deliberate, explicit departure from the reveal-gating convention every prior experiment followed —
not a silent scope change.

This experiment retrains this project's two "best model" pipelines — `weights_svm` (best overall,
restricted scope, `exp_5`) and `weights_kdm_occlusion` (best KDM, `exp_6`) — extended to 10 factors
instead of 9, with the same LOO + full-metric-suite + confusion-matrix + per-case diagnostic depth
already backfilled for both models' original 9-factor versions (2026-08-18/19 backfills in
`exp_5`/`exp_6`'s own `results/`). No new architecture, no new model family — a scope extension of
already-validated pipelines, not a new modeling idea.

## 2. Experimental Setup

### 2a. Shared library change

`src/chimera_task1/features.py`'s `TASK1_VARIABLE_TO_FEATURE_GROUP` gains `"fh": ["cli_fh_binary"]`
— purely additive (every existing key unchanged), so no prior experiment's behavior changes; `fh`
was never a valid key there before (would have raised `KeyError`), so nothing could have silently
depended on its absence.

### 2b. `weights_svm`, 10 factors, restricted scope

Identical to `exp_5`'s original setup (`SVC(kernel="rbf", C=1.0, class_weight="balanced",
probability=True)`, 19-column frame, `build_preprocessor` + `StandardScaler` per fold) except the
factor loop now runs over all 10 `TASK1_FACTORS` (not `IN_SCOPE_FACTORS`). `fh`'s restricted group
is just `["cli_fh_binary"]` — a single binary column, the same shape as `age`/`vol`/`psad`/`cspca`'s
single-column groups.

### 2c. `weights_kdm_occlusion`, 10 factors

Identical to `exp_6`'s original setup (one shared decision-trained KDM backbone per fold, per-factor
occlusion-delta signal, isotonic recalibration) except: (1) the 19-column frame gains a 20th column,
`cli_fh_binary`, joined on **after** `select_exp3_feature_frame` builds the original 19 columns —
the shared frame-selection function itself is not modified, so every past experiment that calls it
still gets exactly 19 columns; (2) the decision-KDM backbone is refit on this 20-column frame (it
needs `cli_fh_binary` as an available input dimension for the occlusion mechanism to occlude when
computing `fh`'s own signal); (3) the factor loop runs over all 10 factors.

### 2d. Evaluation protocol

Leave-one-out (91-fold, pooled) directly — not staged CV-then-LOO — since both underlying models
are already fully validated at 9 factors and this is a scope extension, not a new hypothesis to
screen cheaply first. Full metric suite (accuracy, macro-F1, ordinal error, decisive-set F1, plus
AUROC/Brier with each model's existing caveats — SVM's real `predict_proba`, KDM-occlusion's
one-hot hard-decision approximation, both already established in the 9-factor backfills). Pooled
confusion matrix + classification report across all 10 factors × 91 cases (910 predictions), plus
a per-case breakdown, matching the exact format already produced for both models' 9-factor
versions for direct comparability.

## 3. File Layout

- `experiments/exp_15/scripts/loo_full_metrics_weights_svm_10factor.py`
- `experiments/exp_15/scripts/loo_full_metrics_weights_kdm_occlusion_10factor.py`
- Reuses `experiments/exp_11/scripts/metrics_multiclass.py`, `chimera_task1.reasoning_labels`,
  `chimera_task1.features` (now including `fh`'s restricted group) unchanged.

## 4. Baselines

| Comparison | Mean ordinal error (9-factor) | Mean ordinal error (10-factor, this experiment) |
|---|---|---|
| Baseline (majority class per factor) | 0.413 | *(computed alongside)* |
| `weights_svm` (LOO, 9-factor, backfilled 2026-08-18) | 0.378 | *(this experiment)* |
| `weights_kdm_occlusion` (LOO, 9-factor, backfilled 2026-08-19) | 0.390 | *(this experiment)* |

`fh`'s own naive baseline: majority class is `noted` (64/91 = 70.3%) — per-factor ordinal error and
decisive-set F1 for `fh` are reported alongside the other 9, not just folded into the aggregate,
so the new factor's own difficulty is visible on its own terms.

## 5. Decision Rules

This is a reporting/completeness extension, not a hypothesis test with a pass/fail bar — the goal
is accurate, honestly-caveated numbers for `fh`, not a verdict about whether either model "beats"
anything new. Both the per-factor `fh` numbers and the updated 10-factor aggregate are reported;
the aggregate's shift from the 9-factor number (up or down) is reported as an observation, not
framed as a win or a regression.

## 6. Risks & Mitigations

- **Real information available to a live agent vs. this experiment's training data**: a genuinely
  deployed agent would only have `cli_fh_binary` if it called the reveal tool — this experiment's
  choice (confirmed with the user) trades that realism for a like-for-like extension of the other 9
  factors. Documented explicitly in every output, not hidden.
- **`cli_fh_binary` missingness**: 3/91 labeled cases have `cli_fh_binary` = NaN — handled by
  `build_preprocessor`'s existing median-imputation, same discipline as every other column.
- **`fh`'s label distribution is 3-way, not 4-way** (`decisive` never occurs, 0/91) — `safe_multiclass_auroc`/
  `multiclass_brier_score` already handle a globally-absent class via their existing `labels=`
  guard, no new code needed.

## 7. Next Steps

Implement directly — both scripts are close structural copies of the already-completed 9-factor
backfill scripts (`experiments/exp_5/scripts/loo_full_metrics_weights_svm.py`, `experiments/exp_6/
scripts/loo_full_metrics_weights_kdm_occlusion.py`), extended to 10 factors and the 20-column
frame. No separate plan-mode cycle needed given the small, mechanical nature of the extension.
