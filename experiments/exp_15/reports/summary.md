# exp_15 Report: Retraining the Best Variable-Weights Models to Include `fh`

## 1. Summary

Every weights experiment since `exp_2` excluded `fh` (family history) from modeling, since its
underlying value (`cli_fh_binary`) sits behind a separate MCP tool-reveal action rather than
always being visible like the other 9 schema factors — confirmed directly against the data
(2026-08-19) to have been revealed in **0 of the 91 labeled cases**, ever. Per explicit user
request, `fh` is now included, predicted, and reported the same way the other 9 factors are. The
user was asked and confirmed the specific methodological choice this requires: `cli_fh_binary` is
used **directly as an input feature**, not withheld to simulate a not-yet-revealed value — a
deliberate departure from every prior experiment's reveal-gating convention, made explicitly, not
silently.

Both of this project's "best model" pipelines were retrained at 10 factors (LOO, 91-fold, pooled):
`weights_svm` (best overall, `exp_5`) and `weights_kdm_occlusion` (best KDM, `exp_6`).

| Model | Ordinal error (9→10 factor) | Decisive-set F1 (9→10 factor) |
|---|---|---|
| `weights_svm` | 0.378 → **0.370** | 0.446 → **0.401** |
| `weights_kdm_occlusion` | 0.390 → **0.385** | 0.427 → **0.390** |

## 2. `fh`'s own numbers — a genuine finding, not a formality

`fh` scores **identically across both models**: accuracy=0.703, macro-F1=0.206, ordinal error=0.297,
decisive-set F1=**0.000**, AUROC=None (class `decisive` never occurs for `fh`, 0/91). This is not a
coincidence — it is an exact reproduction of the naive majority-class baseline. `fh`'s true label
distribution is 64 `noted` / 24 `not_used` / 3 `important` / 0 `decisive` (majority class `noted`,
70.3%). A model that always predicts `noted` scores exactly: accuracy = 64/91 = 0.703; ordinal
error = (24×1 + 3×1)/91 = 0.297 (`not_used`→`noted` is 1 rank off, `important`→`noted` is 1 rank
off). Both figures match to three decimals.

**Interpretation**: `cli_fh_binary` alone (a single weak binary flag, 11/91 positive in the labeled
set) is not informative enough for either architecture — a 9-factor-validated SVM or a KDM-backbone
occlusion signal — to beat guessing the majority class. Neither model ever once predicts `important`
for `fh` across all 91 LOO folds, which is exactly why decisive-set F1 is 0.000: the metric
specifically penalizes never identifying any of the true positive cases.

This is a genuinely useful negative result on `fh` specifically, not just a completeness checkbox:
it suggests the original exclusion, while motivated by an information-availability concern (the
feature was never actually revealed in the training data), also happens to coincide with `fh` being
a weak predictor on its own once made available — the two concerns are separate, but this
experiment's numbers show `fh` doesn't obviously become a strong factor just because the modeling
constraint is relaxed.

## 3. Effect on the aggregate

Both models' mean ordinal error moved *down* slightly with `fh` added (0.378→0.370 for SVM,
0.390→0.385 for KDM) — because collapsing to the majority class is not a bad ordinal-distance
strategy on a distribution this skewed (0.297 is below both models' own 9-factor average). Mean
decisive-set F1 moved down meaningfully in both (0.446→0.401 for SVM, 0.427→0.390 for KDM) — `fh`'s
hard 0.000 pulls the 10-factor average down by roughly a tenth of the metric's own scale. The two
official metrics move in opposite directions from adding `fh`, the same kind of divergence
`exp_14`'s KDM regression experiment first surfaced for a different reason.

## 4. Full per-factor results

### `weights_svm` (10-factor, LOO)

| Factor | Accuracy | Macro-F1 | Ordinal error | Decisive-set F1 | AUROC | Brier |
|---|---|---|---|---|---|---|
| age | 0.604 | 0.283 | 0.407 | 0.736 | 0.375 | 0.479 |
| **fh** | **0.703** | **0.206** | **0.297** | **0.000** | — | 0.472 |
| cspca | 0.670 | 0.201 | 0.451 | 0.000 | — | 0.508 |
| pirads | 0.648 | 0.326 | 0.374 | 0.994 | — | 0.472 |
| vol | 0.758 | 0.216 | 0.264 | 0.000 | 0.469 | 0.388 |
| psa | 0.604 | 0.340 | 0.407 | 0.803 | 0.304 | 0.552 |
| comorbidity | 0.692 | 0.205 | 0.308 | 0.000 | — | 0.418 |
| psad | 0.571 | 0.184 | 0.495 | 0.062 | 0.491 | 0.587 |
| dre | 0.747 | 0.347 | 0.253 | 0.583 | 0.285 | 0.380 |
| bx | 0.615 | 0.350 | 0.440 | 0.832 | 0.468 | 0.553 |

**Pooled confusion matrix** (910 = 91 × 10):

| True \ Predicted | not_used | noted | important | decisive |
|---|---|---|---|---|
| not_used | 61 | 60 | 5 | 1 |
| noted | 19 | 339 | 57 | 0 |
| important | 11 | 88 | 167 | 18 |
| decisive | 0 | 10 | 39 | 35 |

Accuracy 0.662, macro-F1 0.604 (precision/recall/f1: not_used 0.670/0.480/0.560, noted
0.682/0.817/0.743, important 0.623/0.588/0.605, decisive 0.648/0.417/0.507).

### `weights_kdm_occlusion` (10-factor, LOO)

| Factor | Accuracy | Macro-F1 | Ordinal error | Decisive-set F1 | AUROC | Brier |
|---|---|---|---|---|---|---|
| age | 0.615 | 0.190 | 0.396 | 0.787 | 0.500 | 0.769 |
| **fh** | **0.703** | **0.206** | **0.297** | **0.000** | — | 0.593 |
| cspca | 0.626 | 0.241 | 0.484 | 0.000 | — | 0.747 |
| pirads | 0.549 | 0.244 | 0.462 | 0.994 | — | 0.901 |
| vol | 0.758 | 0.216 | 0.264 | 0.000 | 0.500 | 0.484 |
| psa | 0.571 | 0.279 | 0.440 | 0.803 | 0.539 | 0.857 |
| comorbidity | 0.692 | 0.205 | 0.308 | 0.000 | — | 0.615 |
| psad | 0.549 | 0.177 | 0.505 | 0.000 | 0.492 | 0.901 |
| dre | 0.769 | 0.338 | 0.242 | 0.486 | 0.577 | 0.462 |
| bx | 0.604 | 0.343 | 0.451 | 0.828 | 0.620 | 0.791 |

**Pooled confusion matrix** (910 = 91 × 10):

| True \ Predicted | not_used | noted | important | decisive |
|---|---|---|---|---|
| not_used | 54 | 68 | 5 | 0 |
| noted | 18 | 332 | 65 | 0 |
| important | 10 | 80 | 188 | 6 |
| decisive | 0 | 11 | 61 | 12 |

Accuracy 0.644, macro-F1 0.527 (precision/recall/f1: not_used 0.659/0.425/0.517, noted
0.676/0.800/0.733, important 0.589/0.662/0.624, decisive 0.667/0.143/0.235).

## 5. What changed in shared code

`src/chimera_task1/features.py`'s `TASK1_VARIABLE_TO_FEATURE_GROUP` gained
`"fh": ["cli_fh_binary"]` — purely additive, every existing key unchanged, so no experiment
`exp_1`–`exp_14` is affected (none of them ever called `restricted_feature_group("fh", ...)`, which
would previously have raised `KeyError`). No other shared library code changed; both retraining
scripts join `cli_fh_binary` into their own copy of the 19-column frame locally, so
`select_exp3_feature_frame`/`select_official_feature_frame` still return exactly what they always
did for every other caller.

## 6. Recommendation

- **Both models' 10-factor numbers are now the ones to cite going forward** if `fh` needs to be
  part of any reported aggregate — `experiments/exp_15/results/loo_full_metrics_weights_{svm,
  kdm_occlusion}_10factor/metrics.json` (plus `per_case.csv` for both, same format as `exp_5`/
  `exp_6`'s 9-factor backfills).
- **`fh` itself is not a promising factor to invest further modeling effort in** on the current
  single-feature (`cli_fh_binary`) representation — it never escapes the majority-class baseline
  under either architecture tried. If `fh` needs to genuinely improve, it would need either a richer
  feature representation (not just the single binary flag) or a fundamentally different approach
  (e.g., modeling it jointly with the reveal-sequence decision it's naturally coupled to, since
  `family_history` is also the reveal section it depends on) — out of scope for this experiment,
  which was a reporting extension, not a new modeling idea.
- **The reveal-gating exclusion this experiment reversed remains methodologically real** — a
  genuinely deployed agent still wouldn't have `cli_fh_binary` without calling the reveal tool
  first. This experiment's numbers describe what's achievable if that constraint is relaxed, not a
  validation that the constraint doesn't matter in a live setting.
