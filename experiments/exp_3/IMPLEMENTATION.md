# exp_3 Implementation Plan — 8-Model Comparison + MRI-PCA(2) + Decorrelated PSA Family

## Context

This implements `experiments/exp_3/DESIGN.md` (reviewed and refined with the user across many
turns this session — feature set, model list, and per-factor restricted groups are all already
locked, not open questions). `exp_1` found weak signal from tabular ML overall; `exp_2` found
schema-restriction helps decision but hurts confidence, and per-factor feature restriction hurts
weights. `exp_3` broadens the model search (SVM, Random Forest, XGBoost, Extra Trees, MLP,
Gaussian Naive Bayes, kNN, KDM) for decision + confidence specifically, on a refined 19-column
feature set (PSA family reduced to `psa`+`psad`, comorbidity fixed to grouped flags, 2-component
MRI-PCA added), to see whether a wider model search — rather than more feature engineering —
finally beats baseline for confidence and weights, and to look for further gains on decision
beyond `exp_2`'s already-positive result.

**This plan, once approved, gets saved as `experiments/exp_3/IMPLEMENTATION.md`** before any
other files are touched, per this project's established convention (see `exp_1`/`exp_2`).

## Files to Change

### 1. `src/chimera_task1/features.py` — one small addition

- `select_exp3_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame`: calls
  the existing `select_official_feature_frame(inp, comorbidity_treatment="flags")` (unchanged,
  already defaults to flags), drops `cli_psap`/`cli_psav` (the two columns removed per the PSA-
  family collinearity discussion), and joins the pre-computed `mri_pca` frame. `mri_pca` is
  built by the runner scripts via the **existing, already-generic**
  `chimera_task1.train_decision.mri_pca_features(inp, n_components=2)` — no changes needed there,
  it already takes `n_components` as a parameter.
- **No changes needed to `restricted_feature_group()` / `TASK1_VARIABLE_TO_FEATURE_GROUP`** —
  verified these already match `exp_3`'s locked restricted-group table exactly (they never
  included `psap`/`psav`/MRI in any group to begin with, so dropping those from the "official"
  frame doesn't affect the restricted groups at all). Called with `comorbidity_treatment="flags"`
  fixed, same as `exp_2`.

### 2. `experiments/exp_3/scripts/models.py` — new, the 8-model registry

A `build_sklearn_models(n_classes: int) -> dict[str, estimator]` function returning the 7
sklearn/xgboost-compatible estimators (KDM is handled separately in each runner via the existing
`fit_predict_kdm`, since it isn't a plain sklearn `Estimator`). Hyperparameters chosen for
N=91–195 (shallow/regularized, not tuned via search — noted as a limitation in the eventual
report, consistent with `exp_1`/`exp_2`'s "not tuned via validation" caveat):

| Model | Key hyperparameters |
|---|---|
| `SVC` | `kernel="rbf", C=1.0, class_weight="balanced", probability=True` |
| `RandomForestClassifier` | `n_estimators=200, max_depth=4, min_samples_leaf=10, class_weight="balanced"` |
| `XGBClassifier` | `n_estimators=100, max_depth=3, min_child_weight=5, learning_rate=0.1, reg_lambda=1.0` |
| `ExtraTreesClassifier` | same shape as Random Forest |
| `MLPClassifier` | `hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, early_stopping=True` |
| `GaussianNB` | defaults |
| `KNeighborsClassifier` | `n_neighbors=7, weights="distance"` |

**Class-imbalance handling differs by model capability** (noted explicitly rather than silently
inconsistent): `class_weight="balanced"` natively for SVC/RF/ExtraTrees; `sample_weight` from
`sklearn.utils.class_weight.compute_sample_weight("balanced", y)` passed to `.fit()` for
XGBoost/GaussianNB (both support `sample_weight`, neither supports `class_weight` for multiclass
the way sklearn's own estimators do); **no rebalancing for MLP/kNN** (neither supports
per-class or per-sample weighting in sklearn) — each `metrics.json` for these two records
`"class_imbalance_handling": "none"` so this isn't silently inconsistent with the other six.

**Feature scaling**: `StandardScaler` applied uniformly for all 8 models (via the same
`build_preprocessor(...)` → `StandardScaler` pattern already used for KDM in `exp_1`/`exp_2`) —
doesn't hurt tree-based models and keeps one consistent preprocessing path instead of a
per-model conditional (resolves the open question flagged in `DESIGN.md` §9).

### 3. `experiments/exp_3/scripts/run_decision.py` and `run_confidence.py`

Each builds the 19-column `exp_3` frame (`select_exp3_feature_frame` + `mri_pca_features(inp,
n_components=2)`), then:
- For the 7 sklearn/xgboost models: reuses the CV pattern from `exp_2/scripts/run_decision.py`
  (`RepeatedStratifiedKFold` + `cross_val_score` for F1, `StratifiedKFold` + `cross_val_predict`
  for ROC-AUC/PR-AUC on decision; the generalized out-of-fold loop for confidence — see below).
- For KDM: reuses `fit_predict_kdm` exactly as `exp_2`'s runners did (`n_classes=2` for decision,
  `n_classes=3` for confidence), no changes.
- **`run_confidence.py` needs one small generalization not available in `train_reasoning.py`**:
  that module's `repeated_out_of_fold_predict()` hardcodes `make_classifier()` (the fixed
  OvR-logistic). `exp_3` needs the same repeated-out-of-fold-CV *shape* but parameterized by an
  arbitrary classifier. Rather than modifying `train_reasoning.py` (kept untouched per this
  project's "shared code stays reproducible" convention — `exp_1`/`exp_2`'s results must remain
  reproducible from that same code), `run_confidence.py` defines its own
  `repeated_out_of_fold_predict_generic(X, y, preprocessor, clf_factory)` — a ~10-line adaptation
  of the existing function with `clf_factory()` substituted for the hardcoded `make_classifier()`
  call, living in the experiment-scripts layer where `exp_2` already established that
  small per-experiment adaptations belong (e.g. `exp_2`'s `run_reveal.py` similarly adapted
  `eval_reveal()`'s logic rather than modifying `train_reasoning.py`).
- Writes `results/decision_<model>/metrics.json` (8 files) and `results/confidence_<model>/metrics.json`
  (8 files), same schema as `exp_1`/`exp_2` (condition, target, features, model, primary
  metric(s), CV description, class-imbalance-handling note where relevant).

### 4. `experiments/exp_3/scripts/run_weights.py` and `run_reveal.py`

Thin adaptations of `exp_2`'s equivalents: same `repeated_out_of_fold_predict()` /
`restricted_feature_group()` reuse, just swapping in `select_exp3_feature_frame(...)` for the
feature frame and dropping the now-settled comorbidity-treatment loop (single "flags" condition
instead of the count/flags pair `exp_2` tested). `run_weights.py` still produces both
`weights_official` and `weights_restricted` (per this session's explicit request to re-test that
variant with the new feature set).

### 5. New dependency: `xgboost`

Not yet installed in the project `.venv` — `pip install xgboost` before running `run_decision.py`
/ `run_confidence.py`.

### 6. No changes to `src/chimera_task1/{train_decision,train_reasoning,train_confidence_kdm}.py`

Same rule as `exp_2` — these stay exactly as they are so `exp_1`/`exp_2`'s results remain
reproducible from the same code. All new logic is additive (`features.py`) or lives in
`experiments/exp_3/scripts/`.

## Verification

1. Unit-check `select_exp3_feature_frame()` — confirm it returns exactly 19 columns (`cli_psap`/
   `cli_psav` absent, `mri_pca_0`/`mri_pca_1`/`mri_missing` present), and that
   `restricted_feature_group()` output is unchanged from `exp_2` (spot-check a few factors).
2. Smoke-test each of the 8 models individually on a small subset before committing to the full
   repeated-CV run (same approach used in earlier sessions to catch API mismatches — e.g. KDM's
   `liblinear`-multiclass issue found this way in `exp_1`).
3. Run all 4 scripts (backgrounding the slower ones — XGBoost/MLP/KDM — as needed, same pattern
   as `exp_2`'s decision/confidence runs); confirm all 19 `results/<condition>/metrics.json`
   files are written and valid JSON.
4. Compare all 19 new numbers against `exp_1`/`exp_2`'s existing results (already on disk) before
   deciding whether/how to write `experiments/exp_3/reports/summary.md`, per this project's
   established review-before-report pattern.
