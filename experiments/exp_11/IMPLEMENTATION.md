# exp_11 Implementation Plan — Direct ARD-KDM for Confidence

## Context

`experiments/exp_11/DESIGN.md` (status: Proposed, reviewed this session) revives `exp_2`/`exp_3`'s
pre-`exp_6` approach — a KDM trained *directly* on the confidence label (a genuine 3-class
classifier, not a signal derived from the decision-trained backbone `exp_6`–`exp_10` have all used).
`exp_3`'s original directly-trained `confidence_kdm` already holds this project's best-ever
confidence macro-F1 (0.508) despite using the worst backbone available at the time (scalar `σ`, an
inferior frame) — nobody has retried it with the two real backbone improvements confirmed since
(`exp_9`'s ARD, the 23-column frame). This experiment tests that exact combination, reusing `exp_9`'s
ARD code entirely unchanged.

`exp_11` is also the **first experiment run under both reporting conventions confirmed this
session** (`experiments/INDEX.md`'s "📈 full metric-suite" and "📐 LOO evaluation protocol" notes) —
full metric suite and mandatory LOO from the start, not backfilled later.

**This plan, once approved, gets saved as `experiments/exp_11/IMPLEMENTATION.md`.**

## Key technical findings from this planning session

1. **`fit_kdm_backbone_ard`/`compute_signals_ard` (`experiments/exp_9/scripts/ard_kernel.py`) are
   already fully generic over `n_classes` — zero changes needed.** `fit_kdm_backbone_ard` builds
   `y_onehot = F.one_hot(yt, n_classes)` and constructs `ARDKDMClassModel(..., dim_y=n_classes, ...)`
   — nothing hardcodes 2 classes anywhere in either function. Calling it with the 3-level confidence
   rank and `n_classes=3` is the *only* change needed versus every prior use on the decision target
   — the same genericity `exp_1`'s scalar-backbone equivalent (`train_confidence_kdm.fit_predict_kdm`)
   already relied on for its own confidence conditions back in `exp_2`/`exp_3`.
2. **No recalibration step is needed at all — this experiment is structurally simpler than
   `exp_9`/`exp_10`'s derived-signal scripts.** `run_signals_23col.py` fits one KDM on decision, then
   fits *five more* small models (isotonic × 3, logistic blend × 2) on top of its derived signals.
   Here, the model **is** the confidence classifier — its own `argmax(probs)` is the prediction, no
   downstream fitting step, no `isotonic_rank`/`blend_rank` helpers needed.
3. **`ordinal_distance()` (`src/chimera_task1/reasoning_labels.py`) takes label *strings* + a
   `rank_map` dict, not integer ranks** — every prior script maps predictions back through
   `CONFIDENCE_LEVELS[pred_rank]` before calling it (see `run_signals_23col.py` line ~180); this
   experiment follows the identical pattern, reusing `ordinal_distance` unchanged.
4. **Multiclass AUROC and Brier score need new small helpers — the first time this project has
   scored a >2-class target with the full metric suite.** Per `experiments/INDEX.md`'s reporting
   note: AUROC is one-vs-rest macro (`sklearn.metrics.roc_auc_score(y_true, proba, multi_class="ovr",
   average="macro", labels=[0,1,2])` — wrapped in a `try/except ValueError` for the edge case where a
   fold's test set happens to be missing a class entirely, e.g. a single-row LOO fold); Brier score
   has no sklearn multiclass built-in (`brier_score_loss` is binary-only) and needs a small explicit
   function: `mean over samples of Σ_k (p_k − 1[y=k])²`. Both are pure, tiny, and go in one new
   shared helper module rather than being duplicated across the three scripts below.
5. **Held-out and LOO both reuse existing infrastructure unchanged.** `mri_pca_train_only`
   (`experiments/exp_3/scripts/holdout_eval.py`) is frame-agnostic, already reused unchanged by
   `exp_9`/`exp_10`. The held-out split itself stays **stratified on `y_decision`, not confidence** —
   "the same fixed 19-case split used since `exp_3`" means the same row indices (`exp_3`'s original
   `holdout_eval.py` scored `confidence_svm`/`confidence_kdm` on this exact decision-stratified
   split too, so this isn't a new convention, just reviving an old one). LOO follows
   `experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py`'s `run_loo()` shape
   (deterministic single pass, pooled out-of-fold prediction, scored once) generalized from 2 to 3
   classes.
6. **Both frames reuse existing frame-selection functions unchanged**: `select_exp3_feature_frame`
   (19-col, `chimera_task1.features`) and `select_exp8_feature_frame` (23-col,
   `experiments/exp_8/scripts/features_v3.py`) — no new frame code, matching `exp_9`'s own dual-frame
   pattern exactly.

## Files to Add

### 1. `experiments/exp_11/scripts/metrics_multiclass.py` — small shared helper module

- `multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float`: builds
  the one-hot true-label matrix, returns `((proba - onehot) ** 2).sum(axis=1).mean()` (finding #4).
- `safe_multiclass_auroc(y_true: np.ndarray, proba: np.ndarray, labels: list[int]) -> float | None`:
  wraps `roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)` in
  `try/except ValueError`, returning `None` (logged, not crashed) if a class is missing from the
  scored set — same discipline as this project's other per-fold degenerate-case handling (`exp_5`'s
  `ValueError`-catch precedent for rare per-factor classes).

### 2. `experiments/exp_11/scripts/run_confidence_direct_ard.py`

CV loop, both frames, in one script (loop over `{"19col": select_exp3_feature_frame, "23col":
select_exp8_feature_frame}`). Per fold: `build_preprocessor(X_frame)` + `StandardScaler` (fit train
rows only — the same two-stage shape every `exp_6`–`exp_9` script already uses, *not*
`exp_10`'s MinMax/one-hot convention, since this experiment isn't testing preprocessing), then
`fit_kdm_backbone_ard(X_train, y_confidence_rank[train_idx], n_classes=3, **ARD_CONFIG)` with
`ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}` (finding #1, `exp_9`'s original
defaults, no search per `DESIGN.md` §2c). Per repeat: pool every fold's out-of-fold `probs` into one
`(91, 3)` array, take `argmax` for predictions, then score once (finding #2's simplification means
no signal/recalibration bookkeeping is needed — just `probs` in, five metrics out):
accuracy (`accuracy_score`), macro-F1 (`f1_score(average="macro", labels=[0,1,2], zero_division=0)`),
ordinal distance (`ordinal_distance` on `CONFIDENCE_LEVELS`-mapped predictions, finding #3),
one-vs-rest AUROC (`safe_multiclass_auroc`), Brier score (`multiclass_brier_score`) — mean±std across
the 10 repeats. Writes `confidence_kdm_direct_ard_19col` and `confidence_kdm_direct_ard_23col`.

### 3. `experiments/exp_11/scripts/holdout_eval_confidence_direct_ard.py`

Both frames, one script. Same split as every prior held-out check
(`train_test_split(..., test_size=0.2, stratify=y_decision, random_state=0)`, finding #5) — decision-
stratified by convention, scored on confidence. `mri_pca_train_only` for the MRI-PCA alignment;
`build_preprocessor` + `StandardScaler` fit on the train portion only. Single-value full metric suite
per frame (no repeats — a one-shot check). Cites `exp_3`'s original `confidence_kdm`/`confidence_svm`
held-out numbers as reference constants if available in that experiment's results, alongside the CV
numbers from script 2 — not recomputed, cited exactly as every prior held-out script cites its CV
counterpart.

### 4. `experiments/exp_11/scripts/loo_confidence_direct_ard.py`

Both frames, one script, following `verify_decision_loo_repeated_holdout.py`'s `run_loo()` shape
(finding #5): `LeaveOneOut()`, 91 folds, `mri_pca_train_only` per fold (train-only-fit MRI-PCA, same
held-out-family discipline that script already established), pool all 91 out-of-fold `probs` into
one `(91, 3)` array, score the full metric suite once (deterministic, no repeat variance to report).

### 5. No changes to `exp_1`–`exp_10`'s scripts, the `kdm` library, or `src/chimera_task1/*.py`

Same rule as every prior experiment. `metrics_multiclass.py` is new, `exp_11`-only code.

## Execution Order

**Priority 1**: `run_confidence_direct_ard.py` (the core CV result), `holdout_eval_confidence_direct_ard.py`
(mandatory per `DESIGN.md` §7).
**Priority 2**: `loo_confidence_direct_ard.py` (mandatory per the same section, first-ever LOO check
for a non-decision subtask).

## Verification

1. **Smoke-test `fit_kdm_backbone_ard` with `n_classes=3`** on a small subset: confirm output shape
   `(n, 3)`, `probs` rows sum to 1, `compute_signals_ard`'s built-in `probs_check_ok` is `True` —
   same discipline as every prior ARD experiment's first check.
2. **Smoke-test `multiclass_brier_score`** against known closed-form cases before trusting real
   numbers: a perfect one-hot prediction should score 0; a uniform `[1/3, 1/3, 1/3]` prediction
   against any true label should score a known constant (`2/3`).
3. **Smoke-test `safe_multiclass_auroc`** on a small subset with all 3 classes present (should return
   a float, not `None`) and confirm the `ValueError` path is actually reachable (construct a
   deliberately single-class test fold and confirm it returns `None` without crashing).
4. **Smoke-test the `CONFIDENCE_LEVELS[pred_rank]` → `ordinal_distance` integration** on one fold:
   confirm the mapped strings are valid keys in `CONFIDENCE_RANK`.
5. Run Priority 1 (background); sanity-check the CV result isn't wildly divergent from `exp_3`'s
   original `confidence_kdm` (0.530 ordinal distance / 0.508 macro-F1) before trusting it — a huge
   swing either direction would suggest a bug, not a finding.
6. Run Priority 2.
7. Compare every result against `DESIGN.md` §4's baseline table and §8's decision-rule branches
   before writing `experiments/exp_11/reports/summary.md`.
