# Experiment Design: BrentMemKDM re-evaluation of the exp_5–exp_8 hard-KNN generation (exp_28)
**Experiment**: experiments/exp_28/ · **Project**: pathology-reasoning · **Date**: 2026-08-19 · **Status**: Complete

---

## 1. Motivation — exp_5–exp_8 still hold the best published LOOCV Macro-F1 in the repo

`exp_5`–`exp_8` are the original *hard-KNN* generation of the Task-1 biopsy-decision track:
unimodal tabular/MRI/text KNN sweeps plus late-fusion soft voting. `exp_8`'s equal-weight
trimodal fusion (LOOCV Macro-F1 **0.7171**) is still the best number anywhere in the repo —
above the Fuzzy-KNN generation (`exp_13`–`16`, best `exp_16` late fusion), the KDM generation
(`exp_23`/`24`, best 0.6694), and the MemKDM generation (`exp_25`–`27`, best `late_fusion_optimal`
0.6174–0.6648). None of `exp_13`–`27` was benchmarked directly against `exp_5`–`8`; every one of
them measured itself against the Fuzzy-KNN numbers instead.

`src/methods/brent_mem_kdm.py` (present in the working tree, not yet exercised by any experiment)
is a `MemKDM` whose only fitted quantity is the RBF bandwidth `sigma` — one per modality — chosen
by a global derivative-free (Brent) search of the mean MCCV Macro-F1. `x`, `y`, `w` are never
trained anywhere, and the search is fit exactly once, globally, never per fold. It exists because
of two findings already on record in this repo:

- `exp_25` found all five Stage-2 joint conditions selecting `sigma_scale=2.0` — the top edge of
  a truncated `{0.5, 1.0, 2.0}` grid — a pattern, not a coincidence, that a discrete grid cannot
  resolve.
- `exp_27` showed that per-fold gradient training of any kind (including `sigma`) turns Phase B
  into 88 independent small fits rather than a genuine frozen-parameter evaluation.

A continuous, globally-fit-once bandwidth addresses both at once, and because `BrentMemKDM.fit()`
builds a model with zero trainable parameters (a pure data-driven init), Phase B is deterministic
and seed-independent — no seed averaging is needed anywhere in this design.

**H (single, primary).** Replacing exp_5–8's per-modality KNN classifiers with `BrentMemKDM`
(same MCCV→LOOCV protocol, same base tabular/text representations, best available MRI
representation — see §2.1) will produce a fusion LOOCV Macro-F1 exceeding exp_8's 0.7171, with
McNemar significance against a recomputed leak-free KNN fusion reference (§2.2). Per-modality
arms (`tab`, `text`) are evaluated as secondary, like-for-like comparisons against exp_5's 0.6333
and exp_7's 0.6988.

## 2. Background — what's reused, what can't be reproduced, and why

Built on `src/evaluation/{data,protocol,metrics,reporting}.py` and `src/methods/{base,mem_kdm,
brent_mem_kdm}.py` — the same harness `exp_23`–`27` already use, reading
`Data/preprocessed_old/task1/` (the schema exp_5–8 used; `resolve_data_dir`/`load_cohort` handle
path resolution and the `'NONE'`-sentinel/`N=88`/54-yes/34-no assertions that correct exp_5–8's
narrated but wrong "56/32" denominators). Does **not** reuse exp_5–8's `train.py` scripts
directly — they hardcode `data/chimera26/preprocessed/task1/`, which does not exist in this
checkout.

### 2.1 `utils/embedding-kit/` is empty — exp_6/exp_8's published numbers are not reproducible targets

`exp_6`'s winning MRI representation was `embedkit_sup` (EmbedKit supervised projection,
`frozen_target_dim=384`), and `exp_8`'s 0.7171 fusion consumes that MRI arm. `utils/
embedding-kit/` is empty in this checkout (same caveat `exp_25`–`27` all carry), so neither
number is reproducible here. Two things follow:

- The MRI representation fed to `BrentMemKDM` is the best **available** one. Reading
  `experiments/exp_6/results/grid_search_results.csv` directly (not assuming embedkit's win
  transfers): among `{raw, pca, corr_0.7, corr_0.8, corr_0.9, corr_0.95}` (excluding
  `embedkit_unsup`/`embedkit_sup`), the best is `pca` (MinMax → PCA@0.90), `k=1`, `uniform`,
  `euclidean`, Phase-A mean Macro-F1 **0.5367** — barely below the embedkit winner's 0.5469, and
  the CSV's top-5 rows tie to 4 decimal places on all four metrics, so the embedkit pick was an
  arbitrary `sort_values().iloc[0]` tie-break in exp_6, not a sweep-supported result.
  `build_mri_features` (`src/evaluation/data.py:218`) supports `pca_variance` directly; no
  correlation-pruning step exists in the shared harness and none is added for this experiment.
- `exp_6`'s 0.5335 and `exp_8`'s 0.7171 are reported as **published-but-not-reproducible-here**
  references, alongside two **recomputed** KNN baselines built inside this experiment's own
  script from the same available `pca` MRI representation (§4, "Reference arms") — the actual
  comparators for C2 and C4.

### 2.2 exp_8's weighted-fusion number was selected on the evaluation set — not reproduced

`exp_8`'s `Optimal-Weighted-Trimodal` condition grid-searches fusion weights by scoring directly
against the 88 LOOCV labels it then reports (`evaluate_probs(p_comb, y_labeled)` on the OOF
predictions) — the same defect `exp_16`'s fusion weights had, which `src/methods/mem_kdm.py`'s
`search_fusion_weights` was written to fix (weights selected on MCCV validation splits, never on
LOOCV output). `Optimal-Weighted-Trimodal` merely **ties** the honest `Equal-Trimodal-Fusion` row
(both 0.7171) in exp_8's own results, so the target number to beat is unaffected either way, but
this experiment does not reproduce the eval-set weight sweep. Its own honest fusion-weight
condition (`fusion_optimal_leakfree`, §3) selects weights on **Stage-1 MCCV validation
probabilities only**.

### 2.3 The frozen sigma is not leak-free — same accepted bias as exp_17/exp_27

`BrentMemKDM.search()` fits `sigma` once against the mean Macro-F1 over the 100 MCCV splits,
which collectively cover the whole labeled cohort. The frozen value used in every LOOCV fold of
Phase B was therefore informed, in aggregate, by data that includes that fold's own held-out
patient — the same shape of bias `exp_17`'s frozen meta-thresholds and `exp_27` §2.3's frozen
`STAGE1_MEAN_SIGMA` carry. Not claimed leak-free; stated here and in the summary.

### 2.4 Required `src/methods/mem_kdm.py` fix — already in the working tree, needs a commit

`BrentMemKDM.to_memkdm()` always builds a model with `x_train=y_train=w_train=False` and every
kernel `trainable=False` — zero `requires_grad` parameters. `MemKDM.fit()`'s working-tree diff
(`git status`: `M src/methods/mem_kdm.py`) guards the optimizer construction and training loop on
`any(p.requires_grad for p in model.parameters())`, skipping both when nothing needs a gradient —
without it, `loss.backward()` raises (`kdm.layers.*`'s `nn.Parameter`s always exist; only
`requires_grad` varies). This fix is a prerequisite for `BrentMemKDM.fit()` to run at all and must
be committed (together with `src/methods/brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`,
and `experiments/exp_25`–`27`, none of which are committed yet) before this experiment's final run,
so `results/git_commit.txt` names the code actually evaluated.

### 2.5 Soft targets consume the confidence label — labeled, not hidden

Alongside the hard `y ∈ {0,1}` arm (exp_5–8's original supervision), every condition is also run
with `exp_13`-style soft targets (`build_targets(y_binary, confidence, CONFIDENCE_CERTAINTY_MAP)`
— clear=1.0/borderline=0.5/uncertain=0.25). The confidence label is itself a separate challenge
target (`MemKDM`/`BrentMemKDM` track this as `target_informed=True`); `exp_13`–`16` used this
supervision and the repo accepted it, but every soft-arm results row is labeled
`target_informed` so a soft-arm win over the hard exp_5–8 baselines is not misread as a pure
method gain.

### 2.6 `scripts/verify_brent_mem_kdm.py` has never been run

No logged output, results artifact, or logbook entry references it. Its check 1 is the only
existing validation of the fast Nadaraya-Watson reduction (`_FoldCache`) against a real
`MemKDM.predict_proba` at the search bounds — where `KDMLayer`'s `out_w` clamp is most likely to
fire. Running it in full (not `--quick`) is Step 0 of this experiment (§4), before any
`BrentMemKDM` number is trusted.

## 3. Conditions

Two supervision arms (`hard`, `soft`) × four primary conditions, plus two secondary conditions
run once each (supervision arm noted per condition):

| # | Condition | Modalities | Search | Replaces | Target |
|---|-----------|-----------|--------|----------|--------|
| C1 | `tab` | tabular | 1-D Brent | exp_5 | **0.6333** (published, reproducible — Step-0 gate) |
| C2 | `mri` | MRI (`pca`) | 1-D Brent | exp_6 | recomputed `knn_mri_pca` reference (§4) |
| C3 | `txt` | text | 1-D Brent | exp_7 | **0.6988** (published, reproducible — Step-0 gate) |
| C4 | `fusion_equal` | tab+mri+txt, soft-voted 1/3 each | 3× independent 1-D Brent | exp_8 | recomputed `knn_fusion_equal` and published **0.7171** |

Secondary (both arms not required — hard only, since they exist to test a structural question,
not to re-litigate supervision):

| # | Condition | Purpose |
|---|-----------|---------|
| S1 | `fusion_optimal_leakfree` | Honest counterpart of exp_8's leaky weighted fusion — simplex weights chosen on Stage-1 MCCV validation probabilities only |
| S2 | `joint_trimodal` | Single product-kernel `BrentMemKDM` over all three modalities, `strategy="coordinate"`. Exploratory: `exp_25`/`26`/`27` refuted joint product-kernel multimodal three times (0.5860/0.6048/0.4943, each below its own run's late fusion). Tests whether a continuous bandwidth changes that finding; this experiment's weight sits on C1–C4, not S2. |

## 4. Representation grids and search budget

Representation is not a `BrentMemKDM` search dimension (it searches `sigma` only) — it is a
caller-side Phase-A grid, scored by the same mean MCCV Macro-F1, resolved by
`select_best` (`src/evaluation/protocol.py:54` — std then `cfg_id` tie-break; exp_5–8 had no
tie-break at all, which is how exp_6 landed on an arbitrary embedkit pick, §2.1):

| Modality | Grid | Source |
|---|---|---|
| `tab` | one representation: `build_tabular_features(..., dre_categories=cohort.dre_categories)` | mandatory, not optional — see below |
| `mri` | `pca_variance ∈ {None, 0.90}` (`raw_l2`, `pca90_l2`) | `build_mri_features` |
| `txt` | `max_features ∈ {500, 2000, None}`, `pca_variance=0.90` | `build_text_features`; 500 = exp_7's winner, 2000 = exp_27's |

**`dre_categories=cohort.dre_categories` is mandatory for `tab`.** `build_tabular_features`'s own
docstring (`src/evaluation/data.py:184`) records that inferring categories from train (exp_5's
behavior, `dre_categories=None`) yields <5 `dre` levels in 49/100 MCCV splits and 3/88 LOOCV
folds — 3 of 5 levels are singletons in the 88-cohort — which crashes `init_kdm_layer`'s
shape-checked `copy_`. This gives the KDM arm a fixed-width one-hot where exp_5's KNN used an
inferred-width one; a stated deviation, already the accepted convention in `exp_23`/`26`/`27`.

**Stated deviation, unavoidable:** `build_mri_features`/`build_text_features` add a MinMaxScaler
and a trailing `Normalizer(norm="l2")` that exp_6/exp_7's original pipelines did not have. The L2
step is what makes an RBF kernel behave like cosine similarity in this KDM lineage (`KDMLayer`
squares kernel values, so a raw cosine kernel would give an anti-aligned −1 the same weight as an
aligned +1 — verified in the `MemKDM` lineage's own module docstring). exp_7's winner used a
genuine `cosine` KNN metric, so this is faithful in spirit for text; for MRI, exp_6's `pca`
winner used `euclidean`, and L2-normalizing makes euclidean rank-equivalent to cosine on unit
vectors. `src/evaluation/data.py` is not edited for this experiment — the KNN reference arms
(§5) use exp_5/6/7's *original* pipelines verbatim so the Step-0 reproduction gates stay exact.

**Search budget** (per `scripts/verify_brent_mem_kdm.py`'s own calibration):

- C1–C4's per-modality searches: `strategy="nested"` (degenerates to plain 1-D Brent for one
  modality), defaults `n_prescan=15, maxiter=20` (~35 evaluations each — cheap on the fast
  Nadaraya-Watson path).
- S2 (`joint_trimodal`): `strategy="coordinate"`, `n_prescan=7, maxiter=10, max_rounds=5` — the
  verification script's own reduced trimodal budget. The nested default at full budget is
  ~(35)³ leaf evaluations per fold-set and is not used here.
- `metric="macro_f1"`, `aggregate="mean"`, `backend="auto"` (fast path), default
  `bounds_mult=(1/32, 32)`. `label_smoothing=0.0` for `hard`; matches `build_targets`'s soft
  encoding for `soft` (no additional label smoothing layered on top).

**Assertion in `train.py`:** folds passed to `run_brent_search` are built only from
`iter_mccv_splits` (the 100 MCCV splits) — never from LOOCV folds, where `n_val=1` makes
Macro-F1 degenerate and `_auroc_scalar` returns a hardcoded 0.5.

## 5. Reference arms (recomputed, because embedkit is unavailable)

Computed inside this experiment's own script, through the same `run_loocv`, for honest
comparison targets where the published exp_6/exp_8 numbers can't be reproduced (§2.1):

- `knn_mri_pca` — `KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")`
  on exp_6's `pca` representation (MinMax → PCA@0.90, no L2) — the best available non-embedkit
  config from exp_6's own grid CSV. Honest **exp_6 reference** for C2.
- `knn_fusion_equal` — equal-weight soft vote of exp_5's tabular KNN + `knn_mri_pca` + exp_7's
  text KNN. Honest **exp_8 reference** for C4.

Both are labeled as recomputed, not published; exp_6's 0.5335 and exp_8's 0.7171 are reported
alongside with a footnote that they are not reproducible in this checkout.

## 6. Step 0 — reproduction gates

- **G0** — `python scripts/verify_brent_mem_kdm.py` (full run, not `--quick`): all 5 checks pass.
- **G1** — exp_5's KNN (`k=3, uniform, euclidean`, exp_5's original un-`dre_categories`-fixed
  pipeline) through `run_loocv` reproduces Macro-F1 `0.6333333333333333`, confusion 46/14/20/8.
- **G2** — exp_7's KNN (`max_features=500`, TF-IDF → PCA@0.90, `k=1, uniform, cosine`, exp_7's
  original pipeline) reproduces `0.6987539367383268`, confusion 42/21/13/12.
- exp_6/exp_8 have **no gate** — §2.1 states why. `train.py` asserts G1/G2 with
  `abs(got - expected) < 1e-12` and aborts the run on failure, matching `exp_25`/`26`'s pattern.
  If a gate misses by a hair, `load_cohort`'s `dtype=str, keep_default_na=False` reading (vs.
  exp_5–8's bare `read_csv`) is the first place to check.

## 7. File Layout

```
experiments/exp_28/
├── DESIGN.md
├── IMPLEMENTATION.md          ← added after this file is approved
├── scripts/train.py            ← self-contained; --smoke flag
├── results/
│   ├── reproduction_gates.json       ← G0/G1/G2 pass/fail + values
│   ├── phasea_grid_tab.csv, _mri.csv, _txt.csv
│   ├── stage1_best_hparams.json      ← per-modality winning rep + sigma_mult/sigma_ref/sigma*
│   ├── sigma_search_traces.json      ← SigmaSearchResult.trace per condition
│   ├── loocv_metrics.json            ← all 4 primary conditions × 2 arms + S1/S2 + reference arms
│   ├── loocv_predictions.csv
│   ├── mcnemar.json
│   └── git_commit.txt
└── reports/
    ├── figures/   ← sigma-vs-MacroF1 curves per modality, confusion matrices, ROC overlay
    └── summary.md
```

## 8. Evaluation Protocol & Decision Rules

Same two-phase MCCV→LOOCV harness as exp_5–8 (`CLAUDE.md`'s protocol; `src/evaluation/protocol.py`):
Phase A = 100 MCCV splits (`experiments/exp_4/results/mccv_design.csv`), select on mean Macro-F1;
Phase B = 88-fold LOOCV, frozen sigma, model re-fit per fold, hyperparameters never re-fit.
Metrics: `binary_metrics` (Macro-F1 primary; accuracy, sensitivity, specificity, AUROC, confusion
counts) + `mcnemar_exact` against each condition's KNN counterpart.

- **H1 (primary):** at least one `BrentMemKDM` arm of C4 (`fusion_equal`, hard or soft) exceeds
  exp_8's 0.7171 LOOCV Macro-F1, with `mcnemar_exact` vs. `knn_fusion_equal` reported.
- **H2 (per-modality, secondary):** C1 vs. 0.6333 and C3 vs. 0.6988 are like-for-like; C2 is
  judged against `knn_mri_pca` only (§2.1 — no like-for-like exp_6 target exists here).
- Given `std_macro_f1 ≈ 0.11` across MCCV splits in this cohort (dwarfing typical inter-config
  gaps <0.02, per `exp_13`/`23`), a Phase-B improvement without a significant McNemar result is
  reported as **not established**, not as a win — the failure mode `exp_23`'s "Partial" verdict
  already recorded.
- S1/S2 are reported as context, not scored against H1/H2.

## 9. Scope

**In scope:** re-running C1–C4 (both supervision arms) and S1/S2 (hard only) under §3–6;
recomputed MRI/fusion reference arms where embedkit is unavailable; reproduction gates G0–G2;
`mcnemar_exact` against every condition's KNN counterpart; `sigma_mult` reported against exp_27's
frozen grid winners (`tab` 0.5, `mri` 1.0, `txt` 0.5) and exp_25's edge-of-grid `sigma_scale=2.0`
pattern, as a secondary readout on whether the continuous search finds something the grid missed.

**Out of scope:** editing `src/evaluation/*` (representation pipelines used as-is); the new-schema
data (`Data/preprocessed/task1/`); reproducing exp_6/exp_8's embedkit-based numbers; reproducing
exp_8's eval-set-leaky weight sweep; re-opening `x_train`/`y_train`/`w_train` as search dimensions
(`BrentMemKDM` fixes all three `False` by construction — this is the method under test, not a
choice made per-experiment); seed averaging (`BrentMemKDM.fit()` is deterministic).

## 10. Next Steps

1. Review and accept this design.
2. `IMPLEMENTATION.md` (concrete build plan, exact execution command) for approval.
3. Commit `src/methods/mem_kdm.py`'s `has_trainable` guard, `src/methods/brent_mem_kdm.py`,
   `scripts/verify_brent_mem_kdm.py`, and `experiments/exp_25`–`27` (all currently uncommitted).
4. Implement `scripts/train.py`, run `--smoke`, then the full run.
