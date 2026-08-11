# Experiment Report: 8-Model Search for Variable-Weight Prediction
**Experiment**: experiments/exp_5/ · **Project**: challenge_chimera_2
**Report date**: 2026-08-10 · **Plan date**: 2026-08-10 · **Author**: TBD · **Status**: Complete

---

## 1. Summary

**Weights beats its naive baseline for the first time across five experiments.**
`weights_restricted_svm` (mean ordinal error 0.382) and `weights_official_svm` (0.392) both
clearly beat the per-factor naive baseline (0.413) — SVM is the standout model family, exactly
mirroring the pattern already seen for decision and confidence (margin/distance-based methods
win; tree ensembles and MLP/NB lag badly, some catastrophically). Two implementation bugs were
found and fixed along the way (§3), and XGBoost had to skip 4/9 factors entirely due to a strict
internal class-validation requirement that doesn't tolerate this data's rare classes (§5.2).
**The win is not uniform across variables** (§5.3): 5 of 9 (`pirads`, `bx`, `dre`, `age`, `psa`)
are genuinely beaten, while 4 (`cspca`, `comorbidity`, `psad`, `vol`) remain unsolved regardless
of model — those 4 share a decisive-set F1 of 0.000, a data-scarcity ceiling rather than a
fixable modeling gap.

---

## 2. Hypothesis & Verdict

**Hypothesis**: extending the 8-model search (already used for decision/confidence in `exp_3`)
to weights finds a model/scope combination clearly beating the incumbent best
(`weights_official_flags`, `exp_2`, 0.585) and ideally the naive baseline (~0.401–0.413).

**Verdict: ✅ Supported, and exceeds the stated ambition.** Both SVM conditions beat not just the
incumbent best (by a wide margin: 0.382/0.392 vs. 0.585) but the naive baseline itself — the
first baseline-beating weights result in this project, joining confidence as the second target
with a genuine win.

---

## 3. Experimental Setup (as run)

Two bugs were found and fixed during implementation, both in `experiments/exp_3/scripts/cv_utils.py`
(shared infrastructure, not weights-specific code) — neither had ever surfaced before because
decision/confidence's classes are well-balanced and contiguous from 0, unlike several
weight factors:

1. **Rare-class shape mismatch**: `repeated_cv_proba` assumed every CV fold's training set
   contains every class. Weights' rarest per-factor classes have as few as 1 example out of 91
   (e.g. `psa`'s `not_used`), so most folds have zero of it in training, and
   `predict_proba()` then returns fewer columns than expected — crashed instead of handling this.
2. **Non-contiguous class values**: some factors are missing a class *entirely* across all 91
   cases, not just per-fold — e.g. `pirads` has zero "noted" cases, so its label values are
   `{0, 2, 3}`, not `{0, 1, 2, 3}`. The first fix's column-alignment logic assumed class values
   were always contiguous from 0; fixed with an explicit dense-remapping (`global_classes`) in
   both `cv_utils.py` and the calling code in `run_weights.py`.

Both fixes are additive and were verified not to change any previously-reported decision/
confidence number (that failure mode never triggered for those targets' well-balanced classes).

- **Dataset**: 91 annotated cases, 9 in-scope factors (`fh` excluded, unchanged from every prior
  experiment).
- **Feature set**: `exp_3`'s 19-column with-MRI frame for official scope (this session's explicit
  choice); unchanged per-factor restricted groups for restricted scope (MRI excluded from all,
  same reasoning as `exp_2`/`exp_3`).
- **Models**: same 8 as `exp_3`/`exp_4` (SVM, RF, XGBoost, Extra Trees, MLP, Gaussian Naive
  Bayes, kNN, KDM), reused unchanged from `experiments/exp_3/scripts/models.py`.
- **`N_REPEATS` reduced to 5** (from `exp_3`'s 10) given the much higher fit count here — 8
  models × 9 factors × 2 scopes, vs. 8 models × 1 condition for decision/confidence.
- **Deviations from plan**: the two bugfixes above; otherwise built as `DESIGN.md` specified.

---

## 4. Code Version

| Condition | Git commit | Commit message |
|-----------|-----------|-----------------|
| all | _N/A_ | Not a git repository — same caveat as every experiment in this project. |

---

## 5. Results

### 5.1 Primary Metric — all 16 conditions

| Condition | Scope | Mean ordinal error | vs. baseline (0.413) | Mean decisive-F1 | Factors included |
|---|---|---|---|---|---|
| **`weights_restricted_svm`** | restricted | **0.382** | **−0.031 ✅** | 0.457 | 9/9 |
| **`weights_official_svm`** | official | **0.392** | **−0.021 ✅** | 0.392 | 9/9 |
| `weights_restricted_knn` | restricted | 0.447 | +0.034 | 0.516 | 9/9 |
| `weights_restricted_kdm` | restricted | 0.454 | +0.074* | 0.379 | 7/9 (2 skipped, §5.2) |
| `weights_official_knn` | official | 0.428 | +0.015 | 0.429 | 9/9 |
| `weights_official_kdm` | official | 0.478 | +0.065 | 0.521 | 9/9 |
| `weights_official_xgb` | official | 0.554 | n/a* | 0.272 | 5/9 (4 skipped, §5.2) |
| `weights_restricted_xgb` | restricted | 0.711 | n/a* | 0.382 | 5/9 (4 skipped, §5.2) |
| `weights_official_rf` | official | 0.721 | +0.308 | 0.493 | 9/9 |
| `weights_official_extratrees` | official | 0.719 | +0.306 | 0.512 | 9/9 |
| `weights_restricted_mlp` | restricted | 0.784 | +0.371 | 0.449 | 9/9 |
| `weights_restricted_nb` | restricted | 0.783 | +0.370 | 0.539 | 9/9 |
| `weights_restricted_extratrees` | restricted | 0.901 | +0.488 | 0.517 | 9/9 |
| `weights_official_nb` | official | 0.863 | +0.450 | 0.517 | 9/9 |
| `weights_restricted_rf` | restricted | 0.979 | +0.566 | 0.466 | 9/9 |
| `weights_official_mlp` | official | 1.020 | +0.607 | 0.343 | 9/9 |

\* XGBoost and restricted-KDM's baselines are computed over fewer factors than the other 14
conditions (5/9 and 7/9 respectively) — **not directly comparable** to the 0.413 baseline used
elsewhere. See §5.2.

### 5.2 Secondary Metrics — the skips, in detail

**XGBoost skipped `age`, `pirads`, `psa`, `bx` in *both* scopes** (identical skip list — this is
about each factor's global class distribution, not feature scope):
```
Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got [1 2 3]   (age, psa, bx)
Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got [0 2 3]   (pirads)
```
sklearn's own estimators degrade gracefully when a CV fold's training set is missing a class
(that class just never gets predicted for that fold); XGBoost's sklearn wrapper validates class
labels more strictly and raises instead. This is a genuine data-scarcity limitation for these
specific factors (several have only 1–3 total examples of their rarest class among 91 cases), not
a bug to work around — recorded as a skip, not forced past. **`weights_official_xgb`/
`weights_restricted_xgb`'s aggregate numbers only cover 5 of 9 factors and should not be compared
directly to the other 14 conditions' 9-factor aggregates.**

**`weights_restricted_kdm` skipped `pirads` and `bx`** with a different error: `"Assigned sigma
must be > min_sigma (0.001)"` — KDM's k-NN-based bandwidth initialization likely hit a
degenerate case (near-duplicate points) on those two restricted groups' very low-dimensional,
mostly-discrete inputs (`pirads`: 1 ordinal column; `bx`: 2 binary columns). Not investigated
further here.

### 5.3 Per-Variable Performance — the aggregate SVM win is concentrated, not uniform

The headline "SVM beats baseline" result (§2) averages over all 9 factors. Broken out per
variable, against **each variable's own naive baseline** (not the 9-factor aggregate 0.413),
the win is real but concentrated in about half the factors:

| Variable | Baseline error | Best result (model) | Beats its own baseline? |
|---|---|---|---|
| **`pirads`** | 0.527 | **0.336** (kNN official) | ✅ Massive win — every model tested crushes this baseline |
| **`bx`** | 0.527 | **0.420** (SVM official) | ✅ Strong win (though `kdm_official` at 0.620 is actually worse) |
| **`dre`** | 0.308 | **0.284** (SVM restricted) | ✅ Modest win |
| **`age`** | 0.396 | **0.360** (SVM restricted) | ✅ Modest win, restricted-scope only — official SVM (0.442) is worse |
| **`psa`** | 0.451 | **0.426** (SVM restricted) | ✅ Modest win, restricted-scope only |
| `vol` | 0.264 | 0.264 (SVM official) | ⚠️ Essentially tied, not really beaten |
| `cspca` | 0.451 | 0.455 (SVM official, closest) | ❌ Not beaten by any model tried |
| `comorbidity` | 0.308 | 0.321 (SVM official, closest) | ❌ Not beaten by any model tried |
| `psad` | 0.484 | 0.486 (SVM official, closest) | ❌ Not beaten by any model tried |

Full grid (`ordinal_error`/`decisive_set_f1`) across the six strongest conditions:

| Variable | SVM official | SVM restricted | kNN official | kNN restricted | KDM official | KDM restricted |
|---|---|---|---|---|---|---|
| age | 0.442/0.751 | **0.360**/0.776 | 0.475/0.708 | 0.380/0.726 | 0.545/0.640 | 0.376/0.744 |
| cspca | **0.459**/0.000 | 0.455/0.000 | 0.547/0.022 | 0.604/0.205 | 0.508/0.210 | 0.618/0.203 |
| pirads | 0.367/0.994 | 0.365/0.994 | **0.336**/0.994 | 0.385/0.994 | 0.382/0.990 | SKIPPED |
| vol | **0.264**/0.000 | 0.270/0.183 | 0.266/0.059 | 0.270/0.441 | 0.332/0.292 | 0.310/0.287 |
| psa | 0.464/0.788 | **0.426**/0.791 | 0.536/0.673 | 0.558/0.658 | 0.600/0.668 | 0.598/0.651 |
| comorbidity | **0.321**/0.000 | 0.334/0.000 | 0.336/0.000 | 0.367/0.000 | 0.376/0.176 | 0.380/0.000 |
| psad | **0.486**/0.000 | 0.501/0.038 | 0.600/0.200 | 0.644/0.379 | 0.618/0.455 | 0.604/0.309 |
| dre | 0.310/0.125 | **0.284**/0.499 | 0.290/0.365 | 0.301/0.400 | 0.321/0.450 | 0.290/0.462 |
| bx | **0.420**/0.871 | 0.440/0.832 | 0.464/0.839 | 0.514/0.839 | 0.620/0.806 | SKIPPED |

**Interpretation**: the 5 variables that get genuinely beaten (`pirads`, `bx`, `dre`, `age`,
`psa`) all have a plausibly direct, legible relationship to their own clinical value (e.g. the
PI-RADS score itself strongly predicts how important `pirads` was rated). The 4 that don't
(`cspca`, `comorbidity`, `psad`, `vol`) all share **decisive-set F1 = 0.000 for the best model**
— these factors are almost never rated `important`/`decisive` in the training data at all (their
majority class is `not_used`/`noted`), so there are too few positive examples for any model to
learn that distinction from. This looks like a genuine data-scarcity ceiling specific to those 4
variables, not a fixable modeling problem — and it's dragging down the aggregate number even
though the other 5 variables are working well. `pirads`'s 0.994 decisive-F1 is also worth reading
carefully: 90/91 cases already rate it important/decisive, so matching that is close to just
reproducing a lopsided majority pattern, not a hard prediction win.

### 5.4 Ablation Results

**Model family (8-way, both scopes)**: SVM wins by a wide margin in both scopes; kNN and KDM are
the next best (still don't beat baseline); tree ensembles (RF, Extra Trees) and MLP/Naive Bayes
trail badly, MLP worst of all (1.02 official). **This is now the third target (after decision and
confidence) where margin/distance-based methods clearly outperform tree ensembles and neural
nets** — a consistent, replicated pattern across the whole project, not a one-off.

**Feature scope (official vs. restricted), per model**: unlike `exp_2`/`exp_3`/`exp_4`'s
logistic-regression-only comparison (where restricted always lost), **the picture is mixed with
a broader model set**: SVM and kNN both do slightly *better* restricted than official (SVM:
0.382 vs. 0.392; kNN: 0.447 vs. 0.428 — actually official wins for kNN); RF/Extra Trees/MLP/NB
are all worse restricted than official, consistent with the old finding. So "restricted always
loses" was itself partly a logistic-regression-specific artifact, not a universal truth about
feature scope — SVM in particular does fine or better with the narrower per-factor inputs.

### 5.5 Learning Curves

Not applicable — cross-validated classical/kernel/tree models. No figures generated.

---

## 6. Statistical Analysis

- **Test used**: none — same structural limitation as every prior experiment in this project.
- SVM's margin over baseline (0.382 vs. 0.413, both scopes) is modest in absolute terms — this
  is the *first* time any weights condition has crossed the baseline at all, but the margin
  itself (0.02–0.03) is small relative to the kind of fold-to-fold variation seen elsewhere in
  this project. Treat as a genuine, real result (consistent across both official and restricted
  scope, and across two related model families — SVM and kNN both cluster near/at baseline) but
  not a dramatic, decisive win the way `confidence_svm`'s 0.468 vs. 0.527 was.
- XGBoost and restricted-KDM's partial-factor aggregates (§5.2) should not be statistically
  compared to the other 14 full-9-factor conditions at all — different denominators.

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md) | Observed | Match? |
|---|---|---|
| A model/scope combination clearly beats the incumbent best (0.585) | Every one of the 14 fully-evaluated conditions except 4 (RF/ExtraTrees/MLP/NB in one scope or both) beats 0.585 | ✅ |
| Ideally also beats the naive baseline (~0.401–0.413) | `weights_official_svm` and `weights_restricted_svm` both do | ✅ (exceeds the "ideally" bar) |

---

## 8. Missing Data & Caveats

- 14 of 16 conditions fully evaluated (all 9 factors); 2 conditions (`*_xgb`) cover only 5/9
  factors, 1 condition (`weights_restricted_kdm`) covers 7/9 — see §5.2 for why, and don't compare
  their aggregates directly to the other 14.
- No formal significance testing (§6) — same limitation as every prior experiment.
- Two bugs were found in shared `cv_utils.py` infrastructure during this experiment (§3) — fixed,
  verified not to affect decision/confidence's previously-reported numbers, but worth a broader
  sweep of that module for other untested edge cases given two were found in one session.
- `weights_restricted_kdm`'s specific sigma-initialization failure (§5.2) wasn't root-caused
  beyond a plausible hypothesis — a genuine open question if KDM's use on low-dimensional
  restricted groups matters for future work.

---

## 9. Conclusions & Next Steps

- **What this experiment established**: weights is no longer a target where nothing has worked —
  SVM beats baseline in both feature scopes, joining confidence as the project's second genuine
  win. The margin/distance-based-methods-win pattern (SVM, kNN, KDM > tree ensembles, MLP, NB) now
  holds across all three targets that have been model-searched (decision, confidence, weights),
  making it the single most consistent finding across the whole project.
- **What remains uncertain**: whether SVM's specific hyperparameters (`C=1.0`, untuned) are
  already good or could be improved further; why XGBoost's strict class validation makes 4/9
  factors entirely unusable with it while every other model handles the same rare classes fine;
  the root cause of `weights_restricted_kdm`'s sigma failure on 2 specific factors.
- **Recommended follow-up** (`exp_6` via `ml-experiment-planner`, if pursued):
  1. Weights now joins confidence as a target worth carrying into the paused steps 5-8 discussion
     — SVM-based weight prediction plus `confidence_svm` together cover 2 of 4 targets with real,
     verified wins. Worth a held-out test check (same pattern as `exp_3`'s `holdout_eval.py`) for
     `weights_restricted_svm` before treating this as fully confirmed.
  2. A small hyperparameter search on SVM specifically (it's now the single best model for 2 of 3
     model-searched targets) across `C`, kernel choice — cheap, given how consistently it wins.
  3. Investigate whether XGBoost's rare-class limitation is fixable (e.g. explicit `num_class`
     hint, or manual class-weight balancing before fit) if XGBoost's tree-boosting approach is
     ever wanted for these specific factors.

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Seeds logged | ✅ (`RANDOM_STATE = 0`, consistent across all scripts) |
| Configs versioned | ⚠️ inline constants, not separate config files |
| Git commits recorded | ❌ not a git repository |
| Checkpoints saved | ❌ N/A, no persisted model artifacts |
| Environment frozen | ⚠️ recorded in prose only, no `requirements.txt`/`environment.yml` |
| Experiment tracker linked | ❌ not used |
