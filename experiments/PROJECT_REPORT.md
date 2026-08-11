# CHIMERA-Agent Task 1 — Consolidated Results Report
**Project**: challenge_chimera_2
**Covers**: `exp_1`, `exp_2`, `exp_3`, the 2026-08-10 decision-label bugfix, and the held-out test
verification
**Report date**: 2026-08-10
**Status**: Current as of this date — see `experiments/INDEX.md` for the live per-experiment index

---

## 1. Executive Summary

Three experiments explored whether small supervised models on Task 1's structured clinical
features can predict the biopsy decision, confidence, per-factor variable weights, and reveal
sequence well enough to justify building the full submission pipeline (steps 5-8 of the original
plan: rubric scorer, MCP+LLM wiring, Docker packaging, platform submission).

**Bottom line**: **confidence is the one target with a genuine, verified win** —
`confidence_svm` beats its naive baseline both under repeated cross-validation (ordinal distance
0.468 vs. baseline 0.527) and on a completely untouched held-out split (0.263 vs. 0.368). Decision
comes close but hasn't cleared its baseline in any experiment (best: 0.733 vs. baseline 0.762,
CV). Weights has not worked under any approach tried. Reveal-sequence consistently beats baseline
across every experiment (~0.83–0.85 vs. 0.783).

A significant data-integrity bug was found and fixed on 2026-08-10, affecting every decision-model
result across all three experiments (see §3) — the corrected numbers throughout this report
reflect that fix.

---

## 2. What Was Done, in Order

1. **`exp_1`** — baseline hybrid-ML pipeline: logistic regression, HistGradientBoosting, and a
   KDM (kernel density matrix) classifier on 47 engineered clinical features.
2. **`exp_2`** — restricted the feature set to the 11 officially-documented Task-1 input
   variables (vs. `exp_1`'s broader engineered set, much of which isn't part of the real
   `structured-prompt.json` schema), added grouped comorbidity flags, added KDM for decision.
   16 fully-crossed conditions (comorbidity treatment × model/scope).
3. **`exp_3`** — broadened the model search to 8 families (SVM, Random Forest, XGBoost, Extra
   Trees, MLP, Gaussian Naive Bayes, kNN, KDM) for decision + confidence, added a 2-component
   MRI-embedding PCA, resolved PSA-family collinearity (kept `psa`+`psad`, dropped `psap`/`psav`).
   19 conditions.
4. **Bugfix (2026-08-10)** — found and corrected a data-integrity bug affecting every decision
   result across all three experiments (§3). All affected `results/*/metrics.json` and
   `reports/summary.md` files were corrected in place.
5. **Held-out verification (2026-08-10)** — built a genuine train/test split (never used for any
   model selection) to sanity-check the leading CV-based findings against unseen data (§5).

---

## 3. The Bugfix (for context — full detail in `exp_1/reports/summary.md`'s erratum)

`target_biopsy_decision` is `NaN` (not "no") for 104 of 195 cases — the same 91 cases carry
*every* ground-truth label across the board, not 195 for decision and 91 for everything else as
originally assumed. The original code computed `y = (df["target_biopsy_decision"] == "yes")` on
the full 195-row merge; pandas' `NaN == "yes"` evaluates to `False`, not an error, so the 104
unlabeled cases were silently coded as `y=0` instead of excluded. This corrupted every decision
F1/ROC-AUC/PR-AUC and the decision naive baseline in `exp_1`–`exp_3`. Confidence, weights, and
reveal-sequence were unaffected (already correctly filtered to N=91 from the start).

Fixed via `train_decision.load_labeled_data()`. Real decision N=91 (not 195), real positive rate
61.5% (not 28.7%), real naive baseline F1=0.762 (not 0.446). This reversed several headline
findings — most notably, `exp_2`'s "first decision win" and `exp_3`'s "Extra Trees wins decision"
conclusions were both artifacts of the bug and no longer hold.

---

## 4. Results by Target — Every Condition Across All Three Experiments

### 4.1 Decision (F1, higher=better) — baseline = 0.762, N=91

| Rank | Condition | Experiment | Model | F1 |
|---|---|---|---|---|
| 1 | `decision_knn` | exp_3 | kNN | 0.733 |
| 2 | `decision_svm` | exp_3 | SVM | 0.730 |
| 3 | `decision_kdm_flags` | exp_2 | KDM | 0.723 |
| 4 | `decision_kdm_count` | exp_2 | KDM | 0.710 |
| 5 | `decision_kdm` | exp_3 | KDM | 0.709 |
| 6 | `decision_hgb_mri_pca` | exp_1 | HistGradientBoosting | 0.705 |
| 7 | `decision_logistic_count` | exp_2 | Logistic Regression | 0.678 |
| 8 | `decision_logistic_flags` | exp_2 | Logistic Regression | 0.665 |
| 8 | `decision_extratrees` | exp_3 | Extra Trees | 0.665 |
| 10 | `decision_logistic_clinical` | exp_1 | Logistic Regression | 0.658 |
| 11 | `decision_hgb_count` | exp_2 | HistGradientBoosting | 0.655 |
| 12 | `decision_hgb_clinical` | exp_1 | HistGradientBoosting | 0.653 |
| 13 | `decision_hgb_flags` | exp_2 | HistGradientBoosting | 0.652 |
| 14 | `decision_xgb` | exp_3 | XGBoost | 0.644 |
| 15 | `decision_logistic_mri_pca` | exp_1 | Logistic Regression | 0.635 |
| 16 | `decision_rf` | exp_3 | Random Forest | 0.618 |
| 17 | `decision_mlp` | exp_3 | MLP | 0.383 |
| 18 | `decision_nb` | exp_3 | Gaussian Naive Bayes | 0.078 |

**No condition beats baseline under CV.** Best gap: −0.029 (kNN). Margin/distance-based methods
(kNN, SVM, KDM) consistently lead; tree ensembles and Naive Bayes lag.

### 4.2 Confidence (ordinal distance, lower=better) — baseline = 0.527, N=91

| Rank | Condition | Experiment | Model | Ordinal distance |
|---|---|---|---|---|
| 1 | **`confidence_svm`** | **exp_3** | **SVM** | **0.468 ✅ beats baseline** |
| 2 | `confidence_kdm` | exp_3 | KDM | 0.530 |
| 3 | `confidence_knn` | exp_3 | kNN | 0.558 |
| 4 | `confidence_kdm` | exp_1 | KDM | 0.564 |
| 5 | `confidence_logistic` | exp_1 | Logistic Regression | 0.576 |
| 6 | `confidence_kdm_count` | exp_2 | KDM | 0.578 |
| 7 | `confidence_kdm_flags` | exp_2 | KDM | 0.598 |
| 8 | `confidence_logistic_flags` | exp_2 | Logistic Regression | 0.657 |
| 9 | `confidence_logistic_count` | exp_2 | Logistic Regression | 0.673 |
| 10 | `confidence_xgb` | exp_3 | XGBoost | 0.678 |
| 11 | `confidence_rf` | exp_3 | Random Forest | 0.748 |
| 12 | `confidence_extratrees` | exp_3 | Extra Trees | 0.778 |
| 13 | `confidence_nb` | exp_3 | Gaussian Naive Bayes | 0.831 |
| 14 | `confidence_mlp` | exp_3 | MLP | 0.856 |

**`confidence_svm` is the only baseline-beating result across all three experiments and every
target.** Caveat (see §5.2): it never predicts the "uncertain" class at all — the win is real on
the aggregate ordinal metric but doesn't mean the model identifies genuine uncertainty.

### 4.3 Variable-weights (mean ordinal error, lower=better) — baseline ≈ 0.40–0.41

| Rank | Condition | Experiment | Model | Ordinal error |
|---|---|---|---|---|
| 1 | `weights_logistic` | exp_1 | OvR Logistic (47-col) | 0.574 |
| 2 | `weights_official_flags` | exp_2 | OvR Logistic | 0.585 |
| 3 | `weights_official_count` | exp_2 | OvR Logistic | 0.600 |
| 4 | `weights_official` | exp_3 | OvR Logistic | 0.614 |
| 5 | `weights_restricted_count` | exp_2 | OvR Logistic (per-factor) | 0.711 |
| 6 | `weights_restricted_flags` | exp_2 | OvR Logistic (per-factor) | 0.720 |
| 6 | `weights_restricted` | exp_3 | OvR Logistic (per-factor) | 0.720 |

**No condition beats baseline in any experiment.** Per-factor (restricted) feature scope is
consistently worse than full-frame (official) scope — a well-replicated negative finding.

### 4.4 Reveal-sequence (set precision, higher=better) — baseline = 0.783

| Rank | Condition | Experiment | Model | Set precision |
|---|---|---|---|---|
| 1 | `reveal_flags` | exp_2 | MultiOutput OvR Logistic | 0.853 |
| 2 | `reveal_count` | exp_2 | MultiOutput OvR Logistic | 0.852 |
| 3 | `reveal_logistic` | exp_1 | MultiOutput OvR Logistic | 0.840 |
| 4 | `reveal` | exp_3 | MultiOutput OvR Logistic | 0.833 |

**All four conditions beat baseline** — the one target where every attempt has worked.

---

## 5. Held-Out Test Verification (2026-08-10)

Everything in §4 is repeated cross-validation on the training set — no split had ever been held
out and left untouched until this step. `experiments/exp_3/scripts/holdout_eval.py` carves out
~20% of the 91 labeled cases (n_test=19, stratified by decision, seed=0), fits MRI-PCA and all
preprocessing on the training portion only (zero leakage into the test rows), refits the leading
models from §4.1–4.2 on the remaining 72 cases, and scores once on the untouched 19.

### 5.1 Results

**Decision** — local baseline ("always yes", train-majority) F1 = 0.774:

| Model | F1 | ROC-AUC |
|---|---|---|
| `svm` | **0.828** | 0.869 |
| `knn` | 0.786 | 0.762 |
| `kdm` | 0.786 | — |

**Confidence** — local baseline ("always clear", train-majority) ordinal distance = 0.368:

| Model | Ordinal distance |
|---|---|
| `svm` | **0.263** |
| `kdm` | 0.368 (ties baseline) |

`confidence_svm` beating baseline **replicates on a completely untouched split** — real
reassurance the CV finding isn't fold-selection luck. Decision models edging past baseline here
(unlike the aggregate CV picture) is intriguing but shouldn't be over-read given n=19 — each case
is worth ~5% of F1 at this size.

### 5.2 Full Classification Reports

**`decision_svm`** (F1=0.828, ROC-AUC=0.869):
```
              precision    recall  f1-score   support

          no      1.000     0.286     0.444         7
         yes      0.706     1.000     0.828        12

    accuracy                          0.737        19
   macro avg      0.853     0.643     0.636        19
weighted avg      0.814     0.737     0.686        19
```

**`decision_knn`** (F1=0.786, ROC-AUC=0.762):
```
              precision    recall  f1-score   support

          no      0.667     0.286     0.400         7
         yes      0.688     0.917     0.786        12

    accuracy                          0.684        19
   macro avg      0.677     0.601     0.593        19
weighted avg      0.680     0.684     0.644        19
```

**`decision_kdm`** (F1=0.786) — identical confusion matrix to kNN on this split (coincidence at
this N, not general equivalence — their CV numbers across 5 folds × 10 repeats were close but not
identical: 0.733 vs. 0.709):
```
              precision    recall  f1-score   support

          no      0.667     0.286     0.400         7
         yes      0.688     0.917     0.786        12

    accuracy                          0.684        19
   macro avg      0.677     0.601     0.593        19
weighted avg      0.680     0.684     0.644        19
```

**`confidence_svm`** (ordinal_distance=0.263):
```
              precision    recall  f1-score   support

   uncertain      0.000     0.000     0.000         2
  borderline      0.500     0.333     0.400         3
       clear      0.824     1.000     0.903        14

    accuracy                          0.789        19
   macro avg      0.441     0.444     0.434        19
weighted avg      0.686     0.789     0.729        19
```

**`confidence_kdm`** (ordinal_distance=0.368):
```
              precision    recall  f1-score   support

   uncertain      0.000     0.000     0.000         2
  borderline      0.500     1.000     0.667         3
       clear      0.917     0.786     0.846        14

    accuracy                          0.737        19
   macro avg      0.472     0.595     0.504        19
weighted avg      0.754     0.737     0.729        19
```

**Consistent caveat across both confidence models and both evaluation methods (CV and held-out):
neither ever correctly identifies the "uncertain" class** (0% recall in every report above, CV
and held-out alike). Both models' apparent wins come from shifting predictions between
"borderline" and "clear" more accurately, not from detecting genuine uncertainty. Worth keeping
in mind for any downstream use where flagging truly uncertain cases matters.

---

## 6. Key Findings

1. **Confidence prediction is the one genuinely working target** — `confidence_svm` beats
   baseline under both CV and a held-out split. It never predicts "uncertain" though; the win is
   about ranking clear-vs-borderline better, not detecting uncertainty per se.
2. **Decision is close but not solved** — best CV result (0.733) sits 0.029 below baseline
   (0.762); the held-out split showed several models edging past their *local* baseline, but at
   n=19 that's not strong evidence either way.
3. **Weights has not worked under any approach tried** across three experiments — feature scope,
   comorbidity treatment, and model family all failed to close the gap to baseline. Per-factor
   restriction actively makes it worse. This looks like a genuine ceiling for the current
   feature set, not an unexplored corner.
4. **Reveal-sequence works reliably** regardless of what else changes — the most robust finding
   in the project.
5. **Margin/distance-based methods (SVM, kNN, KDM) outperform tree ensembles and MLP** on both
   decision and confidence — a real, model-family-level pattern, not the "opposite pattern per
   target" originally (and incorrectly) reported before the bugfix.
6. **Decision and confidence models are currently fully independent** (§ discussed in chat, not
   yet tested) — no shared training, no cross-feeding of predictions. A chained design (decision
   model's predicted probability as a confidence-model input) is an untested idea.
7. **The bugfix itself is a cautionary finding**: a silent `NaN == "yes" → False` coercion
   propagated through two full experiments before being caught by a direct question about sample
   sizes — worth building a data-integrity assertion into the shared loading code going forward.

---

## 7. Recommendations / Next Steps

1. **Confidence is the strongest candidate to resume the paused steps 5-8** (rubric scorer,
   MCP+LLM wiring, Docker packaging, platform submission) — it's the only target with a
   verified, held-out-confirmed win.
2. **Decision deserves one more focused push** before concluding tabular ML has hit its ceiling —
   e.g. ensembling the top 2-3 models (kNN/SVM/KDM), or a modest hyperparameter search on those
   three specifically, given how close it already is to baseline.
3. **Weights and the "uncertain" confidence class both point toward the free-text/LLM-agent
   alternative** (reading the `txt_*` narrative columns directly, or prototyping the organizers'
   full LLM-agent baseline) — structured tabular features alone haven't captured whatever signal
   distinguishes truly uncertain/hard-to-weight cases.
4. **Try the decision→confidence chained-model idea** (§6.6) — cheap to test, currently unexplored.
5. **Add a data-integrity check** (assert expected N / non-null label counts at load time) to
   `src/chimera_task1/` — the concrete lesson from the bugfix.

---

## 8. Where to Find More Detail

- Full per-experiment design/results/write-up: `experiments/exp_{1,2,3}/{DESIGN,IMPLEMENTATION}.md`,
  `results/*/metrics.json`, `reports/summary.md`
- Bugfix erratum (full root-cause detail): `experiments/exp_1/reports/summary.md`
- Held-out evaluation script (reproducible): `experiments/exp_3/scripts/holdout_eval.py`
- Live experiment registry: `experiments/INDEX.md`
- Cross-session project memory: `chimera-task1-paused` memory entry
