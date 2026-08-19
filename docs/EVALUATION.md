# EVALUATION.md — CHIMERA Task 1: Experimental Protocol, Metrics & Reproducibility

**Project**: pathology-reasoning
**Task**: Task 1 — Prostate Biopsy Decision (subtasks 1.1, 1.2, 1.3, 1.4)
**Status**: Frozen protocol — Draft
**Last updated**: 2026-08-18

---

This document defines the **frozen experimental protocol** used to design, select, and
benchmark Task 1 models and agent pipelines. It specifies:

1. **Cohorts and data splits** — the only files and partitions used.
2. **Per-subtask metrics** — exact formulas and the CHIMERA evaluator they correspond to.
3. **Aggregation & selection rules** — how MCCV and LOO results are computed and
   compared, and why LOO is a sanity check, not a selection signal.
4. **Output schema checks** — what "a valid Task 1 submission" means locally.
5. **Reproducibility requirements** — the checklist every experiment must satisfy.

The Challenge ranking (`ranking_score`) is computed externally by the Grand Challenge
evaluator; this document describes the **internal** protocol that produces the model to
submit.

---

## 1. Cohorts and Files Used

| Role | Set | Case ids | Ground truth | Used for |
|------|-----|----------|--------------|----------|
| Training / Selection | `usable_labeled` | 88 | Yes | MCCV search, LOO sanity check, final fit |
| Challenge evaluation | `unlabeled_test` + `excluded_missing_mri` | ~104 | No (internal) | Challenge submission only |
| Excluded | `excluded_missing_mri` (3 w/ GT, 2 w/o GT), `excluded_missing_pirads` | 5 | — | Not modeled |

- **Cohort source**: `experiments/exp_2/DESIGN.md` §2 — `usable_labeled` is the
  experimental cohort.
- **Split source**:
  `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (**frozen**, do not regenerate).
- **Feature matrix**: `data/chimera26/preprocessed/task1/inputs.csv` (195 × 1077).
- **Target matrix**: `data/chimera26/preprocessed/task1/ground_truth.csv` (195 × 27).
- The Challenge uses its own hidden ground-truth files; the evaluator is run externally.

### 1.1 The frozen split file

Columns:

- `case_id`
- `cohort_status`: `usable_labeled` / `unlabeled_test` / `excluded_missing_mri` /
  `excluded_missing_pirads`
- `has_gt`, `has_mri`, `has_pirads` (binary flags)
- `loocv_fold`: 0..87 for `usable_labeled`; -1 for excluded cases
- `mccv_split_00` .. `mccv_split_49`: 0 = train, 1 = validation (on the 88 cases only)

Rules:

- Only rows with `cohort_status == usable_labeled` enter MCCV or LOO.
- MCCV train/validation assignments come from columns `mccv_split_00..49`.
- LOO fold membership comes from `loocv_fold`.
- The order of rows in this file defines the case order for all reporting; do not reshuffle.

### 1.2 Internal vs. Challenge

- **Internal** (this document): MCCV + LOO over the 88 labeled cases.
- **Challenge**: unseen set (~104 cases, no internal labels). The external
  CHIMERA evaluator computes `ranking_score`.

## 2. Task 1 Decomposition

The Task 1 submission consists of four coupled outputs:

| Subtask | Output field | Type | Notes |
|---|---|---|---|
| 1.1 Decision | `biopsy_decision` | `yes` / `no` | binary biopsy indication |
| 1.2 Confidence | `confidence` | `clear` / `borderline` / `uncertain` | ordinal |
| 1.3 Clinical relevance | `variable_weights` | 10 variables, `not_used`/`noted`/`important`/`decisive` | multi-output ordinal |
| 1.4 Reasoning | `free_text`, `reveal_sequence` | text + section list | agent-produced |

The Challenge evaluator enforces a **decision gate**: an incorrect 1.1 sets `case_score = 0`
and stops further scoring. Subtasks 1.2–1.4 can still be selected/benchmarked independently,
but in the end-to-end submission a wrong 1.1 zeroes the case.

## 3. Per-subtask metrics

### 3.1 Subtask 1.1 — Biopsy decision

Primary (use for model selection within each experiment):

- **F1_yes** — F1 for the positive class `yes`.
- **Macro-F1** — unweighted mean of `yes`/`no` F1; reported as a balance guardrail.

Secondary:

- `decision_accuracy`
- `balanced_accuracy`
- `sensitivity` (recall of `yes`), `specificity` (recall of `no`)
- `precision` of `yes`
- `MCC` (Matthews correlation coefficient)
- `PR-AUC` (area under precision-recall curve; preferred over ROC-AUC under imbalance)
- `ROC-AUC` (one-vs-rest, positive class)
- `Brier score` (binary; requires probabilities)
- **Calibration**: reliability curve + `ECE` (expected calibration error)
- `classification_report` (precision/recall/F1 per class)
- `confusion_matrix`

Challenge correspondence: `decision_score` (exact gate), `decision_f1_yes`,
`decision_accuracy`, `decision_gate_pass_rate`.

#### Selection rule

Primary: **F1_yes**. Tie-break by Macro-F1, then balanced accuracy.

#### Ordinal / probability note

- If the model outputs probabilities, the 0.5 threshold is the default gate; do not scan
  thresholds on the LOO results to improve F1_yes. Any thresholding decision is made *before*
  looking at LOO performance (e.g., at the MCCV stage on validation folds only).
- ROC-AUC and PR-AUC are ranking metrics (threshold-free).

### 3.2 Subtask 1.2 — Confidence (ordinal, 3 levels)

Code mapping: `uncertain=0`, `borderline=1`, `clear=2`.

#### Canonical protocol (corrected 2026-08-18)

The original protocol used "normalized ordinal distance" (OD), QWK, and several secondary
metrics that produced misleading rankings (e.g., regression trees with lowest OD but class
collapse). The canonical protocol below supersedes all prior subtask 1.2 metric definitions.

**Primary metric (selection):**

- **MOE_abs** (Mean Ordinal Error, absolute) = `(1/3) * Σ_c mean_{i:y_i=c}(|pred_i - true_i|/2)`.
  Range [0, 1]; lower = better. Class-balanced by construction: each class contributes
  equally regardless of prevalence.

**Tiebreak (when MOE_abs is identical across models):**

- **F1_macro** — unweighted mean of per-class F1.

**Required diagnostics (never for selection):**

- Confusion matrix (3×3, absolute + row-normalized), order `[uncertain, borderline, clear]`.
- Per-class recall: `rec_U`, `rec_B`, `rec_C`.
- Validity rate (structural + output validity).
- Sanity check: `MOE_abs` over the full "always clear" baseline with real MCCV labels
  (exactly 900 predictions) as lower bound reference.

**Selection cascade (in order):**

1. Validity = 100%.
2. `MOE_abs < MOE_abs_baseline` (baseline = always clear, N=900).
3. No zero recall (all three classes must have rec > 0).
4. Minimize `MOE_abs`.
5. Tiebreak: maximize `F1_macro`.

**Baseline (always clear):**

- Predicts `clear` for all cases.
- Evaluated with real MCCV labels (900 pooled predictions, not 88).
- Typical: `MOE_abs = 0.5000`, `F1_macro = 0.2626`.

**Baseline reasoning (from `prediction.json`):**

- The Task 1 agent's own confidence predictions, evaluated on the 88 `usable_labeled`
  cases (LOO).
- Typical: `MOE_abs ≈ 0.4788`, `F1_macro ≈ 0.2962`.
- Used as a reference ceiling for sanity checking only.

**Removed metrics (no longer used for subtask 1.2):**

- `accuracy`, `balanced_accuracy`, QWK, Brier, ECE, AUC, Spearman, MAE, OE,
  per-class precision/recall/F1, error_unclear, error_reverse, bootstrap CIs,
  feature usage, monotonicity.

Challenge correspondence: `confidence_score` (per-case; maps to normalized agreement),
`confidence_weighted_kappa`.

### 3.3 Subtask 1.3 — Clinical relevance (10 variables, ordinal 4 levels)

Code mapping: `not_used=0`, `noted=1`, `important=2`, `decisive=3`.

Each of the **10 variables** is scored independently:

- `age`, `fh`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`

Primary (computed **per variable**, then macro-averaged across the 10):

- **Normalized ordinal distance** = `1 - |pred_code - true_code| / 3` (in [0, 1]).
- `quadratic_weighted_kappa` per variable (macro-averaged across 10).

Secondary (per variable):

- `accuracy`, `Macro-F1`
- `Brier score` (multiclass per variable)
- `ROC-AUC` (multiclass, one-vs-rest per variable)
- `MAE` per variable (ordinal, in code units)
- `classification_report` per variable
- `confusion_matrix` per variable (4×4)

In addition to per-variable metrics, report:

- **Important/decisive set F1**: set F1 over variables where `weight ∈ {important, decisive}`
  (matches the Challenge `important_decisive_factor_score`).
- Macro-average of `ROC-AUC`, `Brier`, `QWK`, `MAE` across the 10 variables.

Challenge correspondence: `variable_weight_score` (normalized ordinal agreement),
`important_decisive_factor_score` (set F1), `variable_weight_weighted_kappa` (cohort-wide
QWK over all variable-weight pairs).

#### Selection rule

Primary: **normalized ordinal distance, macro-averaged across 10 variables**.
Tie-break by important/decisive set F1; then macro-QWK.

#### Rare-class handling

- For variables that are predominantly `not_used`/`noted` in a given CV fold, report the
  number of valid folds (splits where ≥2 classes are present) for QWK / ROC-AUC. Metrics
  that are undefined on a fold are reported as `NaN` and excluded from the mean, with the
  count of valid folds shown.

### 3.4 Subtask 1.4 — Reasoning (agent text + reveal sequence)

This is evaluated **end-to-end** through the agent, not as isolated text metrics.

Primary:

- **Challenge-aligned deterministic `mean_case_score`** (recomputed locally to mirror the
  evaluator with `USE_RATIONALE_JUDGE=0`), aggregated with the 0.225/0.275/0.175/0.150/0.175
  weighting (confidence / var_weight / factor_f1 / tool / section_grounding).
  - Only the **deterministic** part is used for selection (the LLM rationale judge is
    enabled only for the final shortlist, to avoid variable-cost, non-deterministic search).
- **Valid output rate** — fraction of cases producing a fully schema-valid Task 1 submission
  (see §5).
- **Decision/structure consistency** — agreement of `free_text` with the predicted
  `1.1` decision, `1.2` confidence, and `1.3` weights (diagnosed, not selected on).

Secondary (diagnostic; do not drive selection):

- `section_grounding_score` (challenge equivalent)
- `mean_tool_score` / tool-efficiency precision on `reveal_sequence`
- Coverage of evidence: fraction of `important`/`decisive` variables substantiated in
  `free_text`.
- Contradiction rate: fraction of `free_text` claims inconsistent with clinical inputs.
- Precision / recall / F1 / Jaccard of `reveal_sequence` (exact-set match to pathologist).
- Neighbor retrieval quality: `Recall@k`, `MRR`, `nDCG@k` (if neighbor retrieval is used),
  plus end-to-end utility (Δ in `mean_case_score` vs. no-neighbor baseline).
- Latency and cost (generation tokens), for practical comparison.

Challenge correspondence: `rationale_score`, `mean_section_grounding_score`,
`mean_tool_score`, `mean_case_score`, `mean_case_score_among_gate_passed`,
`ranking_score` (full, with judge, at the end).

> Text generation metrics (e.g. BLEU/ROUGE/BERTScore) are reported only as diagnostics;
> they are **not** used for selection because clinical correctness, grounding, and
> consistency are the objectives, not surface textual similarity to expert text.

## 4. Aggregation & Selection Rules

### 4.1 MCCV computation (primary for selection)

For each of the 50 MCCV splits:

1. Train on the 80%-train subset (fixed by the `mccv_split_*` columns).
2. Predict the 20%-validation subset.
3. Score 1.1, 1.2 (where valid), 1.3 (per variable), and 1.4 (agent, deterministic only).
4. The Challenge-equivalent `mean_case_score`, `decision_f1_yes`, and derived
   `ranking_score` (deterministic, no judge) are computed over the validation subset.

Per-split values are stored. The experiment-level report shows, for each metric:

- mean, std, min, max, n_valid_splits

### 4.2 How metrics are aggregated per experiment

One report per **experiment** summarizing all candidate configurations attempted within it:

| Item | What |
|---|---|
| Per-config MCCV summary | Mean ± std for 1.1, 1.2, 1.3 (macro), and 1.4 (deterministic mean_case_score), plus derived challenge-equivalent `ranking_score`. |
| Selected hyperparameters | Best config per selection rule **within the experiment** (§3). |
| LOO summary | Metrics for the selected config only (sanity check). |
| Config log | All hyperparameter sets tried and their MCCV scores. |

### 4.3 Selection rule per experiment

- 1.1 primary: **F1_yes**. Tiebreak by macro-F1, then balanced accuracy.
- 1.2 primary: **MOE_abs** (class-balanced ordinal error). Tiebreak by F1_macro.
  See §3.2 for full cascade.
- 1.3 primary: **normalized ordinal distance, macro-averaged across 10 variables**.
  Tiebreak by important/decisive set F1; then macro-QWK.
- 1.4 primary: deterministic challenge-equivalent `mean_case_score`, and
  challenge-equivalent `ranking_score` (deterministic).
- Final model selection **across experiments**: the experiment whose best configuration
  has the highest MCCV aggregate on its primary metric(s).

### 4.4 LOO computation (sanity check only)

After the selected config is fixed:

1. Run the 88 LOO folds (fixed by `loocv_fold`).
2. In each fold: train on the other 87, predict the held-out sample, using the
   **frozen** selected hyperparameters.
3. Report the same metric suite as MCCV.
4. The LOO result is a sanity check that the MCCV-selected model behaves stably out-of-fold.

### 4.5 LOO must not influence selection

- LOO is computed **once per experiment**, **only** for the MCCV-selected config.
- LOO is **never** used to choose hyperparameters, tune thresholds, or re-rank experiments.
- A LOO/MCCV gap is reported as a diagnostic flag (possible over-selection), not as a
  reason to change the model.
- Final model selection across experiments uses **MCCV only**.

### 4.6 Challenge-equivalent global metric

For end-to-end 1.4 comparisons, the internal deterministic proxy of the leaderboard metric is:

```text
ranking_score_deterministic = (mean_case_score_deterministic + decision_f1_yes) / 2
```

where `mean_case_score_deterministic` uses the no-judge weights (0.225/0.275/0.175/0.150/0.175).
The true Challenge `ranking_score` (with the rationale judge) is computed externally by Grand
Challenge and is **not a selection input** here.

## 5. Output schema validation (Challenge compliance)

A locally produced Task 1 prediction is valid iff it matches the evaluator's `validate_record`
(`docs/chimera-agent-evaluation/evaluation/evaluate.py`):

- It is a JSON object.
- `biopsy_decision` ∈ {`yes`, `no`}.
- `variable_weights` is an object (may omit the judge field `free_text`; the deterministic
  path zeroes missing components per the Challenge weights).
- `confidence`, `reveal_sequence`, `free_text` and `clinical_data` are accepted as-is.

Per-case `case_score` and the dataset aggregates (`decision_f1_yes`,
`mean_case_score`, `mean_case_score_among_gate_passed`,
`mean_section_grounding_score`, `mean_tool_score`) are recomputed locally using the logic
in `evaluate.py` (deterministic mode) so internal 1.4 numbers track the evaluator.

## 6. Experiment artifacts and file layout

Each experiment should produce:

```
experiments/exp_N/
├── DESIGN.md                 # experiment design (this protocol is referenced, not duplicated)
├── IMPLEMENTATION.md         # build plan for the experiment
├── results/
│   ├── <candidate_config>/
│   │   ├── metrics_mccv.json         # per-fold + mean/std for 1.1/1.2/1.3/1.4
│   │   ├── metrics_loo.json          # selected config only (sanity check)
│   │   ├── hyperparameters.json      # the frozen selected hyperparameters
│   │   ├── oof_predictions.csv       # out-of-fold predictions (MCCV + LOO) per subtask
│   │   ├── config_log.json           # all configs tried + their MCCV scores
│   │   └── validation_report.json    # schema-validity + gate pass rate
│   └── challenge_equivalence.json    # deterministic ranking_score proxy (selected config)
└── reports/
    └── summary.md           # comparison table across configs (MCCV-driven)
```

## 7. Reproducibility requirements (frozen + per-run)

The following are **frozen for all experiments** in this campaign and must not change:

- [x] Split file: `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (do not regenerate).
- [x] Cohort: `usable_labeled` (88 cases) for MCCV and LOO; challenge set is held out.
- [x] Case row order in the split file defines reporting order.
- [x] MCCV train/validation assignments from columns `mccv_split_00..49`.
- [x] LOO fold assignments from `loocv_fold`.

The following are **per-experiment choices** (vary between experiments, fixed within one):

- [ ] Random seed for model-internal stochasticity (recorded).
- [ ] Hyperparameter set (recorded in `hyperparameters.json`).
- [ ] Input modality / feature group (clinical / text / MRI / multimodal).
- [ ] Model family / architecture.
- [ ] Imputation / encoding / scaling (fit on train fold only — no global stats).

Per-run hygiene (record before launching each candidate):

- [ ] Git commit hash: `git log -1 --format="%H %s" > results/<config>/git_commit.txt`
- [ ] Working tree was clean (commit or stash changes first).
- [ ] Environment pinned (`docs/chimera-agent-baseline/requirements.txt` / lockfile).
- [ ] Split file hash recorded alongside results.

## 8. Known edge cases and conventions

- **Classes absent from a fold**: ROC-AUC, PR-AUC, and QWK/kappa are undefined on a fold
  where only one class is present; report `NaN` for that fold and show `n_valid_splits`.
- **1.3 per-variable**: metrics are computed per variable, then macro-averaged; never combined
  into a single 10-variable confusion matrix.
- **Thresholding**: any decision threshold chosen on validation folds (MCCV stage) is frozen
  before LOO and challenge; do not tune thresholds on LOO or challenge results.
- **Neighbor index**: built from training cases only within each fold; the held-out case and
  its reasoning are never retrievable for it.
- **1.4 determinism**: selection uses the deterministic (no-judge) evaluator path; the LLM
  rationale judge is reserved for final shortlist reporting.
- **Calibration**: reported on validation subsets (MCCV); not tuned to LOO or challenge.
