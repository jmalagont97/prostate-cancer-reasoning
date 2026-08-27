# Experiment Design: Task3 — Post-Surgical Pathology Text Features for BCR Survival
**Experiment**: experiments/exp_3/
**Project**: challenge_chimera_v3
**Date**: 2026-08-24
**Status**: Complete

---

## 1. Hypothesis

`data/processed/task3/structured.csv`'s 4 bare fields (age, psa, dre, active_treatment_prior_to_surgery)
give a Cox baseline of mean C-index=0.641 (5-fold, `scripts/fit_baseline_models.py`). But `dre` is not
actually a clean category — it's near-unique free text per case (confirmed in
`notebooks/task3_eda.ipynb` §3), so that 0.641 baseline is itself running on a noisy one-hot encoding
of free text, not a genuinely bare-minimum model. `surgical_pathology_report`'s post-surgical findings
(ISUP grade, pT stage, margins, EPE, SVI, lymph-node status) are established real-world prognostic
factors for biochemical recurrence and were confirmed extractable at 100% coverage via closed-vocabulary
regex (`notebooks/task3_eda.ipynb` §5, ported into `scripts/extract_features_task3.py`).

Two separable questions, kept separate rather than conflated into one "did adding stuff help" number:

- **Q1 (encoding quality, no new information)**: does replacing `dre`'s raw-text one-hot encoding
  with a cleaned parse (dre_abnormal/bilateral/ct_stage_ordinal) change performance, holding the
  *information source* constant?
- **Q2 (new information)**: does adding the genuinely new surgical-pathology findings on top of the
  cleaned structured features improve C-index over what's available pre-surgically?

**Protocol note, inherited from task1's exp_1**: 5-fold CV was shown misleading at task1's n=91 (a
leakage-driven "win" collapsed once properly LOOCV-validated). Task3's n=75 is the same order of
magnitude — LOOCV is the primary protocol here too; the existing 5-fold number is reported only for
direct backward-comparability to the already-published 0.641 baseline, not trusted on its own.

Refuted if the LOOCV C-index for the pathology-augmented condition does not exceed the cleaned
structured-only condition (Q2), or if neither condition beats the as-published 0.641 baseline under
LOOCV.

---

## 2. Conditions

| Condition | Features | Question |
|---|---|---|
| `baseline_aspublished` | age, psa, dre (raw text, one-hot), active_treatment_prior_to_surgery (raw text, one-hot) | sanity check: reproduce the existing 0.641 number under the same 5-fold protocol before trusting anything new |
| `structured_cleaned` | age, psa, dre_abnormal, dre_bilateral, dre_ct_stage_ordinal, active_treatment_flag | Q1 |
| `structured_cleaned_plus_pathology` | `structured_cleaned` + isup_grade_group, pt_stage_ordinal, epe_present, margins_positive, svi_present, lvi_present, ln_assessed, ln_positive | Q2 |

All engineered columns from `scripts/extract_features_task3.py` (validated 100% coverage on the
pathology fields, 64–69% on DRE fields — DRE not performed is real missingness, imputed not dropped).

---

## 3. Metric

Harrell's C-index (`lifelines.utils.concordance_index`), the same metric `evaluate.py` uses as task3's
`ranking_score` directly (task3 has no decision gate / tool_score / section_grounding_score — confirmed
earlier this project — C-index alone is the real target metric here, not a proxy).

- **LOOCV** (primary): 75-fold leave-one-out refit; per-case partial-hazard risk pooled across all
  folds, then one global `concordance_index(months, -risk, event)` over all 75 cases.
- **5-fold** (secondary, for continuity with the published 0.641): existing `splits.json` folds,
  mean C-index across folds — same protocol `fit_baseline_models.py` used.

## 4. Promotion rule

Only replace the deployed task3 baseline (`scripts/fit_baseline_models.py`'s `FEATURES["task3"]` +
any downstream agent wiring) if `structured_cleaned_plus_pathology`'s LOOCV C-index measurably beats
`baseline_aspublished`'s LOOCV C-index — the same discipline used throughout task1 (bx-clamp/labs-reveal
were tested and reverted when they didn't win; the confidence feature-set swap was promoted because it
did). No promotion on 5-fold alone, given the established unreliability of 5-fold at this n.
