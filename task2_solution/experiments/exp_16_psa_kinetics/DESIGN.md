# exp_16 Design: PSA-Kinetics + Lab-Panel Features for `decision`

## Background

Re-verifying the challenge's own metric formula (`Task 2 score = (Mean case score + Weighted F1)/2`,
confirmed word-for-word from the official evaluation-repositories page, matching what was already
derived from `evaluate.py` source) surfaced the highest-leverage lever untouched so far:
`decision_score` is both the gate for `mean_case_score` (58/72 currently pass) and half of
`ranking_score` directly via `decision_weighted_f1` — improving it pays twice, unlike anything done
to `confidence`/`variable_weights` in exp_13–15. exp_11/12 never varied `decision`'s *feature set*,
only its model. `psa_trend` and `laboratory_results` (flagged as unused since the original EDA) are
genuinely new signal — confirmed `psa` == `psa_trend`'s last reading == the lab panel's own PSA
entry (same value everywhere), so only the trajectory (velocity) and the *other* lab values (free
PSA, %free PSA — real clinical biomarkers for risk stratification) add anything.

## Hypothesis

Adding PSA-velocity (`psa_velocity_per_year`, `psa_ratio_last_first`) and/or the lab panel (free
PSA, %free PSA, testosterone, creatinine, alkaline phosphatase, LDH, reduced-eGFR flag) to
`decision`'s feature set improves `ranking_score` over the current 10-feature baseline. Stated with
real optimism this time, unlike exp_10's skeptical prior for task3's text features — %free PSA in
particular is a well-established real clinical decision factor (lower %free PSA associated with
higher-grade disease), not a speculative addition.

## Method

New extraction: `scripts/extract_features_task2.py` extended with `parse_psa_velocity()` (from
`psa_trend`, date-aware, years-elapsed denominator) and `parse_labs()` (from `laboratory_results`,
100% coverage on every field used; `HbA1c` excluded, only 36/153 covered, diabetics-only). Swept
independently — psa-kinetics alone, labs alone, both combined — against the 10-feature baseline,
`decision`'s model held at the current promoted default (`knn`), same isolation discipline as
task3's exp_10. A follow-up will re-check the winning feature set against `random_forest` (exp_12's
strongest, if unconfirmed, model candidate) if the feature addition itself proves out.

## What would change the promoted config

Only a feature-set variant that beats the baseline on `ranking_score`, bootstrap-confirmed before
promotion — same bar as every other experiment.
