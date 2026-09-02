# exp_17 Design: Feature-Subset Sweep, All 3 Sub-Tasks, Evaluator-Vocabulary Floor

## Background

Requested directly: test different input subsets for task2, with an explicit minimum-required
floor. Clarified via two questions: (1) scope = all 3 sub-tasks, not `confidence` alone; (2) floor
= the evaluator's 11-variable scored vocabulary (`task2_schema.TASK2_VARIABLES`, confirmed from
`section_variable_mapping.json`): `age, psa, psad, pirads, cspca, ct, bx_isup, bx_gl_prim,
bx_gl_sec, comorbidity, fh`. One gap closed first: `fh`/`family_history` had never been extracted
as a usable numeric feature (only consumed for tool-gating logic) — added
`parse_family_history()` to `scripts/extract_features_task2.py`, same Yes/No/Unknown -> 1.0/0.0/None
convention task3's own `family_history_present` uses (94.8% coverage).

## Method

Floor (`FLOOR_11`, always included, 11 features) swept against the two feature groups already
extracted in exp_16 (`psa_kinetics`: 2 features; `labs`: 7 features) — independently per sub-task,
each held at its own currently-promoted model (`decision`/`confidence`=`knn`,
`variable_weights`=`mord.LogisticIT`). 4 variants × 3 sub-tasks = 12 LOOCV runs.

Note on scope: predictive features are drawn from `engineered_features.csv` regardless of whether
the corresponding tool is in `required_tools` — the same architecture task1's `confidence` config
already uses (`psa_trend`/`lab_results`-derived features consumed without necessarily being
"revealed" in every variant). `required_tools`/`reveal_sequence` is a separate, independently-swept
axis (exp_11) governing whether `variable_weights` may claim high importance for a gated variable
without an ungrounded-variable penalty — not a gate on what the underlying classifier may use.

## What would change the promoted config

Only a feature-set variant that beats the current promoted baseline for that sub-task (which now
includes `family_history_present` in the floor for the first time — itself untested until this
experiment) on `ranking_score`, bootstrap-confirmed before promotion.
