# exp_10 Design: The Remaining Gated Text Sections (radiology, prior notes, family history)

## Background

Task 3 gates five free-text sections behind tool calls in the reference agent, but nothing in the
evaluator scores tool usage directly — only the final prediction. Across exp_3 through exp_9, only
`surgical_pathology_report` was ever turned into features (exp_3). `radiology_report`,
`previous_notes`, and `family_history` were named, explicitly, as an open limitation in every
downstream document (MODEL_CARD.md §7, README.md's honesty notes, `results/summary.json`'s
`known_limitations`) — untried, not refuted.

## Corpus scan (not assumed — confirmed)

All three fields are already loaded into `data/processed/task3/clinical_notes.jsonl` (built earlier,
never consumed). A direct scan of all 75 raw `...clinical-data.json` files found:

- **`radiology_report`**: 75/75 present, one closed template (prostate volume, PSA density,
  PI-RADS, an AI-model cSPCa probability). 74/75 match exactly; 1 case reports two PI-RADS scores
  for two lesions — resolved by taking the max (the standard clinical convention for multifocal
  disease).
- **`previous_notes`**: 30/75 missing entirely (no note recorded — real missingness). Of the 45
  present, 44 follow a closed template (5-ARI status, an optional comorbidity clause, an optional
  Charlson Comorbidity Index); 1 outlier row (smoking-cessation medication, not 5-ARI) still parses
  for the comorbidity sub-pattern.
- **`family_history`**: 4 closed templates cover all 75 cases — "there is/isn't a family history",
  "unknown", or raw-missing (kept alongside "unknown" as the same not-usable-signal case, per this
  project's standing missingness convention).

## New features (added to `scripts/extract_features_task3.py`, additive — existing columns
untouched, `engineered_features.csv` regenerated with 8 new columns)

| Group | Feature | Coverage |
|---|---|---|
| Radiology | `prostate_volume_cc`, `psa_density`, `pirads`, `ai_prob_cspca` | 75/75 (100%) |
| Prior notes | `on_5ari`, `has_comorbidities`, `cci` | 44/75, 30/75, 19/75 |
| Family history | `family_history_present` | 35/75 (46.7%) |

## Hypothesis

Stated skeptically, matching this project's track record: naive feature expansion has repeatedly
hurt elsewhere (task1's exp_2, and 3 of exp_5's 4 embedding-fusion variants). At n=75 with only 19
events, adding up to 8 more features to an already-18-feature model risks overfitting more than it
adds real signal — especially `psa_density`, which is arithmetically related to the already-included
`psa` and could be redundant rather than additive.

## Method

Reuse the exact promoted pipeline (`_CoxEmbeddingAdapter`, penalizer=0.3, prostatectomy-slide
embedding PCA k=5, percentile p=0.90) — only the feature set varies, isolating its effect the same
way exp_5 isolated the embedding-source effect. Test each new group individually plus all three
combined, against the 18-feature baseline, all under full LOOCV. Bootstrap (2,000 resamples) any
variant that beats the baseline on the point estimate before calling it a win.

## What would change the promoted config

Only a feature-set variant that beats the baseline by a bootstrap-significant margin gets promoted
into `task3_config.NUMERIC_FEATURES` and the production/LOOCV/simulation scripts. Anything else is
reported honestly as tried-and-not-promoted, same discipline as every prior experiment.
