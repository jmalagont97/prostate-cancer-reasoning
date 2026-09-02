# exp_11 Design: Task2 First Baseline (Agent Built + Tool-Policy Swept)

## Background

Task 2 (MRI + biopsy risk stratification) had no agent, no scored baseline, and no verified output
contract before this experiment. `notebooks/task2_eda.ipynb` established the real facts: 153 cases
released, only 72 labeled; 4-class `treatment_recommendation` with a severe imbalance
(`watchful_waiting` 2/72); output/scoring contract confirmed from `external/CHIMERA-agent/
evaluation/evaluate.py` to be structurally identical to task1's (two files: a bare decision string +
a `{confidence, variable_weights, reveal_sequence, free_text}` reasoning object), scored by the same
`evaluate_case()` machinery task1 uses. `agent/src/section_variable_mapping.json` confirmed the real
per-variable grounding requirements — not assumed from the webpage's prose.

## Hypothesis

Task1's proven architecture (schema/free_text/subtask_config/run_agent_loocv/local_eval, all built
and validated across exp_1/exp_2) should port directly to task2 with three changes: an 11-variable
vocabulary (task1 has 10), a 4-class decision domain (task1 is binary), and a 6-section reveal
vocabulary (task1 has 5, task2 adds `pathology_report` since a biopsy now exists). A first baseline
using the always-available structured variables (no gated-text feature engineering yet) should
produce a real, honest `ranking_score` — and, per task1's own finding, the tool-grounding policy for
`variable_weights` is worth sweeping rather than assuming, since ground-truth `reveal_sequence` is
always empty (confirmed directly from `labels.csv`) and any tool call therefore costs the full
`tool_score` component.

## Method

- `scripts/extract_features_task2.py`: `ct_ordinal` (reusing task3's exact stage-parsing grammar)
  and `comorbidity_count` (from `pmhx`) — the only two engineered columns needed, since task2's
  `structured.csv` already carries the graded numbers (PI-RADS, ISUP, Gleason) directly, unlike
  task3's free-text-only pathology grading.
- Ported task1's architecture file-for-file: `task2_schema.py`, `task2_targets.py`,
  `task2_free_text.py`, `task2_mock_tools.py` (6 tools, 1 new: `get_pathology_report`),
  `task2_subtask_config.py`, `run_task2_agent_loocv.py`, `local_eval_task2.py`,
  `task2_variant_tester.py`.
- Base feature set (all 3 sub-tasks): `age, psa, psad, pirads, bx_isup, bx_gl_prim, bx_gl_sec,
  cspca, ct_ordinal, comorbidity_count` — 10 numeric features, `knn` (task1's proven winner,
  `n_neighbors=7, weights=distance`).
- LOOCV on the 72 labeled cases only (the 81 unlabeled cases have no ground truth to validate
  against).
- Tool-grounding policy for `variable_weights` swept via `task2_variant_tester.py`: zero tools,
  `get_mri_report` only, `get_pathology_report` only, `get_mri_report`+`get_pathology_report`
  (grounds 7/11 variables), and all three plus `get_family_history` (grounds 9/11, everything but
  the structurally-ungroundable `comorbidity`).

## What would change the promoted config

Only a variant that beats the current default on `ranking_score`, checked with a paired bootstrap
before being called significant — same promotion-gated discipline as every task1/task3 experiment.
