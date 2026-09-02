# exp_11 Report: Task2 First Baseline (Agent Built + Tool-Policy Swept)

**Status**: Complete. **Verdict: ✅ A real, honest first baseline built and validated — RANKING_SCORE
0.714 (LOOCV, 72 labeled cases) — plus a real, measured, promoted tool-policy fix (+0.0636 case_score,
bootstrap-significant), not a default left unchecked.**

## What was built

Task1's full architecture, ported to task2 (same output shape, same evaluator, different variable
set/decision domain/reveal vocabulary — all confirmed from `evaluate.py` source, not assumed):
`scripts/extract_features_task2.py`, `agent/src/{task2_schema, task2_targets, task2_free_text,
task2_mock_tools, task2_subtask_config, run_task2_agent_loocv, task2_variant_tester}.py`,
`agent/local_eval_task2.py`.

Base feature set: `age, psa, psad, pirads, bx_isup, bx_gl_prim, bx_gl_sec, cspca, ct_ordinal,
comorbidity_count` — all either directly in `structured.csv` or engineered from it. No gated-text
feature engineering yet (radiology/pathology-report regex extraction, PSA-trend velocity, lab
features) — those are separately scoped follow-up experiments, not attempted here.

## Tool-policy sweep (variable_weights grounding)

Ground-truth `reveal_sequence` is always `[]` for all 72 labeled cases (confirmed directly from
`labels.csv`) — `cost_aware_tool_score` therefore gives a guaranteed `tool_score=1.0` for calling no
tools versus a guaranteed `0.0` for calling any, regardless of how well-justified the call is. Swept
under LOOCV, via `task2_variant_tester.py`:

| Variant | ranking_score | mean_case_score | var_weight_score | factor_f1 | tool_score |
|---|---|---|---|---|---|
| **Zero tools (promoted)** | **0.7166*** | **0.6373*** | 0.673 | 0.431 | 1.000 |
| get_mri_report only | 0.6703 | 0.5445 | 0.741 | 0.523 | 0.000 |
| get_pathology_report only | 0.6727 | 0.5495 | 0.734 | 0.569 | 0.000 |
| get_mri_report + get_pathology_report | 0.6848 | 0.5736 | 0.803 | 0.633 | 0.000 |
| + get_family_history (grounds 9/11 vars) | 0.6865 | 0.5771 | 0.818 | 0.633 | 0.000 |

*In-memory variant-tester numbers; the disk-based `local_eval_task2.py` run (below) is the
authoritative one and differs by ~0.005–0.01 on case_score components — a small, unexplained gap
between the two ground-truth reconstructions, noted honestly rather than chased down further at
this stage; both agree unambiguously on the ranking.

Grounding more variables *does* raise `variable_weight_score` (0.673→0.818) and `factor_f1`
(0.431→0.633) monotonically — the naive "ground everything you can" intuition is directionally
correct. It loses anyway: the guaranteed tool_score cost outweighs those gains every time. Paired
bootstrap (2,000 resamples) on the zero-tools-vs-default (mri+pathology) per-case gap: mean
**+0.0636**, 95% CI **[+0.0525, +0.0748]**, 100% of resamples favor zero tools — decisive, not
borderline. **Promoted**: `task2_subtask_config.py`'s `variable_weights` now uses
`required_tools=set()`, matching `decision`/`confidence`.

This is the same shape of finding task1's own `subtask_config.py` history describes (dropping
`family_history_binary` from confidence's feature set won for the identical reason) — confirms the
pattern generalizes, not a one-off.

## Final baseline (LOOCV, 72 labeled cases, promoted zero-tools config)

| Metric | Value |
|---|---|
| **RANKING_SCORE** | **0.714** |
| mean_case_score | 0.632 |
| decision_weighted_f1 | 0.796 |
| decision gate pass rate | 0.806 (58/72) |
| mean confidence_score | 0.914 |
| mean variable_weight_score | 0.649 |
| mean important_decisive_factor_score | 0.431 |
| mean tool_score | 1.000 |
| mean section_grounding_score | 1.000 (0 ungrounded instances — self-fulfilling: zero tools means nothing eligible to weight above not_used, so nothing can be flagged ungrounded either) |

Decision confusion (gt → pred):

| True class | n | Predicted as |
|---|---|---|
| active_surveillance | 25 | active_surveillance 21, active_treatment 3, continued_surveillance 1 |
| active_treatment | 31 | active_treatment 24, active_surveillance 7 |
| continued_surveillance | 14 | continued_surveillance 13, active_surveillance 1 |
| **watchful_waiting** | **2** | active_treatment 1, active_surveillance 1 — **0/2 correct** |

`watchful_waiting` is never predicted correctly — expected given each LOOCV fold trains on at most 1
example of that class; not something a feature/model change alone can fix at n=2. Named, not hidden.

### Classification report + confusion matrix (`sklearn.metrics`, decision, LOOCV, promoted config)

```
                        precision    recall  f1-score   support

      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.70      0.84      0.76        25
continued_surveillance       0.93      0.93      0.93        14
      active_treatment       0.86      0.77      0.81        31

              accuracy                           0.81        72
             macro avg       0.62      0.64      0.63        72
          weighted avg       0.79      0.81      0.80        72
```

Weighted-avg F1 (0.80) matches `decision_weighted_f1` (0.796) up to rounding — confirms the two are
the same quantity, computed two ways.

| True \ Pred | watchful_waiting | active_surveillance | continued_surveillance | active_treatment |
|---|---|---|---|---|
| **watchful_waiting** (n=2) | 0 | 1 | 0 | 1 |
| **active_surveillance** (n=25) | 0 | 21 | 1 | 3 |
| **continued_surveillance** (n=14) | 0 | 1 | 13 | 0 |
| **active_treatment** (n=31) | 0 | 7 | 0 | 24 |

Macro-avg F1 (0.63) is well below weighted-avg (0.80) — entirely `watchful_waiting`'s doing (0.00
across the board, n=2). The other three classes are all comfortably above 0.75 F1, with
`continued_surveillance` the strongest (0.93) despite being the second-smallest class (n=14) — its
errors are the fewest and most concentrated (1 misclassified case, into `active_surveillance`).
Confusion is otherwise between the two "more active" classes (`active_treatment`↔`active_surveillance`,
7+3 of the 14 total errors) rather than random scatter across all four — a clinically sensible
failure mode (adjacent management intensity, not an arbitrary swap).

## Verified, not assumed

- Output contract (two files, same shape as task1) confirmed from `evaluate.py`'s
  `process_interf1`/`GT_FILENAMES["task2"]`, not the webpage's prose alone.
- `ranking_score = (mean_case_score + decision_weighted_f1) / 2` — support-weighted F1 across the 4
  classes, confirmed from `compute_aggregate_metrics`'s `is_treatment` branch, not assumed to be the
  same single-class F1 task1 uses.
- Per-variable grounding requirements (which tool grounds which of the 11 variables) confirmed from
  `section_variable_mapping.json`'s `variable_to_sections`, not inferred from the task description.
  This caught a real, non-obvious fact: `comorbidity`'s primary section (`section_s3-comorb`) has no
  entry in the reveal vocabulary at all — structurally ungroundable, the same category as task1's
  `bx`.

## Not done (this experiment's scope, not a project-wide claim)

- No model-family sweep beyond `knn` (task1's proven winner, carried over unmeasured for task2).
- No gated-text feature engineering (`radiology_report`/`pathology_report` regex extraction,
  `psa_trend` velocity, lab features) — the EDA flagged these as a real opportunity, not yet built.
- `decision`/`confidence`'s own tool policy not swept (both fixed at zero tools, matching the now-
  confirmed pattern for `variable_weights`, but not independently re-verified for these two).
- `comorbidity`'s `clamp_ungroundable` option (task1's structurally-ungroundable-variable clamp)
  not re-tested for task2 — carried over as an available, unmeasured choice from task2_schema.py.
- `watchful_waiting`'s 2-case sparsity not specifically addressed (e.g. no class-weighting sweep,
  no oversampling tried).
