# exp_3 Report: Task3 Post-Surgical Pathology Text Features

**Status**: Complete. **Verdict: Q2 (new information) confirmed, promoted. Q1 (encoding cleanup
alone) refuted.**

## Results

| Condition | LOOCV C-index (primary) | 5-fold C-index (continuity check) | n used |
|---|---|---|---|
| `baseline_aspublished` (age, psa, raw-text `dre`/`active_treatment_prior_to_surgery`, one-hot) | 0.7000 | 0.6410 | 75/75 |
| `structured_cleaned` (Q1: same info, cleaned encoding) | 0.6611 | 0.6301 | 75/75 |
| `structured_cleaned_plus_pathology` (Q2: + 8 surgical-pathology fields) | **0.7717** | **0.7312** | 75/75 |

(All numbers after fixing a real regex-alternation bug in `dre_bilateral`'s extraction — see
Issues below — and dropping `active_treatment_flag` from the two engineered conditions, which
demonstrably destabilized LOOCV for a feature carrying almost no signal; see Issues.)

- **Sanity check passed**: `baseline_aspublished`'s 5-fold number (0.6410) reproduces the
  already-published baseline (`scripts/fit_baseline_models.py`, 0.641) essentially exactly — same
  data, same protocol, confirms nothing else in the pipeline drifted.
- **Q1 refuted**: cleaning `dre`'s encoding (raw near-unique free text → parsed
  abnormal/bilateral/cT-stage fields) *hurt* under both protocols (0.700→0.661 LOOCV,
  0.641→0.630 5-fold). The raw one-hot apparently captures some structure the 3-field parse
  doesn't (plausibly: exact phrasing correlates with which clinician/site produced the note, itself
  weakly correlated with outcome via case mix — not something to lean on, but also not the target
  of Q2's promotion decision).
- **Q2 confirmed, decisively**: adding the 8 surgical-pathology fields (ISUP grade group, pT
  stage, EPE, margins, SVI, LVI, lymph-node assessed/positive) on top of the cleaned structured
  set lifts LOOCV C-index from 0.700 → 0.772 (+0.072) and 5-fold from 0.641 → 0.731 (+0.090) —
  consistent direction and a real margin under both protocols, unlike task1's earlier +0.001
  confidence-swap case. C-index ≈0.77 is also in the range real urological literature reports for
  post-prostatectomy BCR prediction using pathology stage/grade (e.g. Partin-table-style models),
  which is a face-validity check this result passes, not just an internal comparison.
- **Not leakage**: all 8 pathology fields are recorded *at the time of surgery* (surgical pathology
  report), strictly before the recurrence clock starts — the same temporal ordering real clinical
  Cox models for BCR use these variables under. `pathology_report` (pre-op biopsy) was checked
  separately and does NOT carry these fields (see notebook §5) — no post-hoc information is
  entering the model.

## Promotion

Per `DESIGN.md`'s rule (promote only if `structured_cleaned_plus_pathology`'s LOOCV C-index beats
`baseline_aspublished`'s LOOCV C-index): **promoted**. `scripts/fit_baseline_models.py`'s task3
config now merges `data/processed/task3/engineered_features.csv` and uses the winning 12-feature
set; `docs/model_training_plan.md`'s T3.6 row marked ✅.

## Issues found and fixed during this experiment

1. **Regex alternation-order bug** in the first draft of the `dre` finding parser: Python regex
   alternation tries options left-to-right and stops at the first match, not the longest — with
   `"was abnormal"` listed before `"was abnormal bilaterally"`, every bilateral case matched the
   shorter alternative first and `dre_bilateral` was silently always 0 across all 75 cases (zero
   variance, invisible until it caused a Cox convergence failure). Fixed by reordering
   alternatives most-specific-first. A reminder that a silently-wrong regex doesn't announce
   itself — it just produces a plausible-looking, wrong column, same lesson as this session's
   earlier SVI keyword-scan iteration in the EDA notebook.
2. **`active_treatment_flag` near-constant column**: 1.0 for exactly 1/75 cases. As a *numeric*
   feature (unlike the raw-text categorical baseline, where sklearn's `OneHotEncoder` simply omits
   categories unseen in a training fold), holding that one case out in LOOCV leaves the column
   literally constant across the other 74 training rows, singularizing the Cox design matrix —
   confirmed as the exact cause of the one non-convergent LOOCV fold in the first run. Dropped from
   both engineered conditions (near-zero real signal, real convergence cost); all three conditions
   now converge on 75/75 LOOCV folds.

## Next steps (not started, out of scope for this experiment)

- T3.7 (IPCW time-dependent AUC cross-check against `evaluate.py`'s other reported metric).
- T3.2/T3.3 (MRI/slide-embedding fusion) — worth testing now that a strong tabular+text baseline
  (0.772) exists to beat, same "does the embedding add anything on top of the best non-embedding
  model" framing used for task2.
- Wiring a task3 agent (analogous to `agent/src/run_agent.py` for task1) was not attempted this
  session — task3 has no confidence/variable-weights/tool_score/section_grounding_score components
  at all (confirmed earlier: `ranking_score` = C-index directly), so its agent shape is much
  simpler than task1's; worth a dedicated pass once T3.7's cross-check is done.
