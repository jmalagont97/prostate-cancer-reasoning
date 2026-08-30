# exp_7 Report: Corrected Task3 Output Contract

**Status**: Complete. **Verdict: a real bug fixed, not a new feature — our output was schema-incomplete.**

## What was wrong

Every prior session in this project treated `Task3Output` (in the reference baseline repo's
`schema.py`) as the full submission contract for task3: `case_id`, `months_to_recurrence`,
`reasoning` — no `event`, with `extra="forbid"` actively rejecting one. This was wrong. The user
directly checked the challenge's own webpage, which states plainly that "Predicted BCR status
within the observation window (1 = BCR; 0 = no BCR)" is required. Cross-checked against the
reference agent's *actual* output-writing code (`run.py`), not just its internal pydantic model —
confirmed: the real submitted file does include `event`. `Task3Output`'s pydantic schema was only
ever used to validate an intermediate LLM form-fill step, never the final write.

## What was fixed

- `task3_schema.py`: added `event: int` (0–1) to our own `Task3Output`.
- `task3_config.py`: added `predict_event(df, horizon, threshold)` to the promoted model, deriving
  a binary call from the survival curve already being computed — no new model.
- **Horizon and threshold swept under LOOCV** against real ground-truth labels, not assumed:
  60 months (5-year BCR, the clinical standard) + threshold 0.7 won, balanced accuracy 0.779 vs. a
  trivial baseline's useless 0.500.
- All three output-producing scripts (`run_task3_agent.py`, `run_task3_agent_loocv.py`,
  `simulate_production_task3.py`) updated to predict and write it.
- `local_eval_task3.py` updated — the diagnostic `event_score` component (previously always 0,
  structurally unreachable) now computes for real: `mean_event_score=0.800`.

## Verified, not assumed

| Check | Result |
|---|---|
| `ranking_score` unchanged after the fix | **0.8765**, exactly as before — confirmed, not assumed (C-index only ever reads *ground-truth* event) |
| Full output audit, both paths (production + LOOCV) | **75/75** schema-valid, including the new field's range check |
| `extra="forbid"` still rejects a truly unexpected field | Confirmed |
| `event=2` (out of range) rejected | Confirmed |
| Diagnostic `mean_event_score` on the real LOOCV output files matches the isolated sweep exactly | **0.800 = 0.800** — confirms correct end-to-end wiring, not just the sweep script in isolation |

## Classification report + confusion matrix (LOOCV, horizon=60mo, threshold=0.7)

Computed with `sklearn.metrics` directly against the honest LOOCV output files on disk (not a
separate ad hoc recomputation) — confirms the wiring end to end, not just the sweep script.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No BCR (0) | 0.902 | 0.821 | 0.860 | 56 |
| BCR (1) | 0.583 | 0.737 | 0.651 | 19 |
| **Accuracy** | | | **0.800** | 75 |
| Macro avg | 0.743 | 0.779 | 0.755 | 75 |
| Weighted avg | 0.821 | 0.800 | 0.807 | 75 |

|              | Predicted: no BCR | Predicted: BCR |
|---|---|---|
| **True: no BCR** (n=56) | 46 (TN) | 10 (FP) |
| **True: BCR** (n=19) | 5 (FN) | 14 (TP) |

Macro-avg recall (0.779) matches the sweep's own reported balanced accuracy exactly; overall
accuracy (0.800) matches the sweep's number for this exact (horizon, threshold) cell. Of 19 true
recurrences, 14 were caught (74% recall); the model leans toward over-flagging risk (10 false
positives) rather than missing a real recurrence (5 false negatives) — the right failure mode to
prefer in a clinical screening context, and a direct consequence of optimizing for balanced
accuracy under this cohort's 75/25 class split rather than raw accuracy.

## Not done

- No bootstrap significance test on the balanced-accuracy result — classification metrics at
  n=75/19-events are noisy, and unlike every `ranking_score` gain in this project, this doesn't
  touch the scored metric, so it wasn't held to the same bar. The neighboring-cell plateau in the
  sweep grid (0.72–0.74 across several nearby horizon/threshold combos) is reassuring but
  observational, not a formal test.
- Not yet pushed to the GitHub repo (`task3_solution/`) or reflected in the two published artifacts
  — deliberately held for a separate go-ahead given the scope.
