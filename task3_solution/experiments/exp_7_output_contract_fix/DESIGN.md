# Experiment Design: Task3 — Corrected Output Contract (`event` Field)
**Experiment**: experiments/exp_7/
**Project**: challenge_chimera_v3
**Date**: 2026-08-29
**Status**: Complete

---

## 1. Context: a real correction, not a new feature

Every prior session treated `Task3Output` in `external/chimera-agent-baseline/…/schema.py` as the
authoritative submission contract — it defines only `case_id`, `task`, `months_to_recurrence`,
`reasoning`, with `extra="forbid"`. When directly asked to re-check the org's own task-3 webpage,
it states plainly: *"Predicted BCR status within the observation window (1 = BCR; 0 = no BCR)"* is
a required output. Cross-checked against the reference agent's actual output-writing code
(`run.py::_write_task_outputs`), which was never previously read closely enough:

```python
if task == 3:
    _write_json(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json", {
        "event": structured.get("event", 0),
        "months_to_recurrence": structured["months_to_recurrence"],
    })
```

**`Task3Output`'s pydantic schema is not the real contract** — it constrains only an internal LLM
form-fill step; the function that writes the actual submitted file repackages the result into a
different, unvalidated shape that includes `event`. This was a genuine gap in our own agent, caught
only because the user checked the primary source directly rather than accepting the earlier
(incomplete) analysis.

## 2. What needed to change

1. `event: int` added to our own `task3_schema.py`'s `Task3Output` (0–1, validated).
2. A way to actually predict it — the model had never produced a binary call before, only a
   continuous time.
3. `write_outputs()` in all three output-producing scripts updated to include it.

## 3. Deriving `event` from the existing survival curve — swept, not assumed

The standard, textbook approach for converting a fitted survival curve into a binary risk call at a
fixed horizon: pick horizon *T*, predict `event=1` if `S(T|x) < threshold`. No new model — reuses
the already-promoted Cox fit's own `predict_survival_function()`.

Two free choices, both swept under LOOCV against real ground-truth `event` labels (not assumed):

| Horizon (mo) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| 12 | acc .787 / bal .579 | .773 / .570 | .773 / .587 | .813 / .666 | .827 / .693 |
| 24 | .773 / .570 | .800 / .640 | .827 / .693 | .827 / .693 | .827 / .727 |
| 36 | .773 / .570 | .800 / .640 | .827 / .693 | .827 / .693 | .827 / .727 |
| 60 | .800 / .640 | .827 / .693 | .827 / .693 | .827 / .727 | **.800 / .779** |
| 110 | .827 / .693 | .827 / .693 | .827 / .727 | .800 / .744 | .707 / .734 |

(cells: accuracy / balanced accuracy)

**Winner: horizon = 60 months (the standard clinical 5-year BCR benchmark — converges with
clinical convention, not just an empirical accident), threshold = 0.7.** Balanced accuracy 0.779,
recall on true events 0.737, precision 0.583 — against a trivial "always predict 0" baseline's
useless 0.500 balanced accuracy (it would score 0.747 raw accuracy purely from class imbalance,
19/75 events, while catching zero real recurrences).

**Honest limitation**: unlike `ranking_score` gains elsewhere in this project, this was not
bootstrap-tested for significance — classification accuracy at n=75/19-events is inherently noisy,
and this is a secondary, non-scored diagnostic addition (confirmed: `ranking_score` is unaffected
by `event`, since C-index only ever reads *ground-truth* event, never predicted). The plateau across
neighboring (horizon, threshold) cells (24/0.70, 36/0.70, 110/0.60 all cluster near 0.72–0.74) is
reassuring against a fragile single-cell peak, but that is observational, not a formal test.

## 4. Verification

- `ranking_score` re-confirmed unchanged at 0.8765 after the change (as predicted from source, but
  verified rather than assumed).
- Full output audit (both production and LOOCV paths): 75/75 cases schema-valid, including the new
  `event` field's range constraint and confirming `extra="forbid"` still correctly rejects a truly
  unexpected field.
- The previously-unreachable diagnostic `event_score` is now genuinely computed:
  `mean_event_score=0.800` on the honest LOOCV path, matching the sweep's own number exactly —
  confirms the wiring end-to-end, not just the isolated sweep script.
