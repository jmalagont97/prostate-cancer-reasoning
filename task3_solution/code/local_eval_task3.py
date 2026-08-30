"""Scores agent/output/task3 against ground truth using the REAL evaluate.py scoring functions,
imported directly (not reimplemented). Mirrors agent/local_eval.py (task1).

ranking_score = concordance_index(gt_months, pred_months, gt_event) -- the ONLY thing that feeds
the leaderboard for task3 (confirmed from evaluate.py source, see the comment at its own
compute_task_aggregate_recurrence-equivalent function). case_score (event/time/rationale composite)
is reported here too, but purely diagnostic -- it does not feed ranking_score. Its event component
is now reachable (experiments/exp_7/ added a real predict_event() to the model and `event` to our
output schema -- corrected after finding schema.py's Task3Output, which we'd been treating as the
full contract, omits `event` entirely; the org's own webpage and the reference agent's real write
path both confirm it's required).
"""

import json
import sys
from pathlib import Path
from statistics import mean

EVAL_DIR = Path(__file__).resolve().parent.parent / "external" / "CHIMERA-agent" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from evaluate import (  # noqa: E402
    TD_AUC_HORIZONS_MONTHS, concordance_index, recurrence_event_score, recurrence_time_score,
    time_dependent_auc,
)

RAW_TASK3 = Path(__file__).resolve().parent.parent / "data" / "train_release_v2" / "train_release" / "task3"
AGENT_OUTPUT = Path(__file__).resolve().parent / "output" / "task3"


def load_gt(case_id: str) -> dict:
    return json.load(open(RAW_TASK3 / case_id / "prostate-time-to-recurrence-or-last-follow-up.json"))


def load_pred(case_dir: Path) -> dict:
    outcome = json.load(open(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json"))
    reasoning = json.load(open(case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json"))
    return {"months_to_recurrence": outcome["months_to_recurrence"], "event": outcome.get("event"),
            "reasoning": reasoning}


def main():
    case_dirs = sorted(p for p in AGENT_OUTPUT.iterdir() if p.is_dir())
    rows = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        gt_path = RAW_TASK3 / case_id
        if not gt_path.exists():
            print(f"[skip] no raw ground truth dir for {case_id}")
            continue
        gt = load_gt(case_id)
        pred = load_pred(case_dir)
        tsc = recurrence_time_score(gt, pred)
        esc = recurrence_event_score(gt, pred)
        rows.append({"case_id": case_id, "gt_months": gt["months_to_recurrence"], "gt_event": gt["event"],
                      "pred_months": pred["months_to_recurrence"], "pred_event": pred["event"],
                      "time_score": tsc, "event_score": esc})

    times = [r["gt_months"] for r in rows]
    preds = [r["pred_months"] for r in rows]
    events = [r["gt_event"] for r in rows]
    c_index = concordance_index(times, preds, events)

    # Diagnostic case_score, matching evaluate.py's own weighting when the rationale judge is
    # unavailable (no Ollama here): event 0.50, time 0.50 -- both now reachable.
    time_scores = [r["time_score"] for r in rows if r["time_score"] is not None]
    event_scores = [r["event_score"] for r in rows if r["event_score"] is not None]
    mean_time_score = mean(time_scores) if time_scores else None
    mean_event_score = mean(event_scores) if event_scores else None
    mean_case_score = (
        mean(0.5 * r["event_score"] + 0.5 * r["time_score"] for r in rows
             if r["event_score"] is not None and r["time_score"] is not None)
    )
    recurrence_event_accuracy = mean_event_score  # same quantity under this 0/1 scoring rule

    # T3.7: the evaluator's OTHER reported metric (not part of ranking_score, which is C-index
    # alone -- this is a cross-check, not a second thing to optimise). If a model's C-index looks
    # good only because of how it orders extreme cases but is poorly calibrated at clinically
    # relevant horizons, this catches it; agreement between the two is reassurance, not a target.
    td_auc = {f"{int(h)}m": time_dependent_auc(times, preds, events, h) for h in TD_AUC_HORIZONS_MONTHS}
    td_auc_values = [v for v in td_auc.values() if v is not None]
    mean_td_auc = mean(td_auc_values) if td_auc_values else None

    print(f"n_cases={len(rows)}")
    print(f"RANKING_SCORE (C-index, the only leaderboard-relevant number)={c_index:.4f}")
    print(f"mean_case_score (diagnostic only, event 50% + time 50%)={mean_case_score:.4f}")
    print(f"  mean_event_score={mean_event_score:.4f}  (BCR-status accuracy vs ground truth)")
    print(f"  mean_time_score={mean_time_score:.4f}")
    print(f"\nT3.7 cross-check -- IPCW time-dependent AUC (diagnostic, not part of ranking_score):")
    for horizon, v in td_auc.items():
        v_str = f"{v:.4f}" if v is not None else "n/a (no case/control pair at this horizon)"
        print(f"  {horizon:>4s}: {v_str}")
    if mean_td_auc is not None:
        print(f"  mean across horizons: {mean_td_auc:.4f}")

    with open(Path(__file__).resolve().parent / "output" / "eval_results_task3.json", "w") as f:
        json.dump({"rows": rows, "ranking_score": c_index, "mean_case_score_diagnostic": mean_case_score,
                    "mean_time_score_diagnostic": mean_time_score,
                    "mean_event_score_diagnostic": mean_event_score,
                    "recurrence_event_accuracy": recurrence_event_accuracy,
                    "time_dependent_auc": td_auc, "mean_time_dependent_auc": mean_td_auc},
                   f, indent=2, default=str)


if __name__ == "__main__":
    main()
