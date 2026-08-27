"""Scores agent/output/task3 against ground truth using the REAL evaluate.py scoring functions,
imported directly (not reimplemented). Mirrors agent/local_eval.py (task1).

ranking_score = concordance_index(gt_months, pred_months, gt_event) -- the ONLY thing that feeds
the leaderboard for task3 (confirmed from evaluate.py source, see the comment at its own
compute_task_aggregate_recurrence-equivalent function). mean_case_score (event/time/rationale
composite) is reported here too, but purely diagnostic -- it does not feed ranking_score, and our
schema (task3_schema.py) never predicts "event" at all, so its event component is structurally
always 0 (also confirmed from source, not a bug to fix here).
"""

import json
import sys
from pathlib import Path
from statistics import mean

EVAL_DIR = Path(__file__).resolve().parent.parent / "external" / "CHIMERA-agent" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from evaluate import (  # noqa: E402
    TD_AUC_HORIZONS_MONTHS, concordance_index, recurrence_time_score, time_dependent_auc,
)

RAW_TASK3 = Path(__file__).resolve().parent.parent / "data" / "train_release_v2" / "train_release" / "task3"
AGENT_OUTPUT = Path(__file__).resolve().parent / "output" / "task3"


def load_gt(case_id: str) -> dict:
    return json.load(open(RAW_TASK3 / case_id / "prostate-time-to-recurrence-or-last-follow-up.json"))


def load_pred(case_dir: Path) -> dict:
    outcome = json.load(open(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json"))
    reasoning = json.load(open(case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json"))
    return {"months_to_recurrence": outcome["months_to_recurrence"], "reasoning": reasoning}


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
        rows.append({"case_id": case_id, "gt_months": gt["months_to_recurrence"], "gt_event": gt["event"],
                      "pred_months": pred["months_to_recurrence"], "time_score": tsc})

    times = [r["gt_months"] for r in rows]
    preds = [r["pred_months"] for r in rows]
    events = [r["gt_event"] for r in rows]
    c_index = concordance_index(times, preds, events)

    # Diagnostic case_score, matching evaluate.py's renormalization when both the event field
    # (schema doesn't have one) and the rationale judge (no Ollama here) are unavailable: pure
    # time_score, weight 1.0 (its 0.35/0.50 share plus the unreachable event share, per source).
    mean_case_score = mean(r["time_score"] for r in rows if r["time_score"] is not None)

    # T3.7: the evaluator's OTHER reported metric (not part of ranking_score, which is C-index
    # alone -- this is a cross-check, not a second thing to optimise). If a model's C-index looks
    # good only because of how it orders extreme cases but is poorly calibrated at clinically
    # relevant horizons, this catches it; agreement between the two is reassurance, not a target.
    td_auc = {f"{int(h)}m": time_dependent_auc(times, preds, events, h) for h in TD_AUC_HORIZONS_MONTHS}
    td_auc_values = [v for v in td_auc.values() if v is not None]
    mean_td_auc = mean(td_auc_values) if td_auc_values else None

    print(f"n_cases={len(rows)}")
    print(f"RANKING_SCORE (C-index, the only leaderboard-relevant number)={c_index:.4f}")
    print(f"mean_time_score (diagnostic only)={mean_case_score:.4f}")
    print("(event_score always 0/unreachable under this schema -- see module docstring)")
    print(f"\nT3.7 cross-check -- IPCW time-dependent AUC (diagnostic, not part of ranking_score):")
    for horizon, v in td_auc.items():
        v_str = f"{v:.4f}" if v is not None else "n/a (no case/control pair at this horizon)"
        print(f"  {horizon:>4s}: {v_str}")
    if mean_td_auc is not None:
        print(f"  mean across horizons: {mean_td_auc:.4f}")

    with open(Path(__file__).resolve().parent / "output" / "eval_results_task3.json", "w") as f:
        json.dump({"rows": rows, "ranking_score": c_index, "mean_time_score_diagnostic": mean_case_score,
                    "time_dependent_auc": td_auc, "mean_time_dependent_auc": mean_td_auc},
                   f, indent=2, default=str)


if __name__ == "__main__":
    main()
