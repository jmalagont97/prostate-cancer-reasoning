"""Honest version of run_task3_agent.py: leave-one-out predictions (each case scored by a model
trained on the OTHER 74 cases), not the in-sample-fit production model. Mirrors task1's
run_agent_loocv.py; writes the two-file GC-style outputs to disk for local_eval_task3.py to score.
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from task3_config import DEFAULT_MODEL_KEY, MODEL_FACTORIES, NUMERIC_FEATURES  # noqa: E402
from task3_free_text import build_free_text  # noqa: E402
from task3_schema import Task3Output  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "task3"


def load_task3():
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")
    df = structured.merge(engineered, on="case_id", how="inner").merge(labels, on="case_id", how="inner")
    for col in NUMERIC_FEATURES + ["age", "psa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def write_outputs(output: Task3Output):
    case_dir = OUTPUT_DIR / output.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json", "w") as f:
        json.dump({"months_to_recurrence": output.months_to_recurrence, "event": output.event}, f)
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json", "w") as f:
        json.dump(output.reasoning, f)


def main():
    df = load_task3()
    n = len(df)
    print(f"Model: {DEFAULT_MODEL_KEY}")

    n_ok, n_fail = 0, 0
    for i in range(n):
        tr, va = df.drop(index=i), df.iloc[[i]]
        case_id = va["case_id"].iloc[0]
        try:
            model = MODEL_FACTORIES[DEFAULT_MODEL_KEY]().fit(tr)
            months = float(model.predict_months(va)[0])
            event = int(model.predict_event(va)[0])
            reasoning = build_free_text(va.iloc[0].to_dict(), months)
            output = Task3Output(case_id=case_id, months_to_recurrence=months, event=event,
                                  reasoning=reasoning)
            write_outputs(output)
            n_ok += 1
        except Exception as exc:
            print(f"FAILED {case_id}: {exc}")
            n_fail += 1

    print(f"Wrote {n_ok} LOOCV (leakage-free) cases, {n_fail} failures, to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
