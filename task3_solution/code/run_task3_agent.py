"""v0 task3 agent (production/in-sample fit -- see run_task3_agent_loocv.py for the honest,
leakage-free evaluation path). Loads the saved task3_config.DEFAULT_MODEL_KEY model, predicts
months_to_recurrence per case, builds template reasoning, validates against Task3Output, writes
the two GC-style output files.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from task3_config import MODELS_DIR, NUMERIC_FEATURES  # noqa: E402
from task3_free_text import build_free_text  # noqa: E402
from task3_schema import Task3Output  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "task3"


def load_task3():
    """Deliberately does NOT merge labels.csv: this is the production/inference path, meant to run
    on genuinely new cases that have no ground truth at all. An earlier version inner-joined
    labels.csv here (copy-pasted from the LOOCV/evaluation scripts, where that join is legitimate
    since they need the true event/months to score against) -- that silently required every case to
    already have a known outcome, which defeats the purpose of a prediction script. Caught during a
    gap-analysis pass, not by a failing test (no unlabeled cases exist in our current data dump to
    trigger it), so worth remembering next time a new script is copied from an eval script."""
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    df = structured.merge(engineered, on="case_id", how="inner")
    for col in NUMERIC_FEATURES + ["age", "psa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def write_outputs(output: Task3Output):
    case_dir = OUTPUT_DIR / output.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    # Matches the real submission contract (confirmed against the org's task-3 webpage and the
    # reference agent's actual write path, run.py::_write_task_outputs -- NOT Task3Output's own
    # pydantic schema in the baseline repo, which turned out to be an incomplete internal-only
    # constraint, see task3_schema.py): {months_to_recurrence, event} outcome object plus a
    # separate reasoning file.
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json", "w") as f:
        json.dump({"months_to_recurrence": output.months_to_recurrence, "event": output.event}, f)
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json", "w") as f:
        json.dump(output.reasoning, f)


def main():
    df = load_task3()
    bundle = joblib.load(MODELS_DIR / "task3_agent_model.joblib")
    model = bundle["model"]
    print(f"Loaded task3 model: {bundle['model_key']}")

    n_ok, n_fail = 0, 0
    for i in range(len(df)):
        row = df.iloc[[i]]
        case_id = row["case_id"].iloc[0]
        try:
            months = float(model.predict_months(row)[0])
            event = int(model.predict_event(row)[0])
            reasoning = build_free_text(row.iloc[0].to_dict(), months)
            output = Task3Output(case_id=case_id, months_to_recurrence=months, event=event,
                                  reasoning=reasoning)
            write_outputs(output)
            n_ok += 1
        except Exception as exc:
            print(f"FAILED {case_id}: {exc}")
            n_fail += 1

    print(f"Wrote {n_ok} cases, {n_fail} failures, to {OUTPUT_DIR}")
    print("NOTE: this is the in-sample-fit production model -- do NOT score these outputs against "
          "the same 75 training cases. Use run_task3_agent_loocv.py's outputs for any real evaluation.")


if __name__ == "__main__":
    main()
