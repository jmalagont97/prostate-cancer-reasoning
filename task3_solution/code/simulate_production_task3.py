"""Simulates a real production submission for ONE task3 case, end to end, as close to the actual
grand-challenge interface as our locally downloaded data allows:

  1. Reads ONLY the case's 3 raw input files (structured-prompt.json, clinical-data.json,
     neural-representations.json) -- never touches data/processed/task3/{structured,engineered_
     features}.csv, which is the batch/offline path every other script in this project uses.
  2. Extracts its engineered features live, via the same single-case parser functions
     extract_features_task3.py uses in bulk (imported, not re-derived).
  3. Refits the promoted model on the OTHER 74 cases only -- this case is excluded from training,
     simulating a genuinely new submission (not something that just happens to be in-sample).
  4. Forces the model's embedding lookups down the raw-file fallback path (task3_embeddings.py's
     _pooled_from_raw_case) by removing this case from the fitted model's compiled-npz maps after
     fitting -- exercises the exact code path a truly new case would need, not the fast cached path
     every other evaluation in this project has used (since all 75 downloaded cases happen to
     already be in the compiled npz).
  5. Validates the prediction against Task3Output and writes the two real GC-shaped output files.

Our current data has 0 genuinely unlabeled task3 cases (confirmed: splits.json), so this can't test
against a truly new case -- the closest honest substitute is "pretend we've never seen this one",
which is exactly what LOOCV already does model-wise; what's NEW here is testing the actual I/O path
(raw files in, GC-shaped files out) rather than the batch CSV pipeline every other script uses.

Usage: python agent/src/simulate_production_task3.py --case_id T3-005
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from task3_config import DEFAULT_MODEL_KEY, MODEL_FACTORIES, NUMERIC_FEATURES  # noqa: E402
from task3_free_text import build_free_text  # noqa: E402
from task3_schema import Task3Output  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from extract_features_task3 import parse_active_treatment, parse_dre, parse_surgical_pathology  # noqa: E402

import pandas as pd  # noqa: E402

RAW_TASK3_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "train_release_v2"
                  / "train_release" / "task3")
TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "task3_production_simulation"


def load_case_from_raw_files(case_id: str) -> dict:
    """Step 1+2: read ONLY this case's own 3 raw input files, extract features live -- the actual
    production interface, not our compiled batch CSVs."""
    case_dir = RAW_TASK3_DIR / case_id
    structured_prompt = json.loads((case_dir / "structured-prompt.json").read_text())
    clinical_data = json.loads(
        (case_dir / "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json").read_text()
    )
    # neural-representations.json isn't read directly here -- task3_embeddings.py's
    # EmbeddingSource._pooled_from_raw_case() reads it, exercised via step 4 below.

    row = {"case_id": case_id, "age": structured_prompt["age"], "psa": structured_prompt["psa"]}
    dre_abnormal, dre_bilateral, _, dre_ct_stage = parse_dre(structured_prompt.get("dre"))
    row["dre_abnormal"] = dre_abnormal
    row["dre_bilateral"] = dre_bilateral
    row["dre_ct_stage_ordinal"] = dre_ct_stage
    row.update(parse_surgical_pathology(clinical_data.get("surgical_pathology_report")))
    return row


def load_training_set_excluding(case_id: str) -> pd.DataFrame:
    """Step 3's data: the other 74 cases, from our compiled batch files -- this part legitimately
    uses the offline pipeline, since refitting needs the full labeled training set, not just one
    case's raw files."""
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")
    df = structured.merge(engineered, on="case_id", how="inner").merge(labels, on="case_id", how="inner")
    for col in NUMERIC_FEATURES + ["age", "psa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["case_id"] != case_id].reset_index(drop=True)


def force_raw_embedding_path(model, case_id: str):
    """Step 4: strip this case from the freshly-fit model's own EmbeddingSource maps, so
    predict_months() is forced down the raw-neural-representations.json fallback path instead of
    the fast compiled-npz path every other script's evaluation has relied on (since all 75
    downloaded cases are already in that compiled file -- a genuinely new case wouldn't be)."""
    sources = getattr(model, "sources", {})
    n_forced = 0
    for source in sources.values():
        if case_id in source.emb:
            del source.emb[case_id]
            del source.mask[case_id]
            n_forced += 1
    return n_forced


def simulate_one_case(case_id: str, verbose: bool = True) -> dict:
    """Runs steps 1-5 for one case; returns {months, reasoning, output_dir, ground_truth (or None)}.
    Shared by the one-shot CLI (__main__ below) and interactive_task3.py's REPL loop."""
    def log(msg):
        if verbose:
            print(msg)

    log("[1/5] Reading raw input files only (structured-prompt.json, clinical-data.json)...")
    case_row_dict = load_case_from_raw_files(case_id)
    case_row = pd.DataFrame([case_row_dict])
    log(f"  Extracted features: { {k: v for k, v in case_row_dict.items() if k != 'case_id'} }")

    log(f"\n[2/5] Loading training set EXCLUDING {case_id} (simulating a genuinely new case)...")
    train_df = load_training_set_excluding(case_id)
    log(f"  Training on {len(train_df)} cases (should be 74)")

    log(f"\n[3/5] Refitting model ({DEFAULT_MODEL_KEY}) on the 74-case training set...")
    model = MODEL_FACTORIES[DEFAULT_MODEL_KEY]().fit(train_df)

    log(f"\n[4/5] Forcing embedding lookups down the raw-file fallback path...")
    n_forced = force_raw_embedding_path(model, case_id)
    log(f"  Forced {n_forced} embedding source(s) to read {case_id}'s raw "
        f"neural-representations.json directly (not the compiled npz)")

    log(f"\n[5/5] Predicting + validating against Task3Output schema...")
    months = float(model.predict_months(case_row)[0])
    event = int(model.predict_event(case_row)[0])
    reasoning = build_free_text(case_row_dict, months)
    output = Task3Output(case_id=case_id, months_to_recurrence=months, event=event, reasoning=reasoning)
    log(f"  Schema-validated OK.")

    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json", "w") as f:
        json.dump({"months_to_recurrence": output.months_to_recurrence, "event": output.event}, f)
    with open(case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json", "w") as f:
        json.dump(output.reasoning, f)
    log(f"  Wrote GC-shaped output files to {case_dir}")

    gt = None
    gt_path = RAW_TASK3_DIR / case_id / "prostate-time-to-recurrence-or-last-follow-up.json"
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())

    return {"case_id": case_id, "months_to_recurrence": months, "event": event, "reasoning": reasoning,
            "output_dir": case_dir, "extracted_features": case_row_dict, "ground_truth": gt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default="T3-005")
    args = parser.parse_args()
    case_id = args.case_id

    print(f"=== Simulating production submission for {case_id} ===\n")
    result = simulate_one_case(case_id)

    print(f"\n=== Result ===")
    print(f"  Predicted months_to_recurrence: {result['months_to_recurrence']:.2f}")
    print(f"  Predicted event (BCR status): {result['event']}")
    print(f"  Reasoning: {result['reasoning']}")

    if result["ground_truth"]:
        gt = result["ground_truth"]
        print(f"\n  [Not part of production -- our own sanity check only, since this case happens to\n"
              f"   have a real label. A genuinely new submission case would have none of this.]")
        print(f"  Ground truth: months_to_recurrence={gt['months_to_recurrence']}, event={gt['event']}")


if __name__ == "__main__":
    main()
