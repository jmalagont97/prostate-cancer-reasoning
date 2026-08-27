"""Interactive, real-time testing loop for task3 -- pick any case from the downloaded data and see
the agent's live prediction immediately, repeatedly, in one running session. Reuses
simulate_production_task3.py's simulate_one_case() exactly (same raw-file-only I/O, same
exclude-and-refit-per-case simulation of a genuinely new submission, same schema validation) --
this file only adds the REPL loop and a nicer per-case summary on top.

Usage: python agent/src/interactive_task3.py
  Commands: a case_id (e.g. T3-005), 'list' (show all available case_ids), 'random' (pick one for
  you), 'quit'/'exit' to stop.
"""

import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from simulate_production_task3 import RAW_TASK3_DIR, simulate_one_case  # noqa: E402


def list_case_ids() -> list[str]:
    return sorted(p.name for p in RAW_TASK3_DIR.iterdir() if p.is_dir())


def print_case_result(case_id: str):
    print(f"\n{'=' * 60}")
    print(f"Case: {case_id}  (refitting on the other 74 cases, reading only its raw input files...)")
    try:
        result = simulate_one_case(case_id, verbose=False)
    except FileNotFoundError:
        print(f"  [error] no raw data folder for '{case_id}' -- try 'list' to see valid case_ids")
        return
    except Exception as exc:
        print(f"  [error] {exc}")
        return

    feats = result["extracted_features"]
    print(f"  Structured/pathology findings used: age={feats.get('age')}, psa={feats.get('psa')}, "
          f"ISUP={feats.get('isup_grade_group')}, pT-ordinal={feats.get('pt_stage_ordinal')}, "
          f"margins_positive={feats.get('margins_positive')}, EPE={feats.get('epe_present')}, "
          f"SVI={feats.get('svi_present')}")
    print(f"\n  PREDICTED months_to_recurrence: {result['months_to_recurrence']:.2f}")
    print(f"  Reasoning: {result['reasoning']}")

    gt = result["ground_truth"]
    if gt:
        event_str = "recurrence observed" if gt["event"] == 1 else "censored (no recurrence seen)"
        print(f"\n  [sanity check only -- a real new case wouldn't have this]")
        print(f"  Ground truth: {gt['months_to_recurrence']} months ({event_str})")
    print(f"  Output files: {result['output_dir']}")


def main():
    all_ids = list_case_ids()
    print(f"=== Task3 interactive testing ({len(all_ids)} cases available) ===")
    print("Enter a case_id (e.g. T3-005), 'list', 'random', or 'quit'/'exit'.\n")

    while True:
        try:
            raw = input("task3> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if raw.lower() == "list":
            for i in range(0, len(all_ids), 10):
                print("  " + ", ".join(all_ids[i:i + 10]))
            continue
        if raw.lower() == "random":
            case_id = random.choice(all_ids)
            print(f"(picked {case_id})")
        else:
            case_id = raw

        if case_id not in all_ids:
            print(f"  '{case_id}' not found. Try 'list' to see valid case_ids.")
            continue

        print_case_result(case_id)


if __name__ == "__main__":
    main()
