"""Backfill macro-F1 into exp_4's already-completed results, per this project's cross-experiment
macro-F1 reporting initiative, extended back to exp_1-exp_5 on 2026-08-13.

Reuses experiments/exp_3/scripts/backfill_macro_f1.py's functions unchanged (same pattern as
exp_4's own run_*.py scripts reusing exp_3's cv_utils.py/models.py) -- only the feature frame
differs (select_exp4_feature_frame, the 16-column clinical-only/no-MRI frame).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_4/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backfill_macro_f1 import run_confidence, run_decision, run_reveal, run_weights  # noqa: E402

import numpy as np

from chimera_task1.features import select_exp4_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_RANK
from chimera_task1.train_decision import load_labeled_data
from chimera_task1.train_reasoning import load_annotated


def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"

    _, inp, df = load_labeled_data()
    y_decision = (df["target_biopsy_decision"] == "yes").astype(int).values
    X_decision = select_exp4_feature_frame(inp)
    print("=== decision ===")
    run_decision(results_dir, X_decision, y_decision)

    ann, inp_ann = load_annotated()
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])
    X_ann = select_exp4_feature_frame(inp_ann)
    print("\n=== confidence ===")
    run_confidence(results_dir, X_ann, y_conf_rank)

    print("\n=== weights ===")
    run_weights(results_dir, ann, X_ann)

    print("\n=== reveal ===")
    run_reveal(results_dir, ann, X_ann)


if __name__ == "__main__":
    main()
