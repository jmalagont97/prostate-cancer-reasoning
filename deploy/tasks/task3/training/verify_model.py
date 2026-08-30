#!/usr/bin/env python3
"""Verify the serialized Task 3 artifact on the annotated cohort."""
from __future__ import annotations

from pathlib import Path

from deploy.tasks.task3.inference import predict

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data" / "raw" / "train_release" / "task3"
MODEL = ROOT / "deploy" / "tasks" / "task3" / "training" / "models" / "task3_cox_pca5.pkl"


def main() -> None:
    cases = sorted(path for path in RAW.iterdir() if path.is_dir())
    outputs = [predict(case, MODEL) for case in cases]
    assert len(outputs) == 75
    assert all(set(output) == {"months_to_recurrence", "event"} for output in outputs)
    assert all(output["months_to_recurrence"] >= 0 for output in outputs)
    assert all(output["event"] in (0, 1) for output in outputs)
    print(f"VERIFIED: {len(outputs)} cases; events_predicted={sum(o['event'] for o in outputs)}")


if __name__ == "__main__":
    main()
