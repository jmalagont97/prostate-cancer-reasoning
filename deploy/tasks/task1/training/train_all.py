#!/usr/bin/env python3
"""
Master Orchestrator for Task 1 Training & Canonical Artifact Generation.
Trains and serializes all 3 self-contained models on 100% of usable data:
  1. Subtask 1.1: deploy/tasks/task1/training/models/subtask1_ensemble.pkl
  2. Subtask 1.2: deploy/tasks/task1/training/models/subtask2_kdm.pkl
  3. Subtask 1.3: deploy/tasks/task1/training/models/subtask3_extratrees.pkl
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task1.training.train_subtask1 import main as train_11
from deploy.tasks.task1.training.train_subtask2 import main as train_12
from deploy.tasks.task1.training.train_subtask3 import main as train_13


def main():
    print("=" * 80)
    print("STARTING FULL RE-TRAINING OF CANONICAL MODELS FOR TASK 1")
    print("=" * 80)

    print("\n--- [1/3] Subtask 1.1: Late Fusion Decision Gate ---")
    train_11()

    print("\n--- [2/3] Subtask 1.2: Scalar Direct-KDM Confidence Classifier ---")
    train_12()

    print("\n--- [3/3] Subtask 1.3: Joint ExtraTrees 23D Variable Weights Meta-Predictor ---")
    train_13()

    print("\n" + "=" * 80)
    print("ALL TASK 1 MODELS TRAINED AND SERIALIZED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
