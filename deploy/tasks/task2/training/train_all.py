#!/usr/bin/env python3
"""
Master Orchestrator for Task 2 Training & Canonical Artifact Generation.
Trains and serializes all 3 self-contained models on 100% of usable data:
  1. Subtask 2.1: deploy/tasks/task2/training/models/subtask1_ensemble.pkl
  2. Subtask 2.2: deploy/tasks/task2/training/models/subtask2_knn.pkl
  3. Subtask 2.3: deploy/tasks/task2/training/models/subtask3_extratrees.pkl
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task2.training.train_subtask1 import main as train_21
from deploy.tasks.task2.training.train_subtask2 import main as train_22
from deploy.tasks.task2.training.train_subtask3 import main as train_23


def main():
    print("=" * 80)
    print("STARTING FULL RE-TRAINING OF CANONICAL MODELS FOR TASK 2")
    print("=" * 80)

    print("\n--- [1/3] Subtask 2.1: Fuzzy-SVR Late Fusion Decision Gate + Neighbor Retriever ---")
    train_21()

    print("\n--- [2/3] Subtask 2.2: Multimodal Fuzzy-KNN Confidence Classifier ---")
    train_22()

    print("\n--- [3/3] Subtask 2.3: Joint 98D ExtraTrees/RF Variable Weights Meta-Predictors ---")
    train_23()

    print("\n" + "=" * 80)
    print("ALL TASK 2 MODELS TRAINED AND SERIALIZED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
