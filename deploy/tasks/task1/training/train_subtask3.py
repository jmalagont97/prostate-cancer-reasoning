#!/usr/bin/env python3
"""
Canonical Training Script for Subtask 1.3: Variable Weights & Decisive Factors Meta-Predictor.
Uses model_definitions.Subtask3JointExtraTrees to ensure global pickling scope.
"""

from pathlib import Path
import joblib
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task1.training.model_definitions import Subtask3JointExtraTrees


def main():
    data_dir = PROJECT_ROOT / "data" / "processed" / "task1"
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task1" / "training" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed task1 data from: {data_dir}")
    df_tab = pd.read_csv(data_dir / "clinical_tabular.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task1.csv")

    labeled_count = df_gt["is_labeled"].sum()
    print(f"Total cohort size: {len(df_gt)}, Labeled cases: {labeled_count}")

    model = Subtask3JointExtraTrees(n_estimators=50, random_state=42)

    print("Fitting canonical Joint ExtraTrees 23D meta-predictor on all labeled cases...")
    model.fit(df_tab, df_gt)

    out_path = models_dir / "subtask3_extratrees.pkl"
    joblib.dump(model, out_path)
    print(f"Saved self-contained model artifact to: {out_path}")


if __name__ == "__main__":
    main()
