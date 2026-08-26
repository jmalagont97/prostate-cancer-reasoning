#!/usr/bin/env python3
"""
Canonical Training Script for Subtask 1.2: Clinical Confidence Classifier.
Uses model_definitions.Subtask2DirectKDMClassifier to ensure global pickling scope.
"""

from pathlib import Path
import joblib
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task1.training.model_definitions import Subtask2DirectKDMClassifier


def main():
    data_dir = PROJECT_ROOT / "data" / "processed" / "task1"
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task1" / "training" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed task1 data from: {data_dir}")
    df_tab = pd.read_csv(data_dir / "clinical_tabular.csv")
    df_txt = pd.read_csv(data_dir / "clinical_texts.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task1.csv")

    labeled_count = df_gt["is_labeled"].sum()
    print(f"Total cohort size: {len(df_gt)}, Labeled cases: {labeled_count}")

    model = Subtask2DirectKDMClassifier(n_mri_pca=2, n_epochs=300, lr=0.01)

    print("Fitting canonical Scalar Direct-KDM classifier on all labeled cases...")
    model.fit(df_tab, df_txt, df_mri, df_gt)

    out_path = models_dir / "subtask2_kdm.pkl"
    joblib.dump(model, out_path)
    print(f"Saved self-contained model artifact to: {out_path}")


if __name__ == "__main__":
    main()
