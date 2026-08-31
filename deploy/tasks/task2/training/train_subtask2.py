#!/usr/bin/env python3
"""
Canonical Training Script for Task 2 Subtask 2.2: Clinical Confidence Fuzzy-KNN.
Serializes Subtask2FuzzyKNNClassifier into deploy/tasks/task2/training/models/subtask2_knn.pkl
"""

from pathlib import Path
import sys
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task2.model_definitions import Subtask2FuzzyKNNClassifier


def main():
    data_dir = PROJECT_ROOT / "data" / "processed" / "task2"
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task2" / "training" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed Task 2 data from: {data_dir}")
    df_tab = pd.read_csv(data_dir / "clinical_tabular_task2.csv")
    df_txt = pd.read_csv(data_dir / "clinical_texts_task2.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings_task2.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task2.csv")

    print(f"Total cases for confidence training: {len(df_gt)}")

    # exp_12 champion configuration: MRI=0.55, Text=0.45
    model = Subtask2FuzzyKNNClassifier(
        weight_tab=0.0,
        weight_mri=0.55,
        weight_txt=0.45,
    )

    print("Fitting canonical Multimodal Fuzzy-KNN classifier on all 70 cases...")
    model.fit(df_tab, df_txt, df_mri, df_gt)

    out_path = models_dir / "subtask2_knn.pkl"
    joblib.dump(model, out_path)
    print(f"[+] Saved Subtask 2.2 model artifact to: {out_path}")


if __name__ == "__main__":
    main()
