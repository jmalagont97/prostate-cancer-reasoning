#!/usr/bin/env python3
"""
Canonical Training Script for Task 2 Subtask 2.1: Treatment Decision Gate Ensemble.
Serializes Subtask1FuzzySVREnsemble into deploy/tasks/task2/training/models/subtask1_ensemble.pkl
"""

import json
from pathlib import Path
import sys
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task2.model_definitions import Subtask1FuzzySVREnsemble


def load_reasonings_map(raw_dir: Path, pids: list) -> dict:
    reasonings = {}
    for pid in pids:
        reason_file = raw_dir / pid / "prostate-treatment-decision-reasoning.json"
        if reason_file.exists():
            try:
                with open(reason_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                txt = data.get("free_text", data.get("reasoning", data.get("clinical_reasoning", "")))
                reasonings[pid] = txt if txt else "Reasoning not found in file."
            except Exception:
                reasonings[pid] = "Error reading reasoning JSON."
        else:
            reasonings[pid] = "Reasoning file does not exist."
    return reasonings


def main():
    data_dir = PROJECT_ROOT / "data" / "processed" / "task2"
    raw_dir = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task2" / "training" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed Task 2 data from: {data_dir}")
    df_tab = pd.read_csv(data_dir / "clinical_tabular_task2.csv")
    df_txt = pd.read_csv(data_dir / "clinical_texts_task2.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings_task2.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task2.csv")

    pids = df_gt["pid"].tolist()
    print(f"Total cohort size: {len(df_gt)}")

    reasonings_map = load_reasonings_map(raw_dir, pids)
    print(f"Extracted clinical reasoning text for {len(reasonings_map)} patients.")

    # exp_17 champion weights: Tab=0.50, MRI=0.05, Text=0.45
    model = Subtask1FuzzySVREnsemble(
        weight_tab=0.50,
        weight_mri=0.05,
        weight_txt=0.45,
        k_neighbors=3,
    )

    print("Fitting canonical Fuzzy-SVR ensemble on all 70 Task 2 cases...")
    model.fit(df_tab, df_txt, df_mri, df_gt, reasonings_map=reasonings_map)

    out_path = models_dir / "subtask1_ensemble.pkl"
    joblib.dump(model, out_path)
    print(f"[+] Saved Subtask 2.1 model artifact to: {out_path}")


if __name__ == "__main__":
    main()
