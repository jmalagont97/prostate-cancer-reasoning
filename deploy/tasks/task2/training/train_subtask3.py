#!/usr/bin/env python3
"""
Canonical Training Script for Task 2 Subtask 2.3: Variable Weights Ensemble.
Serializes Subtask3MultiVariableEnsemble into deploy/tasks/task2/training/models/subtask3_extratrees.pkl
"""

import json
from pathlib import Path
import sys
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tasks.task2.model_definitions import Subtask3MultiVariableEnsemble


def load_var_weights(raw_dir: Path, pids: list) -> dict:
    var_weights_map = {}
    for pid in pids:
        f = raw_dir / pid / "prostate-treatment-decision-reasoning.json"
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                vw = data.get("variable_weights", {})
                var_weights_map[pid] = vw
            except Exception:
                var_weights_map[pid] = {}
        else:
            var_weights_map[pid] = {}
    return var_weights_map


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
    var_weights_map = load_var_weights(raw_dir, pids)
    print(f"Loaded ground truth variable weights for {len(var_weights_map)} cases.")

    model = Subtask3MultiVariableEnsemble()

    print("Fitting canonical 11 Meta-Classifiers on joint 98D multimodal space...")
    model.fit(df_tab, df_txt, df_mri, var_weights_map)

    out_path = models_dir / "subtask3_extratrees.pkl"
    joblib.dump(model, out_path)
    print(f"[+] Saved Subtask 2.3 model artifact to: {out_path}")


if __name__ == "__main__":
    main()
