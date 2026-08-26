#!/usr/bin/env python3
"""
Verification and Test Suite for Task 1 Deployed Models.
Loads all 3 serialized .pkl artifacts and validates prediction on raw case directories.
"""

import json
from pathlib import Path
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Note: importing model_definitions registers classes for pickle
import deploy.tasks.task1.training.model_definitions as model_definitions


def test_models_on_raw_cases():
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task1" / "training" / "models"
    raw_task1_dir = PROJECT_ROOT / "data" / "raw" / "train_release" / "task1"

    print(f"[*] Loading serialized models from: {models_dir}")
    m1 = joblib.load(models_dir / "subtask1_ensemble.pkl")
    m2 = joblib.load(models_dir / "subtask2_kdm.pkl")
    m3 = joblib.load(models_dir / "subtask3_extratrees.pkl")
    print("[+] All 3 models loaded into memory successfully.")

    # Select 2 sample test cases (1 labeled, 1 unlabeled or distinct)
    test_cases = [
        raw_task1_dir / "PT-pseudo_47d8b39fd07f",
        raw_task1_dir / "PT-pseudo_0020cfca66c8"
    ]

    for case_dir in test_cases:
        if not case_dir.exists():
            continue
        print("\n" + "=" * 80)
        print(f"TESTING END-TO-END INFERENCE ON RAW DIRECTORY: {case_dir.name}")
        print("=" * 80)

        # Subtask 1.1
        res1 = m1.predict_raw(case_dir)
        print("\n--- Subtask 1.1 Output (Decision & Neighbors) ---")
        print(f"Case ID: {res1['case_id']}")
        print(f"Decision: {res1['decision']} (Prob: {res1['biopsy_probability']})")
        print(f"Modality Probs: {res1['modality_probabilities']}")
        print(f"Total Unique Neighbors: {res1['neighbors_union']['total_unique_neighbors']}")
        print(f"Neighbor PIDs: {res1['neighbors_union']['neighbor_pids']}")
        print("Neighbor Free Texts (Reasoning):")
        for i, txt in enumerate(res1['neighbors_union']['reasoning_free_texts'], 1):
            print(f"  [{i}] {txt}")

        # Subtask 1.2
        res2 = m2.predict_raw(case_dir)
        print("\n--- Subtask 1.2 Output (Clinical Confidence) ---")
        print(f"Confidence: {res2['confidence']} (Score: {res2['confidence_score']})")
        print(f"Class Probs: {res2['class_probabilities']}")

        # Subtask 1.3
        res3 = m3.predict_raw(case_dir)
        print("\n--- Subtask 1.3 Output (Variable Weights & Decisive Factors) ---")
        print(f"Variable Weights: {res3['variable_weights']}")
        print(f"Decisive Factors: {res3['decisive_factors']}")
        print(f"Important Factors: {res3['important_factors']}")


if __name__ == "__main__":
    test_models_on_raw_cases()
