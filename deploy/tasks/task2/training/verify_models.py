#!/usr/bin/env python3
"""
Verification and Test Suite for Task 2 Deployed Models.
Loads all 3 serialized .pkl artifacts and validates end-to-end prediction on raw case directories (T2-XXX).
"""

from pathlib import Path
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register classes for joblib unpickling
import deploy.tasks.task2.model_definitions as model_definitions


def test_models_on_raw_cases():
    models_dir = PROJECT_ROOT / "deploy" / "tasks" / "task2" / "training" / "models"
    raw_task2_dir = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"

    print(f"[*] Loading serialized models from: {models_dir}")
    m1 = joblib.load(models_dir / "subtask1_ensemble.pkl")
    m2 = joblib.load(models_dir / "subtask2_knn.pkl")
    m3 = joblib.load(models_dir / "subtask3_extratrees.pkl")
    print("[+] All 3 Task 2 models loaded into memory successfully.")

    # Select 2 sample test cases
    test_cases = [
        raw_task2_dir / "T2-004",
        raw_task2_dir / "T2-010"
    ]

    for case_dir in test_cases:
        if not case_dir.exists():
            continue
        print("\n" + "=" * 80)
        print(f"TESTING END-TO-END INFERENCE ON RAW DIRECTORY: {case_dir.name}")
        print("=" * 80)

        # Subtask 2.1
        res1 = m1.predict_raw(case_dir)
        print("\n--- Subtask 2.1 Output (Treatment Decision & Multimodal Neighbors) ---")
        print(f"Case ID: {res1['case_id']}")
        print(f"Decision: {res1['decision']}")
        print(f"Treatment Probs: {res1['treatment_probabilities']}")
        print(f"Modality Probs: {res1['modality_probabilities']}")
        print(f"Total Unique Neighbors: {res1['neighbors_union']['total_unique_neighbors']}")
        print(f"Neighbor PIDs: {res1['neighbors_union']['neighbor_pids']}")
        print("Neighbor Free Texts (Reasoning):")
        for i, txt in enumerate(res1['neighbors_union']['reasoning_free_texts'], 1):
            print(f"  [{i}] ({res1['neighbors_union']['neighbor_pids'][i-1]}): {txt[:120]}...")

        # Subtask 2.2
        res2 = m2.predict_raw(case_dir)
        print("\n--- Subtask 2.2 Output (Clinical Confidence) ---")
        print(f"Confidence: {res2['confidence']} (Ordinal Score: {res2['confidence_score']})")
        print(f"Class Probs: {res2['class_probabilities']}")

        # Subtask 2.3
        res3 = m3.predict_raw(case_dir)
        print("\n--- Subtask 2.3 Output (Variable Weights & Decisive Factors) ---")
        print(f"Variable Weights: {res3['variable_weights']}")
        print(f"Decisive Factors: {res3['decisive_factors']}")
        print(f"Important Factors: {res3['important_factors']}")


if __name__ == "__main__":
    test_models_on_raw_cases()
