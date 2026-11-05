#!/usr/bin/env python3
"""evaluate_test.py

Out-of-sample blind evaluation script for Goal 02 exp_3.
Evaluates the induced model (results/artifacts/model.py) strictly on Test partition (N=46).
Computes:
- Accuracy
- Curva ROC & AUC-ROC
- F1-yes, F1-no, Macro-F1
- Sensitivity (Recall yes) & Specificity (Recall no)
- Confusion Matrix

Generates diagnostic plots in reports/figures/ and saves results/test_metrics.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_test_set() -> Dict[str, Any]:
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent
    exp_dir = scripts_dir.parent
    goal_dir = exp_dir.parent.parent

    data_dir = goal_dir / "data"
    results_dir = exp_dir / "results"
    artifacts_dir = results_dir / "artifacts"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_py_path = artifacts_dir / "model.py"
    if not model_py_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_py_path}")

    # Load dynamic model.py
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for {model_py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict_biopsy = getattr(mod, "predict_biopsy")

    # Load test dataset
    inputs_df = pd.read_csv(data_dir / "inputs.csv")
    outputs_df = pd.read_csv(data_dir / "outputs.csv")[["case_id", "biopsy_decision"]]
    design_df = pd.read_csv(data_dir / "experimental-design.csv")[["case_id", "split"]]

    merged = inputs_df.merge(outputs_df, on="case_id").merge(design_df, on="case_id")
    test_df = merged[merged["split"] == "test"].copy()

    total_test = len(test_df)
    if total_test != 46:
        print(f"Warning: Expected 46 test samples, found {total_test}.")

    y_true = test_df["biopsy_decision"].tolist()
    y_pred = []
    y_prob = []

    for _, row in test_df.iterrows():
        pred_label, prob_yes = predict_biopsy(row.to_dict())
        y_pred.append(pred_label)
        y_prob.append(float(prob_yes))

    y_true_binary = [1 if y == "yes" else 0 for y in y_true]
    y_pred_binary = [1 if y == "yes" else 0 for y in y_pred]

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_yes = float(f1_score(y_true, y_pred, pos_label="yes"))
    f1_no = float(f1_score(y_true, y_pred, pos_label="no"))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    sensitivity = float(recall_score(y_true, y_pred, pos_label="yes"))
    specificity = float(recall_score(y_true, y_pred, pos_label="no"))

    # ROC & AUC
    # If all probabilities are identical (single leaf edge case), handle gracefully
    try:
        auc_roc = float(roc_auc_score(y_true_binary, y_prob))
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob)
    except Exception:
        auc_roc = 0.5
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])

    cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        "dataset_split": "test",
        "total_samples": total_test,
        "class_distribution": test_df["biopsy_decision"].value_counts().to_dict(),
        "accuracy": round(acc, 4),
        "auc_roc": round(auc_roc, 4),
        "f1_yes": round(f1_yes, 4),
        "f1_no": round(f1_no, 4),
        "macro_f1": round(macro_f1, 4),
        "sensitivity_tpr": round(sensitivity, 4),
        "specificity_tnr": round(specificity, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_labels": ["no", "yes"],
            "raw_matrix": cm.tolist(),
        },
    }

    # Save metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 1. Plot ROC Curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Neuro-Symbolic Tree (AUC = {auc_roc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curve — Out-of-Sample Test Evaluation (N = 46)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_fig_path = figures_dir / "roc_curve.png"
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()

    # 2. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Test Partition (N = 46)", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Biopsy", "Biopsy"], fontsize=11)
    plt.yticks(tick_marks, ["No Biopsy", "Biopsy"], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total_test) * 100
            plt.text(
                j, i, f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=12, fontweight="bold"
            )

    plt.ylabel("Ground Truth Clinician Label", fontsize=11, fontweight="bold")
    plt.xlabel("Neuro-Symbolic Tree Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_fig_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()

    # Telemetry
    print(f"\n{'='*70}\n[Test Evaluation Complete] Out-of-Sample Blind Evaluation (N = 46)")
    print(f"Accuracy:         {acc * 100:.2f}% ({tp + tn}/{total_test})")
    print(f"AUC-ROC:          {auc_roc:.4f}")
    print(f"Macro-F1:         {macro_f1:.4f}")
    print(f"F1-Score (Yes):   {f1_yes:.4f}")
    print(f"F1-Score (No):    {f1_no:.4f}")
    print(f"Sensitivity (TPR):{sensitivity * 100:.2f}% ({tp}/{tp + fn})")
    print(f"Specificity (TNR):{specificity * 100:.2f}% ({tn}/{tn + fp})")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved artifacts:")
    print(f"  Metrics JSON: {metrics_path.relative_to(goal_dir)}")
    print(f"  ROC Curve:    {roc_fig_path.relative_to(goal_dir)}")
    print(f"  Confusion:    {cm_fig_path.relative_to(goal_dir)}")

    return metrics


if __name__ == "__main__":
    evaluate_test_set()
