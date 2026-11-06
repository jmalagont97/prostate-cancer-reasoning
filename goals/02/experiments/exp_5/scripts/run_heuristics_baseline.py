#!/usr/bin/env python3
"""run_heuristics_baseline.py

Execution script for Goal 02 exp_5:
Evaluates 3 deterministic clinical heuristic rules (H0, H1, H2) on Train (N=45) and Test (N=46).
Computes classification metrics, generates comparison figures, and renders the triangular
benchmark tradeoff plot (Heuristics vs. CART vs. Neuro-Symbolic).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def get_git_revision_hash(base_path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_git_hash"


def predict_h0(row: pd.Series) -> str:
    """H0: Naive clinical suspicion (PIRADS >= 4 or PSAD >= 0.15)."""
    try:
        pirads = float(row.get("radiology_report__pirads", 0))
    except (ValueError, TypeError):
        pirads = 0.0
    try:
        psad = float(row.get("radiology_report__psa_density", 0.0))
    except (ValueError, TypeError):
        psad = 0.0
    return "yes" if (pirads >= 4.0 or psad >= 0.15) else "no"


def predict_h1(row: pd.Series) -> str:
    """H1: Filtered suspicion (excludes patients with confirmed prior cancer)."""
    try:
        pirads = float(row.get("radiology_report__pirads", 0))
    except (ValueError, TypeError):
        pirads = 0.0
    try:
        psad = float(row.get("radiology_report__psa_density", 0.0))
    except (ValueError, TypeError):
        psad = 0.0

    bx = str(row.get("structured_prompt__bx", "")).strip().lower()
    try:
        prior_cancer = int(row.get("previous_notes__prior_cancer_confirmed", 0))
    except (ValueError, TypeError):
        prior_cancer = 0

    has_prior_cancer = (bx == "positive" or prior_cancer == 1)
    suspicion = (pirads >= 4.0 or psad >= 0.15)
    return "yes" if (suspicion and not has_prior_cancer) else "no"


def predict_h2(row: pd.Series) -> str:
    """H2: EAU stratified clinical guideline rule."""
    try:
        pirads = float(row.get("radiology_report__pirads", 0))
    except (ValueError, TypeError):
        pirads = 0.0
    try:
        psad = float(row.get("radiology_report__psa_density", 0.0))
    except (ValueError, TypeError):
        psad = 0.0

    bx = str(row.get("structured_prompt__bx", "")).strip().lower()
    try:
        prior_cancer = int(row.get("previous_notes__prior_cancer_confirmed", 0))
    except (ValueError, TypeError):
        prior_cancer = 0

    has_prior_cancer = (bx == "positive" or prior_cancer == 1)

    if pirads <= 2.0:
        return "no"
    elif pirads == 3.0:
        return "yes" if psad >= 0.15 else "no"
    else:  # pirads >= 4.0
        return "no" if has_prior_cancer else "yes"


def evaluate_heuristic(
    df: pd.DataFrame, func: Callable[[pd.Series], str], rule_name: str
) -> Dict[str, Any]:
    y_true = df["biopsy_decision"].tolist()
    y_pred = [func(row) for _, row in df.iterrows()]

    acc = float(accuracy_score(y_true, y_pred))
    f1_yes = float(f1_score(y_true, y_pred, pos_label="yes"))
    f1_no = float(f1_score(y_true, y_pred, pos_label="no"))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    sens = float(recall_score(y_true, y_pred, pos_label="yes"))
    spec = float(recall_score(y_true, y_pred, pos_label="no"))
    balanced_acc = float((sens + spec) / 2.0)

    cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "rule_name": rule_name,
        "total_samples": len(df),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_yes": round(f1_yes, 4),
        "f1_no": round(f1_no, 4),
        "sensitivity_tpr": round(sens, 4),
        "specificity_tnr": round(spec, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_labels": ["no", "yes"],
            "raw_matrix": cm.tolist(),
        },
    }


def main() -> None:
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent
    exp_dir = scripts_dir.parent
    goal_dir = exp_dir.parent.parent

    data_dir = goal_dir / "data"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    inputs_df = pd.read_csv(data_dir / "inputs.csv")
    outputs_df = pd.read_csv(data_dir / "outputs.csv")[["case_id", "biopsy_decision"]]
    design_df = pd.read_csv(data_dir / "experimental-design.csv")[["case_id", "split"]]

    merged = inputs_df.merge(outputs_df, on="case_id").merge(design_df, on="case_id")

    train_df = merged[merged["split"] == "train"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    assert len(train_df) == 45, f"Expected 45 train samples, found {len(train_df)}"
    assert len(test_df) == 46, f"Expected 46 test samples, found {len(test_df)}"

    heuristics = [
        ("H0_Naive_Suspicion", predict_h0),
        ("H1_PriorBx_Filtered", predict_h1),
        ("H2_EAU_Stratified_Guideline", predict_h2),
    ]

    # 2. Evaluate
    train_results = {name: evaluate_heuristic(train_df, func, name) for name, func in heuristics}
    test_results = {name: evaluate_heuristic(test_df, func, name) for name, func in heuristics}

    # Save JSON results
    with open(results_dir / "heuristics_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2)

    with open(results_dir / "heuristics_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # Version tracking
    git_hash = get_git_revision_hash(goal_dir)
    version_content = (
        f"Git Revision: {git_hash}\n"
        f"Date: {datetime.now().isoformat()}\n"
        f"Model: Deterministic Clinical Heuristics (H0, H1, H2)\n"
        f"Train samples: {len(train_df)}\n"
        f"Test samples: {len(test_df)}\n"
    )
    with open(results_dir / "version.txt", "w", encoding="utf-8") as f:
        f.write(version_content)

    # 3. Figures
    # Figure 1: Comparison Bar Chart on Test
    metrics_keys = ["accuracy", "macro_f1", "sensitivity_tpr", "specificity_tnr"]
    metric_labels = ["Accuracy", "Macro-F1", "Sensitivity", "Specificity"]

    x = np.arange(len(metric_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]

    for idx, (h_name, _) in enumerate(heuristics):
        res = test_results[h_name]
        vals = [res[k] for k in metrics_keys]
        pos = x + (idx - 1) * width
        rects = plt.bar(pos, vals, width, label=h_name.replace("_", " "), color=colors[idx])
        for r in rects:
            h = r.get_height()
            plt.text(
                r.get_x() + r.get_width() / 2.0,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    plt.ylim([0, 1.18])
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Deterministic Clinical Heuristics — Out-of-Sample Test Evaluation (N = 46)", fontsize=12, fontweight="bold")
    plt.xticks(x, metric_labels, fontsize=10, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "heuristics_comparison_test.png", dpi=300)
    plt.close()

    # Figure 2: Confusion Matrices Panel 1x3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_cmaps = [plt.cm.Oranges, plt.cm.Greens, plt.cm.Blues]

    for idx, (h_name, _) in enumerate(heuristics):
        ax = axes[idx]
        res = test_results[h_name]
        cm = np.array(res["confusion_matrix"]["raw_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=cm_cmaps[idx])
        ax.set_title(f"{h_name.replace('_', ' ')}\nAcc: {res['accuracy']*100:.1f}% | Spec: {res['specificity_tnr']*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Biopsy", "Biopsy"], fontsize=10)
        ax.set_yticklabels(["No Biopsy", "Biopsy"], fontsize=10)
        ax.set_xlabel("Heuristic Prediction", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Ground Truth Clinician", fontsize=10, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = (val / 46) * 100
                ax.text(
                    j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=11, fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices_heuristics.png", dpi=300)
    plt.close()

    # Figure 3: Triangular Benchmark Tradeoff (Heuristics vs. CART vs. Neuro-Symbolic)
    plt.figure(figsize=(9, 7))

    # Points data: (Sensitivity, Specificity, Label, Color, Marker, Size)
    benchmark_points = [
        (test_results["H0_Naive_Suspicion"]["sensitivity_tpr"], test_results["H0_Naive_Suspicion"]["specificity_tnr"], "H0: Naive Suspicion (PIRADS>=4 | PSAD>=0.15)", "#ff7f0e", "o", 140),
        (test_results["H1_PriorBx_Filtered"]["sensitivity_tpr"], test_results["H1_PriorBx_Filtered"]["specificity_tnr"], "H1: Prior Bx Filtered", "#2ca02c", "s", 140),
        (test_results["H2_EAU_Stratified_Guideline"]["sensitivity_tpr"], test_results["H2_EAU_Stratified_Guideline"]["specificity_tnr"], "H2: EAU Stratified Guideline", "#17becf", "^", 140),
        (0.8571, 0.6111, "exp_4: Unconstrained CART (scikit-learn)", "#d62728", "D", 150),
        (0.8571, 0.7778, "exp_3: Neuro-Symbolic Agentic Tree (Ours)", "#9467bd", "*", 260),
    ]

    for sens, spec, label, color, marker, size in benchmark_points:
        plt.scatter(spec, sens, color=color, marker=marker, s=size, label=label, zorder=5, edgecolor="black", linewidth=1.2)
        # Add text callout
        offset_x = 0.02
        offset_y = 0.02
        if "H0" in label:
            offset_y = -0.04
        elif "H1" in label or "H2" in label:
            offset_x = -0.22
            offset_y = -0.05
        elif "CART" in label:
            offset_y = -0.04
        elif "Neuro-Symbolic" in label:
            offset_x = -0.15
            offset_y = 0.035

        plt.annotate(
            label.split(":")[0],
            (spec, sens),
            xytext=(spec + offset_x, sens + offset_y),
            fontsize=10,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    # Reference lines and ideal quadrant
    plt.axvline(0.70, color="gray", linestyle=":", alpha=0.6)
    plt.axhline(0.80, color="gray", linestyle=":", alpha=0.6)
    plt.fill_between([0.70, 1.02], 0.80, 1.02, color="#9467bd", alpha=0.08, label="Clinical Target Zone (High Sens & Spec)")

    plt.xlim([-0.02, 1.05])
    plt.ylim([0.45, 1.05])
    plt.xlabel("Specificity (1 - Overbiopsy Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Sensitivity (Cancer Detection Recall)", fontsize=11, fontweight="bold")
    plt.title("Tripartite Benchmark Tradeoff: Clinical Heuristics vs. CART vs. Neuro-Symbolic\nOut-of-Sample Test Evaluation (N = 46)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "triangular_benchmark_tradeoff.png", dpi=300)
    plt.close()

    # Telemetry
    print(f"\n{'='*70}\n[Heuristics Baseline Execution Complete]")
    for h_name, _ in heuristics:
        tr = train_results[h_name]
        te = test_results[h_name]
        print(f"\n--- {h_name} ---")
        print(f"Train: Acc = {tr['accuracy']:.4f} | Macro-F1 = {tr['macro_f1']:.4f} | Sens = {tr['sensitivity_tpr']:.4f} | Spec = {tr['specificity_tnr']:.4f}")
        print(f"Test:  Acc = {te['accuracy']:.4f} | Macro-F1 = {te['macro_f1']:.4f} | Sens = {te['sensitivity_tpr']:.4f} | Spec = {te['specificity_tnr']:.4f}")
        print(f"Confusion Matrix (Test): {te['confusion_matrix']['raw_matrix']}")
    print(f"\nSaved artifacts to {results_dir.relative_to(goal_dir)}")
    print(f"Saved figures to {figures_dir.relative_to(goal_dir)}")


if __name__ == "__main__":
    main()
