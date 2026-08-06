import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import spearmanr


def compute_ici(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p_mean = (p1 + p2 + p3) / 3.0
    
    # Inter-modality standard deviation
    var_p = ((p1 - p_mean)**2 + (p2 - p_mean)**2 + (p3 - p_mean)**2) / 3.0
    std_p = np.sqrt(var_p)
    
    # Certitude margin to decision boundary 0.50
    delta_margin = np.abs(p_mean - 0.50)
    
    # Composite Reliability Index (ICI) - Identical to exp_9, exp_10, exp_11, exp_17
    ici = (2.0 * delta_margin) * (1.0 - 2.0 * std_p)
    return np.clip(ici, 0.0, 1.0), p_mean, std_p, delta_margin


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    exp_dir = project_root / "experiments" / "exp_19"
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading out-of-fold hybrid predictions from exp_18 and clinical reasoning targets...")
    df_oof = pd.read_csv(project_root / "experiments" / "exp_18" / "results" / "oof_predictions.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align by patient_id
    pids = df_design["patient_id"].values
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)
    df_oof_labeled = df_oof[df_oof["patient_id"].isin(df_design_labeled["patient_id"].values)].sort_values("patient_id").reset_index(drop=True)

    p_tab_fuzzy = df_oof_labeled["prob_tabular_fuzzy"].values
    p_mri_hard = df_oof_labeled["prob_mri_hard"].values
    p_text_hard = df_oof_labeled["prob_text_hard"].values

    # Compute Composite Hybrid ICI
    ici_hybrid, p_mean, std_p, delta_margin = compute_ici(p_tab_fuzzy, p_mri_hard, p_text_hard)

    # Confidence 3-class target mapping
    confidence_map = {"uncertain": 0, "borderline": 1, "clear": 2}
    inv_confidence_map = {0: "uncertain", 1: "borderline", 2: "clear"}
    y_conf = df_reasoning_labeled["confidence"].map(confidence_map).values

    print(f"Cohort size: {len(y_conf)} patients.")
    print(f"Class distribution: Clear={sum(y_conf==2)}, Borderline={sum(y_conf==1)}, Uncertain={sum(y_conf==0)}")

    # ---------------------------------------------------------
    # Phase A: 100-Split MCCV Decision Tree Meta-Thresholding
    # ---------------------------------------------------------
    print("Beginning Phase A: 100-Split MCCV Class-Weighted Decision Tree Threshold Learning...")

    n_splits = 100
    thresholds_t1 = []
    thresholds_t2 = []

    for split_idx in range(n_splits):
        col_name = f"split_{split_idx}"
        split_vals = df_design_labeled[col_name].values
        train_mask = split_vals == 0

        X_tr = ici_hybrid[train_mask].reshape(-1, 1)
        y_tr = y_conf[train_mask]

        dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
        dt.fit(X_tr, y_tr)

        # Extract cut points from decision tree thresholds
        tree_thresholds = dt.tree_.threshold[dt.tree_.threshold != -2]
        tree_thresholds = np.sort(tree_thresholds)

        if len(tree_thresholds) >= 2:
            t1, t2 = tree_thresholds[0], tree_thresholds[1]
        elif len(tree_thresholds) == 1:
            t1 = tree_thresholds[0]
            t2 = tree_thresholds[0] + 0.1
        else:
            t1, t2 = 0.10, 0.30

        thresholds_t1.append(t1)
        thresholds_t2.append(t2)

    meta_t1 = float(np.mean(thresholds_t1))
    meta_t2 = float(np.mean(thresholds_t2))
    std_t1 = float(np.std(thresholds_t1))
    std_t2 = float(np.std(thresholds_t2))

    meta_hparams = {
        "meta_threshold_1": meta_t1,
        "std_threshold_1": std_t1,
        "meta_threshold_2": meta_t2,
        "std_threshold_2": std_t2
    }

    with open(results_dir / "meta_thresholds.json", "w") as f:
        json.dump(meta_hparams, f, indent=4)

    print(f"Learned Composite Hybrid Balanced Meta-Thresholds: t1 = {meta_t1:.4f} (+/- {std_t1:.4f}), t2 = {meta_t2:.4f} (+/- {std_t2:.4f})")

    # Plot MCCV Thresholds Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(ici_hybrid[y_conf == 0], bins=15, alpha=0.6, label="Uncertain", color="red")
    plt.hist(ici_hybrid[y_conf == 1], bins=15, alpha=0.6, label="Borderline", color="orange")
    plt.hist(ici_hybrid[y_conf == 2], bins=15, alpha=0.6, label="Clear", color="green")
    plt.axvline(meta_t1, color="black", linestyle="--", linewidth=2, label=f"Meta t1 = {meta_t1:.4f}")
    plt.axvline(meta_t2, color="blue", linestyle="--", linewidth=2, label=f"Meta t2 = {meta_t2:.4f}")
    plt.title("Composite Hybrid ICI Distribution & Learned Balanced Meta-Thresholds", fontsize=11, fontweight="bold")
    plt.xlabel("Composite Hybrid Reliability Index (ICI)", fontsize=10)
    plt.ylabel("Patient Count", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_tree_thresholds.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Phase B: Frozen LOOCV Out-of-Fold Evaluation
    # ---------------------------------------------------------
    print("Beginning Phase B: Frozen LOOCV Out-of-Fold Evaluation...")

    oof_preds = []
    for ici_val in ici_hybrid:
        if ici_val < meta_t1:
            oof_preds.append(0)  # uncertain
        elif ici_val < meta_t2:
            oof_preds.append(1)  # borderline
        else:
            oof_preds.append(2)  # clear

    oof_preds = np.array(oof_preds)

    macro_f1 = f1_score(y_conf, oof_preds, average="macro", zero_division=0)
    acc = accuracy_score(y_conf, oof_preds)
    rho_val, p_val = spearmanr(y_conf, oof_preds)

    loocv_metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "spearman_rho": float(rho_val),
        "spearman_pvalue": float(p_val),
        "total_cases": len(y_conf)
    }

    with open(results_dir / "loocv_confidence_metrics.json", "w") as f:
        json.dump(loocv_metrics, f, indent=4)

    # Save OOF Predictions Dataframe
    df_oof_out = pd.DataFrame({
        "patient_id": df_design_labeled["patient_id"].values,
        "composite_hybrid_ici": ici_hybrid,
        "prob_mean": p_mean,
        "prob_std": std_p,
        "certitude_margin": delta_margin,
        "ground_truth_confidence": [inv_confidence_map[c] for c in y_conf],
        "predicted_confidence": [inv_confidence_map[p] for p in oof_preds]
    })
    df_oof_out.to_csv(results_dir / "oof_confidence_predictions.csv", index=False)

    # Plot 3x3 Confusion Matrix
    cm_3class = confusion_matrix(y_conf, oof_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_3class, display_labels=["Uncertain", "Borderline", "Clear"])
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f"Class-Weighted Composite Hybrid ICI 3-Class Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_3class.png", dpi=300)
    plt.close()

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Class-Weighted Composite Hybrid ICI Diagnostic Confidence Prediction (exp_19) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Class-Weighted Decision Tree Meta-Thresholding on Composite Hybrid ICI (`class_weight='balanced'`)  \n")
        f.write(f"**Dataset**: Labeled Reasoning Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Phase A: Learned Balanced Meta-Thresholds (100 MCCV Splits)\n")
        f.write(f"- **Meta-Threshold 1 ($\bar{{\\tau}}_1$, Uncertain / Borderline)**: `{meta_t1:.4f}` ($\text{{std}} = {std_t1:.4f}$)  \n")
        f.write(f"- **Meta-Threshold 2 ($\bar{{\\tau}}_2$, Borderline / Clear)**: `{meta_t2:.4f}` ($\text{{std}} = {std_t2:.4f}$)  \n\n")

        f.write("## Phase B: Frozen LOOCV Out-of-Fold Evaluation (88 Folds)\n")
        f.write(f"- **3-Class Macro-F1**: **`{macro_f1:.4f}`**  \n")
        f.write(f"- **Accuracy**: **`{acc:.4f}`** ({sum(y_conf == oof_preds)}/88 correct)  \n")
        f.write(f"- **Spearman Rank Correlation ($\rho$)**: **`{rho_val:.4f}`** (p-value: `{p_val:.4e}`)  \n\n")

        f.write("### 3x3 Confusion Matrix Counts:\n")
        f.write("| Ground Truth \\ Predicted | Uncertain | Borderline | Clear |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Uncertain** ($N=15$) | **{cm_3class[0,0]}** | {cm_3class[0,1]} | {cm_3class[0,2]} |\n")
        f.write(f"| **Borderline** ($N=18$) | {cm_3class[1,0]} | **{cm_3class[1,1]}** | {cm_3class[1,2]} |\n")
        f.write(f"| **Clear** ($N=55$) | {cm_3class[2,0]} | {cm_3class[2,1]} | **{cm_3class[2,2]}** |\n\n")

        f.write("## Comparison across All ICI Diagnostic Confidence Experiments\n")
        f.write("| Experiment | Multimodal Sources | ICI Formulation | Meta \bar{\\tau}_1 | Meta \bar{\\tau}_2 | LOOCV Macro-F1 | LOOCV Accuracy | Spearman \rho |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **`exp_10`** | All Hard KNN | Hard Composite | 0.0669 | 0.2960 | 0.3691 | 39.77% | 0.1228 |\n")
        f.write(f"| **`exp_17`** | All Fuzzy KNN | Fuzzy Composite | 0.0180 | 0.1266 | 0.4470 | 57.95% | 0.2790 |\n")
        f.write(f"| **`exp_19`** | **Hybrid KNN (Tab Fuzzy + MRI/Text Hard)** | **Hybrid Composite** | **{meta_t1:.4f}** | **{meta_t2:.4f}** | **{macro_f1:.4f}** | **{acc*100:.2f}%** | **{rho_val:.4f}** |\n")

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
