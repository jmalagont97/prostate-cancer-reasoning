#!/usr/bin/env python3
"""run_cart_baseline.py

Execution script for Goal 02 exp_4:
Trains an unconstrained classical CART Decision Tree on the canonical Train partition (N=45)
using the 10 Ground Truth canonical features.
Evaluates both on Train (N=45) and blind out-of-sample on Test (N=46).
Generates metrics, serializes model artifacts, and renders diagnostic plots.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FEATURES = [
    "radiology_report__pirads",
    "radiology_report__psa_density",
    "structured_prompt__bx",
    "laboratory_results__psa",
    "structured_prompt__age",
    "structured_prompt__dre",
    "clinical_data__family_history",
    "clinical_data__cspca",
    "radiology_report__prostate_volume",
    "previous_notes__prior_cancer_confirmed",
]

CATEGORICAL_FEATURES = [
    "structured_prompt__bx",
    "structured_prompt__dre",
    "clinical_data__family_history",
]

NUMERICAL_FEATURES = [
    "radiology_report__pirads",
    "radiology_report__psa_density",
    "laboratory_results__psa",
    "structured_prompt__age",
    "clinical_data__cspca",
    "radiology_report__prostate_volume",
    "previous_notes__prior_cancer_confirmed",
]


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


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, split_name: str
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    f1_yes = float(f1_score(y_true, y_pred, pos_label=1))
    f1_no = float(f1_score(y_true, y_pred, pos_label=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    sens = float(recall_score(y_true, y_pred, pos_label=1))
    spec = float(recall_score(y_true, y_pred, pos_label=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "dataset_split": split_name,
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "f1_yes": round(f1_yes, 4),
        "f1_no": round(f1_no, 4),
        "macro_f1": round(macro_f1, 4),
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
    artifacts_dir = results_dir / "artifacts"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical datasets
    inputs_df = pd.read_csv(data_dir / "inputs.csv")
    outputs_df = pd.read_csv(data_dir / "outputs.csv")[["case_id", "biopsy_decision"]]
    design_df = pd.read_csv(data_dir / "experimental-design.csv")[["case_id", "split"]]

    merged = inputs_df.merge(outputs_df, on="case_id").merge(design_df, on="case_id")

    train_df = merged[merged["split"] == "train"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    assert len(train_df) == 45, f"Expected 45 train samples, found {len(train_df)}"
    assert len(test_df) == 46, f"Expected 46 test samples, found {len(test_df)}"

    # 2. Preprocess data (impute missing categories as 'Missing')
    for cat_col in CATEGORICAL_FEATURES:
        train_df[cat_col] = train_df[cat_col].fillna("Missing").astype(str)
        test_df[cat_col] = test_df[cat_col].fillna("Missing").astype(str)

    X_train_raw = train_df[FEATURES]
    y_train = (train_df["biopsy_decision"] == "yes").astype(int).values

    X_test_raw = test_df[FEATURES]
    y_test = (test_df["biopsy_decision"] == "yes").astype(int).values

    # 3. Fit preprocessor strictly on Train
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()
    all_feature_names = cat_feature_names + NUMERICAL_FEATURES

    # 4. Fit unconstrained CART Decision Tree
    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 5. Inferences
    y_train_pred = clf.predict(X_train)
    y_train_prob = clf.predict_proba(X_train)[:, 1]

    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    # 6. Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob, "train")
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, "test")

    # Add tree topology details
    tree_depth = int(clf.get_depth())
    tree_leaves = int(clf.get_n_leaves())
    train_metrics["tree_depth"] = tree_depth
    train_metrics["tree_leaves"] = tree_leaves
    test_metrics["tree_depth"] = tree_depth
    test_metrics["tree_leaves"] = tree_leaves

    # Save metrics JSON
    with open(results_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 7. Serialize artifacts
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "classifier": clf,
        "feature_names": all_feature_names,
    }
    joblib.dump(pipeline_artifact, artifacts_dir / "cart_model.joblib")

    # Text rules
    rules_text = export_text(clf, feature_names=all_feature_names)
    with open(artifacts_dir / "tree_rules.txt", "w", encoding="utf-8") as f:
        f.write(rules_text)

    # Version tracking
    git_hash = get_git_revision_hash(goal_dir)
    version_content = (
        f"Git Revision: {git_hash}\n"
        f"Date: {datetime.now().isoformat()}\n"
        f"Model: DecisionTreeClassifier (unconstrained CART)\n"
        f"Scikit-learn random_state: 42\n"
        f"Tree Depth: {tree_depth}\n"
        f"Leaf Nodes: {tree_leaves}\n"
    )
    with open(results_dir / "version.txt", "w", encoding="utf-8") as f:
        f.write(version_content)

    # 8. Diagnostic figures
    # Figure 1: ROC Curve on Test
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="#d62728",
        lw=2.5,
        label=f"Unconstrained CART (AUC = {test_metrics['auc_roc']:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="grey",
        lw=1.5,
        linestyle="--",
        label="Random Chance (AUC = 0.500)",
    )
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(
        "ROC Curve — CART Baseline Out-of-Sample Test (N = 46)",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Figure 2: Confusion Matrix on Test
    cm = np.array(test_metrics["confusion_matrix"]["raw_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    plt.title("Confusion Matrix — CART Baseline (Test N = 46)", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Biopsy", "Biopsy"], fontsize=11)
    plt.yticks(tick_marks, ["No Biopsy", "Biopsy"], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / 46) * 100
            plt.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.ylabel("Ground Truth Clinician Label", fontsize=11, fontweight="bold")
    plt.xlabel("CART Baseline Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Figure 3: Full Tree Structure
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=all_feature_names,
        class_names=["No Biopsy", "Biopsy"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title(
        f"Unconstrained CART Decision Tree Topology (Depth: {tree_depth}, Leaves: {tree_leaves})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_tree_structure.png", dpi=300)
    plt.close()

    # Figure 4: Train vs Test Overfitting Comparison
    metrics_to_plot = [
        ("Accuracy", train_metrics["accuracy"], test_metrics["accuracy"]),
        ("AUC-ROC", train_metrics["auc_roc"], test_metrics["auc_roc"]),
        ("Macro-F1", train_metrics["macro_f1"], test_metrics["macro_f1"]),
        ("Sensitivity", train_metrics["sensitivity_tpr"], test_metrics["sensitivity_tpr"]),
        ("Specificity", train_metrics["specificity_tnr"], test_metrics["specificity_tnr"]),
    ]

    labels = [m[0] for m in metrics_to_plot]
    train_vals = [m[1] for m in metrics_to_plot]
    test_vals = [m[2] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, train_vals, width, label="Train (N = 45)", color="#1f77b4")
    plt.bar(x + width / 2, test_vals, width, label="Test (N = 46)", color="#d62728")

    for i in range(len(labels)):
        plt.text(
            x[i] - width / 2,
            train_vals[i] + 0.02,
            f"{train_vals[i]:.2f}",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
        plt.text(
            x[i] + width / 2,
            test_vals[i] + 0.02,
            f"{test_vals[i]:.2f}",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    plt.ylim([0, 1.15])
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title(
        "Empirical Overfitting Gap: Train (100% Memorization) vs. Out-of-Sample Test",
        fontsize=12,
        fontweight="bold",
    )
    plt.xticks(x, labels, fontsize=10, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "train_vs_test_overfitting.png", dpi=300)
    plt.close()

    # Telemetry
    print(f"\n{'='*70}\n[CART Baseline Execution Complete]")
    print(f"Tree Topology: Depth = {tree_depth}, Total Leaves = {tree_leaves}")
    print(
        f"Train (N=45): Acc = {train_metrics['accuracy']:.4f} | AUC = {train_metrics['auc_roc']:.4f} | Macro-F1 = {train_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test  (N=46): Acc = {test_metrics['accuracy']:.4f} | AUC = {test_metrics['auc_roc']:.4f} | Macro-F1 = {test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test Sens: {test_metrics['sensitivity_tpr']:.4f} | Test Spec: {test_metrics['specificity_tnr']:.4f}"
    )
    print(f"Saved artifacts to {results_dir.relative_to(goal_dir)}")
    print(f"Saved figures to {figures_dir.relative_to(goal_dir)}")


if __name__ == "__main__":
    main()
