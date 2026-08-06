import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, roc_curve
from sklearn.model_selection import LeaveOneOut

# Define paths relative to this script
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]


def main():
    exp_dir = project_root / "experiments" / "exp_13"
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align datasets by patient_id
    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    # Filter labeled complete-case cohort (N=88) matching exp_5
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    pids_labeled = df_dec_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values

    # Construct uncertainty-guided soft targets (\tilde{y}_j)
    # clear -> c_j = 1.00, borderline -> c_j = 0.50, uncertain -> c_j = 0.25
    confidence_certainty_map = {
        "clear": 1.00,
        "borderline": 0.50,
        "uncertain": 0.25
    }
    c_weights = df_reasoning_labeled["confidence"].map(confidence_certainty_map).fillna(1.00).values

    # Soft target formula:
    # y = 1 -> \tilde{y} = 0.50 + 0.50 * c_j
    # y = 0 -> \tilde{y} = 0.50 - 0.50 * c_j
    y_soft = np.where(y_binary == 1, 0.50 + 0.50 * c_weights, 0.50 - 0.50 * c_weights).astype(np.float32)

    print(f"Total labeled cases for Fuzzy KNN: {len(y_binary)} (Biopsy positive: {sum(y_binary==1)}, Biopsy negative: {sum(y_binary==0)})")

    num_cols = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols = ["dre"]

    # Grid search parameters (identical to exp_5)
    k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25]
    weights_list = ["uniform", "distance"]
    metrics_list = ["euclidean", "manhattan", "cosine"]

    grid = []
    for k in k_list:
        for w in weights_list:
            for m in metrics_list:
                grid.append({"k": k, "weights": w, "metric": m})

    print(f"Total grid search configurations: {len(grid)}")

    # ---------------------------------------------------------
    # Phase A: 100-Split MCCV Grid Search
    # ---------------------------------------------------------
    print("Beginning Phase A: 100-Split MCCV Grid Search...")

    n_splits = 100
    grid_scores = {i: {"macro_f1": [], "acc": [], "sens": [], "spec": []} for i in range(len(grid))}

    for split_idx in range(n_splits):
        col_name = f"split_{split_idx}"
        split_vals = df_design_labeled[col_name].values  # 0 = train, 1 = val
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        # Preprocessing per split
        scaler = MinMaxScaler()
        X_tr_num = scaler.fit_transform(df_tab_labeled.iloc[train_idx][num_cols])
        X_va_num = scaler.transform(df_tab_labeled.iloc[val_idx][num_cols])

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_cat = ohe.fit_transform(df_tab_labeled.iloc[train_idx][cat_cols])
        X_va_cat = ohe.transform(df_tab_labeled.iloc[val_idx][cat_cols])

        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_va = np.hstack([X_va_num, X_va_cat])

        y_tr_soft = y_soft[train_idx]
        y_va_true = y_binary[val_idx]

        for cfg_idx, cfg in enumerate(grid):
            knn = KNeighborsRegressor(
                n_neighbors=cfg["k"],
                weights=cfg["weights"],
                metric=cfg["metric"]
            )
            knn.fit(X_tr, y_tr_soft)
            p_val_soft = knn.predict(X_va)
            y_val_pred = (p_val_soft >= 0.50).astype(int)

            macro_f1 = f1_score(y_va_true, y_val_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_va_true, y_val_pred)
            
            tn, fp, fn, tp = confusion_matrix(y_va_true, y_val_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            grid_scores[cfg_idx]["macro_f1"].append(macro_f1)
            grid_scores[cfg_idx]["acc"].append(acc)
            grid_scores[cfg_idx]["sens"].append(sens)
            grid_scores[cfg_idx]["spec"].append(spec)

        if (split_idx + 1) % 25 == 0:
            print(f"Completed MCCV splits: {split_idx + 1}/100.")

    # Aggregate MCCV results
    results_rows = []
    for cfg_idx, cfg in enumerate(grid):
        mean_f1 = np.mean(grid_scores[cfg_idx]["macro_f1"])
        std_f1 = np.std(grid_scores[cfg_idx]["macro_f1"])
        mean_acc = np.mean(grid_scores[cfg_idx]["acc"])
        mean_sens = np.mean(grid_scores[cfg_idx]["sens"])
        mean_spec = np.mean(grid_scores[cfg_idx]["spec"])

        results_rows.append({
            "cfg_id": cfg_idx,
            "k": cfg["k"],
            "weights": cfg["weights"],
            "metric": cfg["metric"],
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "mean_acc": mean_acc,
            "mean_sens": mean_sens,
            "mean_spec": mean_spec
        })

    df_grid = pd.DataFrame(results_rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    df_grid.to_csv(results_dir / "grid_search_results.csv", index=False)

    best_cfg = df_grid.iloc[0]
    best_hparams = {
        "k": int(best_cfg["k"]),
        "weights": str(best_cfg["weights"]),
        "metric": str(best_cfg["metric"]),
        "mccv_mean_macro_f1": float(best_cfg["mean_macro_f1"]),
        "mccv_std_macro_f1": float(best_cfg["std_macro_f1"]),
        "mccv_mean_accuracy": float(best_cfg["mean_acc"])
    }

    with open(results_dir / "best_hparams.json", "w") as f:
        json.dump(best_hparams, f, indent=4)

    print("\nPhase A Complete. Best Hyperparameters Found:")
    print(json.dumps(best_hparams, indent=4))

    # Plot MCCV Grid Search Curves
    plt.figure(figsize=(10, 6))
    for w in weights_list:
        for m in metrics_list:
            subset = df_grid[(df_grid["weights"] == w) & (df_grid["metric"] == m)].sort_values("k")
            plt.plot(subset["k"], subset["mean_macro_f1"], marker='o', label=f"weights={w}, metric={m}")

    plt.title("Tabular Fuzzy KNN 100-Split MCCV Grid Search Performance", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Neighbors (k)", fontsize=10)
    plt.ylabel("Mean Validation Macro-F1", fontsize=10)
    plt.xticks(k_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "grid_search_curves.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Phase B: LOOCV Evaluation on 88 Complete Labeled Cases
    # ---------------------------------------------------------
    print("\nBeginning Phase B: Leave-One-Out (LOOCV) Evaluation on N=88 Complete Labeled Cohort...")

    loo = LeaveOneOut()
    oof_p_soft = []
    oof_y_true = []

    for train_idx, val_idx in loo.split(y_binary):
        scaler_loo = MinMaxScaler()
        X_tr_num = scaler_loo.fit_transform(df_tab_labeled.iloc[train_idx][num_cols])
        X_va_num = scaler_loo.transform(df_tab_labeled.iloc[val_idx][num_cols])

        ohe_loo = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_cat = ohe_loo.fit_transform(df_tab_labeled.iloc[train_idx][cat_cols])
        X_va_cat = ohe_loo.transform(df_tab_labeled.iloc[val_idx][cat_cols])

        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_va = np.hstack([X_va_num, X_va_cat])

        y_tr_soft = y_soft[train_idx]

        knn_best = KNeighborsRegressor(
            n_neighbors=best_hparams["k"],
            weights=best_hparams["weights"],
            metric=best_hparams["metric"]
        )
        knn_best.fit(X_tr, y_tr_soft)
        p_val_soft = knn_best.predict(X_va)[0]

        oof_p_soft.append(p_val_soft)
        oof_y_true.append(y_binary[val_idx[0]])

    oof_p_soft = np.array(oof_p_soft)
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = (oof_p_soft >= 0.50).astype(int)

    # Compute LOOCV Metrics
    final_macro_f1 = f1_score(oof_y_true, oof_y_pred, average="macro", zero_division=0)
    final_acc = accuracy_score(oof_y_true, oof_y_pred)
    
    tn, fp, fn, tp = confusion_matrix(oof_y_true, oof_y_pred, labels=[0, 1]).ravel()
    final_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    final_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    final_auroc = roc_auc_score(oof_y_true, oof_p_soft)
    final_brier = brier_score_loss(oof_y_true, oof_p_soft)

    loocv_metrics = {
        "macro_f1": float(final_macro_f1),
        "accuracy": float(final_acc),
        "sensitivity": float(final_sens),
        "specificity": float(final_spec),
        "auroc": float(final_auroc),
        "brier_score": float(final_brier),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_cases": int(len(oof_y_true))
    }

    with open(results_dir / "loocv_metrics.json", "w") as f:
        json.dump(loocv_metrics, f, indent=4)

    # Save OOF Predictions Dataframe
    df_oof = pd.DataFrame({
        "patient_id": pids_labeled,
        "ground_truth_biopsy": oof_y_true,
        "confidence_annotation": df_reasoning_labeled["confidence"].values,
        "certainty_weight": c_weights,
        "predicted_soft_prob": oof_p_soft,
        "predicted_biopsy": oof_y_pred
    })
    df_oof.to_csv(results_dir / "oof_predictions.csv", index=False)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn, fp], [fn, tp]]), display_labels=["No Biopsy", "Biopsy"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Tabular Fuzzy KNN LOOCV Confusion Matrix (Macro-F1: {final_macro_f1:.4f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(oof_y_true, oof_p_soft)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Fuzzy KNN ROC (AUC = {final_auroc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.title("Tabular Fuzzy KNN LOOCV ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Tabular Fuzzy KNN Sweep & LOOCV (exp_13) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`)  \n")
        f.write(f"**Dataset**: Labeled Complete-Case Tabular Clinical Data ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Phase A: 100-Split MCCV Grid Search Results\n")
        f.write(f"- **Best Hyperparameters Found**:\n")
        f.write(f"  - `n_neighbors` (k): `{best_hparams['k']}`  \n")
        f.write(f"  - `weights`: `{best_hparams['weights']}`  \n")
        f.write(f"  - `metric`: `{best_hparams['metric']}`  \n")
        f.write(f"  - **Mean Validation Macro-F1**: `{best_hparams['mccv_mean_macro_f1']:.4f}` ($\text{{std}} = {best_hparams['mccv_std_macro_f1']:.4f}$)  \n\n")

        f.write("## Phase B: Leave-One-Out (LOOCV) Generalization Performance\n")
        f.write(f"- **Out-of-Fold Macro-F1**: **`{final_macro_f1:.4f}`**  \n")
        f.write(f"- **Out-of-Fold Accuracy**: **`{final_acc:.4f}`** ({tp+tn}/88 correct)  \n")
        f.write(f"- **Sensitivity (Yes class)**: **`{final_sens:.4f}`** ({tp}/56 correct)  \n")
        f.write(f"- **Specificity (No class)**: **`{final_spec:.4f}`** ({tn}/32 correct)  \n")
        f.write(f"- **AUROC**: **`{final_auroc:.4f}`**  \n")
        f.write(f"- **Brier Calibration Score**: **`{final_brier:.4f}`**  \n\n")

        f.write("### 2x2 Confusion Matrix Counts:\n")
        f.write("| Ground Truth \\ Predicted | No Biopsy | Biopsy |\n")
        f.write("|:---|:---:|:---:|\n")
        f.write(f"| **No Biopsy** ($N=32$) | **{tn}** | {fp} |\n")
        f.write(f"| **Biopsy** ($N=56$) | {fn} | **{tp}** |\n\n")

        f.write("## Comparison against Baseline (exp_5 Standard Tabular KNN)\n")
        f.write("| Model / Harness | Hyperparameters (k, w, m) | MCCV Mean Macro-F1 | LOOCV Macro-F1 | LOOCV Accuracy | Sensitivity | Specificity | AUROC |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **`exp_5` (Standard KNN)** | k=3, uniform, euclidean | 0.6218 | 0.6333 | 68.18% | 0.8519 | 0.4118 | — |\n")
        f.write(f"| **`exp_13` (Fuzzy KNN)** | k={best_hparams['k']}, {best_hparams['weights']}, {best_hparams['metric']} | **{best_hparams['mccv_mean_macro_f1']:.4f}** | **{final_macro_f1:.4f}** | **{final_acc*100:.2f}%** | **{final_sens:.4f}** | **{final_spec:.4f}** | **{final_auroc:.4f}** |\n")

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
