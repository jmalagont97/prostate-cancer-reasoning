import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import torch
from scipy.stats import mode
import sys
# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

# Import EmbedKit
from embedkit import EmbedKit

def get_pruned_feature_indices(X_train, threshold):
    # X_train is a numpy array of shape (N, D)
    # Compute correlation matrix
    corr = np.corrcoef(X_train, rowvar=False)
    # Handle NaNs (e.g. constant features)
    corr = np.nan_to_num(corr)
    abs_corr = np.abs(corr)
    
    keep_indices = []
    dropped = np.zeros(abs_corr.shape[0], dtype=bool)
    
    for i in range(abs_corr.shape[0]):
        if dropped[i]:
            continue
        keep_indices.append(i)
        # Find all features collinear with i and drop them
        collinear = abs_corr[i] > threshold
        dropped[collinear] = True
        
    return keep_indices

def main():
    # Define paths relative to this script
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    exp_dir = project_root / "experiments" / "exp_6"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for EmbedKit projection models.")

    print("Loading datasets...")
    df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align by patient_id
    pids = df_design["patient_id"].values
    df_mri = df_mri[df_mri["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    assert (df_mri["patient_id"] == df_dec["patient_id"]).all()
    assert (df_mri["patient_id"] == df_design["patient_id"]).all()

    # Filter labeled cases only (since grid search and validation is on labeled cohort)
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_mri_labeled = df_mri[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    X_mri_raw = df_mri_labeled.drop(columns=["patient_id"]).values.astype(np.float32)
    label_map = {"yes": 1, "no": 0}
    inv_label_map = {1: "yes", 0: "no"}
    y_labeled = df_dec_labeled["biopsy_decision"].map(label_map).values

    # Sweep parameters
    n_neighbors_list = [1, 3, 5, 7, 9, 11, 15, 21]
    weights_list = ["uniform", "distance"]
    metric_list = ["euclidean", "cosine"]
    representation_names = ["raw", "pca", "embedkit_unsup", "embedkit_sup", "corr_0.7", "corr_0.8", "corr_0.9", "corr_0.95"]

    # Struct to keep tracked scores
    # Key: (rep_name, k, w, m) -> list of validation scores across 100 splits
    grid_scores = {}
    for r in representation_names:
        for k in n_neighbors_list:
            for w in weights_list:
                for m in metric_list:
                    grid_scores[(r, k, w, m)] = {
                        "f1": [], "acc": [], "sens": [], "spec": []
                    }

    # Tracking EmbedKit dynamic dimensions
    mccv_dims_u = []
    mccv_dims_s = []

    split_cols = [f"split_{i}" for i in range(100)]

    print("Beginning Phase A: 100-split MCCV Parameter Sweep...")

    for split_idx, col in enumerate(split_cols):
        split_vals = df_design_labeled[col].values
        train_mask = split_vals == 0
        val_mask = split_vals == 1

        y_train = y_labeled[train_mask]
        y_val = y_labeled[val_mask]

        # 1. Raw representation (MinMax scaled)
        scaler = MinMaxScaler()
        X_tr_raw = scaler.fit_transform(X_mri_raw[train_mask])
        X_va_raw = scaler.transform(X_mri_raw[val_mask])

        # 2. PCA (90% variance)
        pca = PCA(n_components=0.90, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_raw)
        X_va_pca = pca.transform(X_va_raw)

        # 3. EmbedKit Unsupervised (Self-supervised)
        ek_u = EmbedKit(mode="self_supervised", target_dim="auto", epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_u = ek_u.fit_transform(X_tr_raw)
        X_va_u = ek_u.transform(X_va_raw)
        dim_u = int(ek_u._config["target_dim"])
        mccv_dims_u.append(dim_u)

        # 4. EmbedKit Supervised
        ek_s = EmbedKit(mode="supervised", target_dim="auto", epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_s = ek_s.fit_transform(X_tr_raw, y_train)
        X_va_s = ek_s.transform(X_va_raw)
        dim_s = int(ek_s._config["target_dim"])
        mccv_dims_s.append(dim_s)

        # 5. Correlation Pruning (different thresholds)
        corr_reps = {}
        for th in [0.70, 0.80, 0.90, 0.95]:
            keep_indices = get_pruned_feature_indices(X_tr_raw, th)
            X_tr_c = X_tr_raw[:, keep_indices]
            X_va_c = X_va_raw[:, keep_indices]
            corr_reps[th] = (X_tr_c, X_va_c)

        # Collect representation splits
        reps = {
            "raw": (X_tr_raw, X_va_raw),
            "pca": (X_tr_pca, X_va_pca),
            "embedkit_unsup": (X_tr_u, X_va_u),
            "embedkit_sup": (X_tr_s, X_va_s),
            "corr_0.7": corr_reps[0.70],
            "corr_0.8": corr_reps[0.80],
            "corr_0.9": corr_reps[0.90],
            "corr_0.95": corr_reps[0.95]
        }

        # Evaluate KNN on each representation
        for r_name, (X_tr, X_va) in reps.items():
            for k in n_neighbors_list:
                for w in weights_list:
                    for m in metric_list:
                        knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
                        knn.fit(X_tr, y_train)
                        preds = knn.predict(X_va)

                        # Metrics
                        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
                        acc = accuracy_score(y_val, preds)
                        
                        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
                        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                        grid_scores[(r_name, k, w, m)]["f1"].append(f1)
                        grid_scores[(r_name, k, w, m)]["acc"].append(acc)
                        grid_scores[(r_name, k, w, m)]["sens"].append(sens)
                        grid_scores[(r_name, k, w, m)]["spec"].append(spec)

        if (split_idx + 1) % 10 == 0:
            print(f"Processed splits: {split_idx + 1}/100. Average dynamic target dimension - Unsupervised: {np.mean(mccv_dims_u):.1f}, Supervised: {np.mean(mccv_dims_s):.1f}")

    # Consolidate results
    results_list = []
    for (r, k, w, m), metrics in grid_scores.items():
        results_list.append({
            "representation": r,
            "n_neighbors": k,
            "weights": w,
            "metric": m,
            "mean_macro_f1": np.mean(metrics["f1"]),
            "mean_accuracy": np.mean(metrics["acc"]),
            "mean_sensitivity": np.mean(metrics["sens"]),
            "mean_specificity": np.mean(metrics["spec"])
        })

    df_results = pd.DataFrame(results_list)
    grid_csv_path = results_dir / "grid_search_results.csv"
    df_results.to_csv(grid_csv_path, index=False)
    print(f"Saved all sweep results to: {grid_csv_path}")

    # Select the best configuration
    best_row = df_results.sort_values(by="mean_macro_f1", ascending=False).iloc[0]
    best_rep = best_row["representation"]
    best_k = int(best_row["n_neighbors"])
    best_w = best_row["weights"]
    best_m = best_row["metric"]
    best_f1 = best_row["mean_macro_f1"]

    best_hparams = {
        "representation": best_rep,
        "n_neighbors": best_k,
        "weights": best_w,
        "metric": best_m,
        "mean_macro_f1": best_f1
    }

    # If the winning representation uses EmbedKit, compute mode dimension
    mode_dim = None
    if best_rep == "embedkit_unsup":
        mode_dim = int(mode(mccv_dims_u, keepdims=False).mode)
        best_hparams["frozen_target_dim"] = mode_dim
    elif best_rep == "embedkit_sup":
        mode_dim = int(mode(mccv_dims_s, keepdims=False).mode)
        best_hparams["frozen_target_dim"] = mode_dim

    hparams_json_path = results_dir / "best_hparams.json"
    with open(hparams_json_path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"Best Configuration: Representation={best_rep}, k={best_k}, weights={best_w}, metric={best_m} (Mean Macro-F1: {best_f1:.4f})")
    if mode_dim is not None:
        print(f"Frozen EmbedKit target dimension for Phase B (LOOCV): {mode_dim}")
    print(f"Saved best parameters to: {hparams_json_path}")

    # Plot validation curves for each representation under the best weights/metric KNN settings
    print("Generating validation curves plot...")
    plt.figure(figsize=(12, 6))
    for r in representation_names:
        df_r = df_results[
            (df_results["representation"] == r) & 
            (df_results["weights"] == best_w) & 
            (df_results["metric"] == best_m)
        ].sort_values("n_neighbors")
        plt.plot(df_r["n_neighbors"], df_r["mean_macro_f1"], marker='o', label=f"Rep: {r}")
        
    plt.title(f"MRI KNN Grid Search: Validation Macro-F1 vs Neighbors (Weights: {best_w}, Metric: {best_m})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Neighbors (k)", fontsize=11)
    plt.ylabel("Mean Macro-F1 (100 MCCV Splits)", fontsize=11)
    plt.xticks(n_neighbors_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    curves_png_path = figures_dir / "grid_search_curves.png"
    plt.savefig(curves_png_path, dpi=300)
    plt.close()
    print(f"Saved validation curves visualization to: {curves_png_path}")

    print("Beginning Phase B: Leave-One-Out Cross-Validation (LOOCV) final evaluation...")
    loo = LeaveOneOut()
    oof_preds = []
    oof_probs = []
    oof_targets = []
    oof_patient_ids = []

    for train_idx, val_idx in loo.split(X_mri_raw):
        X_train_raw_split = X_mri_raw[train_idx]
        X_val_raw_split = X_mri_raw[val_idx]
        y_train = y_labeled[train_idx]
        y_val = y_labeled[val_idx]
        
        pid_val = df_mri_labeled.iloc[val_idx[0]]["patient_id"]
        
        # Fit MinMaxScaler strictly on training subset
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_train_raw_split)
        X_va_scaled = scaler.transform(X_val_raw_split)
        
        # Apply the frozen winning representation technique
        if best_rep == "raw":
            X_tr_proj, X_va_proj = X_tr_scaled, X_va_scaled
        elif best_rep == "pca":
            pca = PCA(n_components=0.90, random_state=42)
            X_tr_proj = pca.fit_transform(X_tr_scaled)
            X_va_proj = pca.transform(X_va_scaled)
        elif best_rep == "embedkit_unsup":
            # Use frozen mode target dimension
            ek_u = EmbedKit(mode="self_supervised", target_dim=mode_dim, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_proj = ek_u.fit_transform(X_tr_scaled)
            X_va_proj = ek_u.transform(X_va_scaled)
        elif best_rep == "embedkit_sup":
            # Use frozen mode target dimension
            ek_s = EmbedKit(mode="supervised", target_dim=mode_dim, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_proj = ek_s.fit_transform(X_tr_scaled, y_train)
            X_va_proj = ek_s.transform(X_va_scaled)
        elif best_rep.startswith("corr_"):
            th = float(best_rep.split("_")[1])
            keep_indices = get_pruned_feature_indices(X_tr_scaled, th)
            X_tr_proj = X_tr_scaled[:, keep_indices]
            X_va_proj = X_va_scaled[:, keep_indices]
        
        # Fit optimal KNN on projected training cases
        knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, metric=best_m)
        knn.fit(X_tr_proj, y_train)
        
        pred = knn.predict(X_va_proj)[0]
        prob = knn.predict_proba(X_va_proj)[0, 1]
        
        oof_preds.append(pred)
        oof_probs.append(prob)
        oof_targets.append(y_val[0])
        oof_patient_ids.append(pid_val)

    # Compute LOOCV metrics
    oof_preds = np.array(oof_preds)
    oof_probs = np.array(oof_probs)
    oof_targets = np.array(oof_targets)

    loocv_f1 = f1_score(oof_targets, oof_preds, average='macro')
    loocv_acc = accuracy_score(oof_targets, oof_preds)
    
    tn, fp, fn, tp = confusion_matrix(oof_targets, oof_preds, labels=[0, 1]).ravel()
    loocv_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    loocv_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Final LOOCV Metrics:")
    print(f"  Macro-F1: {loocv_f1:.4f}")
    print(f"  Accuracy: {loocv_acc:.4f} ({tp+tn}/{len(oof_targets)})")
    print(f"  Sensitivity (Sensitivity of Yes): {loocv_sens:.4f}")
    print(f"  Specificity (Specificity of No): {loocv_spec:.4f}")

    loocv_metrics = {
        "macro_f1": loocv_f1,
        "accuracy": loocv_acc,
        "sensitivity": loocv_sens,
        "specificity": loocv_spec,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

    metrics_json_path = results_dir / "loocv_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(loocv_metrics, f, indent=4)
    print(f"Saved LOOCV metrics to: {metrics_json_path}")

    # Save out-of-fold predictions
    df_predictions = pd.DataFrame({
        "patient_id": oof_patient_ids,
        "target": [inv_label_map[t] for t in oof_targets],
        "prediction": [inv_label_map[p] for p in oof_preds],
        "probability": oof_probs
    })
    predictions_csv_path = results_dir / "loocv_predictions.csv"
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"Saved LOOCV predictions to: {predictions_csv_path}")

    # Plot confusion matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(oof_targets, oof_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"LOOCV Confusion Matrix ({best_rep}, k={best_k}, weights={best_w}, metric={best_m})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visualization to: {cm_png_path}")

    # Write summary.md report
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# MRI Representations Model Selection (exp_6) Summary Report\n\n")
        f.write(f"**Date**: 2026-08-04  \n")
        f.write(f"**Model**: K-Nearest Neighbors Classifier on MRI Embeddings  \n")
        f.write(f"**Dataset**: Labeled Complete-Case MRI Dataset ($N_{{labeled}} = 88$)  \n\n")
        
        f.write("## Phase A: 100-Split MCCV Grid Search Results\n")
        f.write("- **Best Configuration Found**:\n")
        f.write(f"  - **Representation**: `{best_rep}`  \n")
        f.write(f"  - `n_neighbors` (k): `{best_k}`  \n")
        f.write(f"  - `weights`: `{best_w}`  \n")
        f.write(f"  - `metric`: `{best_m}`  \n")
        if mode_dim is not None:
            f.write(f"  - **Frozen EmbedKit Target Dimension**: `{mode_dim}`  \n")
        f.write(f"  - **Mean Validation Macro-F1**: `{best_f1:.4f}`  \n\n")

        f.write("### Mean Dynamic Dimensions Logged for EmbedKit:\n")
        f.write(f"- Unsupervised mode dimension: {np.mean(mccv_dims_u):.1f} (std: {np.std(mccv_dims_u):.1f})  \n")
        f.write(f"- Supervised mode dimension: {np.mean(mccv_dims_s):.1f} (std: {np.std(mccv_dims_s):.1f})  \n\n")

        f.write("### Top 5 Hyperparameter Configurations:\n")
        df_top = df_results.sort_values(by="mean_macro_f1", ascending=False).head(5)
        f.write("| Rank | Representation | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for idx, row in enumerate(df_top.itertuples(), 1):
            f.write(f"| {idx} | {row.representation} | {row.n_neighbors} | {row.weights} | {row.metric} | {row.mean_macro_f1:.4f} | {row.mean_accuracy:.4f} | {row.mean_sensitivity:.4f} | {row.mean_specificity:.4f} |\n")
        f.write("\n")

        f.write("## Phase B: Leave-One-Out (LOOCV) Generalization Performance\n")
        f.write(f"The optimal representation and KNN configuration was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:\n\n")
        f.write(f"- **Final Macro-F1**: `{loocv_f1:.4f}`  \n")
        f.write(f"- **Final Accuracy**: `{loocv_acc:.4f}` ({tp+tn} correct out of 88 cases)  \n")
        f.write(f"- **Sensitivity (Yes class)**: `{loocv_sens:.4f}` (Correctly identified {tp} out of 56 yes cases)  \n")
        f.write(f"- **Specificity (No class)**: `{loocv_spec:.4f}` (Correctly identified {tn} out of 32 no cases)  \n\n")
        
        f.write("### Confusion Matrix Counts:\n")
        f.write(f"- True Negatives (TN): `{tn}`  \n")
        f.write(f"- False Positives (FP): `{fp}`  \n")
        f.write(f"- False Negatives (FN): `{fn}`  \n")
        f.write(f"- True Positives (TP): `{tp}`  \n\n")

    print(f"Summary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
