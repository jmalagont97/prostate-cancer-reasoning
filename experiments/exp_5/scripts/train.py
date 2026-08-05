import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

def main():
    # Define paths relative to this script
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    exp_dir = project_root / "experiments" / "exp_5"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align by patient_id
    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    assert (df_tab["patient_id"] == df_dec["patient_id"]).all()
    assert (df_tab["patient_id"] == df_design["patient_id"]).all()

    # Filter labeled cases only (since grid search and validation is on labeled cohort)
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    X_labeled = df_tab_labeled.drop(columns=["patient_id"])
    label_map = {"yes": 1, "no": 0}
    inv_label_map = {1: "yes", 0: "no"}
    y_labeled = df_dec_labeled["biopsy_decision"].map(label_map).values

    num_cols = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols = ["dre"]

    # Define hyperparameter grid
    n_neighbors_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25]
    weights_list = ["uniform", "distance"]
    metric_list = ["euclidean", "manhattan", "cosine"]

    results_list = []
    split_cols = [f"split_{i}" for i in range(100)]

    print("Beginning Phase A: 100-split MCCV Parameter Sweep...")
    
    # Progress counter
    total_combinations = len(n_neighbors_list) * len(weights_list) * len(metric_list)
    comb_idx = 0

    for k in n_neighbors_list:
        for w in weights_list:
            for m in metric_list:
                f1_scores = []
                acc_scores = []
                sens_scores = []
                spec_scores = []
                
                for col in split_cols:
                    split_vals = df_design_labeled[col].values
                    train_mask = split_vals == 0
                    val_mask = split_vals == 1
                    
                    X_train = X_labeled[train_mask].copy()
                    y_train = y_labeled[train_mask]
                    X_val = X_labeled[val_mask].copy()
                    y_val = y_labeled[val_mask]
                    
                    # Fit pipeline strictly on train
                    scaler = MinMaxScaler()
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    
                    X_train_num = scaler.fit_transform(X_train[num_cols])
                    X_train_cat = ohe.fit_transform(X_train[cat_cols])
                    X_train_processed = np.hstack([X_train_num, X_train_cat])
                    
                    X_val_num = scaler.transform(X_val[num_cols])
                    X_val_cat = ohe.transform(X_val[cat_cols])
                    X_val_processed = np.hstack([X_val_num, X_val_cat])
                    
                    # Train KNN
                    knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
                    knn.fit(X_train_processed, y_train)
                    
                    # Predict
                    preds = knn.predict(X_val_processed)
                    
                    # Metrics
                    f1 = f1_score(y_val, preds, average='macro', zero_division=0)
                    acc = accuracy_score(y_val, preds)
                    
                    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    f1_scores.append(f1)
                    acc_scores.append(acc)
                    sens_scores.append(sens)
                    spec_scores.append(spec)
                
                mean_f1 = np.mean(f1_scores)
                mean_acc = np.mean(acc_scores)
                mean_sens = np.mean(sens_scores)
                mean_spec = np.mean(spec_scores)
                
                results_list.append({
                    "n_neighbors": k,
                    "weights": w,
                    "metric": m,
                    "mean_macro_f1": mean_f1,
                    "mean_accuracy": mean_acc,
                    "mean_sensitivity": mean_sens,
                    "mean_specificity": mean_spec
                })
                
                comb_idx += 1
                if comb_idx % 10 == 0 or comb_idx == total_combinations:
                    print(f"Sweep Progress: {comb_idx}/{total_combinations} configurations evaluated.")

    # Convert results to DataFrame and save
    df_results = pd.DataFrame(results_list)
    grid_csv_path = results_dir / "grid_search_results.csv"
    df_results.to_csv(grid_csv_path, index=False)
    print(f"Saved all sweep results to: {grid_csv_path}")

    # Find the best hyperparameters based on mean_macro_f1
    best_row = df_results.sort_values(by="mean_macro_f1", ascending=False).iloc[0]
    best_k = int(best_row["n_neighbors"])
    best_w = best_row["weights"]
    best_m = best_row["metric"]
    best_f1 = best_row["mean_macro_f1"]

    best_hparams = {
        "n_neighbors": best_k,
        "weights": best_w,
        "metric": best_m,
        "mean_macro_f1": best_f1
    }

    hparams_json_path = results_dir / "best_hparams.json"
    with open(hparams_json_path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"Best Hyperparameters: k={best_k}, weights={best_w}, metric={best_m} (Mean Macro-F1: {best_f1:.4f})")
    print(f"Saved best parameters to: {hparams_json_path}")

    # Plot grid search curves
    print("Generating validation curves plot...")
    plt.figure(figsize=(10, 5))
    
    # Plot for the best weights setting
    df_subset = df_results[df_results["weights"] == best_w]
    for m in metric_list:
        df_m = df_subset[df_subset["metric"] == m].sort_values("n_neighbors")
        plt.plot(df_m["n_neighbors"], df_m["mean_macro_f1"], marker='o', label=f"Metric: {m}")
        
    plt.title(f"Tabular KNN Grid Search: Validation Macro-F1 vs Neighbors (Weights: {best_w})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Neighbors (k)", fontsize=11)
    plt.ylabel("Mean Macro-F1 (100 MCCV Splits)", fontsize=11)
    plt.xticks(n_neighbors_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
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

    for train_idx, val_idx in loo.split(X_labeled):
        X_train, X_val = X_labeled.iloc[train_idx].copy(), X_labeled.iloc[val_idx].copy()
        y_train, y_val = y_labeled[train_idx], y_labeled[val_idx]
        
        pid_val = df_tab_labeled.iloc[val_idx[0]]["patient_id"]
        
        # Fit pipeline strictly on training subset
        scaler = MinMaxScaler()
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_train_cat = ohe.fit_transform(X_train[cat_cols])
        X_train_processed = np.hstack([X_train_num, X_train_cat])
        
        X_val_num = scaler.transform(X_val[num_cols])
        X_val_cat = ohe.transform(X_val[cat_cols])
        X_val_processed = np.hstack([X_val_num, X_val_cat])
        
        # Train optimal KNN
        knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, metric=best_m)
        knn.fit(X_train_processed, y_train)
        
        pred = knn.predict(X_val_processed)[0]
        # Get probability for the 'yes' class (class 1)
        prob = knn.predict_proba(X_val_processed)[0, 1]
        
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

    # Save LOOCV metrics to json
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

    # Save OOF predictions to CSV
    df_predictions = pd.DataFrame({
        "patient_id": oof_patient_ids,
        "target": [inv_label_map[t] for t in oof_targets],
        "prediction": [inv_label_map[p] for p in oof_preds],
        "probability": oof_probs
    })
    predictions_csv_path = results_dir / "loocv_predictions.csv"
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"Saved LOOCV out-of-fold predictions to: {predictions_csv_path}")

    # Save confusion matrix figure
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(oof_targets, oof_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"LOOCV Confusion Matrix (k={best_k}, weights={best_w}, metric={best_m})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visualization to: {cm_png_path}")

    # Write summary.md report
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Tabular KNN Model Selection (exp_5) Summary Report\n\n")
        f.write(f"**Date**: 2026-08-04  \n")
        f.write(f"**Model**: K-Nearest Neighbors Classifier  \n")
        f.write(f"**Dataset**: Labeled Complete-Case Tabular Clinical Data ($N_{{labeled}} = 88$)  \n\n")
        
        f.write("## Phase A: 100-Split MCCV Grid Search Results\n")
        f.write("- **Best Hyperparameters Found**:\n")
        f.write(f"  - `n_neighbors` (k): `{best_k}`  \n")
        f.write(f"  - `weights`: `{best_w}`  \n")
        f.write(f"  - `metric`: `{best_m}`  \n")
        f.write(f"  - **Mean Validation Macro-F1**: `{best_f1:.4f}`  \n\n")
        
        f.write("### Top 5 Hyperparameter Configurations:\n")
        df_top = df_results.sort_values(by="mean_macro_f1", ascending=False).head(5)
        f.write("| Rank | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for idx, row in enumerate(df_top.itertuples(), 1):
            f.write(f"| {idx} | {row.n_neighbors} | {row.weights} | {row.metric} | {row.mean_macro_f1:.4f} | {row.mean_accuracy:.4f} | {row.mean_sensitivity:.4f} | {row.mean_specificity:.4f} |\n")
        f.write("\n")

        f.write("## Phase B: Leave-One-Out (LOOCV) Generalization Performance\n")
        f.write(f"The optimal hyperparameter configuration ($k={best_k}$, weights={best_w}, metric={best_m}) was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:\n\n")
        f.write(f"- **Final Macro-F1**: `{loocv_f1:.4f}`  \n")
        f.write(f"- **Final Accuracy**: `{loocv_acc:.4f}` ({tp+tn} correct out of 88 cases)  \n")
        f.write(f"- **Sensitivity (Yes class)**: `{loocv_sens:.4f}` (Correctly identified {tp} out of 56 yes cases)  \n")
        f.write(f"- **Specificity (No class)**: `{loocv_spec:.4f}` (Correctly identified {tn} out of 32 no cases)  \n\n")
        
        f.write("### Confusion Matrix Counts:\n")
        f.write(f"- True Negatives (TN): `{tn}`  \n")
        f.write(f"- False Positives (FP): `{fp}`  \n")
        f.write(f"- False Negatives (FN): `{fn}`  \n")
        f.write(f"- True Positives (TP): `{tp}`  \n\n")

        f.write("## Preprocessing Pipeline Details\n")
        f.write("- **Numerical variables**: scaled dynamically per fold using `MinMaxScaler` onto $[0, 1]$ interval.  \n")
        f.write("- **Categorical variables**: one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`.  \n")
        f.write("- **Feature space**: concatenated output size = 8 features (7 numerical, 1 category one-hot column for `dre`).  \n")

    print(f"Summary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
