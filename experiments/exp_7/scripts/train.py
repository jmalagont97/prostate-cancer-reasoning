import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import torch
from scipy.stats import mode

# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

# Import EmbedKit
from embedkit import EmbedKit

def get_pruned_feature_indices(X_train, threshold):
    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr)
    abs_corr = np.abs(corr)
    
    keep_indices = []
    dropped = np.zeros(abs_corr.shape[0], dtype=bool)
    
    for i in range(abs_corr.shape[0]):
        if dropped[i]:
            continue
        keep_indices.append(i)
        collinear = abs_corr[i] > threshold
        dropped[collinear] = True
        
    return keep_indices

def main():
    exp_dir = project_root / "experiments" / "exp_7"
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for EmbedKit projection models.")

    print("Loading datasets...")
    df_text = pd.read_csv(data_dir / "clinical_prompts.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align by patient_id
    pids = df_design["patient_id"].values
    df_text = df_text[df_text["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    assert (df_text["patient_id"] == df_dec["patient_id"]).all()
    assert (df_text["patient_id"] == df_design["patient_id"]).all()

    # Filter labeled cases only
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_text_labeled = df_text[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    import spacy
    print("Loading spaCy model en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    print("Preprocessing text corpus with spaCy (lowercasing, punctuation removal, stop words removal, lemmatization)...")
    raw_texts = df_text_labeled["clinical_prompt_text"].values
    cleaned_texts = []
    for doc in nlp.pipe(raw_texts, batch_size=50):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_texts.append(" ".join(tokens))
    text_corpus_labeled = np.array(cleaned_texts)

    label_map = {"yes": 1, "no": 0}
    inv_label_map = {1: "yes", 0: "no"}
    y_labeled = df_dec_labeled["biopsy_decision"].map(label_map).values

    # Sweep parameters
    max_features_list = [100, 300, 500, 1000, None]
    n_neighbors_list = [1, 3, 5, 7, 9, 11, 15, 21]
    weights_list = ["uniform", "distance"]
    metric_list = ["euclidean", "cosine"]
    representation_names = ["raw", "pca", "embedkit_unsup", "embedkit_sup", "corr_0.7", "corr_0.8", "corr_0.9", "corr_0.95"]

    # Tracking metrics across 100 splits
    grid_scores = {}
    for mf in max_features_list:
        mf_str = "None" if mf is None else str(mf)
        for r in representation_names:
            for k in n_neighbors_list:
                for w in weights_list:
                    for m in metric_list:
                        grid_scores[(mf_str, r, k, w, m)] = {
                            "f1": [], "acc": [], "sens": [], "spec": []
                        }

    # Dynamic target dimensions for EmbedKit per (mf, split)
    dims_u_log = { ("None" if mf is None else str(mf)): [] for mf in max_features_list }
    dims_s_log = { ("None" if mf is None else str(mf)): [] for mf in max_features_list }

    split_cols = [f"split_{i}" for i in range(100)]

    print("Beginning Phase A: 100-split MCCV Parameter Sweep (TF-IDF + Representations + KNN)...")

    for split_idx, col in enumerate(split_cols):
        split_vals = df_design_labeled[col].values
        train_mask = split_vals == 0
        val_mask = split_vals == 1

        train_texts = text_corpus_labeled[train_mask]
        val_texts = text_corpus_labeled[val_mask]
        y_train = y_labeled[train_mask]
        y_val = y_labeled[val_mask]

        for mf in max_features_list:
            mf_str = "None" if mf is None else str(mf)

            # Fit TF-IDF strictly on train
            vec = TfidfVectorizer(max_features=mf, norm='l2')
            X_tr_tfidf = vec.fit_transform(train_texts).toarray().astype(np.float32)
            X_va_tfidf = vec.transform(val_texts).toarray().astype(np.float32)

            # 1. Raw representation
            X_tr_raw, X_va_raw = X_tr_tfidf, X_va_tfidf

            # 2. PCA (90% variance)
            n_components_pca = min(X_tr_raw.shape[0], X_tr_raw.shape[1])
            if n_components_pca > 1:
                pca = PCA(n_components=0.90, random_state=42)
                X_tr_pca = pca.fit_transform(X_tr_raw)
                X_va_pca = pca.transform(X_va_raw)
            else:
                X_tr_pca, X_va_pca = X_tr_raw, X_va_raw

            # 3. EmbedKit Unsupervised
            ek_u = EmbedKit(mode="self_supervised", target_dim="auto", epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_u = ek_u.fit_transform(X_tr_raw)
            X_va_u = ek_u.transform(X_va_raw)
            dim_u = int(ek_u._config["target_dim"])
            dims_u_log[mf_str].append(dim_u)

            # 4. EmbedKit Supervised
            ek_s = EmbedKit(mode="supervised", target_dim="auto", epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_s = ek_s.fit_transform(X_tr_raw, y_train)
            X_va_s = ek_s.transform(X_va_raw)
            dim_s = int(ek_s._config["target_dim"])
            dims_s_log[mf_str].append(dim_s)

            # 5. Correlation Pruning
            corr_reps = {}
            for th in [0.70, 0.80, 0.90, 0.95]:
                keep_indices = get_pruned_feature_indices(X_tr_raw, th)
                X_tr_c = X_tr_raw[:, keep_indices]
                X_va_c = X_va_raw[:, keep_indices]
                corr_reps[th] = (X_tr_c, X_va_c)

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

            # Evaluate KNN
            for r_name, (X_tr, X_va) in reps.items():
                for k in n_neighbors_list:
                    for w in weights_list:
                        for m in metric_list:
                            knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
                            knn.fit(X_tr, y_train)
                            preds = knn.predict(X_va)

                            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
                            acc = accuracy_score(y_val, preds)
                            
                            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                            grid_scores[(mf_str, r_name, k, w, m)]["f1"].append(f1)
                            grid_scores[(mf_str, r_name, k, w, m)]["acc"].append(acc)
                            grid_scores[(mf_str, r_name, k, w, m)]["sens"].append(sens)
                            grid_scores[(mf_str, r_name, k, w, m)]["spec"].append(spec)

        if (split_idx + 1) % 10 == 0:
            print(f"Processed splits: {split_idx + 1}/100.")

    # Consolidate results
    results_list = []
    for (mf_str, r, k, w, m), metrics in grid_scores.items():
        results_list.append({
            "max_features": mf_str,
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

    # Find the best configuration
    best_row = df_results.sort_values(by="mean_macro_f1", ascending=False).iloc[0]
    best_mf_str = best_row["max_features"]
    best_mf = None if best_mf_str == "None" else int(best_mf_str)
    best_rep = best_row["representation"]
    best_k = int(best_row["n_neighbors"])
    best_w = best_row["weights"]
    best_m = best_row["metric"]
    best_f1 = best_row["mean_macro_f1"]

    best_hparams = {
        "max_features": best_mf_str,
        "representation": best_rep,
        "n_neighbors": best_k,
        "weights": best_w,
        "metric": best_m,
        "mean_macro_f1": best_f1
    }

    mode_dim = None
    if best_rep == "embedkit_unsup":
        mode_dim = int(mode(dims_u_log[best_mf_str], keepdims=False).mode)
        best_hparams["frozen_target_dim"] = mode_dim
    elif best_rep == "embedkit_sup":
        mode_dim = int(mode(dims_s_log[best_mf_str], keepdims=False).mode)
        best_hparams["frozen_target_dim"] = mode_dim

    hparams_json_path = results_dir / "best_hparams.json"
    with open(hparams_json_path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"Best Configuration: max_features={best_mf_str}, Representation={best_rep}, k={best_k}, weights={best_w}, metric={best_m} (Mean Macro-F1: {best_f1:.4f})")
    if mode_dim is not None:
        print(f"Frozen EmbedKit target dimension for Phase B (LOOCV): {mode_dim}")
    print(f"Saved best parameters to: {hparams_json_path}")

    # Plot validation curves
    print("Generating validation curves plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot mean_macro_f1 vs n_neighbors for best max_features and best weights/metric across representations
    df_subset = df_results[
        (df_results["max_features"] == best_mf_str) & 
        (df_results["weights"] == best_w) & 
        (df_results["metric"] == best_m)
    ]
    for r in representation_names:
        df_r = df_subset[df_subset["representation"] == r].sort_values("n_neighbors")
        plt.plot(df_r["n_neighbors"], df_r["mean_macro_f1"], marker='o', label=f"Rep: {r}")
        
    plt.title(f"Text TF-IDF KNN Grid Search (max_features={best_mf_str}, weights={best_w}, metric={best_m})", fontsize=12, fontweight="bold")
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

    for train_idx, val_idx in loo.split(text_corpus_labeled):
        train_texts_split = text_corpus_labeled[train_idx]
        val_texts_split = text_corpus_labeled[val_idx]
        y_train = y_labeled[train_idx]
        y_val = y_labeled[val_idx]
        
        pid_val = df_text_labeled.iloc[val_idx[0]]["patient_id"]
        
        # Fit TF-IDF on 87 cases
        vec = TfidfVectorizer(max_features=best_mf, norm='l2')
        X_tr_tfidf_split = vec.fit_transform(train_texts_split).toarray().astype(np.float32)
        X_va_tfidf_split = vec.transform(val_texts_split).toarray().astype(np.float32)
        
        # Apply the frozen winning representation technique
        if best_rep == "raw":
            X_tr_proj, X_va_proj = X_tr_tfidf_split, X_va_tfidf_split
        elif best_rep == "pca":
            pca = PCA(n_components=0.90, random_state=42)
            X_tr_proj = pca.fit_transform(X_tr_tfidf_split)
            X_va_proj = pca.transform(X_va_tfidf_split)
        elif best_rep == "embedkit_unsup":
            ek_u = EmbedKit(mode="self_supervised", target_dim=mode_dim, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_proj = ek_u.fit_transform(X_tr_tfidf_split)
            X_va_proj = ek_u.transform(X_va_tfidf_split)
        elif best_rep == "embedkit_sup":
            ek_s = EmbedKit(mode="supervised", target_dim=mode_dim, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
            X_tr_proj = ek_s.fit_transform(X_tr_tfidf_split, y_train)
            X_va_proj = ek_s.transform(X_va_tfidf_split)
        elif best_rep.startswith("corr_"):
            th = float(best_rep.split("_")[1])
            keep_indices = get_pruned_feature_indices(X_tr_tfidf_split, th)
            X_tr_proj = X_tr_tfidf_split[:, keep_indices]
            X_va_proj = X_va_tfidf_split[:, keep_indices]
        
        # Fit optimal KNN
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
    plt.title(f"LOOCV Confusion Matrix ({best_rep}, max_features={best_mf_str}, k={best_k})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visualization to: {cm_png_path}")

    # Write summary.md report
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Text TF-IDF Representations Model Selection (exp_7) Summary Report\n\n")
        f.write(f"**Date**: 2026-08-04  \n")
        f.write(f"**Model**: K-Nearest Neighbors Classifier on Clinical Text TF-IDF  \n")
        f.write(f"**Dataset**: Labeled Complete-Case Text Dataset ($N_{{labeled}} = 88$)  \n\n")
        
        f.write("## Phase A: 100-Split MCCV Grid Search Results\n")
        f.write("- **Best Configuration Found**:\n")
        f.write(f"  - **Vocabulary Size (`max_features`)**: `{best_mf_str}`  \n")
        f.write(f"  - **Representation**: `{best_rep}`  \n")
        f.write(f"  - `n_neighbors` (k): `{best_k}`  \n")
        f.write(f"  - `weights`: `{best_w}`  \n")
        f.write(f"  - `metric`: `{best_m}`  \n")
        if mode_dim is not None:
            f.write(f"  - **Frozen EmbedKit Target Dimension**: `{mode_dim}`  \n")
        f.write(f"  - **Mean Validation Macro-F1**: `{best_f1:.4f}`  \n\n")

        f.write("### Top 5 Hyperparameter Configurations:\n")
        df_top = df_results.sort_values(by="mean_macro_f1", ascending=False).head(5)
        f.write("| Rank | max_features | Representation | k | Weights | Distance Metric | Mean Macro-F1 | Mean Accuracy | Mean Sensitivity | Mean Specificity |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for idx, row in enumerate(df_top.itertuples(), 1):
            f.write(f"| {idx} | {row.max_features} | {row.representation} | {row.n_neighbors} | {row.weights} | {row.metric} | {row.mean_macro_f1:.4f} | {row.mean_accuracy:.4f} | {row.mean_sensitivity:.4f} | {row.mean_specificity:.4f} |\n")
        f.write("\n")

        f.write("## Phase B: Leave-One-Out (LOOCV) Generalization Performance\n")
        f.write(f"The optimal text representation and KNN configuration was frozen and evaluated using a Leave-One-Out loop over the 88 complete cases:\n\n")
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
