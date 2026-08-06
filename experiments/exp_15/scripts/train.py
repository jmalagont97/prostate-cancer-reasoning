import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, roc_curve
from sklearn.model_selection import LeaveOneOut
import torch
import spacy

# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

from embedkit import EmbedKit


def transform_text_representation(rep_name, X_tr_tfidf, X_va_tfidf, y_tr_soft, device):
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_tfidf)
    X_va_scaled = scaler.transform(X_va_tfidf)

    if rep_name == "raw":
        return X_tr_scaled, X_va_scaled
    elif rep_name == "pca":
        # Handle case where n_samples < n_features or variance thresholding
        n_comp = min(X_tr_scaled.shape[0], X_tr_scaled.shape[1])
        if n_comp < 2:
            return X_tr_scaled, X_va_scaled
        pca = PCA(n_components=0.90, random_state=42)
        X_tr_proj = pca.fit_transform(X_tr_scaled)
        X_va_proj = pca.transform(X_va_scaled)
        return X_tr_proj, X_va_proj
    elif rep_name == "embedkit_unsup":
        ek = EmbedKit(mode="unsupervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_proj = ek.fit_transform(X_tr_scaled)
        X_va_proj = ek.transform(X_va_scaled)
        return X_tr_proj, X_va_proj
    elif rep_name == "embedkit_sup":
        ek = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_proj = ek.fit_transform(X_tr_scaled, y_tr_soft)
        X_va_proj = ek.transform(X_va_scaled)
        return X_tr_proj, X_va_proj
    else:
        raise ValueError(f"Unknown representation: {rep_name}")


def main():
    exp_dir = project_root / "experiments" / "exp_15"
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
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align datasets by patient_id
    pids = df_design["patient_id"].values
    df_text = df_text[df_text["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    # Filter labeled complete cases (N=88) matching exp_7
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_text_labeled = df_text[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    pids_labeled = df_dec_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values

    # Construct uncertainty-guided soft targets (\tilde{y}_j)
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

    print("Pre-processing text prompts using spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    raw_texts = df_text_labeled["clinical_prompt_text"].values
    cleaned_texts = []
    for doc in nlp.pipe(raw_texts, batch_size=50):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_texts.append(" ".join(tokens))
    text_corpus_labeled = np.array(cleaned_texts)

    mf_list = [100, 300, 500, 1000, None]
    rep_list = ["raw", "pca", "embedkit_unsup", "embedkit_sup"]
    k_list = [1, 3, 5, 7, 9, 11, 15, 21]
    weights_list = ["uniform", "distance"]
    metrics_list = ["euclidean", "cosine"]

    grid = []
    for mf in mf_list:
        for rep in rep_list:
            for k in k_list:
                for w in weights_list:
                    for m in metrics_list:
                        grid.append({"max_features": mf, "representation": rep, "k": k, "weights": w, "metric": m})

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

        y_tr_soft = y_soft[train_idx]
        y_va_true = y_binary[val_idx]

        # Vectorize TF-IDF per max_features for this split
        tfidf_cache = {}
        for mf in mf_list:
            vec = TfidfVectorizer(max_features=mf, norm='l2')
            X_tr_tfidf = vec.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
            X_va_tfidf = vec.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)
            tfidf_cache[mf] = (X_tr_tfidf, X_va_tfidf)

        # Cache projected representations per (mf, rep) for this split
        rep_cache = {}
        for mf in mf_list:
            X_tr_tfidf, X_va_tfidf = tfidf_cache[mf]
            for rep in rep_list:
                X_tr_proj, X_va_proj = transform_text_representation(rep, X_tr_tfidf, X_va_tfidf, y_tr_soft, device)
                rep_cache[(mf, rep)] = (X_tr_proj, X_va_proj)

        for cfg_idx, cfg in enumerate(grid):
            mf = cfg["max_features"]
            rep = cfg["representation"]
            X_tr_proj, X_va_proj = rep_cache[(mf, rep)]

            knn = KNeighborsRegressor(
                n_neighbors=cfg["k"],
                weights=cfg["weights"],
                metric=cfg["metric"]
            )
            knn.fit(X_tr_proj, y_tr_soft)
            p_val_soft = knn.predict(X_va_proj)
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
            "max_features": str(cfg["max_features"]),
            "representation": cfg["representation"],
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
    best_mf_val = None if best_cfg["max_features"] == "None" else int(best_cfg["max_features"])
    best_hparams = {
        "max_features": str(best_cfg["max_features"]),
        "representation": str(best_cfg["representation"]),
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

    # Plot MCCV Grid Search Curves per max_features
    plt.figure(figsize=(10, 6))
    for mf in mf_list:
        mf_str = str(mf)
        subset = df_grid[(df_grid["max_features"] == mf_str) & (df_grid["representation"] == best_hparams["representation"]) & (df_grid["weights"] == best_hparams["weights"]) & (df_grid["metric"] == best_hparams["metric"])].sort_values("k")
        if not subset.empty:
            plt.plot(subset["k"], subset["mean_macro_f1"], marker='o', label=f"max_features={mf_str}")

    plt.title(f"Text Fuzzy KNN 100-Split MCCV Grid Search (rep={best_hparams['representation']}, weights={best_hparams['weights']}, metric={best_hparams['metric']})", fontsize=11, fontweight="bold")
    plt.xlabel("Number of Neighbors (k)", fontsize=10)
    plt.ylabel("Mean Validation Macro-F1", fontsize=10)
    plt.xticks(k_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
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

    best_rep = best_hparams["representation"]

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(y_binary)):
        y_tr_soft = y_soft[train_idx]

        vec_loo = TfidfVectorizer(max_features=best_mf_val, norm='l2')
        X_tr_tfidf = vec_loo.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
        X_va_tfidf = vec_loo.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)

        X_tr_proj, X_va_proj = transform_text_representation(best_rep, X_tr_tfidf, X_va_tfidf, y_tr_soft, device)

        knn_best = KNeighborsRegressor(
            n_neighbors=best_hparams["k"],
            weights=best_hparams["weights"],
            metric=best_hparams["metric"]
        )
        knn_best.fit(X_tr_proj, y_tr_soft)
        p_val_soft = knn_best.predict(X_va_proj)[0]

        oof_p_soft.append(p_val_soft)
        oof_y_true.append(y_binary[val_idx[0]])

        if (fold_idx + 1) % 20 == 0:
            print(f"Processed LOOCV folds: {fold_idx + 1}/88.")

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
    plt.title(f"Text Fuzzy KNN LOOCV Confusion Matrix (Macro-F1: {final_macro_f1:.4f})", fontsize=11, fontweight="bold")
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
    plt.title("Text Fuzzy KNN LOOCV ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Text Fuzzy KNN Sweep & LOOCV (exp_15) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Distance-Weighted Fuzzy KNN Regressor (`KNeighborsRegressor`) on TF-IDF Text Prompts  \n")
        f.write(f"**Dataset**: Labeled Complete-Case Text Dataset ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Phase A: 100-Split MCCV Grid Search Results\n")
        f.write(f"- **Best Configuration Found**:\n")
        f.write(f"  - **Vocabulary Size (`max_features`)**: `{best_hparams['max_features']}`  \n")
        f.write(f"  - **Representation**: `{best_hparams['representation']}`  \n")
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

        f.write("## Comparison against Baseline (exp_7 Standard Text KNN)\n")
        f.write("| Model / Harness | max_features | Representation | Hyperparameters (k, w, m) | MCCV Mean Macro-F1 | LOOCV Macro-F1 | LOOCV Accuracy | Sensitivity | Specificity | AUROC |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **`exp_7` (Standard KNN)** | 500 | pca | k=1, uniform, cosine | 0.6329 | 0.6988 | 71.59% | 0.7778 | 0.6176 | — |\n")
        f.write(f"| **`exp_15` (Fuzzy KNN)** | **{best_hparams['max_features']}** | **{best_hparams['representation']}** | k={best_hparams['k']}, {best_hparams['weights']}, {best_hparams['metric']} | **{best_hparams['mccv_mean_macro_f1']:.4f}** | **{final_macro_f1:.4f}** | **{final_acc*100:.2f}%** | **{final_sens:.4f}** | **{final_spec:.4f}** | **{final_auroc:.4f}** |\n")

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
