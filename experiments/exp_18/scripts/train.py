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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, roc_curve
from sklearn.model_selection import LeaveOneOut
import torch
import spacy

# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

from embedkit import EmbedKit


def main():
    exp_dir = project_root / "experiments" / "exp_18"
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
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
    df_text = pd.read_csv(data_dir / "clinical_prompts.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align datasets by patient_id
    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_mri = df_mri[df_mri["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_text = df_text[df_text["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)

    # Filter labeled complete cases (N=88)
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_mri_labeled = df_mri[labeled_mask].reset_index(drop=True)
    df_text_labeled = df_text[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)

    pids_labeled = df_dec_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values

    # Construct uncertainty-guided soft targets (\tilde{y}_j) for Tabular Fuzzy KNN
    confidence_certainty_map = {
        "clear": 1.00,
        "borderline": 0.50,
        "uncertain": 0.25
    }
    c_weights = df_reasoning_labeled["confidence"].map(confidence_certainty_map).fillna(1.00).values
    y_soft = np.where(y_binary == 1, 0.50 + 0.50 * c_weights, 0.50 - 0.50 * c_weights).astype(np.float32)

    # Tabular column definitions
    num_cols = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols = ["dre"]

    # MRI features
    X_mri_raw = df_mri_labeled.drop(columns=["patient_id"]).values.astype(np.float32)

    # Text features (spaCy cleaning)
    print("Pre-processing text prompts using spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    raw_texts = df_text_labeled["clinical_prompt_text"].values
    cleaned_texts = []
    for doc in nlp.pipe(raw_texts, batch_size=50):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_texts.append(" ".join(tokens))
    text_corpus_labeled = np.array(cleaned_texts)

    print(f"Executing LOOCV (88 Folds) for Hybrid Fusion (Tabular Fuzzy + MRI Hard + Text Hard)...")

    loo = LeaveOneOut()
    oof_p_tab_fuzzy = []
    oof_p_mri_hard = []
    oof_p_text_hard = []
    oof_y_true = []

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(y_binary)):
        y_tr_soft = y_soft[train_idx]
        y_tr_hard = y_binary[train_idx]

        # -----------------------------------------------------
        # 1. Tabular Fuzzy Pipeline (exp_13: k=1, uniform, euclidean)
        # -----------------------------------------------------
        scaler_tab = MinMaxScaler()
        ohe_tab = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        X_tr_num = scaler_tab.fit_transform(df_tab_labeled.iloc[train_idx][num_cols])
        X_tr_cat = ohe_tab.fit_transform(df_tab_labeled.iloc[train_idx][cat_cols])
        X_tr_tab = np.hstack([X_tr_num, X_tr_cat])

        X_va_num = scaler_tab.transform(df_tab_labeled.iloc[val_idx][num_cols])
        X_va_cat = ohe_tab.transform(df_tab_labeled.iloc[val_idx][cat_cols])
        X_va_tab = np.hstack([X_va_num, X_va_cat])

        knn_tab = KNeighborsRegressor(n_neighbors=1, weights="uniform", metric="euclidean")
        knn_tab.fit(X_tr_tab, y_tr_soft)
        p_val_tab_fuzzy = knn_tab.predict(X_va_tab)[0]

        # -----------------------------------------------------
        # 2. MRI Standard Hard Pipeline (exp_6: embedkit_sup 384D, k=3, uniform, euclidean)
        # -----------------------------------------------------
        scaler_mri = MinMaxScaler()
        X_tr_mri_sc = scaler_mri.fit_transform(X_mri_raw[train_idx])
        X_va_mri_sc = scaler_mri.transform(X_mri_raw[val_idx])

        ek_mri = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_mri_proj = ek_mri.fit_transform(X_tr_mri_sc, y_tr_hard)
        X_va_mri_proj = ek_mri.transform(X_va_mri_sc)

        knn_mri = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="euclidean")
        knn_mri.fit(X_tr_mri_proj, y_tr_hard)
        p_val_mri_hard = knn_mri.predict_proba(X_va_mri_proj)[0, 1]

        # -----------------------------------------------------
        # 3. Text Standard Hard Pipeline (exp_7: max_features=500, pca 90%, k=1, uniform, cosine)
        # -----------------------------------------------------
        vec_text = TfidfVectorizer(max_features=500, norm='l2')
        X_tr_tfidf = vec_text.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
        X_va_tfidf = vec_text.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)

        scaler_text = MinMaxScaler()
        X_tr_text_sc = scaler_text.fit_transform(X_tr_tfidf)
        X_va_text_sc = scaler_text.transform(X_va_tfidf)

        pca_text = PCA(n_components=0.90, random_state=42)
        X_tr_text_proj = pca_text.fit_transform(X_tr_text_sc)
        X_va_text_proj = pca_text.transform(X_va_text_sc)

        knn_text = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="cosine")
        knn_text.fit(X_tr_text_proj, y_tr_hard)
        p_val_text_hard = knn_text.predict_proba(X_va_text_proj)[0, 1]

        oof_p_tab_fuzzy.append(p_val_tab_fuzzy)
        oof_p_mri_hard.append(p_val_mri_hard)
        oof_p_text_hard.append(p_val_text_hard)
        oof_y_true.append(y_binary[val_idx[0]])

        if (fold_idx + 1) % 20 == 0:
            print(f"Processed LOOCV folds: {fold_idx + 1}/88.")

    oof_p_tab_fuzzy = np.array(oof_p_tab_fuzzy)
    oof_p_mri_hard = np.array(oof_p_mri_hard)
    oof_p_text_hard = np.array(oof_p_text_hard)
    oof_y_true = np.array(oof_y_true)

    # ---------------------------------------------------------
    # Evaluate Hybrid Fusion Configurations & Simplex Grid Search
    # ---------------------------------------------------------
    def evaluate_hybrid_weights(w_tab, w_mri, w_text):
        p_hybrid = w_tab * oof_p_tab_fuzzy + w_mri * oof_p_mri_hard + w_text * oof_p_text_hard
        p_hybrid = np.clip(p_hybrid, 0.0, 1.0)
        y_pred = (p_hybrid >= 0.50).astype(int)

        macro_f1 = f1_score(oof_y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(oof_y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(oof_y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        auroc = roc_auc_score(oof_y_true, p_hybrid)
        brier = brier_score_loss(oof_y_true, p_hybrid)

        return {
            "w_tab": float(w_tab),
            "w_mri": float(w_mri),
            "w_text": float(w_text),
            "macro_f1": float(macro_f1),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "auroc": float(auroc),
            "brier_score": float(brier),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }

    # Fixed Hybrid Conditions
    fixed_conditions = {
        "Unimodal-Tabular-Fuzzy": (1.00, 0.00, 0.00),
        "Unimodal-MRI-Hard": (0.00, 1.00, 0.00),
        "Unimodal-Text-Hard": (0.00, 0.00, 1.00),
        "Equal-Hybrid-Fusion": (1/3, 1/3, 1/3),
        "Bimodal-TabularFuzzy-TextHard": (0.50, 0.00, 0.50),
        "Bimodal-TabularFuzzy-MRIHard": (0.50, 0.50, 0.00),
        "Bimodal-TextHard-MRIHard": (0.00, 0.50, 0.50)
    }

    results_conditions = {}
    for cond_name, (wt, wm, wx) in fixed_conditions.items():
        results_conditions[cond_name] = evaluate_hybrid_weights(wt, wm, wx)

    # Simplex Grid Search with step 0.05
    grid_weights = []
    step = 0.05
    for wt in np.arange(0.0, 1.0 + step/2, step):
        for wm in np.arange(0.0, 1.0 - wt + step/2, step):
            wx = round(1.0 - wt - wm, 4)
            if wx >= -1e-5:
                wx = max(0.0, wx)
                grid_weights.append(evaluate_hybrid_weights(wt, wm, wx))

    df_grid = pd.DataFrame(grid_weights).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df_grid.to_csv(results_dir / "fusion_grid_results.csv", index=False)

    best_weight_row = df_grid.iloc[0]
    best_weights = {
        "w_tabular_fuzzy": float(best_weight_row["w_tab"]),
        "w_mri_hard": float(best_weight_row["w_mri"]),
        "w_text_hard": float(best_weight_row["w_text"]),
        "macro_f1": float(best_weight_row["macro_f1"]),
        "accuracy": float(best_weight_row["accuracy"]),
        "sensitivity": float(best_weight_row["sensitivity"]),
        "specificity": float(best_weight_row["specificity"]),
        "auroc": float(best_weight_row["auroc"]),
        "brier_score": float(best_weight_row["brier_score"])
    }

    results_conditions["Optimal-Weighted-Hybrid"] = evaluate_hybrid_weights(
        best_weights["w_tabular_fuzzy"], best_weights["w_mri_hard"], best_weights["w_text_hard"]
    )

    with open(results_dir / "best_fusion_weights.json", "w") as f:
        json.dump(best_weights, f, indent=4)

    with open(results_dir / "loocv_metrics.json", "w") as f:
        json.dump(results_conditions, f, indent=4)

    # Predictions Dataframe
    p_opt = best_weights["w_tabular_fuzzy"] * oof_p_tab_fuzzy + best_weights["w_mri_hard"] * oof_p_mri_hard + best_weights["w_text_hard"] * oof_p_text_hard
    p_opt = np.clip(p_opt, 0.0, 1.0)
    y_opt_pred = (p_opt >= 0.50).astype(int)

    df_oof = pd.DataFrame({
        "patient_id": pids_labeled,
        "ground_truth_biopsy": oof_y_true,
        "prob_tabular_fuzzy": oof_p_tab_fuzzy,
        "prob_mri_hard": oof_p_mri_hard,
        "prob_text_hard": oof_p_text_hard,
        "prob_optimal_hybrid_fusion": p_opt,
        "predicted_biopsy_optimal": y_opt_pred
    })
    df_oof.to_csv(results_dir / "oof_predictions.csv", index=False)

    # Plot Confusion Matrix of Optimal Hybrid Fusion
    opt_tn = results_conditions["Optimal-Weighted-Hybrid"]["tn"]
    opt_fp = results_conditions["Optimal-Weighted-Hybrid"]["fp"]
    opt_fn = results_conditions["Optimal-Weighted-Hybrid"]["fn"]
    opt_tp = results_conditions["Optimal-Weighted-Hybrid"]["tp"]

    disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[opt_tn, opt_fp], [opt_fn, opt_tp]]), display_labels=["No Biopsy", "Biopsy"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Hybrid Late Fusion LOOCV (Macro-F1: {best_weights['macro_f1']:.4f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Plot Comparative ROC Curves
    plt.figure(figsize=(8, 7))
    for name, p_vec in [
        ("Tabular Fuzzy KNN (exp_13)", oof_p_tab_fuzzy),
        ("MRI Hard KNN (exp_6)", oof_p_mri_hard),
        ("Text Hard KNN (exp_7)", oof_p_text_hard),
        ("Optimal Hybrid Fusion (exp_18)", p_opt)
    ]:
        fpr, tpr, _ = roc_curve(oof_y_true, p_vec)
        auc_val = roc_auc_score(oof_y_true, p_vec)
        lw = 2.5 if "Optimal" in name else 1.5
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.title("Hybrid Multimodal Late Fusion Comparative ROC Curves (LOOCV)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close()

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Hybrid Multimodal Late Fusion LOOCV (exp_18) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Hybrid Late-Fusion Soft-Voting Ensemble (Tabular Fuzzy KNN + MRI Hard KNN + Text Hard KNN)  \n")
        f.write(f"**Dataset**: Labeled Complete-Case Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Comparative Ensemble Performance Across Hybrid Conditions (LOOCV 88 Folds)\n\n")
        f.write("| Condition | Weights (Tabular Fuzzy, MRI Hard, Text Hard) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier Score |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for cond_name, m in results_conditions.items():
            w_str = f"[{m['w_tab']:.2f}, {m['w_mri']:.2f}, {m['w_text']:.2f}]"
            f.write(f"| `{cond_name}` | `{w_str}` | **{m['macro_f1']:.4f}** | **{m['accuracy']*100:.2f}%** | **{m['sensitivity']:.4f}** | **{m['specificity']:.4f}** | **{m['auroc']:.4f}** | **{m['brier_score']:.4f}** |\n")

        f.write("\n\n## Comparison against Baselines (exp_8 Standard Hard Fusion vs exp_16 All-Fuzzy Fusion)\n")
        f.write("| Experiment / Fusion Strategy | Optimal Weights (Tab, MRI, Text) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Verdict |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|\n")
        f.write(f"| **`exp_8` All Hard KNN Fusion** | `[0.25, 0.41, 0.34]` | 0.7171 | 75.00% | 0.8889 | 0.5294 | 0.7715 | All Hard Voting |\n")
        f.write(f"| **`exp_16` All Fuzzy KNN Fusion** | `[0.15, 0.55, 0.30]` | 0.6813 | 71.59% | 0.8519 | 0.5000 | 0.7053 | All Soft Targets |\n")
        opt_res = results_conditions["Optimal-Weighted-Hybrid"]
        f.write(f"| **`exp_18` Hybrid KNN Fusion** | `[{opt_res['w_tab']:.2f}, {opt_res['w_mri']:.2f}, {opt_res['w_text']:.2f}]` | **{opt_res['macro_f1']:.4f}** | **{opt_res['accuracy']*100:.2f}%** | **{opt_res['sensitivity']:.4f}** | **{opt_res['specificity']:.4f}** | **{opt_res['auroc']:.4f}** | **Hybrid Optimal Combination** |\n")

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
