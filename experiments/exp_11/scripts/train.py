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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import spearmanr
import torch
import spacy

# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

from embedkit import EmbedKit


def compute_ici(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p_mean = (p1 + p2 + p3) / 3.0
    
    # Inter-modality standard deviation
    var_p = ((p1 - p_mean)**2 + (p2 - p_mean)**2 + (p3 - p_mean)**2) / 3.0
    std_p = np.sqrt(var_p)
    
    # Certitude margin
    delta_margin = np.abs(p_mean - 0.50)
    
    # Composite Reliability Index (ICI)
    ici = (2.0 * delta_margin) * (1.0 - 2.0 * std_p)
    return np.clip(ici, 0.0, 1.0), p_mean, std_p, delta_margin


def main():
    exp_dir = project_root / "experiments" / "exp_11"
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
    df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
    df_text = pd.read_csv(data_dir / "clinical_prompts.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align datasets by patient_id
    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_mri = df_mri[df_mri["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_text = df_text[df_text["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)

    # Filter to labeled cohort with valid confidence annotations (N=88)
    confidence_mask = df_reasoning["confidence"] != "NONE"
    df_tab_labeled = df_tab[confidence_mask].reset_index(drop=True)
    df_mri_labeled = df_mri[confidence_mask].reset_index(drop=True)
    df_text_labeled = df_text[confidence_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[confidence_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[confidence_mask].reset_index(drop=True)

    pids_labeled = df_dec_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_biopsy = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values

    confidence_map = {"uncertain": 0, "borderline": 1, "clear": 2}
    inv_confidence_map = {0: "uncertain", 1: "borderline", 2: "clear"}
    y_confidence = df_reasoning_labeled["confidence"].map(confidence_map).values

    print(f"Total labeled cases for confidence task: {len(y_confidence)}")
    print(f"Confidence distribution: {df_reasoning_labeled['confidence'].value_counts().to_dict()}")

    # Preprocess text with spaCy once at top
    print("Loading spaCy model en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    raw_texts = df_text_labeled["clinical_prompt_text"].values
    cleaned_texts = []
    for doc in nlp.pipe(raw_texts, batch_size=50):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_texts.append(" ".join(tokens))
    text_corpus_labeled = np.array(cleaned_texts)

    X_mri_raw = df_mri_labeled.drop(columns=["patient_id"]).values.astype(np.float32)
    num_cols_tab = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols_tab = ["dre"]

    # ---------------------------------------------------------
    # Pure Dynamic LOOCV Loop (88 Folds)
    # ---------------------------------------------------------
    print("Beginning Pure Dynamic LOOCV Loop across 88 folds...")

    loo = LeaveOneOut()
    oof_p_tab = []
    oof_p_mri = []
    oof_p_text = []

    fold_tau1_list = []
    fold_tau2_list = []
    oof_pred_conf_idx = []

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(y_confidence)):
        y_tr_biopsy = y_biopsy[train_idx]
        y_tr_conf = y_confidence[train_idx]

        # 1. Fit unimodal models on 87 training cases
        # Tabular
        scaler_tab = MinMaxScaler()
        X_tr_num = scaler_tab.fit_transform(df_tab_labeled.iloc[train_idx][num_cols_tab])
        X_va_num = scaler_tab.transform(df_tab_labeled.iloc[val_idx][num_cols_tab])

        ohe_tab = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_tr_cat = ohe_tab.fit_transform(df_tab_labeled.iloc[train_idx][cat_cols_tab])
        X_va_cat = ohe_tab.transform(df_tab_labeled.iloc[val_idx][cat_cols_tab])

        X_tr_tab = np.hstack([X_tr_num, X_tr_cat])
        X_va_tab = np.hstack([X_va_num, X_va_cat])

        knn_tab = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_tab.fit(X_tr_tab, y_tr_biopsy)
        p_tab_tr = knn_tab.predict_proba(X_tr_tab)[:, 1]
        p_tab_val = knn_tab.predict_proba(X_va_tab)[0, 1]

        # MRI
        scaler_mri = MinMaxScaler()
        X_tr_mri = scaler_mri.fit_transform(X_mri_raw[train_idx])
        X_va_mri = scaler_mri.transform(X_mri_raw[val_idx])

        ek_s = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_mri_proj = ek_s.fit_transform(X_tr_mri, y_tr_biopsy)
        X_va_mri_proj = ek_s.transform(X_va_mri)

        knn_mri = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_mri.fit(X_tr_mri_proj, y_tr_biopsy)
        p_mri_tr = knn_mri.predict_proba(X_tr_mri_proj)[:, 1]
        p_mri_val = knn_mri.predict_proba(X_va_mri_proj)[0, 1]

        # Text
        vec_text = TfidfVectorizer(max_features=500, norm='l2')
        X_tr_tfidf = vec_text.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
        X_va_tfidf = vec_text.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)

        pca_text = PCA(n_components=0.90, random_state=42)
        X_tr_text_proj = pca_text.fit_transform(X_tr_tfidf)
        X_va_text_proj = pca_text.transform(X_va_tfidf)

        knn_text = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='cosine')
        knn_text.fit(X_tr_text_proj, y_tr_biopsy)
        p_text_tr = knn_text.predict_proba(X_tr_text_proj)[:, 1]
        p_text_val = knn_text.predict_proba(X_va_text_proj)[0, 1]

        oof_p_tab.append(p_tab_val)
        oof_p_mri.append(p_mri_val)
        oof_p_text.append(p_text_val)

        # 2. Compute training ICI for the 87 training cases
        ici_tr, _, _, _ = compute_ici(p_tab_tr, p_mri_tr, p_text_tr)

        # 3. Fit DecisionTreeClassifier dynamically on the 87 training cases for fold_idx
        dt_fold = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
        dt_fold.fit(ici_tr.reshape(-1, 1), y_tr_conf)

        tree = dt_fold.tree_
        thresholds = sorted([t for t in tree.threshold if t != -2])

        if len(thresholds) >= 2:
            tau1_fold = thresholds[0]
            tau2_fold = thresholds[1]
        elif len(thresholds) == 1:
            tau1_fold = thresholds[0] * 0.5
            tau2_fold = thresholds[0]
        else:
            tau1_fold = 0.10
            tau2_fold = 0.30

        fold_tau1_list.append(tau1_fold)
        fold_tau2_list.append(tau2_fold)

        # 4. Compute test ICI and classify held-out test patient using dynamic fold thresholds
        ici_val, _, _, _ = compute_ici([p_tab_val], [p_mri_val], [p_text_val])
        ici_val_scalar = ici_val[0]

        if ici_val_scalar < tau1_fold:
            oof_pred_conf_idx.append(0)  # uncertain
        elif ici_val_scalar < tau2_fold:
            oof_pred_conf_idx.append(1)  # borderline
        else:
            oof_pred_conf_idx.append(2)  # clear

        if (fold_idx + 1) % 20 == 0:
            print(f"Processed dynamic LOOCV folds: {fold_idx + 1}/88.")

    oof_ici, oof_p_mean, oof_std, oof_delta = compute_ici(oof_p_tab, oof_p_mri, oof_p_text)
    oof_pred_conf_idx = np.array(oof_pred_conf_idx)

    # Save fold thresholds per fold
    df_fold_thresholds = pd.DataFrame({
        "fold_idx": np.arange(len(fold_tau1_list)),
        "held_out_patient_id": pids_labeled,
        "dynamic_tau_1": fold_tau1_list,
        "dynamic_tau_2": fold_tau2_list
    })
    thresh_csv_path = results_dir / "dynamic_thresholds_per_fold.csv"
    df_fold_thresholds.to_csv(thresh_csv_path, index=False)
    print(f"Saved dynamic thresholds per fold to: {thresh_csv_path}")

    # Calculate metrics
    macro_f1 = f1_score(y_confidence, oof_pred_conf_idx, average='macro')
    acc = accuracy_score(y_confidence, oof_pred_conf_idx)
    rho_val, p_val = spearmanr(oof_ici, y_confidence)

    cm_3class = confusion_matrix(y_confidence, oof_pred_conf_idx, labels=[0, 1, 2])

    loocv_metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "spearman_rho": float(rho_val),
        "spearman_pvalue": float(p_val),
        "confusion_matrix": cm_3class.tolist(),
        "mean_dynamic_tau_1": float(np.mean(fold_tau1_list)),
        "std_dynamic_tau_1": float(np.std(fold_tau1_list)),
        "mean_dynamic_tau_2": float(np.mean(fold_tau2_list)),
        "std_dynamic_tau_2": float(np.std(fold_tau2_list))
    }

    metrics_json_path = results_dir / "loocv_confidence_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(loocv_metrics, f, indent=4)
    print(f"Saved LOOCV confidence metrics to: {metrics_json_path}")

    # Save patient-level predictions dataframe
    df_predictions = pd.DataFrame({
        "patient_id": pids_labeled,
        "ground_truth_confidence": [inv_confidence_map[c] for c in y_confidence],
        "predicted_confidence": [inv_confidence_map[c] for c in oof_pred_conf_idx],
        "ici_score": oof_ici,
        "dynamic_tau_1": fold_tau1_list,
        "dynamic_tau_2": fold_tau2_list,
        "prob_mean": oof_p_mean,
        "prob_std": oof_std,
        "certitude_margin": oof_delta,
        "prob_tabular": oof_p_tab,
        "prob_mri": oof_p_mri,
        "prob_text": oof_p_text
    })
    predictions_csv_path = results_dir / "loocv_confidence_predictions.csv"
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"Saved LOOCV predictions to: {predictions_csv_path}")

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------
    print("Generating dynamic thresholds evolution plot across LOOCV folds...")
    plt.figure(figsize=(9, 5))
    plt.plot(df_fold_thresholds["fold_idx"], df_fold_thresholds["dynamic_tau_1"], color='orange', marker='o', markersize=3, label=r'Dynamic Fold $\tau_1^{(i)}$ (Uncertain/Borderline)')
    plt.plot(df_fold_thresholds["fold_idx"], df_fold_thresholds["dynamic_tau_2"], color='green', marker='s', markersize=3, label=r'Dynamic Fold $\tau_2^{(i)}$ (Borderline/Clear)')
    plt.axhline(np.mean(fold_tau1_list), color='darkorange', linestyle='--', label=f'Mean $\\tau_1 = {np.mean(fold_tau1_list):.4f}$')
    plt.axhline(np.mean(fold_tau2_list), color='darkgreen', linestyle='--', label=f'Mean $\\tau_2 = {np.mean(fold_tau2_list):.4f}$')
    plt.title("Evolution of Dynamic Local Decision Tree Cut-Points Across 88 LOOCV Folds", fontsize=11, fontweight="bold")
    plt.xlabel("LOOCV Fold Index (Held-Out Patient)", fontsize=10)
    plt.ylabel("Learned Threshold Value", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    thresh_png_path = figures_dir / "dynamic_thresholds_evolution.png"
    plt.savefig(thresh_png_path, dpi=300)
    plt.close()
    print(f"Saved thresholds plot to: {thresh_png_path}")

    print("Generating 3x3 Confusion Matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_3class, display_labels=["uncertain", "borderline", "clear"])
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f"Pure Dynamic LOOCV 3-Class Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix_3class.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_png_path}")

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Dynamic Out-of-Fold Diagnostic Confidence Prediction (exp_11) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Pure Dynamic LOOCV Decision Tree Thresholding ($ICI \\to \\text{{confidence}}$)  \n")
        f.write(f"**Dataset**: Labeled Reasoning Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Dynamic Threshold Stability across 88 LOOCV Folds\n")
        f.write(f"- **Mean Dynamic $\\tau_1$ (Uncertain / Borderline)**: `{np.mean(fold_tau1_list):.4f}` ($\text{{std}} = {np.std(fold_tau1_list):.4f}$)  \n")
        f.write(f"- **Mean Dynamic $\\tau_2$ (Borderline / Clear)**: `{np.mean(fold_tau2_list):.4f}` ($\text{{std}} = {np.std(fold_tau2_list):.4f}$)  \n\n")

        f.write("## Out-of-Fold Evaluation Metrics (88 Folds)\n")
        f.write(f"- **3-Class Macro-F1**: **`{macro_f1:.4f}`**  \n")
        f.write(f"- **Accuracy**: **`{acc:.4f}`** ({accuracy_score(y_confidence, oof_pred_conf_idx, normalize=False)}/88 correct)  \n")
        f.write(f"- **Spearman Rank Correlation ($\rho$)**: **`{rho_val:.4f}`** (p-value: `{p_val:.4e}`)  \n\n")

        f.write("### 3x3 Confusion Matrix Counts:\n")
        f.write("| Ground Truth \\ Predicted | Uncertain | Borderline | Clear |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Uncertain** | {cm_3class[0, 0]} | {cm_3class[0, 1]} | {cm_3class[0, 2]} |\n")
        f.write(f"| **Borderline** | {cm_3class[1, 0]} | {cm_3class[1, 1]} | {cm_3class[1, 2]} |\n")
        f.write(f"| **Clear** | {cm_3class[2, 0]} | {cm_3class[2, 1]} | {cm_3class[2, 2]} |\n")

    print(f"Summary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
