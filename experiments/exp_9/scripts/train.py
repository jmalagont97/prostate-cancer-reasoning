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
from sklearn.model_selection import LeaveOneOut
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
    exp_dir = project_root / "experiments" / "exp_9"
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

    # Filter to labeled cohort with valid confidence annotations (N=91)
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
    # Phase A: Meta-Threshold Learning over 100 MCCV Splits
    # ---------------------------------------------------------
    print("Beginning Phase A: 100 MCCV Splits Meta-Threshold Learning...")

    tau1_list = []
    tau2_list = []

    # Map mccv_design patient_id to our 91-labeled index
    pid_to_labeled_idx = {pid: i for i, pid in enumerate(pids_labeled)}

    for split_idx in range(100):
        split_col = f"split_{split_idx}"
        split_roles = df_design[df_design["patient_id"].isin(pids_labeled)].set_index("patient_id").loc[pids_labeled, split_col].values

        train_mask = (split_roles == 0)
        val_mask = (split_roles == 1)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) == 0 or len(val_indices) == 0:
            continue

        # 1. Fit unimodal models on training split
        # Tabular
        scaler_tab = MinMaxScaler()
        X_tr_num = scaler_tab.fit_transform(df_tab_labeled.iloc[train_indices][num_cols_tab])
        X_va_num = scaler_tab.transform(df_tab_labeled.iloc[val_indices][num_cols_tab])

        ohe_tab = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_tr_cat = ohe_tab.fit_transform(df_tab_labeled.iloc[train_indices][cat_cols_tab])
        X_va_cat = ohe_tab.transform(df_tab_labeled.iloc[val_indices][cat_cols_tab])

        X_tr_tab = np.hstack([X_tr_num, X_tr_cat])
        X_va_tab = np.hstack([X_va_num, X_va_cat])

        knn_tab = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_tab.fit(X_tr_tab, y_biopsy[train_indices])
        p_tab_val = knn_tab.predict_proba(X_va_tab)[:, 1]

        # MRI
        scaler_mri = MinMaxScaler()
        X_tr_mri = scaler_mri.fit_transform(X_mri_raw[train_indices])
        X_va_mri = scaler_mri.transform(X_mri_raw[val_indices])

        ek_s = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_mri_proj = ek_s.fit_transform(X_tr_mri, y_biopsy[train_indices])
        X_va_mri_proj = ek_s.transform(X_va_mri)

        knn_mri = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_mri.fit(X_tr_mri_proj, y_biopsy[train_indices])
        p_mri_val = knn_mri.predict_proba(X_va_mri_proj)[:, 1]

        # Text
        vec_text = TfidfVectorizer(max_features=500, norm='l2')
        X_tr_tfidf = vec_text.fit_transform(text_corpus_labeled[train_indices]).toarray().astype(np.float32)
        X_va_tfidf = vec_text.transform(text_corpus_labeled[val_indices]).toarray().astype(np.float32)

        pca_text = PCA(n_components=0.90, random_state=42)
        X_tr_text_proj = pca_text.fit_transform(X_tr_tfidf)
        X_va_text_proj = pca_text.transform(X_va_tfidf)

        knn_text = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='cosine')
        knn_text.fit(X_tr_text_proj, y_biopsy[train_indices])
        p_text_val = knn_text.predict_proba(X_va_text_proj)[:, 1]

        # 2. Compute ICI on validation split
        ici_val, _, _, _ = compute_ici(p_tab_val, p_mri_val, p_text_val)

        # 3. Fit 1D DecisionTree on validation ICI vs confidence labels
        dt = DecisionTreeClassifier(max_depth=2, random_state=42)
        dt.fit(ici_val.reshape(-1, 1), y_confidence[val_indices])

        tree = dt.tree_
        thresholds = sorted(tree.threshold[tree.threshold != -2])

        if len(thresholds) >= 2:
            tau1_list.append(thresholds[0])
            tau2_list.append(thresholds[1])
        elif len(thresholds) == 1:
            tau1_list.append(thresholds[0] * 0.5)
            tau2_list.append(thresholds[0])

        if (split_idx + 1) % 20 == 0:
            print(f"Processed MCCV splits: {split_idx + 1}/100.")

    bar_tau_1 = float(np.mean(tau1_list))
    bar_tau_2 = float(np.mean(tau2_list))

    meta_thresholds = {
        "bar_tau_1": bar_tau_1,
        "bar_tau_2": bar_tau_2,
        "std_tau_1": float(np.std(tau1_list)),
        "std_tau_2": float(np.std(tau2_list)),
        "num_splits_evaluated": len(tau1_list)
    }

    meta_json_path = results_dir / "meta_thresholds.json"
    with open(meta_json_path, "w") as f:
        json.dump(meta_thresholds, f, indent=4)
    print(f"Saved meta-thresholds to: {meta_json_path}")
    print(f"Learned Meta-Thresholds: bar_tau_1 = {bar_tau_1:.4f}, bar_tau_2 = {bar_tau_2:.4f}")

    # ---------------------------------------------------------
    # Phase B: Frozen LOOCV Evaluation (91 Folds)
    # ---------------------------------------------------------
    print("Beginning Phase B: Frozen LOOCV Evaluation across 91 folds...")

    loo = LeaveOneOut()
    oof_p_tab = []
    oof_p_mri = []
    oof_p_text = []

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(y_confidence)):
        y_tr_biopsy = y_biopsy[train_idx]

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
        p_tab = knn_tab.predict_proba(X_va_tab)[0, 1]

        # MRI
        scaler_mri = MinMaxScaler()
        X_tr_mri = scaler_mri.fit_transform(X_mri_raw[train_idx])
        X_va_mri = scaler_mri.transform(X_mri_raw[val_idx])

        ek_s = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_mri_proj = ek_s.fit_transform(X_tr_mri, y_tr_biopsy)
        X_va_mri_proj = ek_s.transform(X_va_mri)

        knn_mri = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_mri.fit(X_tr_mri_proj, y_tr_biopsy)
        p_mri = knn_mri.predict_proba(X_va_mri_proj)[0, 1]

        # Text
        vec_text = TfidfVectorizer(max_features=500, norm='l2')
        X_tr_tfidf = vec_text.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
        X_va_tfidf = vec_text.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)

        pca_text = PCA(n_components=0.90, random_state=42)
        X_tr_text_proj = pca_text.fit_transform(X_tr_tfidf)
        X_va_text_proj = pca_text.transform(X_va_tfidf)

        knn_text = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='cosine')
        knn_text.fit(X_tr_text_proj, y_tr_biopsy)
        p_text = knn_text.predict_proba(X_va_text_proj)[0, 1]

        oof_p_tab.append(p_tab)
        oof_p_mri.append(p_mri)
        oof_p_text.append(p_text)

        if (fold_idx + 1) % 30 == 0:
            print(f"Processed LOOCV folds: {fold_idx + 1}/91.")

    oof_ici, oof_p_mean, oof_std, oof_delta = compute_ici(oof_p_tab, oof_p_mri, oof_p_text)

    # Classify confidence applying frozen meta-thresholds
    oof_pred_conf_idx = []
    for ici_val in oof_ici:
        if ici_val < bar_tau_1:
            oof_pred_conf_idx.append(0)  # uncertain
        elif ici_val < bar_tau_2:
            oof_pred_conf_idx.append(1)  # borderline
        else:
            oof_pred_conf_idx.append(2)  # clear

    oof_pred_conf_idx = np.array(oof_pred_conf_idx)

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
        "meta_threshold_1": bar_tau_1,
        "meta_threshold_2": bar_tau_2
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
    print("Generating decision tree thresholds distribution plot...")
    plt.figure(figsize=(8, 5))
    plt.hist(tau1_list, bins=15, alpha=0.6, color='orange', label=r'Learned $\tau_1$ (Uncertain / Borderline)')
    plt.hist(tau2_list, bins=15, alpha=0.6, color='green', label=r'Learned $\tau_2$ (Borderline / Clear)')
    plt.axvline(bar_tau_1, color='darkorange', linestyle='--', linewidth=2, label=f'Meta $\\bar{{\\tau}}_1 = {bar_tau_1:.4f}$')
    plt.axvline(bar_tau_2, color='darkgreen', linestyle='--', linewidth=2, label=f'Meta $\\bar{{\\tau}}_2 = {bar_tau_2:.4f}$')
    plt.title("Distribution of Decision Tree ICI Cut-Points (100 MCCV Splits)", fontsize=11, fontweight="bold")
    plt.xlabel("Composite Reliability Index (ICI) Threshold", fontsize=10)
    plt.ylabel("Frequency across Splits", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    thresh_png_path = figures_dir / "decision_tree_thresholds.png"
    plt.savefig(thresh_png_path, dpi=300)
    plt.close()
    print(f"Saved thresholds plot to: {thresh_png_path}")

    print("Generating 3x3 Confusion Matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_3class, display_labels=["uncertain", "borderline", "clear"])
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f"LOOCV 3-Class Confidence Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix_3class.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_png_path}")

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Out-of-Fold Diagnostic Confidence Prediction (exp_9) Summary Report\n\n")
        f.write("**Date**: 2026-08-04  \n")
        f.write(f"**Model**: Out-of-Fold ICI + Decision Tree Meta-Thresholding (MCCV $\\to$ LOOCV)  \n")
        f.write(f"**Dataset**: Labeled Reasoning Cohort ($N_{{labeled}} = 91$)  \n\n")

        f.write("## Phase A: Learned Meta-Thresholds (100 MCCV Splits)\n")
        f.write(f"- **Meta-Threshold 1 ($\bar{{\\tau}}_1$, Uncertain / Borderline)**: `{bar_tau_1:.4f}` ($\text{{std}} = {np.std(tau1_list):.4f}$)  \n")
        f.write(f"- **Meta-Threshold 2 ($\bar{{\\tau}}_2$, Borderline / Clear)**: `{bar_tau_2:.4f}` ($\text{{std}} = {np.std(tau2_list):.4f}$)  \n\n")

        f.write("## Phase B: Frozen LOOCV Out-of-Fold Evaluation (91 Folds)\n")
        f.write(f"- **3-Class Macro-F1**: **`{macro_f1:.4f}`**  \n")
        f.write(f"- **Accuracy**: **`{acc:.4f}`** ({accuracy_score(y_confidence, oof_pred_conf_idx, normalize=False)}/91 correct)  \n")
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
