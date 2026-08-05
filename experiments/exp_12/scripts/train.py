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
    var_p = ((p1 - p_mean)**2 + (p2 - p_mean)**2 + (p3 - p_mean)**2) / 3.0
    std_p = np.sqrt(var_p)
    delta_margin = np.abs(p_mean - 0.50)
    ici = (2.0 * delta_margin) * (1.0 - 2.0 * std_p)
    return np.clip(ici, 0.0, 1.0), p_mean, std_p, delta_margin


def main():
    exp_dir = project_root / "experiments" / "exp_12"
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
    # LOOCV 3D Probability State Vector Loop (88 Folds)
    # ---------------------------------------------------------
    print("Beginning LOOCV 3D Probability State Vector Loop across 88 folds...")

    loo = LeaveOneOut()
    oof_p_tab = []
    oof_p_mri = []
    oof_p_text = []

    feature_importances_list = []
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

        # 2. Build 3D Probability Matrix for 87 training cases: [p_tab, p_mri, p_text]
        P_tr = np.column_stack([p_tab_tr, p_mri_tr, p_text_tr])

        # 3. Fit DecisionTreeClassifier(max_depth=3, class_weight='balanced') on 3D probability matrix
        dt_fold = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        dt_fold.fit(P_tr, y_tr_conf)

        feature_importances_list.append(dt_fold.feature_importances_)

        # 4. Predict test patient using 3D probability vector
        P_va = np.array([[p_tab_val, p_mri_val, p_text_val]])
        pred_conf_idx = dt_fold.predict(P_va)[0]
        oof_pred_conf_idx.append(pred_conf_idx)

        if (fold_idx + 1) % 20 == 0:
            print(f"Processed 3D vector LOOCV folds: {fold_idx + 1}/88.")

    oof_ici, oof_p_mean, oof_std, oof_delta = compute_ici(oof_p_tab, oof_p_mri, oof_p_text)
    oof_pred_conf_idx = np.array(oof_pred_conf_idx)

    mean_importances = np.mean(feature_importances_list, axis=0)
    std_importances = np.std(feature_importances_list, axis=0)

    feature_importances_dict = {
        "p_tabular": float(mean_importances[0]),
        "p_mri": float(mean_importances[1]),
        "p_text": float(mean_importances[2]),
        "std_p_tabular": float(std_importances[0]),
        "std_p_mri": float(std_importances[1]),
        "std_p_text": float(std_importances[2])
    }

    feat_imp_path = results_dir / "feature_importances.json"
    with open(feat_imp_path, "w") as f:
        json.dump(feature_importances_dict, f, indent=4)
    print(f"Saved feature importances to: {feat_imp_path}")

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
        "feature_importances": feature_importances_dict
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
    print("Generating modal feature importances bar plot...")
    plt.figure(figsize=(7, 5))
    modalities = ["p_tabular", "p_mri", "p_text"]
    imp_values = [mean_importances[0], mean_importances[1], mean_importances[2]]
    imp_errors = [std_importances[0], std_importances[1], std_importances[2]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = plt.bar(modalities, imp_values, yerr=imp_errors, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title("Modal Feature Importances in 3D Probability Decision Tree", fontsize=11, fontweight="bold")
    plt.xlabel("Unimodal Probability Output Feature", fontsize=10)
    plt.ylabel("Gini Feature Importance", fontsize=10)
    plt.ylim(0, max(imp_values) + max(imp_errors) + 0.15)
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.tight_layout()
    bar_png_path = figures_dir / "feature_importance_bar.png"
    plt.savefig(bar_png_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {bar_png_path}")

    print("Generating 3x3 Confusion Matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_3class, display_labels=["uncertain", "borderline", "clear"])
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f"LOOCV 3D Probability Vector Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix_3class.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_png_path}")

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# 3D Probability State Vector Diagnostic Confidence Prediction (exp_12) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write(f"**Model**: Class-Weighted 3D Probability Vector Decision Tree ($p = [p_{{tab}}, p_{{mri}}, p_{{text}}] \\to \\text{{confidence}}$)  \n")
        f.write(f"**Dataset**: Labeled Reasoning Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Modal Feature Importances\n")
        f.write(f"- **Tabular Probability ($p_{{\\text{{tab}}}}$)**: `{mean_importances[0]:.4f}` ($\text{{std}} = {std_importances[0]:.4f}$)  \n")
        f.write(f"- **MRI Probability ($p_{{\\text{{mri}}}}$)**: `{mean_importances[1]:.4f}` ($\text{{std}} = {std_importances[1]:.4f}$)  \n")
        f.write(f"- **Text Probability ($p_{{\\text{{text}}}}$)**: `{mean_importances[2]:.4f}` ($\text{{std}} = {std_importances[2]:.4f}$)  \n\n")

        f.write("## Out-of-Fold Evaluation Metrics (88 LOOCV Folds)\n")
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
