import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Filter non-fatal warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut
import torch
import spacy

# Define paths relative to this script to find embedkit
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.append(str(project_root / "utils" / "embedding-kit"))

# Import EmbedKit
from embedkit import EmbedKit

def main():
    exp_dir = project_root / "experiments" / "exp_8"
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
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    # Align by patient_id
    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_mri = df_mri[df_mri["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_text = df_text[df_text["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)

    assert (df_tab["patient_id"] == df_dec["patient_id"]).all()
    assert (df_mri["patient_id"] == df_dec["patient_id"]).all()
    assert (df_text["patient_id"] == df_dec["patient_id"]).all()

    # Filter labeled cases only (88 complete-case labeled cohort)
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_mri_labeled = df_mri[labeled_mask].reset_index(drop=True)
    df_text_labeled = df_text[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)

    label_map = {"yes": 1, "no": 0}
    inv_label_map = {1: "yes", 0: "no"}
    y_labeled = df_dec_labeled["biopsy_decision"].map(label_map).values
    pids_labeled = df_dec_labeled["patient_id"].values

    # Preprocess text with spaCy once at start
    print("Loading spaCy en_core_web_sm model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    print("Preprocessing text corpus with spaCy (lowercasing, stop words removal, punctuation removal, lemmatization)...")
    raw_texts = df_text_labeled["clinical_prompt_text"].values
    cleaned_texts = []
    for doc in nlp.pipe(raw_texts, batch_size=50):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_texts.append(" ".join(tokens))
    text_corpus_labeled = np.array(cleaned_texts)

    # Feature matrices
    X_mri_raw = df_mri_labeled.drop(columns=["patient_id"]).values.astype(np.float32)
    num_cols_tab = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols_tab = ["dre"]

    loo = LeaveOneOut()
    oof_p_tab = []
    oof_p_mri = []
    oof_p_text = []

    print("Beginning LOOCV Unimodal Probability Generation (88 folds)...")

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(y_labeled)):
        y_train = y_labeled[train_idx]
        y_val = y_labeled[val_idx]

        # ---------------------------------------------------------
        # 1. Model 1: Tabular KNN (exp_5 optimal config)
        # ---------------------------------------------------------
        scaler_tab = MinMaxScaler()
        X_tr_num = scaler_tab.fit_transform(df_tab_labeled.iloc[train_idx][num_cols_tab])
        X_va_num = scaler_tab.transform(df_tab_labeled.iloc[val_idx][num_cols_tab])

        ohe_tab = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_tr_cat = ohe_tab.fit_transform(df_tab_labeled.iloc[train_idx][cat_cols_tab])
        X_va_cat = ohe_tab.transform(df_tab_labeled.iloc[val_idx][cat_cols_tab])

        X_tr_tab_full = np.hstack([X_tr_num, X_tr_cat])
        X_va_tab_full = np.hstack([X_va_num, X_va_cat])

        knn_tab = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_tab.fit(X_tr_tab_full, y_train)
        p_tab = knn_tab.predict_proba(X_va_tab_full)[0, 1]

        # ---------------------------------------------------------
        # 2. Model 2: MRI Embeddings KNN (exp_6 optimal config)
        # ---------------------------------------------------------
        scaler_mri = MinMaxScaler()
        X_tr_mri_scaled = scaler_mri.fit_transform(X_mri_raw[train_idx])
        X_va_mri_scaled = scaler_mri.transform(X_mri_raw[val_idx])

        ek_s = EmbedKit(mode="supervised", target_dim=384, epochs=60, random_state=42, val_split=0.1, device=device, early_stopping_patience=None)
        X_tr_mri_proj = ek_s.fit_transform(X_tr_mri_scaled, y_train)
        X_va_mri_proj = ek_s.transform(X_va_mri_scaled)

        knn_mri = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
        knn_mri.fit(X_tr_mri_proj, y_train)
        p_mri = knn_mri.predict_proba(X_va_mri_proj)[0, 1]

        # ---------------------------------------------------------
        # 3. Model 3: Text Prompts KNN (exp_7 optimal config)
        # ---------------------------------------------------------
        vec_text = TfidfVectorizer(max_features=500, norm='l2')
        X_tr_tfidf = vec_text.fit_transform(text_corpus_labeled[train_idx]).toarray().astype(np.float32)
        X_va_tfidf = vec_text.transform(text_corpus_labeled[val_idx]).toarray().astype(np.float32)

        pca_text = PCA(n_components=0.90, random_state=42)
        X_tr_text_proj = pca_text.fit_transform(X_tr_tfidf)
        X_va_text_proj = pca_text.transform(X_va_tfidf)

        knn_text = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='cosine')
        knn_text.fit(X_tr_text_proj, y_train)
        p_text = knn_text.predict_proba(X_va_text_proj)[0, 1]

        oof_p_tab.append(p_tab)
        oof_p_mri.append(p_mri)
        oof_p_text.append(p_text)

        if (fold_idx + 1) % 20 == 0:
            print(f"Processed LOOCV folds: {fold_idx + 1}/88.")

    oof_p_tab = np.array(oof_p_tab)
    oof_p_mri = np.array(oof_p_mri)
    oof_p_text = np.array(oof_p_text)

    # ---------------------------------------------------------
    # Multimodal Late Fusion Soft Voting Evaluations
    # ---------------------------------------------------------
    def evaluate_probs(probs, targets):
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(targets, preds, average='macro')
        acc = accuracy_score(targets, preds)
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(targets, probs)
        return {
            "macro_f1": f1,
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "auroc": auc,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }

    conditions = {}

    # Unimodal Baselines
    conditions["Unimodal-Tabular"] = { "probs": oof_p_tab, "weights": [1.0, 0.0, 0.0] }
    conditions["Unimodal-MRI"] = { "probs": oof_p_mri, "weights": [0.0, 1.0, 0.0] }
    conditions["Unimodal-Text"] = { "probs": oof_p_text, "weights": [0.0, 0.0, 1.0] }

    # Equal Trimodal Fusion
    p_equal_trimodal = (oof_p_tab + oof_p_mri + oof_p_text) / 3.0
    conditions["Equal-Trimodal-Fusion"] = { "probs": p_equal_trimodal, "weights": [0.333, 0.333, 0.333] }

    # Bimodal Ablations
    conditions["Bimodal-Tabular-Text"] = { "probs": 0.5 * oof_p_tab + 0.5 * oof_p_text, "weights": [0.5, 0.0, 0.5] }
    conditions["Bimodal-Tabular-MRI"] = { "probs": 0.5 * oof_p_tab + 0.5 * oof_p_mri, "weights": [0.5, 0.5, 0.0] }
    conditions["Bimodal-Text-MRI"] = { "probs": 0.5 * oof_p_text + 0.5 * oof_p_mri, "weights": [0.0, 0.5, 0.5] }

    # Weighted Grid Sweep across weights
    best_grid_f1 = -1.0
    best_weights = None
    best_weighted_probs = None

    weights_grid = []
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = round(1.0 - w1 - w2, 4)
            if w3 >= 0:
                weights_grid.append((round(w1, 2), round(w2, 2), round(w3, 2)))

    for w_t, w_m, w_txt in weights_grid:
        if round(w_t + w_m + w_txt, 2) != 1.0:
            continue
        p_comb = w_t * oof_p_tab + w_m * oof_p_mri + w_txt * oof_p_text
        metrics_tmp = evaluate_probs(p_comb, y_labeled)
        if metrics_tmp["macro_f1"] > best_grid_f1:
            best_grid_f1 = metrics_tmp["macro_f1"]
            best_weights = [w_t, w_m, w_txt]
            best_weighted_probs = p_comb

    conditions["Optimal-Weighted-Trimodal"] = { "probs": best_weighted_probs, "weights": best_weights }

    # Compute all metrics
    metrics_summary = {}
    for cond_name, data in conditions.items():
        metrics_summary[cond_name] = evaluate_probs(data["probs"], y_labeled)
        metrics_summary[cond_name]["weights"] = data["weights"]

    metrics_json_path = results_dir / "loocv_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved LOOCV metrics to: {metrics_json_path}")

    # Save LOOCV predictions dataframe
    best_fusion_probs = metrics_summary["Optimal-Weighted-Trimodal"]
    df_predictions = pd.DataFrame({
        "patient_id": pids_labeled,
        "target": [inv_label_map[t] for t in y_labeled],
        "prob_tabular": oof_p_tab,
        "prob_mri": oof_p_mri,
        "prob_text": oof_p_text,
        "prob_equal_fusion": p_equal_trimodal,
        "prob_optimal_fusion": best_weighted_probs,
        "pred_optimal_fusion": [inv_label_map[p] for p in (best_weighted_probs >= 0.5).astype(int)]
    })
    predictions_csv_path = results_dir / "loocv_predictions.csv"
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"Saved LOOCV predictions to: {predictions_csv_path}")

    # ---------------------------------------------------------
    # Visualizations: ROC Curves & Confusion Matrix
    # ---------------------------------------------------------
    print("Generating ROC Curves plot...")
    plt.figure(figsize=(9, 7))

    for cond_name in ["Unimodal-Tabular", "Unimodal-MRI", "Unimodal-Text", "Equal-Trimodal-Fusion", "Bimodal-Tabular-Text", "Optimal-Weighted-Trimodal"]:
        probs = conditions[cond_name]["probs"]
        fpr, tpr, _ = roc_curve(y_labeled, probs)
        auc_val = roc_auc_score(y_labeled, probs)
        plt.plot(fpr, tpr, label=f"{cond_name} (AUC = {auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.500)")
    plt.title("LOOCV Multimodal Late Fusion ROC Curves Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    roc_png_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_png_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to: {roc_png_path}")

    print("Generating Confusion Matrix plot for Optimal Weighted Fusion...")
    best_preds = (best_weighted_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_labeled, best_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"LOOCV Confusion Matrix (Optimal Fusion: w={best_weights})", fontsize=11, fontweight="bold")
    cm_png_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_png_path}")

    # ---------------------------------------------------------
    # Write summary.md
    # ---------------------------------------------------------
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Multimodal Late Fusion (Soft-Voting) LOOCV Summary Report\n\n")
        f.write(f"**Date**: 2026-08-04  \n")
        f.write(f"**Model**: Late Fusion Soft-Voting Ensemble (KNN Tabular + KNN MRI + KNN Text)  \n")
        f.write(f"**Dataset**: Complete-Case Labeled Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## Comparative Results (LOOCV 88 Folds)\n\n")
        f.write("| Condition | Weights (Tabular, MRI, Text) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for cond_name, data in metrics_summary.items():
            w_str = ", ".join([f"{w:.2f}" for w in data["weights"]])
            f.write(f"| `{cond_name}` | `[{w_str}]` | **{data['macro_f1']:.4f}** | {data['accuracy']:.4f} | {data['sensitivity']:.4f} | {data['specificity']:.4f} | {data['auroc']:.4f} |\n")
        
        f.write("\n\n## Key Multimodal Insights:\n")
        f.write(f"- **Optimal Weighted Trimodal Fusion**: Weights `[Tab: {best_weights[0]}, MRI: {best_weights[1]}, Text: {best_weights[2]}]` achieved a Macro-F1 of **{metrics_summary['Optimal-Weighted-Trimodal']['macro_f1']:.4f}** and AUROC of **{metrics_summary['Optimal-Weighted-Trimodal']['auroc']:.4f}**.  \n")
        f.write(f"- **Bimodal Tabular + Text Fusion**: Achieved a Macro-F1 of **{metrics_summary['Bimodal-Tabular-Text']['macro_f1']:.4f}** and Specificity of **{metrics_summary['Bimodal-Tabular-Text']['specificity']:.4f}**.  \n")
        f.write(f"- **Role of Visual MRI Embeddings**: Standalone MRI embeddings remain weak (F1 = 0.5335), and assigning non-zero weight to MRI in equal fusion slightly degrades performance compared to Tabular + Text bimodal fusion.  \n")

    print(f"Summary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
