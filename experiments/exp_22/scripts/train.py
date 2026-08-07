import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import shap

warnings.filterwarnings("ignore")


class DistanceWeightedFuzzyKNN:
    def __init__(self, n_neighbors=1, metric='euclidean', m=2.0):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m = m

    def fit(self, X, y_soft):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_soft = np.array(y_soft, dtype=np.float64)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn.fit(self.X_train)
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances, indices = self.nn.kneighbors(X)

        if self.n_neighbors == 1:
            return self.y_soft[indices[:, 0]]

        probs = []
        for i in range(len(X)):
            dists = distances[i]
            idxs = indices[i]

            if np.any(dists == 0):
                zero_mask = dists == 0
                val = np.mean(self.y_soft[idxs[zero_mask]])
                probs.append(val)
            else:
                weights = 1.0 / (dists ** (2.0 / (self.m - 1.0)))
                sum_weights = np.sum(weights)
                val = np.sum(weights * self.y_soft[idxs]) / sum_weights
                probs.append(val)

        return np.array(probs)


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    exp_dir = project_root / "experiments" / "exp_22"
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"
    brain_dir = Path("/home/jmalagont/.gemini/antigravity-cli/brain/7884c29e-c602-4c6a-bff9-83df54c2ad16")
    brain_figures_dir = brain_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    brain_dir.mkdir(parents=True, exist_ok=True)
    brain_figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tabular clinical features, urologist reasoning annotations, and targets...")
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)

    # Filter complete-case labeled cohort (N=88)
    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_tr = df_tab[labeled_mask].reset_index(drop=True)
    df_reasoning_tr = df_reasoning[labeled_mask].reset_index(drop=True)
    df_dec_tr = df_dec[labeled_mask].reset_index(drop=True)

    pids_labeled = df_dec_tr["patient_id"].values
    n_samples = len(df_dec_tr)
    print(f"Complete-Case Labeled Cohort Size: N = {n_samples}")

    # Soft Target Labels Construction for Tabular Fuzzy KNN
    biopsy_label_map = {"yes": 1, "no": 0, "BIOPSY": 1, "NO_BIOPSY": 0}
    y_tr_binary = df_dec_tr["biopsy_decision"].map(biopsy_label_map).values

    confidence_map = {"uncertain": 0.25, "borderline": 0.50, "clear": 1.00}
    c_tr = df_reasoning_tr["confidence"].map(confidence_map).fillna(1.00).values
    y_tr_soft = np.where(y_tr_binary == 1, 0.50 + 0.50 * c_tr, 0.50 - 0.50 * c_tr)

    feature_cols = ["age", "psa", "vol", "pirads", "psad", "dre"]
    num_cols = ["age", "psa", "vol", "pirads", "psad"]
    cat_cols = ["dre"]

    # Map Ground-Truth Urologist Reasoning Weights
    weight_map = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
    gt_weights_df = pd.DataFrame()
    for col in feature_cols:
        gt_weights_df[col] = df_reasoning_tr[f"weight_{col}"].map(weight_map).fillna(0).astype(int)

    # Preprocess Tabular Features
    medians = df_tab_tr[num_cols].median()
    df_num = df_tab_tr[num_cols].fillna(medians)
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df_num)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df_tab_tr[cat_cols].fillna("normal"))
    X_all = np.hstack([X_num, X_cat])

    # 1. Compute Full SHAP Vectors for All 88 Patients
    print("\nComputing Full SHAP Vectors across 88 labeled cases...")
    all_shap_vectors = np.zeros((n_samples, len(feature_cols)))

    for i in range(n_samples):
        train_idx = [j for j in range(n_samples) if j != i]
        val_idx = [i]

        X_tr = X_all[train_idx]
        y_tr = y_tr_soft[train_idx]
        X_val = X_all[val_idx]

        model = DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')
        model.fit(X_tr, y_tr)

        bg_samples = shap.sample(X_tr, min(20, len(X_tr)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg_samples, silent=True)
        shap_vals_raw = explainer.shap_values(X_val)

        if isinstance(shap_vals_raw, list):
            s_val = shap_vals_raw[0][0]
        else:
            s_val = shap_vals_raw[0]

        s_features = np.zeros(len(feature_cols))
        s_features[:5] = s_val[:5]
        s_features[5] = np.sum(s_val[5:])
        all_shap_vectors[i] = np.abs(s_features)

    # 2. LOOCV Multivariate Decision Tree Training & Out-of-Fold Prediction
    print("\nRunning Leave-One-Out Cross-Validation (LOOCV $N=88$) with Multivariate SHAP Input Decision Trees...")
    oof_pred_weights = np.zeros((n_samples, len(feature_cols)), dtype=int)

    for i in range(n_samples):
        train_idx = [j for j in range(n_samples) if j != i]
        val_idx = i

        X_tr_shap = all_shap_vectors[train_idx]  # 6D SHAP Vector Matrix (87 x 6)
        X_val_shap = all_shap_vectors[val_idx].reshape(1, -1)  # 6D SHAP Vector (1 x 6)

        # Fit independent multivariate decision tree for each feature j
        for j, col in enumerate(feature_cols):
            y_tr_j = gt_weights_df.iloc[train_idx][col].values

            # Fit 6D multivariate decision tree
            tree_clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
            tree_clf.fit(X_tr_shap, y_tr_j)

            pred_j = tree_clf.predict(X_val_shap)[0]
            oof_pred_weights[i, j] = pred_j

    # 3. Independent Multi-Metric Evaluation per Feature
    print("\nEvaluating Out-of-Fold Independent Feature Metrics (Macro-F1, Accuracy, Spearman Rho)...")
    oof_predictions_df = pd.DataFrame({"patient_id": pids_labeled})
    feature_attribution_metrics = {}
    feature_confusion_matrices = {}

    class_names = ["not_used", "noted", "important", "decisive"]
    fig_grid, axes_grid = plt.subplots(2, 3, figsize=(15, 9))
    axes_grid = axes_grid.flatten()

    for j, col in enumerate(feature_cols):
        gt_w = gt_weights_df[col].values
        pred_w = oof_pred_weights[:, j]

        oof_predictions_df[f"gt_weight_{col}"] = gt_w
        oof_predictions_df[f"pred_weight_{col}"] = pred_w

        rho_v, p_v = spearmanr(gt_w, pred_w)
        if np.isnan(rho_v):
            rho_v, p_v = 0.0, 1.0

        macro_f1_v = float(f1_score(gt_w, pred_w, average="macro", zero_division=0))
        acc_v = float(accuracy_score(gt_w, pred_w))
        cm_v = confusion_matrix(gt_w, pred_w, labels=[0, 1, 2, 3])

        feature_confusion_matrices[col] = cm_v.tolist()

        sens_c_list = []
        spec_c_list = []
        for c in range(4):
            tp_c = cm_v[c, c]
            fn_c = np.sum(cm_v[c, :]) - tp_c
            fp_c = np.sum(cm_v[:, c]) - tp_c
            tn_c = np.sum(cm_v) - (tp_c + fn_c + fp_c)

            sens_c = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0
            spec_c = float(tn_c / (tn_c + fp_c)) if (tn_c + fp_c) > 0 else 0.0
            sens_c_list.append(sens_c)
            spec_c_list.append(spec_c)

        feature_attribution_metrics[col] = {
            "spearman_rho": float(rho_v),
            "spearman_pvalue": float(p_v),
            "macro_f1": macro_f1_v,
            "accuracy": acc_v,
            "mean_shap_magnitude": float(np.mean(all_shap_vectors[:, j])),
            "mean_sensitivity": float(np.mean(sens_c_list)),
            "mean_specificity": float(np.mean(spec_c_list)),
            "per_class_sensitivity": {class_names[c]: sens_c_list[c] for c in range(4)},
            "per_class_specificity": {class_names[c]: spec_c_list[c] for c in range(4)},
            "confusion_matrix": cm_v.tolist()
        }

        # Individual 4x4 Confusion Matrix Image
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        disp_ind = ConfusionMatrixDisplay(confusion_matrix=cm_v, display_labels=class_names)
        disp_ind.plot(cmap=plt.cm.Blues, ax=ax_ind, values_format="d")
        ax_ind.set_title(f"Multivariate SHAP Tree: '{col}' 4x4 Confusion Matrix\n(Rho: {rho_v:.4f}, p={p_v:.4e}, Macro-F1: {macro_f1_v:.4f})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / f"cm_{col}.png", dpi=300)
        plt.savefig(brain_figures_dir / f"cm_{col}.png", dpi=300)
        plt.close(fig_ind)

        # Plot on Grid
        disp_grid = ConfusionMatrixDisplay(confusion_matrix=cm_v, display_labels=["not", "noted", "imp", "dec"])
        disp_grid.plot(cmap=plt.cm.Blues, ax=axes_grid[j], colorbar=False, values_format="d")
        axes_grid[j].set_title(f"'{col}' (Rho={rho_v:.3f}, F1={macro_f1_v:.3f})", fontsize=10, fontweight="bold")

    plt.suptitle("Multivariate SHAP Vector Input Decision Tree 4x4 Confusion Matrices (exp_22 LOOCV)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices_all_features.png", dpi=300)
    plt.savefig(brain_figures_dir / "confusion_matrices_all_features.png", dpi=300)
    plt.close(fig_grid)

    oof_predictions_df.to_csv(results_dir / "oof_multivariate_shap_predictions.csv", index=False)

    with open(results_dir / "feature_attribution_metrics.json", "w") as f:
        json.dump(feature_attribution_metrics, f, indent=4)

    with open(results_dir / "feature_confusion_matrices.json", "w") as f:
        json.dump(feature_confusion_matrices, f, indent=4)

    with open(brain_dir / "feature_attribution_metrics.json", "w") as f:
        json.dump(feature_attribution_metrics, f, indent=4)

    with open(brain_dir / "feature_confusion_matrices.json", "w") as f:
        json.dump(feature_confusion_matrices, f, indent=4)

    # 4. Generate Comparative Histograms (Macro-F1 & Absolute Spearman Rho)
    macro_f1s = [feature_attribution_metrics[col]["macro_f1"] for col in feature_cols]
    abs_rhos = [abs(feature_attribution_metrics[col]["spearman_rho"]) for col in feature_cols]
    raw_rhos = [feature_attribution_metrics[col]["spearman_rho"] for col in feature_cols]

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(16, 6))

    colors_f1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_cols)))
    bars_a = axes_hist[0].bar(feature_cols, macro_f1s, color=colors_f1, edgecolor="black", linewidth=1.2, width=0.55)
    for bar in bars_a:
        yval = bar.get_height()
        axes_hist[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
    axes_hist[0].set_title("A) 4-Class Macro-F1 Score per Feature (exp_22)", fontsize=12, fontweight="bold")
    axes_hist[0].set_xlabel("Clinical Feature", fontsize=10, fontweight="bold")
    axes_hist[0].set_ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    axes_hist[0].set_ylim(0, max(macro_f1s) + 0.05)
    axes_hist[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    colors_rho = plt.cm.GnBu(np.linspace(0.4, 0.9, len(feature_cols)))
    bars_b = axes_hist[1].bar(feature_cols, abs_rhos, color=colors_rho, edgecolor="black", linewidth=1.2, width=0.55)
    for i_b, bar in enumerate(bars_b):
        yval = bar.get_height()
        raw_val = raw_rhos[i_b]
        sign_str = "+" if raw_val >= 0 else "-"
        axes_hist[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"|{yval:.4f}| ({sign_str})", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
    axes_hist[1].set_title("B) Absolute Spearman Correlation |Rho| per Feature (exp_22)", fontsize=12, fontweight="bold")
    axes_hist[1].set_xlabel("Clinical Feature", fontsize=10, fontweight="bold")
    axes_hist[1].set_ylabel("Absolute Spearman Correlation |Rho|", fontsize=10, fontweight="bold")
    axes_hist[1].set_ylim(0, max(abs_rhos) + 0.06)
    axes_hist[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Multivariate SHAP Vector Input Decision Tree Attribution Summary (exp_22)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "histograms_combined_f1_rho_exp22.png", dpi=300)
    plt.savefig(brain_figures_dir / "histograms_combined_f1_rho_exp22.png", dpi=300)
    plt.close(fig_hist)

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Multivariate SHAP Vector Input Decision Tree Clinical Relevance Attribution (exp_22) Summary Report\n\n")
        f.write("**Date**: 2026-08-06  \n")
        f.write("**Model**: Multivariate 6D SHAP Vector `DecisionTreeClassifier(max_depth=3, class_weight='balanced')`  \n")
        f.write(f"**Validation Protocol**: Direct Out-of-Fold LOOCV ($N = {n_samples}$)  \n\n")

        f.write("## 1. Frozen LOOCV Out-of-Fold Performance Summary per Feature\n")
        f.write("| Clinical Feature | Spearman Rank $\\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean \|SHAP\| |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for col in feature_cols:
            m = feature_attribution_metrics[col]
            rho_fmt = f"**`{m['spearman_rho']:.4f}`**" if m["spearman_pvalue"] < 0.05 else f"`{m['spearman_rho']:.4f}`"
            f.write(f"| **`{col}`** | {rho_fmt} | `{m['spearman_pvalue']:.4e}` | `{m['macro_f1']:.4f}` | `{m['accuracy']*100:.2f}%` | `{m['mean_shap_magnitude']:.4f}` |\n")

        f.write("\n## 2. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)\n\n")
        for col in feature_cols:
            m = feature_attribution_metrics[col]
            cm_v = m["confusion_matrix"]
            f.write(f"### Feature: `{col}` (Spearman $\\rho = {m['spearman_rho']:.4f}$, Macro-F1 = `{m['macro_f1']:.4f}`)\n")
            f.write("| Ground Truth \\ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for r_idx, c_name in enumerate(class_names):
                row_vals = cm_v[r_idx]
                f.write(f"| **{c_name} ({r_idx})** | ")
                for c_idx in range(4):
                    if r_idx == c_idx:
                        f.write(f"**{row_vals[c_idx]}** | ")
                    else:
                        f.write(f"{row_vals[c_idx]} | ")
                f.write(f"{sum(row_vals)} |\n")
            f.write("\n")

    with open(brain_dir / "exp_22_summary.md", "w") as f:
        with open(summary_md_path, "r") as f_src:
            f.write(f_src.read())

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
