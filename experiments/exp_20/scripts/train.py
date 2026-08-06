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
from sklearn.metrics import f1_score, accuracy_score

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


def preprocess_data(df_tr, df_te, num_cols, cat_cols):
    df_tr_num = df_tr[num_cols].copy()
    df_te_num = df_te[num_cols].copy()

    # Impute missing continuous values with training median
    tr_medians = df_tr_num.median()
    df_tr_num = df_tr_num.fillna(tr_medians)
    df_te_num = df_te_num.fillna(tr_medians)

    scaler = MinMaxScaler()
    X_tr_num = scaler.fit_transform(df_tr_num)
    X_te_num = scaler.transform(df_te_num)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_cat = ohe.fit_transform(df_tr[cat_cols].fillna("normal"))
    X_te_cat = ohe.transform(df_te[cat_cols].fillna("normal"))

    X_tr = np.hstack([X_tr_num, X_tr_cat])
    X_te = np.hstack([X_te_num, X_te_cat])
    return X_tr, X_te, scaler, ohe, tr_medians


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    exp_dir = project_root / "experiments" / "exp_20"
    data_dir = project_root / "data" / "chimera26" / "preprocessed" / "task1"
    results_dir = exp_dir / "results"
    reports_dir = exp_dir / "reports"
    figures_dir = reports_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tabular clinical features, reasoning annotations, and biopsy decision targets...")
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv")
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv")
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv")
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    patient_ids = df_design_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0, "BIOPSY": 1, "NO_BIOPSY": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values

    # Confidence map
    confidence_map = {"uncertain": 0.25, "borderline": 0.50, "clear": 1.00}
    c_k = df_reasoning_labeled["confidence"].map(confidence_map).values
    y_soft = np.where(y_binary == 1, 0.50 + 0.50 * c_k, 0.50 - 0.50 * c_k)

    num_cols = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
    cat_cols = ["dre"]
    feature_cols = num_cols + cat_cols

    weight_col_map = {
        "age": "weight_age", "psa": "weight_psa", "vol": "weight_vol",
        "pirads": "weight_pirads", "dre": "weight_dre", "psad": "weight_psad",
        "psav": "weight_psa", "psap": "weight_psa"
    }

    weight_map = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
    
    Y_weights = {}
    for col in feature_cols:
        weight_col = weight_col_map[col]
        mapped_vals = df_reasoning_labeled[weight_col].map(lambda v: weight_map.get(str(v).lower(), 0)).values
        Y_weights[col] = mapped_vals

    print("Beginning Phase A: 100-Split MCCV Feature-Independent Decision Tree Meta-Threshold Learning...")

    n_splits = 100
    meta_thresholds_per_feature = {}

    for col in feature_cols:
        t1_list, t2_list, t3_list = [], [], []

        for split_idx in range(n_splits):
            col_split = f"split_{split_idx}"
            split_vals = df_design_labeled[col_split].values
            train_mask = split_vals == 0

            df_tr_raw = df_tab_labeled.iloc[train_mask].copy()
            y_soft_tr = y_soft[train_mask]
            y_weight_tr = Y_weights[col][train_mask]

            X_tr, _, scaler, ohe, tr_medians = preprocess_data(df_tr_raw, df_tr_raw, num_cols, cat_cols)

            model_fuzzy = DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')
            model_fuzzy.fit(X_tr, y_soft_tr)

            p_base_tr = model_fuzzy.predict_proba(X_tr)

            # Mask feature col with training median/mode
            df_tr_perturbed = df_tr_raw.copy()
            if col in cat_cols:
                substitute_val = df_tr_raw[col].mode()[0] if not df_tr_raw[col].mode().empty else "normal"
            else:
                substitute_val = df_tr_raw[col].median()

            df_tr_perturbed[col] = substitute_val
            
            # Preprocess perturbed dataset
            df_tr_pert_num = df_tr_perturbed[num_cols].fillna(tr_medians)
            X_tr_pert_num = scaler.transform(df_tr_pert_num)
            X_tr_pert_cat = ohe.transform(df_tr_perturbed[cat_cols].fillna("normal"))
            X_tr_perturbed = np.hstack([X_tr_pert_num, X_tr_pert_cat])

            p_pert_tr = model_fuzzy.predict_proba(X_tr_perturbed)
            delta_p_tr = np.abs(p_base_tr - p_pert_tr).reshape(-1, 1)

            # Fit 1D Class-Weighted Decision Tree for THIS feature
            dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
            dt.fit(delta_p_tr, y_weight_tr)

            tree_thresholds = dt.tree_.threshold[dt.tree_.threshold != -2]
            tree_thresholds = np.sort(np.unique(tree_thresholds))

            if len(tree_thresholds) >= 3:
                t1, t2, t3 = tree_thresholds[0], tree_thresholds[1], tree_thresholds[2]
            elif len(tree_thresholds) == 2:
                t1, t2, t3 = tree_thresholds[0], tree_thresholds[1], tree_thresholds[1] + 0.05
            elif len(tree_thresholds) == 1:
                t1, t2, t3 = tree_thresholds[0], tree_thresholds[0] + 0.05, tree_thresholds[0] + 0.10
            else:
                t1, t2, t3 = 0.01, 0.05, 0.15

            t1_list.append(t1)
            t2_list.append(t2)
            t3_list.append(t3)

        meta_t1 = float(np.mean(t1_list))
        meta_t2 = float(np.mean(t2_list))
        meta_t3 = float(np.mean(t3_list))

        meta_thresholds_per_feature[col] = {
            "tau_1": meta_t1, "std_tau_1": float(np.std(t1_list)),
            "tau_2": meta_t2, "std_tau_2": float(np.std(t2_list)),
            "tau_3": meta_t3, "std_tau_3": float(np.std(t3_list))
        }

    with open(results_dir / "meta_thresholds.json", "w") as f:
        json.dump(meta_thresholds_per_feature, f, indent=4)

    print("Learned Feature-Independent Meta-Thresholds for 8 Tabular Features:")
    for col, ths in meta_thresholds_per_feature.items():
        print(f"  {col:12s}: tau1={ths['tau_1']:.4f}, tau2={ths['tau_2']:.4f}, tau3={ths['tau_3']:.4f}")

    # ---------------------------------------------------------
    # Phase B: Frozen LOOCV Out-of-Fold Evaluation
    # ---------------------------------------------------------
    print("\nBeginning Phase B: Frozen LOOCV Out-of-Fold Evaluation...")

    n_samples = len(df_design_labeled)
    oof_displacements = {col: np.zeros(n_samples) for col in feature_cols}
    oof_pred_weights = {col: np.zeros(n_samples, dtype=int) for col in feature_cols}

    for fold_i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[fold_i] = False

        df_tr_raw = df_tab_labeled.iloc[train_mask].copy()
        df_te_raw = df_tab_labeled.iloc[[fold_i]].copy()
        y_soft_tr = y_soft[train_mask]

        X_tr, X_te, scaler, ohe, tr_medians = preprocess_data(df_tr_raw, df_te_raw, num_cols, cat_cols)

        model_fuzzy = DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')
        model_fuzzy.fit(X_tr, y_soft_tr)

        p_base_te = model_fuzzy.predict_proba(X_te)[0]

        for col in feature_cols:
            if col in cat_cols:
                substitute_val = df_tr_raw[col].mode()[0] if not df_tr_raw[col].mode().empty else "normal"
            else:
                substitute_val = df_tr_raw[col].median()

            df_te_perturbed = df_te_raw.copy()
            df_te_perturbed[col] = substitute_val

            df_te_pert_num = df_te_perturbed[num_cols].fillna(tr_medians)
            X_te_pert_num = scaler.transform(df_te_pert_num)
            X_te_pert_cat = ohe.transform(df_te_perturbed[cat_cols].fillna("normal"))
            X_te_perturbed = np.hstack([X_te_pert_num, X_te_pert_cat])

            p_pert_te = model_fuzzy.predict_proba(X_te_perturbed)[0]
            delta_p_te = abs(p_base_te - p_pert_te)

            oof_displacements[col][fold_i] = delta_p_te

            tau1 = meta_thresholds_per_feature[col]["tau_1"]
            tau2 = meta_thresholds_per_feature[col]["tau_2"]
            tau3 = meta_thresholds_per_feature[col]["tau_3"]

            if delta_p_te < tau1:
                pred_w = 0  # not_used
            elif delta_p_te < tau2:
                pred_w = 1  # noted
            elif delta_p_te < tau3:
                pred_w = 2  # important
            else:
                pred_w = 3  # decisive

            oof_pred_weights[col][fold_i] = pred_w

    # Evaluate Metrics & Confusion Matrices per Feature
    feature_metrics = {}
    cm_dict = {}
    inv_weight_map = {0: "not_used", 1: "noted", 2: "important", 3: "decisive"}
    class_labels = ["not_used", "noted", "important", "decisive"]

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    # Setup multi-plot grid figure (2 rows x 4 cols)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        gt_w = Y_weights[col]
        pr_w = oof_pred_weights[col]

        rho_val, p_val = spearmanr(gt_w, pr_w)

        if np.isnan(rho_val):
            rho_val = 0.0
            p_val = 1.0

        acc = accuracy_score(gt_w, pr_w)
        macro_f1 = f1_score(gt_w, pr_w, average="macro", zero_division=0)
        
        # 4x4 Confusion Matrix
        cm = confusion_matrix(gt_w, pr_w, labels=[0, 1, 2, 3])
        cm_dict[col] = cm.tolist()

        feature_metrics[col] = {
            "spearman_rho": float(rho_val),
            "spearman_pvalue": float(p_val),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "mean_displacement": float(np.mean(oof_displacements[col])),
            "max_displacement": float(np.max(oof_displacements[col])),
            "confusion_matrix": cm.tolist()
        }

        # Plot individual feature 4x4 confusion matrix
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        disp_ind = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp_ind.plot(cmap=plt.cm.Purples, ax=ax_ind, values_format="d")
        ax_ind.set_title(f"Feature: '{col}' Confusion Matrix\n(Rho: {rho_val:.4f}, Macro-F1: {macro_f1:.4f})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig_ind.savefig(figures_dir / f"cm_{col}.png", dpi=300)
        plt.close(fig_ind)

        # Plot in grid
        disp_grid = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not", "noted", "imp", "dec"])
        disp_grid.plot(cmap=plt.cm.Blues, ax=axes[idx], colorbar=False, values_format="d")
        axes[idx].set_title(f"'{col}' (Rho={rho_val:.3f})", fontsize=10, fontweight="bold")

    plt.suptitle("4x4 Confusion Matrices Across 8 Tabular Clinical Features (LOOCV)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices_all_features.png", dpi=300)
    plt.close(fig)

    with open(results_dir / "feature_attribution_metrics.json", "w") as f:
        json.dump(feature_metrics, f, indent=4)

    with open(results_dir / "feature_confusion_matrices.json", "w") as f:
        json.dump(cm_dict, f, indent=4)

    # Save OOF Attributions Dataframe
    df_oof_out = pd.DataFrame({"patient_id": patient_ids})
    for col in feature_cols:
        df_oof_out[f"delta_p_{col}"] = oof_displacements[col]
        df_oof_out[f"gt_weight_{col}"] = [inv_weight_map[w] for w in Y_weights[col]]
        df_oof_out[f"pred_weight_{col}"] = [inv_weight_map[w] for w in oof_pred_weights[col]]

    df_oof_out.to_csv(results_dir / "oof_feature_attributions.csv", index=False)

    # Plot Per-Feature Spearman Correlation & Macro-F1 Bar Chart
    plt.figure(figsize=(10, 5))
    rhos = [feature_metrics[c]["spearman_rho"] for c in feature_cols]
    f1s = [feature_metrics[c]["macro_f1"] for c in feature_cols]
    
    x = np.arange(len(feature_cols))
    width = 0.35

    plt.bar(x - width/2, rhos, width, label="Spearman Rho (ρ)", color="steelblue")
    plt.bar(x + width/2, f1s, width, label="4-Class Macro-F1", color="darkseagreen")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xticks(x, feature_cols, rotation=30, fontsize=10)
    plt.ylabel("Metric Score", fontsize=10)
    plt.title("Clinical Feature Relevance Attribution: Spearman Rho & 4-Class Macro-F1", fontsize=11, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_bar.png", dpi=300)
    plt.close()

    # Write summary.md with detailed 4x4 Confusion Matrices per feature
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Clinical Feature Relevance Attribution via Mode/Median Perturbation (exp_20) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write("**Model**: Feature-Independent Class-Weighted Decision Trees on Fuzzy KNN Probability Displacements  \n")
        f.write(f"**Cohort**: Labeled Complete-Case Cohort ($N_{{labeled}} = 88$)  \n\n")

        f.write("## 1. Feature-Independent Meta-Thresholds (100 MCCV Splits)\n")
        f.write("| Clinical Feature | Meta-Threshold 1 ($\\bar{\\tau}_1$) | Meta-Threshold 2 ($\\bar{\\tau}_2$) | Meta-Threshold 3 ($\\bar{\\tau}_3$) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for col in feature_cols:
            ths = meta_thresholds_per_feature[col]
            f.write(f"| **`{col}`** | `{ths['tau_1']:.4f}` | `{ths['tau_2']:.4f}` | `{ths['tau_3']:.4f}` |\n")

        f.write("\n## 2. Frozen LOOCV Out-of-Fold Performance Summary\n")
        f.write("| Clinical Feature | Spearman Rank $\\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean Displacement $\\Delta p$ |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for col in feature_cols:
            m = feature_metrics[col]
            sig_str = "**" if m["spearman_pvalue"] < 0.05 else ""
            f.write(f"| **`{col}`** | {sig_str}`{m['spearman_rho']:.4f}`{sig_str} | `{m['spearman_pvalue']:.4e}` | `{m['macro_f1']:.4f}` | `{m['accuracy']*100:.2f}%` | `{m['mean_displacement']:.4f}` |\n")

        f.write("\n## 3. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)\n\n")
        for col in feature_cols:
            cm = np.array(feature_metrics[col]["confusion_matrix"])
            f.write(f"### Feature: `{col}` (Spearman $\\rho = {feature_metrics[col]['spearman_rho']:.4f}$, Macro-F1 = `{feature_metrics[col]['macro_f1']:.4f}`)\n")
            f.write("| Ground Truth \\ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
            f.write(f"| **not_used (0)** | **{cm[0,0]}** | {cm[0,1]} | {cm[0,2]} | {cm[0,3]} | {np.sum(cm[0,:])} |\n")
            f.write(f"| **noted (1)** | {cm[1,0]} | **{cm[1,1]}** | {cm[1,2]} | {cm[1,3]} | {np.sum(cm[1,:])} |\n")
            f.write(f"| **important (2)** | {cm[2,0]} | {cm[2,1]} | **{cm[2,2]}** | {cm[2,3]} | {np.sum(cm[2,:])} |\n")
            f.write(f"| **decisive (3)** | {cm[3,0]} | {cm[3,1]} | {cm[3,2]} | **{cm[3,3]}** | {np.sum(cm[3,:])} |\n\n")

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
