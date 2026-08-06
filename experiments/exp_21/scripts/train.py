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


def extract_decision_tree_thresholds(X_1d, y_class, n_classes=4):
    clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    clf.fit(X_1d.reshape(-1, 1), y_class)

    thresholds = sorted(clf.tree_.threshold[clf.tree_.threshold != -2])

    if len(thresholds) >= n_classes - 1:
        return thresholds[:n_classes - 1]

    quantiles = np.quantile(X_1d, np.linspace(0.25, 0.75, n_classes - 1))

    full_thresholds = []
    for q in quantiles:
        if len(thresholds) > 0:
            closest = min(thresholds, key=lambda t: abs(t - q))
            full_thresholds.append(closest)
        else:
            full_thresholds.append(q)

    return sorted(full_thresholds)


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    exp_dir = project_root / "experiments" / "exp_21"
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
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

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

    # Preprocess Tabular Features
    medians = df_tab_tr[num_cols].median()
    df_num = df_tab_tr[num_cols].fillna(medians)
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df_num)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df_tab_tr[cat_cols].fillna("normal"))
    X_all = np.hstack([X_num, X_cat])

    # Mapping from OHE back to original 8 feature groups
    # First 7 are numeric, last ones belong to DRE
    # Compute LOOCV SHAP Shapley Values per sample
    print("\nComputing LOOCV SHAP Shapley Values across 88 labeled cases...")
    oof_shap_values = np.zeros((n_samples, len(feature_cols)))

    for i in range(n_samples):
        train_idx = [j for j in range(n_samples) if j != i]
        val_idx = [i]

        X_tr = X_all[train_idx]
        y_tr = y_tr_soft[train_idx]
        X_val = X_all[val_idx]

        model = DistanceWeightedFuzzyKNN(n_neighbors=1, metric='euclidean')
        model.fit(X_tr, y_tr)

        # Background sample for KernelExplainer
        bg_samples = shap.sample(X_tr, min(20, len(X_tr)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg_samples, silent=True)
        shap_vals_raw = explainer.shap_values(X_val)

        if isinstance(shap_vals_raw, list):
            s_val = shap_vals_raw[0][0]
        else:
            s_val = shap_vals_raw[0]

        # Map SHAP values to 6 feature groups
        # First 5 numeric: age, psa, vol, pirads, psad
        s_features = np.zeros(len(feature_cols))
        s_features[:5] = s_val[:5]
        # Sum DRE OHE components for feature index 5
        s_features[5] = np.sum(s_val[5:])

        oof_shap_values[i] = np.abs(s_features)

    # ---------------------------------------------------------
    # Phase A: 100 MCCV Split Feature-Independent Decision Tree Meta-Threshold Learning
    # ---------------------------------------------------------
    print("\nBeginning Phase A: 100-Split MCCV Feature-Independent Decision Tree Meta-Threshold Learning...")
    weight_map = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
    gt_weights_df = pd.DataFrame()
    for col in feature_cols:
        gt_weights_df[col] = df_reasoning_tr[f"weight_{col}"].map(weight_map).fillna(0).astype(int)

    mccv_thresholds = {col: [] for col in feature_cols}

    for s_idx in range(100):
        col_name = f"split_{s_idx}"
        split_assignments = df_design[df_design["patient_id"].isin(pids_labeled)].sort_values("patient_id")[col_name].values
        tr_mask_s = split_assignments == 0

        for col in feature_cols:
            shap_train = oof_shap_values[tr_mask_s, feature_cols.index(col)]
            y_train_w = gt_weights_df.iloc[tr_mask_s][col].values

            taus = extract_decision_tree_thresholds(shap_train, y_train_w, n_classes=4)
            mccv_thresholds[col].append(taus)

    learned_meta_thresholds = {}
    print("\nLearned SHAP Feature-Independent Meta-Thresholds for 8 Tabular Features:")
    for col in feature_cols:
        taus_arr = np.array(mccv_thresholds[col])
        t1, t2, t3 = np.mean(taus_arr[:, 0]), np.mean(taus_arr[:, 1]), np.mean(taus_arr[:, 2])
        learned_meta_thresholds[col] = {
            "tau1": float(t1),
            "tau2": float(t2),
            "tau3": float(t3),
            "tau1_std": float(np.std(taus_arr[:, 0])),
            "tau2_std": float(np.std(taus_arr[:, 1])),
            "tau3_std": float(np.std(taus_arr[:, 2]))
        }
        print(f"  {col:<12}: tau1={t1:.4f}, tau2={t2:.4f}, tau3={t3:.4f}")

    with open(results_dir / "meta_thresholds_shap.json", "w") as f:
        json.dump(learned_meta_thresholds, f, indent=4)

    # ---------------------------------------------------------
    # Phase B: Frozen LOOCV Out-of-Fold Multi-Metric Evaluation
    # ---------------------------------------------------------
    print("\nBeginning Phase B: Frozen LOOCV Out-of-Fold Multi-Metric Evaluation...")
    oof_predictions_df = pd.DataFrame({"patient_id": pids_labeled})
    feature_attribution_metrics = {}
    feature_confusion_matrices = {}

    class_names = ["not_used", "noted", "important", "decisive"]
    fig_grid, axes_grid = plt.subplots(2, 3, figsize=(15, 9))
    axes_grid = axes_grid.flatten()

    for idx, col in enumerate(feature_cols):
        shap_col_val = oof_shap_values[:, feature_cols.index(col)]
        gt_w = gt_weights_df[col].values

        t1 = learned_meta_thresholds[col]["tau1"]
        t2 = learned_meta_thresholds[col]["tau2"]
        t3 = learned_meta_thresholds[col]["tau3"]

        pred_w = []
        for val in shap_col_val:
            if val < t1:
                pred_w.append(0)
            elif val < t2:
                pred_w.append(1)
            elif val < t3:
                pred_w.append(2)
            else:
                pred_w.append(3)

        pred_w = np.array(pred_w)

        oof_predictions_df[f"shap_abs_{col}"] = shap_col_val
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
            "mean_shap_magnitude": float(np.mean(shap_col_val)),
            "mean_sensitivity": float(np.mean(sens_c_list)),
            "mean_specificity": float(np.mean(spec_c_list)),
            "per_class_sensitivity": {class_names[c]: sens_c_list[c] for c in range(4)},
            "per_class_specificity": {class_names[c]: spec_c_list[c] for c in range(4)},
            "confusion_matrix": cm_v.tolist()
        }

        # Individual 4x4 Confusion Matrix Image
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        disp_ind = ConfusionMatrixDisplay(confusion_matrix=cm_v, display_labels=class_names)
        disp_ind.plot(cmap=plt.cm.Purples, ax=ax_ind, values_format="d")
        ax_ind.set_title(f"SHAP Feature: '{col}' 4x4 Confusion Matrix\n(Rho: {rho_v:.4f}, p={p_v:.4e}, Macro-F1: {macro_f1_v:.4f})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / f"cm_{col}.png", dpi=300)
        plt.savefig(brain_figures_dir / f"cm_{col}.png", dpi=300)
        plt.close(fig_ind)

        # Plot on Grid
        disp_grid = ConfusionMatrixDisplay(confusion_matrix=cm_v, display_labels=["not", "noted", "imp", "dec"])
        disp_grid.plot(cmap=plt.cm.Purples, ax=axes_grid[idx], colorbar=False, values_format="d")
        axes_grid[idx].set_title(f"'{col}' (Rho={rho_v:.3f})", fontsize=10, fontweight="bold")

    plt.suptitle("SHAP 4x4 Confusion Matrices Across 8 Tabular Clinical Features (LOOCV)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices_all_features.png", dpi=300)
    plt.savefig(brain_figures_dir / "confusion_matrices_all_features.png", dpi=300)
    plt.close(fig_grid)

    oof_predictions_df.to_csv(results_dir / "oof_shap_attributions.csv", index=False)

    with open(results_dir / "feature_attribution_metrics.json", "w") as f:
        json.dump(feature_attribution_metrics, f, indent=4)

    with open(results_dir / "feature_confusion_matrices.json", "w") as f:
        json.dump(feature_confusion_matrices, f, indent=4)

    with open(brain_dir / "feature_attribution_metrics.json", "w") as f:
        json.dump(feature_attribution_metrics, f, indent=4)

    with open(brain_dir / "feature_confusion_matrices.json", "w") as f:
        json.dump(feature_confusion_matrices, f, indent=4)

    # Global SHAP Bar Plot
    mean_shaps = [feature_attribution_metrics[col]["mean_shap_magnitude"] for col in feature_cols]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(feature_cols, mean_shaps, color="purple", edgecolor="black")
    plt.title("Global Feature Importance (Mean Absolute SHAP Magnitude)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=10)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_bar.png", dpi=300)
    plt.savefig(brain_figures_dir / "shap_summary_bar.png", dpi=300)
    plt.close()

    # Write summary.md
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Clinical Feature Relevance Attribution via SHAP Shapley Values (exp_21) Summary Report\n\n")
        f.write("**Date**: 2026-08-05  \n")
        f.write("**Model**: SHAP `KernelExplainer` on Distance-Weighted Tabular Fuzzy KNN (`exp_13`)  \n")
        f.write(f"**Cohort**: Complete-Case Labeled Cohort ($N = {n_samples}$)  \n\n")

        f.write("## 1. SHAP Feature-Independent Meta-Thresholds (100 MCCV Splits)\n")
        f.write("| Clinical Feature | Meta-Threshold 1 ($\bar{\\tau}_1$) | Meta-Threshold 2 ($\bar{\\tau}_2$) | Meta-Threshold 3 ($\bar{\\tau}_3$) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for col in feature_cols:
            m_t = learned_meta_thresholds[col]
            f.write(f"| **`{col}`** | `{m_t['tau1']:.4f}` | `{m_t['tau2']:.4f}` | `{m_t['tau3']:.4f}` |\n")

        f.write("\n## 2. Frozen LOOCV Out-of-Fold Performance Summary\n")
        f.write("| Clinical Feature | Spearman Rank $\\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean \|SHAP\| |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for col in feature_cols:
            m = feature_attribution_metrics[col]
            rho_fmt = f"**`{m['spearman_rho']:.4f}`**" if m["spearman_pvalue"] < 0.05 else f"`{m['spearman_rho']:.4f}`"
            f.write(f"| **`{col}`** | {rho_fmt} | `{m['spearman_pvalue']:.4e}` | `{m['macro_f1']:.4f}` | `{m['accuracy']*100:.2f}%` | `{m['mean_shap_magnitude']:.4f}` |\n")

        f.write("\n## 3. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)\n\n")
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

    with open(brain_dir / "exp_21_summary.md", "w") as f:
        with open(summary_md_path, "r") as f_src:
            f.write(f_src.read())

    print(f"\nSummary report written to: {summary_md_path}")

if __name__ == "__main__":
    main()
