"""
Replication of Composite Fuzzy ICI for Subtask 1.2 Clinical Confidence (exp_7).
Standardized 30-split Multidimensional MCCV Evaluation with Real-Time Compact Reporting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Ensure local imports work
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXP_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"))
sys.path.insert(0, str(SCRIPT_DIR))

from chimera_metrics.core.types import CONF_MAP
from chimera_metrics.task1.reasoning import compute_confidence_score
from models import (
    CustomKNNPredictor,
    MRIPreprocessor,
    MetaThresholdTreeFitter,
    TabularPreprocessor,
    TextPreprocessor,
    compute_composite_ici,
)

ORD_NAMES = ["uncertain", "borderline", "clear"]
ORD_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}


def compute_moe_abs(y_true_ord: np.ndarray, y_pred_ord: np.ndarray) -> float:
    errs = []
    for c in range(3):
        mask = (y_true_ord == c)
        if np.sum(mask) == 0:
            continue
        errs.append(np.mean(np.abs(y_pred_ord[mask] - y_true_ord[mask]) / 2.0))
    return float(np.mean(errs)) if errs else 0.0


def compute_chimera_s_conf(y_true_ord: np.ndarray, y_pred_ord: np.ndarray) -> float:
    scores = []
    for yt, yp in zip(y_true_ord, y_pred_ord):
        gt_dict = {"confidence": ORD_NAMES[int(yt)]}
        pred_dict = {"confidence": ORD_NAMES[int(yp)]}
        s = compute_confidence_score(gt_dict, pred_dict)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def compute_per_class_recall(y_true_ord: np.ndarray, y_pred_ord: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true_ord, y_pred_ord, labels=[0, 1, 2])
    recalls = {}
    for i, name in enumerate(ORD_NAMES):
        row_sum = np.sum(cm[i, :])
        recalls[name] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
    return recalls


def load_canonical_data():
    data_dir = PROJECT_ROOT / "data" / "processed" / "task1"
    
    df_tab = pd.read_csv(data_dir / "clinical_tabular.csv")
    df_txt = pd.read_csv(data_dir / "clinical_texts.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task1.csv")
    df_splits = pd.read_csv(data_dir / "mccv_splits_30.csv")

    # Standardize ID column
    id_col = "pid" if "pid" in df_gt.columns else "case_id"
    for df in [df_tab, df_txt, df_mri, df_splits]:
        if "pid" not in df.columns and "case_id" in df.columns:
            df["pid"] = df["case_id"]
        elif "case_id" not in df.columns and "pid" in df.columns:
            df["case_id"] = df["pid"]

    # Filter labeled subset (N=91)
    df_gt_labeled = df_gt[df_gt["confidence"].notna()].copy()
    labeled_pids = set(df_gt_labeled["pid"].unique())
    
    df_tab = df_tab[df_tab["pid"].isin(labeled_pids)].reset_index(drop=True)
    df_txt = df_txt[df_txt["pid"].isin(labeled_pids)].reset_index(drop=True)
    df_mri = df_mri[df_mri["pid"].isin(labeled_pids)].reset_index(drop=True)
    df_gt_labeled = df_gt_labeled.sort_values("pid").reset_index(drop=True)
    df_splits = df_splits[df_splits["pid"].isin(labeled_pids)].reset_index(drop=True)

    pid_order = df_gt_labeled["pid"].values
    
    df_tab = df_tab.set_index("pid").reindex(pid_order).reset_index()
    df_txt = df_txt.set_index("pid").reindex(pid_order).reset_index()
    df_mri = df_mri.set_index("pid").reindex(pid_order).reset_index()
    df_splits = df_splits.set_index("pid").reindex(pid_order).reset_index()

    # Targets
    dec_col = "decision" if "decision" in df_gt_labeled.columns else "target_biopsy_decision"
    conf_col = "confidence" if "confidence" in df_gt_labeled.columns else "target_confidence"
    
    y_decision = df_gt_labeled[dec_col].astype(str).str.lower().tolist()
    y_confidence = df_gt_labeled[conf_col].astype(str).str.lower().tolist()
    y_conf_ord = np.array([ORD_MAP[c] for c in y_confidence], dtype=int)

    # MRI feature matrix
    mri_feature_cols = [c for c in df_mri.columns if c not in ["pid", "case_id", "split", "modality"]]
    X_mri = df_mri[mri_feature_cols].values

    # Text corpus
    text_col = "full_encounter_narrative" if "full_encounter_narrative" in df_txt.columns else "text"
    texts = df_txt[text_col].fillna("").astype(str).tolist()

    return {
        "pids": pid_order,
        "df_tab": df_tab,
        "texts": texts,
        "X_mri": X_mri,
        "y_decision": y_decision,
        "y_confidence": y_confidence,
        "y_conf_ord": y_conf_ord,
        "df_splits": df_splits,
    }


def run_experiment():
    start_time = time.time()
    results_dir = EXP_DIR / "results"
    reports_dir = EXP_DIR / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" 🔬 EXPERIMENT 7: REPLICATION OF COMPOSITE FUZZY ICI (MCCV-30)")
    print(" Subtask 1.2: Clinical Confidence Prediction (uncertain, borderline, clear)")
    print("=" * 80)

    data = load_canonical_data()
    N = len(data["pids"])
    conf_counts = pd.Series(data["y_confidence"]).value_counts().to_dict()
    print(f" Loaded {N} cases (clear: {conf_counts.get('clear', 0)}, borderline: {conf_counts.get('borderline', 0)}, uncertain: {conf_counts.get('uncertain', 0)})")
    print(f" Protocol: 30-Split Stratified MCCV (N_train=73, N_val=18)")
    print("-" * 80)

    # Frozen exp_4 optimal weights
    weights_optimal_fusion = (0.55, 0.20, 0.25)

    conditions = [
        {"id": "C0_Baseline_Majority", "name": "Majority Class Baseline (clear)", "type": "baseline"},
        {"id": "C1_Canonical_Fuzzy_ICI", "name": "Canonical Composite Fuzzy ICI (Balanced Regressor)", "type": "reg_balanced", "weights": None},
        {"id": "C1_clf_Fuzzy_ICI", "name": "Composite Fuzzy ICI (Balanced Classifier)", "type": "clf_balanced", "weights": None},
        {"id": "C2_Optimal_Weighted_ICI", "name": "Optimal-Weighted Fuzzy ICI (0.55/0.20/0.25)", "type": "reg_balanced", "weights": weights_optimal_fusion},
        {"id": "C3_Unweighted_UniformTree", "name": "Unweighted Fuzzy ICI (Uniform Regressor)", "type": "reg_uniform", "weights": None},
    ]

    all_condition_results = {}
    all_split_metrics = {cond["id"]: [] for cond in conditions}
    all_predictions = {cond["id"]: [] for cond in conditions}

    total_splits = 30
    inner_k_folds = 5

    for cond in conditions:
        cid = cond["id"]
        cname = cond["name"]
        print(f"\n▶ Evaluating Condition: {cid} — {cname}")
        cond_dir = results_dir / cid
        cond_dir.mkdir(parents=True, exist_ok=True)

        for split_idx in range(total_splits):
            split_col = f"split_{split_idx}"
            split_mask = data["df_splits"][split_col].values
            train_idx = np.where(split_mask == "train")[0]
            val_idx = np.where(split_mask == "val")[0]

            y_val_true = data["y_conf_ord"][val_idx]

            if cond["type"] == "baseline":
                y_pred_val = np.full(len(val_idx), ORD_MAP["clear"], dtype=int)
                ici_val = np.zeros(len(val_idx))
            else:
                # 1. Inner Cross-Fitting on train_idx to generate OOF probabilities (p_t, p_m, p_x)
                n_inner = len(train_idx) // inner_k_folds
                p_t_oof = np.zeros(len(train_idx), dtype=float)
                p_m_oof = np.zeros(len(train_idx), dtype=float)
                p_x_oof = np.zeros(len(train_idx), dtype=float)

                for k in range(inner_k_folds):
                    ivs = k * n_inner
                    ive = (k + 1) * n_inner if k < inner_k_folds - 1 else len(train_idx)
                    inner_val_idx = train_idx[ivs:ive]
                    inner_tr_idx = np.concatenate([train_idx[:ivs], train_idx[ive:]])

                    # Tabular Inner
                    tab_prep_in = TabularPreprocessor(spearman_tau=None).fit(data["df_tab"].iloc[inner_tr_idx])
                    X_tab_in_tr = tab_prep_in.transform(data["df_tab"].iloc[inner_tr_idx])
                    X_tab_in_val = tab_prep_in.transform(data["df_tab"].iloc[inner_val_idx])
                    knn_t_in = CustomKNNPredictor(k=5, metric="euclidean", weights="distance", is_fuzzy=True)
                    knn_t_in.fit(X_tab_in_tr, [data["y_decision"][i] for i in inner_tr_idx], [data["y_confidence"][i] for i in inner_tr_idx])
                    p_t_oof[ivs:ive] = knn_t_in.predict_proba(X_tab_in_val)

                    # MRI Inner
                    mri_prep_in = MRIPreprocessor(n_components=16).fit(data["X_mri"][inner_tr_idx])
                    X_mri_in_tr = mri_prep_in.transform(data["X_mri"][inner_tr_idx])
                    X_mri_in_val = mri_prep_in.transform(data["X_mri"][inner_val_idx])
                    knn_m_in = CustomKNNPredictor(k=1, metric="euclidean", weights="uniform", is_fuzzy=False)
                    knn_m_in.fit(X_mri_in_tr, [data["y_decision"][i] for i in inner_tr_idx], [data["y_confidence"][i] for i in inner_tr_idx])
                    p_m_oof[ivs:ive] = knn_m_in.predict_proba(X_mri_in_val)

                    # Text Inner
                    txt_prep_in = TextPreprocessor(max_features=1000, n_components=None).fit([data["texts"][i] for i in inner_tr_idx])
                    X_txt_in_tr = txt_prep_in.transform([data["texts"][i] for i in inner_tr_idx])
                    X_txt_in_val = txt_prep_in.transform([data["texts"][i] for i in inner_val_idx])
                    knn_x_in = CustomKNNPredictor(k=3, metric="cosine", weights="uniform", is_fuzzy=True)
                    knn_x_in.fit(X_txt_in_tr, [data["y_decision"][i] for i in inner_tr_idx], [data["y_confidence"][i] for i in inner_tr_idx])
                    p_x_oof[ivs:ive] = knn_x_in.predict_proba(X_txt_in_val)

                # Compute train ICI
                ici_train = compute_composite_ici(p_t_oof, p_m_oof, p_x_oof, weights=cond["weights"])
                y_train_ord = data["y_conf_ord"][train_idx]

                # Fit 1D Decision Tree Meta-Thresholds
                if cond["type"] == "reg_balanced":
                    tree_fitter = MetaThresholdTreeFitter(mode="regressor", criterion="absolute_error", use_balanced_weights=True)
                elif cond["type"] == "clf_balanced":
                    tree_fitter = MetaThresholdTreeFitter(mode="classifier", use_balanced_weights=True)
                else:  # reg_uniform
                    tree_fitter = MetaThresholdTreeFitter(mode="regressor", criterion="absolute_error", use_balanced_weights=False)

                tree_fitter.fit(ici_train, y_train_ord)

                # 2. Fit full unimodal models on train_idx and evaluate on val_idx
                tab_prep = TabularPreprocessor(spearman_tau=None).fit(data["df_tab"].iloc[train_idx])
                X_tab_tr = tab_prep.transform(data["df_tab"].iloc[train_idx])
                X_tab_val = tab_prep.transform(data["df_tab"].iloc[val_idx])
                knn_t = CustomKNNPredictor(k=5, metric="euclidean", weights="distance", is_fuzzy=True)
                knn_t.fit(X_tab_tr, [data["y_decision"][i] for i in train_idx], [data["y_confidence"][i] for i in train_idx])
                p_t_val = knn_t.predict_proba(X_tab_val)

                mri_prep = MRIPreprocessor(n_components=16).fit(data["X_mri"][train_idx])
                X_mri_tr = mri_prep.transform(data["X_mri"][train_idx])
                X_mri_val = mri_prep.transform(data["X_mri"][val_idx])
                knn_m = CustomKNNPredictor(k=1, metric="euclidean", weights="uniform", is_fuzzy=False)
                knn_m.fit(X_mri_tr, [data["y_decision"][i] for i in train_idx], [data["y_confidence"][i] for i in train_idx])
                p_m_val = knn_m.predict_proba(X_mri_val)

                txt_prep = TextPreprocessor(max_features=1000, n_components=None).fit([data["texts"][i] for i in train_idx])
                X_txt_tr = txt_prep.transform([data["texts"][i] for i in train_idx])
                X_txt_val = txt_prep.transform([data["texts"][i] for i in val_idx])
                knn_x = CustomKNNPredictor(k=3, metric="cosine", weights="uniform", is_fuzzy=True)
                knn_x.fit(X_txt_tr, [data["y_decision"][i] for i in train_idx], [data["y_confidence"][i] for i in train_idx])
                p_x_val = knn_x.predict_proba(X_txt_val)

                ici_val = compute_composite_ici(p_t_val, p_m_val, p_x_val, weights=cond["weights"])
                y_pred_val = tree_fitter.predict(ici_val)

            # Compute Split Metrics
            f1_m = f1_score(y_val_true, y_pred_val, average="macro", zero_division=0)
            acc = accuracy_score(y_val_true, y_pred_val)
            moe = compute_moe_abs(y_val_true, y_pred_val)
            s_conf = compute_chimera_s_conf(y_val_true, y_pred_val)
            recalls = compute_per_class_recall(y_val_true, y_pred_val)

            split_res = {
                "split": split_idx,
                "macro_f1": float(f1_m),
                "accuracy": float(acc),
                "moe_abs": float(moe),
                "s_conf": float(s_conf),
                "rec_uncertain": recalls["uncertain"],
                "rec_borderline": recalls["borderline"],
                "rec_clear": recalls["clear"],
            }
            all_split_metrics[cid].append(split_res)

            for i, vi in enumerate(val_idx):
                all_predictions[cid].append({
                    "split": split_idx,
                    "case_id": data["pids"][vi],
                    "y_true": int(y_val_true[i]),
                    "y_true_name": ORD_NAMES[int(y_val_true[i])],
                    "y_pred": int(y_pred_val[i]),
                    "y_pred_name": ORD_NAMES[int(y_pred_val[i])],
                    "ici_val": float(ici_val[i]),
                })

            # Compact live status update
            print(f"\r  [{split_idx+1:02d}/30] F1: {f1_m:.4f} | Acc: {acc*100:.1f}% | MOE: {moe:.4f} | S_conf: {s_conf:.4f}", end="", flush=True)

        print()  # newline after splits
        
        # Aggregate condition results
        df_sp = pd.DataFrame(all_split_metrics[cid])
        y_true_all = np.array([p["y_true"] for p in all_predictions[cid]], dtype=int)
        y_pred_all = np.array([p["y_pred"] for p in all_predictions[cid]], dtype=int)
        ici_all = np.array([p["ici_val"] for p in all_predictions[cid]], dtype=float)

        spearman_rho, spearman_p = spearmanr(ici_all, y_true_all)
        cm_pooled = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2])

        summary = {
            "condition_id": cid,
            "name": cname,
            "mean_macro_f1": float(df_sp["macro_f1"].mean()),
            "std_macro_f1": float(df_sp["macro_f1"].std()),
            "mean_accuracy": float(df_sp["accuracy"].mean()),
            "std_accuracy": float(df_sp["accuracy"].std()),
            "mean_moe_abs": float(df_sp["moe_abs"].mean()),
            "std_moe_abs": float(df_sp["moe_abs"].std()),
            "mean_s_conf": float(df_sp["s_conf"].mean()),
            "std_s_conf": float(df_sp["s_conf"].std()),
            "mean_rec_uncertain": float(df_sp["rec_uncertain"].mean()),
            "mean_rec_borderline": float(df_sp["rec_borderline"].mean()),
            "mean_rec_clear": float(df_sp["rec_clear"].mean()),
            "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
            "confusion_matrix_pooled": cm_pooled.tolist(),
        }
        all_condition_results[cid] = summary

        # Save condition predictions
        pd.DataFrame(all_predictions[cid]).to_csv(cond_dir / "predictions.csv", index=False)
        with open(cond_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

    # 4. Statistical Testing (Wilcoxon vs C0)
    wilcoxon_results = {}
    c0_f1s = [s["macro_f1"] for s in all_split_metrics["C0_Baseline_Majority"]]
    for cond in conditions[1:]:
        cid = cond["id"]
        c_f1s = [s["macro_f1"] for s in all_split_metrics[cid]]
        try:
            stat, p_val = wilcoxon(c_f1s, c0_f1s, alternative="greater")
            wilcoxon_results[cid] = {"statistic": float(stat), "p_value": float(p_val), "significant_005": bool(p_val < 0.05)}
        except Exception as e:
            wilcoxon_results[cid] = {"error": str(e)}

    # 5. Output Tables and Plots
    print("\n" + "=" * 80)
    print(" 📊 FINAL RESULTS SUMMARY — SUBTASK 1.2 MCCV-30 (FROZEN exp_4 UNIMODAL)")
    print("=" * 80)
    
    summary_rows = []
    for cid, s in all_condition_results.items():
        w_p = wilcoxon_results.get(cid, {}).get("p_value", np.nan)
        sig_str = f"p={w_p:.4f}*" if w_p < 0.05 else (f"p={w_p:.4f}" if not np.isnan(w_p) else "-")
        summary_rows.append({
            "Condition": cid,
            "Macro-F1 (std)": f"{s['mean_macro_f1']:.4f} (±{s['std_macro_f1']:.4f})",
            "Accuracy": f"{s['mean_accuracy']*100:.2f}%",
            "MOE_abs": f"{s['mean_moe_abs']:.4f}",
            "S_conf": f"{s['mean_s_conf']:.4f}",
            "Spearman ρ (p)": f"{s['spearman_rho']:.4f} ({s['spearman_p']:.4e})" if s['spearman_rho'] != 0 else "N/A",
            "Rec (U/B/C)": f"{s['mean_rec_uncertain']:.2f} / {s['mean_rec_borderline']:.2f} / {s['mean_rec_clear']:.2f}",
            "Wilcoxon vs C0": sig_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Save summary tables and results
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)
    with open(results_dir / "aggregate_metrics.json", "w") as f:
        json.dump(all_condition_results, f, indent=2)
    with open(results_dir / "wilcoxon_tests.json", "w") as f:
        json.dump(wilcoxon_results, f, indent=2)

    # Plot Confusion Matrices
    plot_conditions = ["C0_Baseline_Majority", "C1_Canonical_Fuzzy_ICI", "C2_Optimal_Weighted_ICI", "C3_Unweighted_UniformTree"]
    fig, axes = plt.subplots(1, len(plot_conditions), figsize=(4.5 * len(plot_conditions), 4))
    for ax, cid in zip(axes, plot_conditions):
        cm = np.array(all_condition_results[cid]["confusion_matrix_pooled"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=ORD_NAMES, yticklabels=ORD_NAMES, vmin=0, vmax=1)
        ax.set_title(f"{cid}\nMacro-F1: {all_condition_results[cid]['mean_macro_f1']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    # Plot ICI distribution across true confidence classes
    fig, ax = plt.subplots(figsize=(6, 4))
    preds_c1 = pd.DataFrame(all_predictions["C1_Canonical_Fuzzy_ICI"])
    for name in ORD_NAMES:
        sub = preds_c1[preds_c1["y_true_name"] == name]["ici_val"]
        sns.kdeplot(sub, label=f"True: {name} (N={len(sub)})", ax=ax, fill=True, alpha=0.3)
    ax.set_title(f"Composite Fuzzy ICI Distribution (ρ = {all_condition_results['C1_Canonical_Fuzzy_ICI']['spearman_rho']:.4f}, p = {all_condition_results['C1_Canonical_Fuzzy_ICI']['spearman_p']:.4e})")
    ax.set_xlabel("Continuous Composite ICI")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "ici_distribution.png", dpi=150)
    plt.close()

    # Version file
    with open(results_dir / "version.txt", "w") as f:
        f.write(f"exp_7: Composite Fuzzy ICI on MCCV-30 (Frozen exp_4 Modalities)\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runtime: {time.time() - start_time:.2f}s\n")

    print(f"\n✅ All results and figures saved to {results_dir.relative_to(PROJECT_ROOT)} and {reports_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_experiment()
