#!/usr/bin/env python3
"""
Systematic Hyperparameter and Late-Fusion Search for Task 2 Clinical Confidence (exp_18).
Protocol: Multi-subtask MCCV-30 (data/processed/task2/mccv_splits_30.csv).
Metrics: chimera_metrics.task2.reasoning.compute_confidence_score.
Zero data leakage: Scalers and TF-IDF fit exclusively on train folds.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Inject chimera_metrics
root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir / "utils" / "pkg" / "chimera_metrics"))

from chimera_metrics.splits import load_mccv_splits
from chimera_metrics.task2.reasoning import compute_confidence_score

# Ordinal Soft-Target Map
CONF_VAL_MAP = {"uncertain": 0.25, "borderline": 0.5, "clear": 1.0}

T2_NUMERIC_COLS = [
    "age", "psa", "months", "pirads", "psad", "psav", "psap", "vol", "cspca",
    "bx_isup", "bx_gl_prim", "bx_gl_sec",
    "vitals_weight_kg", "vitals_height_cm", "vitals_bmi",
    "vitals_bp_sys", "vitals_bp_dia", "vitals_hr_bpm", "smoking_pack_years",
    "ipss_score", "allergies_count", "comorbidity_count",
    "comorb_diabetes", "comorb_hypertension", "comorb_cardiovascular",
    "comorb_respiratory", "comorb_renal", "comorb_surgery", "comorb_cancer_other",
]
T2_CATEGORICAL_COLS = ["dre", "bx", "ct"]


class Task2TabularPreprocessor:
    def __init__(self) -> None:
        self.impute_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_columns_: Dict[str, List[str]] = {}
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                med = df[col].median()
                self.impute_medians[col] = 0.0 if pd.isna(med) else float(med)
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                mode = df[col].mode()
                self.cat_modes[col] = str(mode.iloc[0]) if len(mode) > 0 else "unknown"

        X_train = self._build_features(df, fit_cats=True)
        self.scaler = StandardScaler().fit(X_train)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df, fit_cats=False)
        assert self.scaler is not None
        return self.scaler.transform(X)

    def _build_features(self, df: pd.DataFrame, fit_cats: bool) -> np.ndarray:
        num_parts: List[np.ndarray] = []
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                vals = df[col].fillna(self.impute_medians.get(col, 0.0)).to_numpy(dtype=float)
            else:
                vals = np.zeros(len(df), dtype=float)
            num_parts.append(vals.reshape(-1, 1))

        cat_parts: List[np.ndarray] = []
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                series = df[col].fillna(self.cat_modes.get(col, "unknown")).astype(str)
            else:
                series = pd.Series(["unknown"] * len(df))

            if fit_cats:
                dummies = pd.get_dummies(series, prefix=col)
                self.cat_columns_[col] = list(dummies.columns)
                cat_parts.append(dummies.to_numpy(dtype=float))
            else:
                dummies = pd.get_dummies(series, prefix=col)
                aligned = dummies.reindex(columns=self.cat_columns_[col], fill_value=0)
                cat_parts.append(aligned.to_numpy(dtype=float))

        all_parts = num_parts + cat_parts
        return np.hstack(all_parts)


def continuous_to_label(val: float) -> str:
    if val < 0.375:
        return "uncertain"
    elif val < 0.75:
        return "borderline"
    else:
        return "clear"


def generate_simplex_weights(step: float = 0.05) -> List[Tuple[float, float, float]]:
    n = int(round(1.0 / step))
    weights = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            weights.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
    return weights


def evaluate_subtask2(y_true_labels: List[str], y_pred_labels: List[str]) -> Dict[str, Any]:
    s_conf_list = []
    for gt_lbl, pred_lbl in zip(y_true_labels, y_pred_labels):
        gt_dict = {"confidence": gt_lbl}
        pred_dict = {"confidence": pred_lbl}
        s_conf = compute_confidence_score(gt_dict, pred_dict)
        s_conf_list.append(s_conf)
    
    mean_s_conf = float(np.mean(s_conf_list))
    acc = float(accuracy_score(y_true_labels, y_pred_labels))
    
    classes = ["clear", "borderline", "uncertain"]
    macro_f1 = float(f1_score(y_true_labels, y_pred_labels, labels=classes, average="macro", zero_division=0))
    f1_per_class = f1_score(y_true_labels, y_pred_labels, labels=classes, average=None, zero_division=0)
    
    return {
        "s_conf": mean_s_conf,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_clear": float(f1_per_class[0]),
        "f1_borderline": float(f1_per_class[1]),
        "f1_uncertain": float(f1_per_class[2]),
    }


def main():
    print("=" * 80)
    print("RUNNING CHIMERA TASK 2 CONFIDENCE (SUBTASK 2.2) MCCV-30 BENCHMARK (exp_18)")
    print("=" * 80)
    
    data_dir = root_dir / "data" / "processed" / "task2"
    exp_dir = root_dir / "experiments" / "exp_18"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df_tab = pd.read_csv(data_dir / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(data_dir / "mri_embeddings_task2.csv")
    df_txt = pd.read_csv(data_dir / "clinical_texts_task2.csv")
    df_gt = pd.read_csv(data_dir / "ground_truth_task2.csv")
    splits_df = load_mccv_splits(str(data_dir / "mccv_splits_30.csv"))
    
    pids = df_gt["pid"].tolist()
    assert df_tab["pid"].tolist() == pids
    assert df_mri["pid"].tolist() == pids
    assert df_txt["pid"].tolist() == pids
    assert splits_df["pid"].tolist() == pids
    
    mri_feature_cols = [c for c in df_mri.columns if c != "pid"]
    
    y_raw_labels = df_gt["confidence"].tolist()
    y_soft_targets = np.array([CONF_VAL_MAP[lbl] for lbl in y_raw_labels], dtype=float)
    
    # -------------------------------------------------------------------------
    # PHASE 1: UNIMODAL GRID SEARCH
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Unimodal Hyperparameter Grid Search ---")
    
    tab_grid = [{"k": k, "weights": w} for k in [1, 3, 5, 7] for w in ["uniform", "distance"]]
    mri_grid = [{"k": k, "weights": w} for k in [1, 3, 5] for w in ["uniform", "distance"]]
    txt_grid = [{"k": k, "weights": w, "max_features": mf} for k in [1, 3, 5] for w in ["uniform", "distance"] for mf in [500, None]]
    
    def evaluate_grid(grid_configs, mod_type):
        best_cfg = None
        best_score = -1.0
        
        for cfg in grid_configs:
            split_scores = []
            for s_idx in range(30):
                col = f"split_{s_idx}"
                train_mask = (splits_df[col] == "train").values
                val_mask = (splits_df[col] == "val").values
                
                y_tr_soft = y_soft_targets[train_mask]
                y_va_lbls = [y_raw_labels[i] for i, v in enumerate(val_mask) if v]
                
                if mod_type == "tab":
                    df_tab_tr = df_tab[train_mask].copy()
                    df_tab_va = df_tab[val_mask].copy()
                    prep = Task2TabularPreprocessor().fit(df_tab_tr)
                    X_tr = prep.transform(df_tab_tr)
                    X_va = prep.transform(df_tab_va)
                elif mod_type == "mri":
                    scaler_mri = StandardScaler().fit(df_mri.loc[train_mask, mri_feature_cols].values)
                    X_tr = scaler_mri.transform(df_mri.loc[train_mask, mri_feature_cols].values)
                    X_va = scaler_mri.transform(df_mri.loc[val_mask, mri_feature_cols].values)
                elif mod_type == "txt":
                    tfidf = TfidfVectorizer(max_features=cfg["max_features"], lowercase=True)
                    s_tr = pd.Series(df_txt.loc[train_mask, "full_encounter_narrative"]).fillna("").tolist()
                    s_va = pd.Series(df_txt.loc[val_mask, "full_encounter_narrative"]).fillna("").tolist()
                    X_tr = tfidf.fit_transform(s_tr).toarray()
                    X_va = tfidf.transform(s_va).toarray()
                
                knn = KNeighborsRegressor(n_neighbors=cfg["k"], weights=cfg["weights"])
                knn.fit(X_tr, y_tr_soft)
                y_pred_cont = knn.predict(X_va)
                y_pred_lbls = [continuous_to_label(v) for v in y_pred_cont]
                
                res = evaluate_subtask2(y_va_lbls, y_pred_lbls)
                split_scores.append(res["s_conf"])
            
            mean_score = float(np.mean(split_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_cfg = cfg
        return best_cfg, best_score

    best_tab_cfg, tab_s_conf = evaluate_grid(tab_grid, "tab")
    best_mri_cfg, mri_s_conf = evaluate_grid(mri_grid, "mri")
    best_txt_cfg, txt_s_conf = evaluate_grid(txt_grid, "txt")
    
    print(f"  Best Tabular Config: {best_tab_cfg} -> S_conf: {tab_s_conf:.4f}")
    print(f"  Best MRI Config    : {best_mri_cfg} -> S_conf: {mri_s_conf:.4f}")
    print(f"  Best Text Config   : {best_txt_cfg} -> S_conf: {txt_s_conf:.4f}")
    
    # -------------------------------------------------------------------------
    # PHASE 2: MCCV-30 EVALUATION & SIMPLEX FUSION
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Full MCCV-30 Evaluation & Simplex Fusion Search ---")
    
    simplex_weights = generate_simplex_weights(step=0.05)
    
    c1_static_splits = []
    val_y_true_all_splits = []
    val_preds_tab_all = []
    val_preds_mri_all = []
    val_preds_txt_all = []
    
    # Static config (exp_12): tab=(k=1, uniform), mri=(k=1, uniform), txt=(k=3, unif, mf=500), w=(0.00, 0.55, 0.45)
    for s_idx in range(30):
        col = f"split_{s_idx}"
        train_mask = (splits_df[col] == "train").values
        val_mask = (splits_df[col] == "val").values
        
        y_tr_soft = y_soft_targets[train_mask]
        y_va_lbls = [y_raw_labels[i] for i, v in enumerate(val_mask) if v]
        val_y_true_all_splits.append(y_va_lbls)
        
        # Preprocess Tabular
        df_tab_tr = df_tab[train_mask].copy()
        df_tab_va = df_tab[val_mask].copy()
        prep = Task2TabularPreprocessor().fit(df_tab_tr)
        X_tab_tr = prep.transform(df_tab_tr)
        X_tab_va = prep.transform(df_tab_va)
        
        # MRI
        scaler_mri = StandardScaler().fit(df_mri.loc[train_mask, mri_feature_cols].values)
        X_mri_tr = scaler_mri.transform(df_mri.loc[train_mask, mri_feature_cols].values)
        X_mri_va = scaler_mri.transform(df_mri.loc[val_mask, mri_feature_cols].values)
        
        # Text for Best Config
        tfidf_opt = TfidfVectorizer(max_features=best_txt_cfg["max_features"], lowercase=True)
        s_tr = pd.Series(df_txt.loc[train_mask, "full_encounter_narrative"]).fillna("").tolist()
        s_va = pd.Series(df_txt.loc[val_mask, "full_encounter_narrative"]).fillna("").tolist()
        X_txt_tr = tfidf_opt.fit_transform(s_tr).toarray()
        X_txt_va = tfidf_opt.transform(s_va).toarray()
        
        # Train tuned models
        knn_tab = KNeighborsRegressor(n_neighbors=best_tab_cfg["k"], weights=best_tab_cfg["weights"]).fit(X_tab_tr, y_tr_soft)
        knn_mri = KNeighborsRegressor(n_neighbors=best_mri_cfg["k"], weights=best_mri_cfg["weights"]).fit(X_mri_tr, y_tr_soft)
        knn_txt = KNeighborsRegressor(n_neighbors=best_txt_cfg["k"], weights=best_txt_cfg["weights"]).fit(X_txt_tr, y_tr_soft)
        
        p_tab = knn_tab.predict(X_tab_va)
        p_mri = knn_mri.predict(X_mri_va)
        p_txt = knn_txt.predict(X_txt_va)
        
        val_preds_tab_all.append(p_tab)
        val_preds_mri_all.append(p_mri)
        val_preds_txt_all.append(p_txt)
        
        # Static baseline model (exp_12)
        tfidf_static = TfidfVectorizer(max_features=500, lowercase=True)
        X_txt_tr_stat = tfidf_static.fit_transform(s_tr).toarray()
        X_txt_va_stat = tfidf_static.transform(s_va).toarray()
        
        k_tab_st = KNeighborsRegressor(n_neighbors=1, weights="uniform").fit(X_tab_tr, y_tr_soft)
        k_mri_st = KNeighborsRegressor(n_neighbors=1, weights="uniform").fit(X_mri_tr, y_tr_soft)
        k_txt_st = KNeighborsRegressor(n_neighbors=3, weights="uniform").fit(X_txt_tr_stat, y_tr_soft)
        
        p_st = 0.00 * k_tab_st.predict(X_tab_va) + 0.55 * k_mri_st.predict(X_mri_va) + 0.45 * k_txt_st.predict(X_txt_va_stat)
        lbls_st = [continuous_to_label(v) for v in p_st]
        c1_static_splits.append(evaluate_subtask2(y_va_lbls, lbls_st))
        
    # Search optimal simplex weights on val predictions
    best_weights = None
    best_simplex_score = -1.0
    for w in simplex_weights:
        w_tab, w_mri, w_txt = w
        split_s_confs = []
        for s_idx in range(30):
            p_ens = w_tab * val_preds_tab_all[s_idx] + w_mri * val_preds_mri_all[s_idx] + w_txt * val_preds_txt_all[s_idx]
            lbls_ens = [continuous_to_label(v) for v in p_ens]
            res = evaluate_subtask2(val_y_true_all_splits[s_idx], lbls_ens)
            split_s_confs.append(res["s_conf"])
        mean_s_conf = float(np.mean(split_s_confs))
        if mean_s_conf > best_simplex_score:
            best_simplex_score = mean_s_conf
            best_weights = w
            
    print(f"  Optimal Simplex Weights (Tab, MRI, Txt): {best_weights} -> Mean S_conf: {best_simplex_score:.4f}")
    
    # Collect final per-split metrics for C2 Tuned
    c2_tuned_splits = []
    w_tab, w_mri, w_txt = best_weights
    for s_idx in range(30):
        p_ens = w_tab * val_preds_tab_all[s_idx] + w_mri * val_preds_mri_all[s_idx] + w_txt * val_preds_txt_all[s_idx]
        lbls_ens = [continuous_to_label(v) for v in p_ens]
        c2_tuned_splits.append(evaluate_subtask2(val_y_true_all_splits[s_idx], lbls_ens))
        
    # Aggregate Metrics
    def aggregate(splits_list):
        keys = splits_list[0].keys()
        agg = {}
        for k in keys:
            vals = [s[k] for s in splits_list]
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }
        return agg

    agg_c1 = aggregate(c1_static_splits)
    agg_c2 = aggregate(c2_tuned_splits)
    
    results = {
        "C1_Static_exp12": agg_c1,
        "C2_Fuzzy_KNN_Tuned": agg_c2,
    }
    
    print("\n" + "=" * 80)
    print("FINAL MCCV-30 RESULTS SUMMARY (CHIMERA_METRICS EVALUATION)")
    print("=" * 80)
    print(f"C1 (Static Baseline exp_12):")
    print(f"  S_conf   : {agg_c1['s_conf']['mean']:.4f} +/- {agg_c1['s_conf']['std']:.4f}")
    print(f"  Macro-F1 : {agg_c1['macro_f1']['mean']:.4f} +/- {agg_c1['macro_f1']['std']:.4f}")
    print(f"  Accuracy : {agg_c1['accuracy']['mean']:.4f} +/- {agg_c1['accuracy']['std']:.4f}")
    print(f"  F1 clear : {agg_c1['f1_clear']['mean']:.4f} | F1 borderline: {agg_c1['f1_borderline']['mean']:.4f}")
    print(f"\nC2 (Fuzzy-KNN Tuned):")
    print(f"  S_conf   : {agg_c2['s_conf']['mean']:.4f} +/- {agg_c2['s_conf']['std']:.4f}")
    print(f"  Macro-F1 : {agg_c2['macro_f1']['mean']:.4f} +/- {agg_c2['macro_f1']['std']:.4f}")
    print(f"  Accuracy : {agg_c2['accuracy']['mean']:.4f} +/- {agg_c2['accuracy']['std']:.4f}")
    print(f"  F1 clear : {agg_c2['f1_clear']['mean']:.4f} | F1 borderline: {agg_c2['f1_borderline']['mean']:.4f}")
    
    # Save artifacts
    with open(results_dir / "best_hyperparams.json", "w") as f:
        json.dump({
            "tabular": best_tab_cfg,
            "mri": best_mri_cfg,
            "text": best_txt_cfg,
        }, f, indent=2)
        
    with open(results_dir / "best_fusion_weights.json", "w") as f:
        json.dump({
            "weights": {
                "tabular": best_weights[0],
                "mri": best_weights[1],
                "text": best_weights[2],
            },
            "best_s_conf": best_simplex_score,
        }, f, indent=2)
        
    with open(results_dir / "split_metrics.json", "w") as f:
        json.dump({
            "C1_Static_exp12": c1_static_splits,
            "C2_Fuzzy_KNN_Tuned": c2_tuned_splits,
        }, f, indent=2)
        
    with open(results_dir / "aggregate_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[+] All artifacts successfully saved to {results_dir}")

if __name__ == "__main__":
    main()
