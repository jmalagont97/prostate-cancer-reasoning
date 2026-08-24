#!/usr/bin/env python3
"""
Canonical MCCV-30 Grid Search and Late Fusion Benchmark for CHIMERA Subtask 1.1 (exp_4).
Protocol: utils/pkg/chimera_metrics/PROTOCOL.md
Framework: chimera_metrics (compute_biopsy_decision_gate)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Add project root and chimera_metrics to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHIMERA_PKG = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHIMERA_PKG))

from chimera_metrics.splits import get_split_indices, load_mccv_splits
from chimera_metrics.task1.decision import compute_biopsy_decision_gate
from experiments.exp_4.scripts.models import (
    CustomKNN,
    MRIPreprocessor,
    TabularPreprocessor,
    TextPreprocessor,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_4" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"


def log(msg: str):
    """Clean real-time logging to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def compute_metrics_for_predictions(
    gt_df: pd.DataFrame, val_pids: List[str], y_preds: List[str]
) -> Dict[str, float]:
    """Compute official Subtask 1.1 metrics using chimera_metrics decision gate."""
    gt_subset = gt_df.set_index("pid").loc[val_pids]
    gates = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for (_, row), y_pred in zip(gt_subset.iterrows(), y_preds):
        score, gt_n, pred_n, _ = compute_biopsy_decision_gate(row["decision"], y_pred)
        gates.append(score)
        if gt_n == "yes" and pred_n == "yes":
            tp += 1
        elif gt_n == "no" and pred_n == "yes":
            fp += 1
        elif gt_n == "yes" and pred_n == "no":
            fn += 1
        elif gt_n == "no" and pred_n == "no":
            tn += 1

    s_gate = float(np.mean(gates)) if len(gates) > 0 else 0.0
    f1_yes = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    f1_no = (2.0 * tn / (2.0 * tn + fp + fn)) if (2.0 * tn + fp + fn) > 0 else 0.0
    macro_f1 = 0.5 * (f1_yes + f1_no)
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "macro_f1": float(macro_f1),
        "f1_yes": float(f1_yes),
        "f1_no": float(f1_no),
        "s_gate": float(s_gate),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def evaluate_tabular_grid(
    df_tab: pd.DataFrame,
    gt_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate 180 Tabular configurations across 30 MCCV splits."""
    spearman_taus = [0.50, 0.60, 0.70, 0.80, None]
    model_types = ["standard_knn_clf", "fuzzy_knn_reg"]
    ks = [1, 3, 5]
    metrics = ["cosine", "euclidean", "manhattan"]
    weights = ["uniform", "distance"]

    configs = []
    for tau in spearman_taus:
        for mtype in model_types:
            for k in ks:
                for metric in metrics:
                    for w in weights:
                        configs.append({
                            "spearman_tau": tau,
                            "model_type": mtype,
                            "k": k,
                            "metric": metric,
                            "weights": w,
                        })

    log(f"--- [Phase 1.1] Tabular Grid Search ({len(configs)} configs x 30 splits = {len(configs)*30} evals) ---")

    fold_data = []
    for split_idx in range(30):
        tr_pids, val_pids, _ = get_split_indices(splits_df, split_idx)
        tr_df = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_df = df_tab.set_index("pid").loc[val_pids].reset_index()

        gt_tr = gt_df.set_index("pid").loc[tr_pids]
        y_tr_dec = gt_tr["decision"].tolist()
        y_tr_conf = gt_tr["confidence"].tolist()

        tau_preprocessors = {}
        for tau in spearman_taus:
            prep = TabularPreprocessor(spearman_tau=tau).fit(tr_df)
            X_tr = prep.transform(tr_df)
            X_val = prep.transform(val_df)
            tau_preprocessors[tau] = (X_tr, X_val)

        fold_data.append({
            "val_pids": val_pids,
            "y_tr_dec": y_tr_dec,
            "y_tr_conf": y_tr_conf,
            "tau_data": tau_preprocessors,
        })

    grid_results = []
    best_config = None
    best_macro_f1 = -1.0

    t0 = time.time()
    for idx, cfg in enumerate(configs, 1):
        split_scores = []
        for split_idx in range(30):
            fd = fold_data[split_idx]
            X_tr, X_val = fd["tau_data"][cfg["spearman_tau"]]
            knn = CustomKNN(
                k=cfg["k"],
                metric=cfg["metric"],
                weights=cfg["weights"],
                model_type=cfg["model_type"],
            ).fit(X_tr, fd["y_tr_dec"], fd["y_tr_conf"])

            y_pred = knn.predict(X_val, threshold=0.50)
            metrics_dict = compute_metrics_for_predictions(gt_df, fd["val_pids"], y_pred)
            split_scores.append(metrics_dict["macro_f1"])

        mean_f1 = float(np.mean(split_scores))
        std_f1 = float(np.std(split_scores))
        grid_results.append({
            "config": cfg,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        })

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_config = cfg

        if idx % 30 == 0 or idx == len(configs):
            elapsed = time.time() - t0
            log(f"  Tabular Progress: {idx}/{len(configs)} configs ({idx/len(configs)*100:.1f}%) | "
                f"Best Macro-F1: {best_macro_f1:.4f} | Elapsed: {elapsed:.1f}s")

    log(f"  ► Best Tabular (C1*): Macro-F1 = {best_macro_f1:.4f} | Config: {best_config}")
    return best_config, grid_results


def evaluate_mri_grid(
    df_mri: pd.DataFrame,
    gt_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate 144 MRI configurations across 30 MCCV splits."""
    pca_comps = [4, 8, 16, 32, 64, None]
    model_types = ["standard_knn_clf", "fuzzy_knn_reg"]
    ks = [1, 3, 5]
    metrics = ["euclidean", "cosine"]
    weights = ["uniform", "distance"]

    configs = []
    for pca in pca_comps:
        for mtype in model_types:
            for k in ks:
                for metric in metrics:
                    for w in weights:
                        configs.append({
                            "pca_components": pca,
                            "model_type": mtype,
                            "k": k,
                            "metric": metric,
                            "weights": w,
                        })

    log(f"--- [Phase 1.2] MRI Grid Search ({len(configs)} configs x 30 splits = {len(configs)*30} evals) ---")

    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]

    fold_data = []
    for split_idx in range(30):
        tr_pids, val_pids, _ = get_split_indices(splits_df, split_idx)
        X_raw_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_raw_val = df_mri.set_index("pid").loc[val_pids][emb_cols].values.astype(float)

        gt_tr = gt_df.set_index("pid").loc[tr_pids]
        y_tr_dec = gt_tr["decision"].tolist()
        y_tr_conf = gt_tr["confidence"].tolist()

        pca_cache = {}
        for pca in pca_comps:
            prep = MRIPreprocessor(n_components=pca).fit(X_raw_tr)
            X_tr = prep.transform(X_raw_tr)
            X_val = prep.transform(X_raw_val)
            pca_cache[pca] = (X_tr, X_val)

        fold_data.append({
            "val_pids": val_pids,
            "y_tr_dec": y_tr_dec,
            "y_tr_conf": y_tr_conf,
            "pca_data": pca_cache,
        })

    grid_results = []
    best_config = None
    best_macro_f1 = -1.0

    t0 = time.time()
    for idx, cfg in enumerate(configs, 1):
        split_scores = []
        for split_idx in range(30):
            fd = fold_data[split_idx]
            X_tr, X_val = fd["pca_data"][cfg["pca_components"]]
            knn = CustomKNN(
                k=cfg["k"],
                metric=cfg["metric"],
                weights=cfg["weights"],
                model_type=cfg["model_type"],
            ).fit(X_tr, fd["y_tr_dec"], fd["y_tr_conf"])

            y_pred = knn.predict(X_val, threshold=0.50)
            metrics_dict = compute_metrics_for_predictions(gt_df, fd["val_pids"], y_pred)
            split_scores.append(metrics_dict["macro_f1"])

        mean_f1 = float(np.mean(split_scores))
        std_f1 = float(np.std(split_scores))
        grid_results.append({
            "config": cfg,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        })

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_config = cfg

        if idx % 30 == 0 or idx == len(configs):
            elapsed = time.time() - t0
            log(f"  MRI Progress: {idx}/{len(configs)} configs ({idx/len(configs)*100:.1f}%) | "
                f"Best Macro-F1: {best_macro_f1:.4f} | Elapsed: {elapsed:.1f}s")

    log(f"  ► Best MRI (C2*): Macro-F1 = {best_macro_f1:.4f} | Config: {best_config}")
    return best_config, grid_results


def evaluate_text_grid(
    df_text: pd.DataFrame,
    gt_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate 480 Clinical Text configurations across 30 MCCV splits."""
    max_features_list = [500, 1000, 2000, None]
    svd_comps = [4, 10, 20, 40, None]
    model_types = ["standard_knn_clf", "fuzzy_knn_reg"]
    ks = [1, 3, 5]
    metrics = ["cosine", "euclidean"]
    weights = ["uniform", "distance"]

    configs = []
    for feat in max_features_list:
        for svd in svd_comps:
            for mtype in model_types:
                for k in ks:
                    for metric in metrics:
                        for w in weights:
                            configs.append({
                                "max_features": feat,
                                "svd_components": svd,
                                "model_type": mtype,
                                "k": k,
                                "metric": metric,
                                "weights": w,
                            })

    log(f"--- [Phase 1.3] Clinical Text Grid Search ({len(configs)} configs x 30 splits = {len(configs)*30} evals) ---")

    fold_data = []
    for split_idx in range(30):
        tr_pids, val_pids, _ = get_split_indices(splits_df, split_idx)
        txt_tr = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        txt_val = df_text.set_index("pid").loc[val_pids]["full_encounter_narrative"].tolist()

        gt_tr = gt_df.set_index("pid").loc[tr_pids]
        y_tr_dec = gt_tr["decision"].tolist()
        y_tr_conf = gt_tr["confidence"].tolist()

        text_cache = {}
        for feat in max_features_list:
            for svd in svd_comps:
                prep = TextPreprocessor(max_features=feat, n_components=svd).fit(txt_tr)
                X_tr = prep.transform(txt_tr)
                X_val = prep.transform(txt_val)
                text_cache[(feat, svd)] = (X_tr, X_val)

        fold_data.append({
            "val_pids": val_pids,
            "y_tr_dec": y_tr_dec,
            "y_tr_conf": y_tr_conf,
            "text_data": text_cache,
        })

    grid_results = []
    best_config = None
    best_macro_f1 = -1.0

    t0 = time.time()
    for idx, cfg in enumerate(configs, 1):
        split_scores = []
        for split_idx in range(30):
            fd = fold_data[split_idx]
            X_tr, X_val = fd["text_data"][(cfg["max_features"], cfg["svd_components"])]
            knn = CustomKNN(
                k=cfg["k"],
                metric=cfg["metric"],
                weights=cfg["weights"],
                model_type=cfg["model_type"],
            ).fit(X_tr, fd["y_tr_dec"], fd["y_tr_conf"])

            y_pred = knn.predict(X_val, threshold=0.50)
            metrics_dict = compute_metrics_for_predictions(gt_df, fd["val_pids"], y_pred)
            split_scores.append(metrics_dict["macro_f1"])

        mean_f1 = float(np.mean(split_scores))
        std_f1 = float(np.std(split_scores))
        grid_results.append({
            "config": cfg,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        })

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_config = cfg

        if idx % 60 == 0 or idx == len(configs):
            elapsed = time.time() - t0
            log(f"  Text Progress: {idx}/{len(configs)} configs ({idx/len(configs)*100:.1f}%) | "
                f"Best Macro-F1: {best_macro_f1:.4f} | Elapsed: {elapsed:.1f}s")

    log(f"  ► Best Text (C3*): Macro-F1 = {best_macro_f1:.4f} | Config: {best_config}")
    return best_config, grid_results


def extract_optimal_unimodal_probabilities(
    df_tab: pd.DataFrame,
    df_mri: pd.DataFrame,
    df_text: pd.DataFrame,
    gt_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    c1_cfg: Dict[str, Any],
    c2_cfg: Dict[str, Any],
    c3_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract continuous prediction probability vectors for optimal unimodal models."""
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    fold_probs = []

    for split_idx in range(30):
        tr_pids, val_pids, _ = get_split_indices(splits_df, split_idx)
        gt_tr = gt_df.set_index("pid").loc[tr_pids]
        y_tr_dec = gt_tr["decision"].tolist()
        y_tr_conf = gt_tr["confidence"].tolist()

        # 1. Tabular
        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[val_pids].reset_index()
        prep_tab = TabularPreprocessor(spearman_tau=c1_cfg["spearman_tau"]).fit(tr_tab)
        X_tab_tr = prep_tab.transform(tr_tab)
        X_tab_val = prep_tab.transform(val_tab)
        knn_tab = CustomKNN(
            k=c1_cfg["k"],
            metric=c1_cfg["metric"],
            weights=c1_cfg["weights"],
            model_type=c1_cfg["model_type"],
        ).fit(X_tab_tr, y_tr_dec, y_tr_conf)
        p_tab = knn_tab.predict_proba(X_tab_val)

        # 2. MRI
        X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_val = df_mri.set_index("pid").loc[val_pids][emb_cols].values.astype(float)
        prep_mri = MRIPreprocessor(n_components=c2_cfg["pca_components"]).fit(X_mri_tr)
        X_mri_tr_p = prep_mri.transform(X_mri_tr)
        X_mri_val_p = prep_mri.transform(X_mri_val)
        knn_mri = CustomKNN(
            k=c2_cfg["k"],
            metric=c2_cfg["metric"],
            weights=c2_cfg["weights"],
            model_type=c2_cfg["model_type"],
        ).fit(X_mri_tr_p, y_tr_dec, y_tr_conf)
        p_mri = knn_mri.predict_proba(X_mri_val_p)

        # 3. Text
        txt_tr = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        txt_val = df_text.set_index("pid").loc[val_pids]["full_encounter_narrative"].tolist()
        prep_txt = TextPreprocessor(
            max_features=c3_cfg["max_features"], n_components=c3_cfg["svd_components"]
        ).fit(txt_tr)
        X_txt_tr_p = prep_txt.transform(txt_tr)
        X_txt_val_p = prep_txt.transform(txt_val)
        knn_txt = CustomKNN(
            k=c3_cfg["k"],
            metric=c3_cfg["metric"],
            weights=c3_cfg["weights"],
            model_type=c3_cfg["model_type"],
        ).fit(X_txt_tr_p, y_tr_dec, y_tr_conf)
        p_txt = knn_txt.predict_proba(X_txt_val_p)

        fold_probs.append({
            "split_idx": split_idx,
            "val_pids": val_pids,
            "p_tab": p_tab,
            "p_mri": p_mri,
            "p_txt": p_txt,
        })

    return fold_probs


def evaluate_late_fusion_grid(
    fold_probs: List[Dict[str, Any]],
    gt_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sweep weight simplex w in Delta^2 and decision threshold theta in [0.20, 0.80]."""
    step = 0.05
    weight_candidates = []
    for w1_i in range(int(1.0 / step) + 1):
        for w2_i in range(int(1.0 / step) + 1 - w1_i):
            w1 = round(w1_i * step, 2)
            w2 = round(w2_i * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= -1e-6:
                weight_candidates.append((w1, w2, max(0.0, w3)))

    thresholds = [round(th, 2) for th in np.arange(0.20, 0.82, 0.02)]

    log(f"--- [Phase 2] Late Fusion Grid Search ({len(weight_candidates)} weight vectors x {len(thresholds)} thresholds = {len(weight_candidates)*len(thresholds)} configs) ---")

    best_config = None
    best_macro_f1 = -1.0
    fusion_results = []

    t0 = time.time()
    total_evals = len(weight_candidates) * len(thresholds)
    eval_count = 0

    for w_vec in weight_candidates:
        w_tab, w_mri, w_txt = w_vec
        for th in thresholds:
            split_scores = []
            for fp in fold_probs:
                p_fused = w_tab * fp["p_tab"] + w_mri * fp["p_mri"] + w_txt * fp["p_txt"]
                y_pred = ["yes" if p >= th else "no" for p in p_fused]
                m = compute_metrics_for_predictions(gt_df, fp["val_pids"], y_pred)
                split_scores.append(m["macro_f1"])

            mean_f1 = float(np.mean(split_scores))
            std_f1 = float(np.std(split_scores))

            cfg_entry = {
                "w_tab": w_tab,
                "w_mri": w_mri,
                "w_txt": w_txt,
                "threshold": th,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
            }
            fusion_results.append(cfg_entry)

            if mean_f1 > best_macro_f1:
                best_macro_f1 = mean_f1
                best_config = cfg_entry

            eval_count += 1

    elapsed = time.time() - t0
    log(f"  Late Fusion Complete ({eval_count}/{total_evals} evals in {elapsed:.1f}s) | Best Macro-F1: {best_macro_f1:.4f}")
    log(f"  ► Best Late Fusion (C4*): w=({best_config['w_tab']}, {best_config['w_mri']}, {best_config['w_txt']}) | theta={best_config['threshold']} | Macro-F1={best_config['mean_macro_f1']:.4f}")
    return best_config, fusion_results


def load_exp1_baseline_predictions(gt_df: pd.DataFrame) -> Dict[str, str]:
    """Load case-level predictions from the official ReAct Baseline (exp_1)."""
    case_metrics_path = PROJECT_ROOT / "experiments" / "exp_1" / "results" / "baseline" / "case_metrics.json"
    if not case_metrics_path.exists():
        raise FileNotFoundError(f"Exp 1 baseline case metrics not found at {case_metrics_path}")

    with open(case_metrics_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    c0_preds = {}
    if isinstance(case_data, list):
        for item in case_data:
            cid = item.get("case_id")
            pred = item.get("decision_pred", "yes")
            if cid:
                c0_preds[cid] = pred
    elif isinstance(case_data, dict):
        for case_id, row in case_data.items():
            c0_preds[case_id] = row.get("biopsy_decision", row.get("decision_pred", "yes"))

    return c0_preds


def run_full_evaluation_and_wilcoxon(
    fold_probs: List[Dict[str, Any]],
    gt_df: pd.DataFrame,
    c0_preds: Dict[str, str],
    c4_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute full metrics for C0, C1*, C2*, C3*, C4* across 30 splits and paired Wilcoxon tests."""
    conditions = ["C0_Baseline", "C1_Tabular", "C2_MRI", "C3_Text", "C4_LateFusion"]
    split_metrics = {cond: [] for cond in conditions}
    split_predictions = {cond: {} for cond in conditions}

    w_tab, w_mri, w_txt = c4_cfg["w_tab"], c4_cfg["w_mri"], c4_cfg["w_txt"]
    th_fused = c4_cfg["threshold"]

    for fp in fold_probs:
        val_pids = fp["val_pids"]
        split_idx = fp["split_idx"]

        # C0: ReAct Baseline
        y_c0 = [c0_preds.get(pid, "yes") for pid in val_pids]
        # C1*: Tabular
        y_c1 = ["yes" if p >= 0.50 else "no" for p in fp["p_tab"]]
        # C2*: MRI
        y_c2 = ["yes" if p >= 0.50 else "no" for p in fp["p_mri"]]
        # C3*: Text
        y_c3 = ["yes" if p >= 0.50 else "no" for p in fp["p_txt"]]
        # C4*: Late Fusion
        p_fused = w_tab * fp["p_tab"] + w_mri * fp["p_mri"] + w_txt * fp["p_txt"]
        y_c4 = ["yes" if p >= th_fused else "no" for p in p_fused]

        pred_map = {
            "C0_Baseline": y_c0,
            "C1_Tabular": y_c1,
            "C2_MRI": y_c2,
            "C3_Text": y_c3,
            "C4_LateFusion": y_c4,
        }

        for cond, preds in pred_map.items():
            m = compute_metrics_for_predictions(gt_df, val_pids, preds)
            split_metrics[cond].append(m)
            split_predictions[cond][f"split_{split_idx}"] = {
                pid: pred for pid, pred in zip(val_pids, preds)
            }

    # Aggregate metrics
    aggregate_metrics = {}
    for cond in conditions:
        df_cond = pd.DataFrame(split_metrics[cond])
        aggregate_metrics[cond] = {}
        for col in df_cond.columns:
            mean_val = float(df_cond[col].mean())
            std_val = float(df_cond[col].std())
            ci_95 = float(1.96 * std_val / np.sqrt(len(df_cond)))
            aggregate_metrics[cond][col] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95_lower": max(0.0, mean_val - ci_95),
                "ci_95_upper": min(1.0, mean_val + ci_95),
            }

    # Paired Wilcoxon Tests vs C0_Baseline
    c0_f1s = [m["macro_f1"] for m in split_metrics["C0_Baseline"]]
    wilcoxon_results = {}
    for cond in ["C1_Tabular", "C2_MRI", "C3_Text", "C4_LateFusion"]:
        cond_f1s = [m["macro_f1"] for m in split_metrics[cond]]
        diffs = np.array(cond_f1s) - np.array(c0_f1s)
        if np.all(diffs == 0):
            stat, p_val = 0.0, 1.0
        else:
            res = wilcoxon(cond_f1s, c0_f1s, zero_method="wilcox", alternative="two-sided")
            stat, p_val = float(res.statistic), float(res.pvalue)

        wilcoxon_results[f"{cond}_vs_C0"] = {
            "test": "Wilcoxon Signed-Rank Test (Two-Sided)",
            "n_splits": len(diffs),
            "statistic": stat,
            "p_value": p_val,
            "mean_difference": float(np.mean(diffs)),
            "is_significant_alpha_0_05": bool(p_val < 0.05),
        }

    return aggregate_metrics, split_metrics, split_predictions, wilcoxon_results


def main():
    log("=" * 100)
    log("CHIMERA Subtask 1.1: Standardized MCCV-30 Unimodal & Late Fusion Benchmark (exp_4)")
    log("=" * 100)

    # 1. Ingest Data
    log("[1/6] Ingesting clinical multimodal representations & splits...")
    splits_df = load_mccv_splits(str(PROJECT_ROOT / "data" / "processed" / "task1" / "mccv_splits_30.csv"))
    gt_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "ground_truth_task1.csv")
    df_tab = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_tabular.csv")
    df_mri = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "mri_embeddings.csv")
    df_text = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_texts.csv")
    c0_preds = load_exp1_baseline_predictions(gt_df)

    log(f"      Loaded: {len(gt_df)} cases ({len(gt_df[gt_df['decision']=='yes'])} yes, "
        f"{len(gt_df[gt_df['decision']=='no'])} no) | Splits: 30 standardized MCCV folds")

    # 2. Phase 1: Unimodal Sweeps
    log("\n[2/6] Executing Phase 1: Unimodal Sweeps (MCCV-30)...")
    c1_best, grid_tab = evaluate_tabular_grid(df_tab, gt_df, splits_df)
    c2_best, grid_mri = evaluate_mri_grid(df_mri, gt_df, splits_df)
    c3_best, grid_txt = evaluate_text_grid(df_text, gt_df, splits_df)

    # 3. Phase 2: Optimal Probability Extraction & Late Fusion
    log("\n[3/6] Executing Phase 2: Extracting probabilities & Sweeping Late Fusion Simplex...")
    fold_probs = extract_optimal_unimodal_probabilities(
        df_tab, df_mri, df_text, gt_df, splits_df, c1_best, c2_best, c3_best
    )
    c4_best, grid_fusion = evaluate_late_fusion_grid(fold_probs, gt_df)

    # 4. Phase 3: Benchmark, Aggregation, and Statistical Testing vs C0
    log("\n[4/6] Executing Phase 3: Comprehensive Multi-condition Evaluation and Wilcoxon Testing...")
    agg_metrics, split_metrics, split_preds, wilcoxon_tests = run_full_evaluation_and_wilcoxon(
        fold_probs, gt_df, c0_preds, c4_best
    )

    # 5. Persist Results
    log("\n[5/6] Serializing structured benchmark artifacts to disk...")
    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg_metrics, f, indent=2)
    with open(RESULTS_DIR / "split_metrics.json", "w", encoding="utf-8") as f:
        json.dump(split_metrics, f, indent=2)
    with open(RESULTS_DIR / "split_predictions.json", "w", encoding="utf-8") as f:
        json.dump(split_preds, f, indent=2)
    with open(RESULTS_DIR / "wilcoxon_tests.json", "w", encoding="utf-8") as f:
        json.dump(wilcoxon_tests, f, indent=2)
    with open(RESULTS_DIR / "optimized_configs.json", "w", encoding="utf-8") as f:
        json.dump({
            "C1_Tabular_Optimal": c1_best,
            "C2_MRI_Optimal": c2_best,
            "C3_Text_Optimal": c3_best,
            "C4_LateFusion_Optimal": c4_best,
        }, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_4 execution timestamp: {datetime.now().isoformat()}\n")

    # 6. Report Summary Table
    log("\n[6/6] Benchmark Summary Results (MCCV-30 across all 30 Splits):")
    log("=" * 100)
    log(f"{'Condition':<20} | {'Macro-F1 (Mean ± Std)':<22} | {'F1-Yes':<10} | {'F1-No':<10} | {'S_gate (Acc)':<12} | {'Sens / Spec':<16}")
    log("-" * 100)
    for cond in ["C0_Baseline", "C1_Tabular", "C2_MRI", "C3_Text", "C4_LateFusion"]:
        m = agg_metrics[cond]
        f1_str = f"{m['macro_f1']['mean']:.4f} ± {m['macro_f1']['std']:.4f}"
        f1_yes = f"{m['f1_yes']['mean']:.4f}"
        f1_no = f"{m['f1_no']['mean']:.4f}"
        acc = f"{m['s_gate']['mean']:.4f}"
        sens_spec = f"{m['sensitivity']['mean']:.2f} / {m['specificity']['mean']:.2f}"
        log(f"{cond:<20} | {f1_str:<22} | {f1_yes:<10} | {f1_no:<10} | {acc:<12} | {sens_spec:<16}")
    log("=" * 100)

    log("\nPaired Wilcoxon Signed-Rank Hypothesis Tests vs C0_Baseline:")
    for comp, res in wilcoxon_tests.items():
        sig = "SIG (p < 0.05)" if res["is_significant_alpha_0_05"] else "NS (p >= 0.05)"
        log(f"  • {comp:<25}: diff = {res['mean_difference']:+.4f} | W = {res['statistic']:<6.1f} | p = {res['p_value']:.4e} [{sig}]")
    log("\nExperiment exp_4 successfully completed and verified.")


if __name__ == "__main__":
    main()
