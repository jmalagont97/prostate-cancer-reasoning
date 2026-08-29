#!/usr/bin/env python3
"""
Exact MCCV-30 Replication of exp_12 Scalar Direct-KDM for CHIMERA Subtask 1.2 Clinical Confidence.
Zero Data Leakage: Strictly excludes all biopsy/pathology-derived fields (bx_positive, bx_missing, isup).
Evaluates clean 23col, 19col, ordinal-weighted supervision, and baseline across 30 stratified folds.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Ensure project and chimera_metrics imports are available
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"))

from chimera_metrics.splits import get_split_indices, load_mccv_splits
from chimera_metrics.task1.reasoning import compute_confidence_score

from models import (
    CONFIDENCE_CLASSES,
    CONFIDENCE_MAP,
    INV_CONFIDENCE_MAP,
    Exp12FeatureBuilder,
    ScalarDirectKDM,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_6" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def compute_fold_metrics(y_true_rank: np.ndarray, y_pred_rank: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true_rank, y_pred_rank))
    macro_f1 = float(f1_score(y_true_rank, y_pred_rank, average="macro", labels=[0, 1, 2], zero_division=0))
    
    # Official S_conf score
    s_conf_list = [1.0 - abs(p - t) / 2.0 for t, p in zip(y_true_rank, y_pred_rank)]
    s_conf_mean = float(np.mean(s_conf_list))
    ord_dist = 1.0 - s_conf_mean
    
    # Balanced MOE_abs
    class_moe = []
    for k in [0, 1, 2]:
        mask = (y_true_rank == k)
        if np.any(mask):
            class_moe.append(float(np.mean(np.abs(y_pred_rank[mask] - k) / 2.0)))
    moe_abs = float(np.mean(class_moe)) if class_moe else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "s_conf": s_conf_mean,
        "ordinal_distance": ord_dist,
        "moe_abs": moe_abs,
    }


def main() -> None:
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 80)
    log("STARTING EXPERIMENT exp_6: DIRECT REPLICATION OF exp_12 SCALAR-KDM (SUBTASK 1.2)")
    log("Protocol: MCCV-30 (73 train / 18 val) | Strict Zero-Leakage Preprocessing")
    log("Conditions: C1 (23col Clean) | C2 (19col Clean) | C3 (23col Ordinal-Weighted)")
    log("=" * 80)

    # 1. Load Data
    df_tab = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_tabular.csv")
    df_txt = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_texts.csv")
    df_mri = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "mri_embeddings.csv")
    df_gt = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "ground_truth_task1.csv")
    splits_df = load_mccv_splits(str(PROJECT_ROOT / "data" / "processed" / "task1" / "mccv_splits_30.csv"))

    labeled_gt = df_gt[df_gt["confidence"].notna()].copy()
    labeled_pids = set(labeled_gt["pid"].values)
    gt_map = dict(zip(labeled_gt["pid"], labeled_gt["confidence"]))

    log(f"Dataset: 91 labeled cases. Class counts: {labeled_gt['confidence'].value_counts().to_dict()}")

    CONDITIONS = [
        "C1_Scalar_KDM_23col_clean",
        "C2_Scalar_KDM_19col_clean",
        "C3_Scalar_KDM_23col_ordinal_weighted",
        "Baseline_Majority_Clear",
    ]

    results: Dict[str, Dict[str, List[float]]] = {
        cond: {"accuracy": [], "macro_f1": [], "s_conf": [], "ordinal_distance": [], "moe_abs": []}
        for cond in CONDITIONS
    }

    all_preds: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in CONDITIONS}

    log("-" * 80)
    log(f"{'Fold':<6} | {'C1 F1':<8} {'C1 Dist':<8} | {'C2 F1':<8} {'C2 Dist':<8} | {'C3 F1':<8} {'C3 Dist':<8} | {'Base F1':<8}")
    log("-" * 80)

    for split_idx in range(30):
        tr_pids_raw, val_pids_raw, _ = get_split_indices(splits_df, split_idx)
        tr_pids = [p for p in tr_pids_raw if p in labeled_pids]
        val_pids = [p for p in val_pids_raw if p in labeled_pids]

        y_tr_labels = [gt_map[p] for p in tr_pids]
        y_val_labels = [gt_map[p] for p in val_pids]

        y_tr = np.array([CONFIDENCE_MAP[l] for l in y_tr_labels])
        y_val = np.array([CONFIDENCE_MAP[l] for l in y_val_labels])

        # --- C1: 23col Clean ---
        builder_23 = Exp12FeatureBuilder(frame_tag="23col", n_mri_pca=2).fit(df_tab, df_txt, df_mri, tr_pids)
        X_tr_23 = builder_23.transform(df_tab, df_txt, df_mri, tr_pids)
        X_val_23 = builder_23.transform(df_tab, df_txt, df_mri, val_pids)

        model_c1 = ScalarDirectKDM(supervision_type="hard", random_state=split_idx).fit(X_tr_23, y_tr)
        preds_c1 = model_c1.predict(X_val_23)
        m_c1 = compute_fold_metrics(y_val, preds_c1)
        for k, v in m_c1.items():
            results["C1_Scalar_KDM_23col_clean"][k].append(v)

        for pid, t_lbl, p_rank in zip(val_pids, y_val_labels, preds_c1):
            all_preds["C1_Scalar_KDM_23col_clean"].append({
                "split": split_idx, "pid": pid, "ground_truth": t_lbl, "prediction": INV_CONFIDENCE_MAP[p_rank]
            })

        # --- C2: 19col Clean ---
        builder_19 = Exp12FeatureBuilder(frame_tag="19col", n_mri_pca=2).fit(df_tab, df_txt, df_mri, tr_pids)
        X_tr_19 = builder_19.transform(df_tab, df_txt, df_mri, tr_pids)
        X_val_19 = builder_19.transform(df_tab, df_txt, df_mri, val_pids)

        model_c2 = ScalarDirectKDM(supervision_type="hard", random_state=split_idx).fit(X_tr_19, y_tr)
        preds_c2 = model_c2.predict(X_val_19)
        m_c2 = compute_fold_metrics(y_val, preds_c2)
        for k, v in m_c2.items():
            results["C2_Scalar_KDM_19col_clean"][k].append(v)

        for pid, t_lbl, p_rank in zip(val_pids, y_val_labels, preds_c2):
            all_preds["C2_Scalar_KDM_19col_clean"].append({
                "split": split_idx, "pid": pid, "ground_truth": t_lbl, "prediction": INV_CONFIDENCE_MAP[p_rank]
            })

        # --- C3: 23col Ordinal Weighted ---
        model_c3 = ScalarDirectKDM(supervision_type="ordinal_weighted", random_state=split_idx).fit(X_tr_23, y_tr)
        preds_c3 = model_c3.predict(X_val_23)
        m_c3 = compute_fold_metrics(y_val, preds_c3)
        for k, v in m_c3.items():
            results["C3_Scalar_KDM_23col_ordinal_weighted"][k].append(v)

        for pid, t_lbl, p_rank in zip(val_pids, y_val_labels, preds_c3):
            all_preds["C3_Scalar_KDM_23col_ordinal_weighted"].append({
                "split": split_idx, "pid": pid, "ground_truth": t_lbl, "prediction": INV_CONFIDENCE_MAP[p_rank]
            })

        # --- Baseline Majority Clear ---
        preds_base = np.full_like(y_val, CONFIDENCE_MAP["clear"])
        m_base = compute_fold_metrics(y_val, preds_base)
        for k, v in m_base.items():
            results["Baseline_Majority_Clear"][k].append(v)

        log(
            f"Fold {split_idx+1:02d} | "
            f"{m_c1['macro_f1']:.4f}   {m_c1['ordinal_distance']:.4f}   | "
            f"{m_c2['macro_f1']:.4f}   {m_c2['ordinal_distance']:.4f}   | "
            f"{m_c3['macro_f1']:.4f}   {m_c3['ordinal_distance']:.4f}   | "
            f"{m_base['macro_f1']:.4f}"
        )

    log("=" * 80)
    log("AGGREGATE RESULTS ACROSS 30 FOLDS (MCCV-30)")
    log("=" * 80)

    summary_records = []
    for cond in CONDITIONS:
        cond_dir = RESULTS_DIR / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        res_dict = {
            "condition": cond,
            "macro_f1_mean": float(np.mean(results[cond]["macro_f1"])),
            "macro_f1_std": float(np.std(results[cond]["macro_f1"])),
            "s_conf_mean": float(np.mean(results[cond]["s_conf"])),
            "s_conf_std": float(np.std(results[cond]["s_conf"])),
            "ordinal_distance_mean": float(np.mean(results[cond]["ordinal_distance"])),
            "ordinal_distance_std": float(np.std(results[cond]["ordinal_distance"])),
            "moe_abs_mean": float(np.mean(results[cond]["moe_abs"])),
            "moe_abs_std": float(np.std(results[cond]["moe_abs"])),
            "accuracy_mean": float(np.mean(results[cond]["accuracy"])),
            "accuracy_std": float(np.std(results[cond]["accuracy"])),
            "n_folds": 30,
        }

        with open(cond_dir / "metrics.json", "w") as f:
            json.dump(res_dict, f, indent=2)

        pred_df = pd.DataFrame(all_preds[cond])
        pred_df.to_csv(cond_dir / "predictions.csv", index=False)

        with open(cond_dir / "version.txt", "w") as f:
            f.write(f"exp_6 replication of exp_12 on MCCV-30 | timestamp: {datetime.now().isoformat()}\n")

        summary_records.append(res_dict)
        log(
            f"[{cond:<36}] Macro-F1: {res_dict['macro_f1_mean']:.4f} ± {res_dict['macro_f1_std']:.4f} | "
            f"Ord-Dist: {res_dict['ordinal_distance_mean']:.4f} ± {res_dict['ordinal_distance_std']:.4f} | "
            f"MOE_abs: {res_dict['moe_abs_mean']:.4f} | Acc: {res_dict['accuracy_mean']:.4f}"
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    log("=" * 80)
    log(f"All artifacts saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
