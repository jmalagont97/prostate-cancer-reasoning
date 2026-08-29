#!/usr/bin/env python3
"""
Canonical MCCV-30 BrentMemKDM Continuous-Bandwidth Replication for CHIMERA Subtask 1.1 (exp_5).
Protocol: utils/pkg/chimera_metrics/PROTOCOL.md
Framework: chimera_metrics (compute_biopsy_decision_gate)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

# Add project root and chimera_metrics to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHIMERA_PKG = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHIMERA_PKG))

from chimera_metrics.splits import get_split_indices, load_mccv_splits
from chimera_metrics.task1.decision import compute_biopsy_decision_gate
from experiments.exp_5.scripts.models import (
    BrentMemKDM,
    MRIPreprocessor,
    TabularPreprocessor,
    TextPreprocessor,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_5" / "results"
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


def optimize_sigma_brent(
    X_train: np.ndarray,
    y_tr_dec: List[str],
    y_tr_conf: List[str],
    X_val: np.ndarray,
    val_pids: List[str],
    gt_df: pd.DataFrame,
    is_fuzzy: bool,
    metric: str,
    n_prescan: int = 15,
    bounds_mult: Tuple[float, float] = (1.0 / 32.0, 32.0),
    xatol: float = 1e-2,
    maxiter: int = 20,
) -> Tuple[float, float]:
    """
    Optimize the continuous RBF bandwidth sigma for a single modality using
    log-uniform prescan + bounded Brent search on out-of-fold MCCV Macro-F1.
    Returns: (sigma_optimal, macro_f1_optimal)
    """
    model = BrentMemKDM(is_fuzzy=is_fuzzy, metric=metric).fit(X_train, y_tr_dec, y_tr_conf)
    sigma_ref = BrentMemKDM.estimate_reference_sigma(X_train, k_neighbor=3, metric=metric)

    def neg_macro_f1(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        probs = model.predict_proba(X_val, sigma=sigma)
        y_pred = ["yes" if p >= 0.50 else "no" for p in probs]
        m = compute_metrics_for_predictions(gt_df, val_pids, y_pred)
        return -m["macro_f1"]

    # Phase A: Log-Uniform Prescan to bracket the optimum
    log_bounds = (np.log(sigma_ref * bounds_mult[0]), np.log(sigma_ref * bounds_mult[1]))
    prescan_points = np.linspace(log_bounds[0], log_bounds[1], n_prescan)
    prescan_scores = [neg_macro_f1(ls) for ls in prescan_points]

    best_i = int(np.argmin(prescan_scores))
    lo = prescan_points[max(0, best_i - 1)]
    hi = prescan_points[min(len(prescan_points) - 1, best_i + 1)]

    # Phase B: Bounded Brent Refinement
    res = minimize_scalar(
        neg_macro_f1,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    sigma_star = float(np.exp(res.x))
    f1_star = float(-res.fun)
    return sigma_star, f1_star


def run_brent_unimodal_phase(
    df_tab: pd.DataFrame,
    df_mri: pd.DataFrame,
    df_text: pd.DataFrame,
    gt_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    is_fuzzy: bool,
) -> Dict[str, Any]:
    """
    Phase 1: For each MCCV split and each modality, run Brent bandwidth optimization.
    Stores per-split optimal sigmas and out-of-fold probabilities.
    """
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    per_split_results = []

    for split_idx in range(30):
        tr_pids, val_pids, _ = get_split_indices(splits_df, split_idx)
        gt_tr = gt_df.set_index("pid").loc[tr_pids]
        y_tr_dec = gt_tr["decision"].tolist()
        y_tr_conf = gt_tr["confidence"].tolist()

        # --- Tabular ---
        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[val_pids].reset_index()
        prep_tab = TabularPreprocessor(spearman_tau=None).fit(tr_tab)
        X_tab_tr = prep_tab.transform(tr_tab)
        X_tab_val = prep_tab.transform(val_tab)
        sig_tab, f1_tab = optimize_sigma_brent(
            X_tab_tr, y_tr_dec, y_tr_conf, X_tab_val, val_pids, gt_df, is_fuzzy, "euclidean"
        )
        model_tab = BrentMemKDM(is_fuzzy=is_fuzzy, metric="euclidean").fit(X_tab_tr, y_tr_dec, y_tr_conf)
        p_tab = model_tab.predict_proba(X_tab_val, sigma=sig_tab)

        # --- MRI ---
        X_mri_raw_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_raw_val = df_mri.set_index("pid").loc[val_pids][emb_cols].values.astype(float)
        prep_mri = MRIPreprocessor(n_components=16).fit(X_mri_raw_tr)
        X_mri_tr = prep_mri.transform(X_mri_raw_tr)
        X_mri_val = prep_mri.transform(X_mri_raw_val)
        sig_mri, f1_mri = optimize_sigma_brent(
            X_mri_tr, y_tr_dec, y_tr_conf, X_mri_val, val_pids, gt_df, is_fuzzy, "euclidean"
        )
        model_mri = BrentMemKDM(is_fuzzy=is_fuzzy, metric="euclidean").fit(X_mri_tr, y_tr_dec, y_tr_conf)
        p_mri = model_mri.predict_proba(X_mri_val, sigma=sig_mri)

        # --- Text ---
        txt_tr = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        txt_val = df_text.set_index("pid").loc[val_pids]["full_encounter_narrative"].tolist()
        prep_txt = TextPreprocessor(max_features=1000, n_components=None).fit(txt_tr)
        X_txt_tr = prep_txt.transform(txt_tr)
        X_txt_val = prep_txt.transform(txt_val)
        sig_txt, f1_txt = optimize_sigma_brent(
            X_txt_tr, y_tr_dec, y_tr_conf, X_txt_val, val_pids, gt_df, is_fuzzy, "cosine"
        )
        model_txt = BrentMemKDM(is_fuzzy=is_fuzzy, metric="cosine").fit(X_txt_tr, y_tr_dec, y_tr_conf)
        p_txt = model_txt.predict_proba(X_txt_val, sigma=sig_txt)

        per_split_results.append({
            "split_idx": split_idx,
            "val_pids": val_pids,
            "sigmas": {"tab": sig_tab, "mri": sig_mri, "txt": sig_txt},
            "unimodal_f1": {"tab": f1_tab, "mri": f1_mri, "txt": f1_txt},
            "p_tab": p_tab,
            "p_mri": p_mri,
            "p_txt": p_txt,
        })

        log(
            f"  Split {split_idx+1:>2}/30 | "
            f"s*=(tab:{sig_tab:.3f}, mri:{sig_mri:.3f}, txt:{sig_txt:.3f}) | "
            f"F1=(t:{f1_tab:.3f}, m:{f1_mri:.3f}, x:{f1_txt:.3f})"
        )

    return {"per_split": per_split_results, "is_fuzzy": is_fuzzy}


def evaluate_late_fusion_grid(
    phase_result: Dict[str, Any],
    gt_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sweep weight simplex w in Delta^2 and decision threshold theta on out-of-fold probabilities."""
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

    fold_probs = phase_result["per_split"]
    log(f"--- [Phase 2] Late Fusion Grid Search ({len(weight_candidates)} weight vectors x {len(thresholds)} thresholds) ---")

    best_config = None
    best_macro_f1 = -1.0
    fusion_results = []
    t0 = time.time()

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

    elapsed = time.time() - t0
    log(f"  Late Fusion Complete ({elapsed:.1f}s) | Best Macro-F1: {best_macro_f1:.4f}")
    log(f"  ► Best Late Fusion (C5*): w=({best_config['w_tab']}, {best_config['w_mri']}, {best_config['w_txt']}) | theta={best_config['threshold']} | Macro-F1={best_config['mean_macro_f1']:.4f}")
    return best_config, fusion_results


def load_exp1_baseline_predictions() -> Dict[str, str]:
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
    phase_result: Dict[str, Any],
    gt_df: pd.DataFrame,
    c0_preds: Dict[str, str],
    c5_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute full metrics for C0-C5 across 30 splits and paired Wilcoxon tests."""
    conditions = [
        "C0_Baseline",
        "C1_Tabular_KDM",
        "C2_MRI_KDM",
        "C3_Text_KDM",
        "C4_EqualFusion",
        "C5_OptimalFusion",
    ]
    split_metrics = {cond: [] for cond in conditions}
    split_predictions = {cond: {} for cond in conditions}

    w_tab, w_mri, w_txt = c5_cfg["w_tab"], c5_cfg["w_mri"], c5_cfg["w_txt"]
    th_fused = c5_cfg["threshold"]

    for fp in phase_result["per_split"]:
        val_pids = fp["val_pids"]
        split_idx = fp["split_idx"]

        y_c0 = [c0_preds.get(pid, "yes") for pid in val_pids]
        y_c1 = ["yes" if p >= 0.50 else "no" for p in fp["p_tab"]]
        y_c2 = ["yes" if p >= 0.50 else "no" for p in fp["p_mri"]]
        y_c3 = ["yes" if p >= 0.50 else "no" for p in fp["p_txt"]]
        p_eq = (fp["p_tab"] + fp["p_mri"] + fp["p_txt"]) / 3.0
        y_c4 = ["yes" if p >= 0.50 else "no" for p in p_eq]
        p_opt = w_tab * fp["p_tab"] + w_mri * fp["p_mri"] + w_txt * fp["p_txt"]
        y_c5 = ["yes" if p >= th_fused else "no" for p in p_opt]

        pred_map = {
            "C0_Baseline": y_c0,
            "C1_Tabular_KDM": y_c1,
            "C2_MRI_KDM": y_c2,
            "C3_Text_KDM": y_c3,
            "C4_EqualFusion": y_c4,
            "C5_OptimalFusion": y_c5,
        }

        for cond, preds in pred_map.items():
            m = compute_metrics_for_predictions(gt_df, val_pids, preds)
            split_metrics[cond].append(m)
            split_predictions[cond][f"split_{split_idx}"] = {
                pid: pred for pid, pred in zip(val_pids, preds)
            }

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

    c0_f1s = [m["macro_f1"] for m in split_metrics["C0_Baseline"]]
    wilcoxon_results = {}
    for cond in conditions[1:]:
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
    t_start = time.time()
    log("=" * 100)
    log("CHIMERA Subtask 1.1: BrentMemKDM Multimodal Continuous-Bandwidth Replication (exp_5)")
    log("=" * 100)

    # 1. Ingest Data
    log("[1/6] Ingesting clinical multimodal representations & splits...")
    splits_df = load_mccv_splits(str(PROJECT_ROOT / "data" / "processed" / "task1" / "mccv_splits_30.csv"))
    gt_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "ground_truth_task1.csv")
    df_tab = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_tabular.csv")
    df_mri = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "mri_embeddings.csv")
    df_text = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "task1" / "clinical_texts.csv")
    c0_preds = load_exp1_baseline_predictions()

    log(f"      Loaded: {len(gt_df)} cases ({len(gt_df[gt_df['decision']=='yes'])} yes, "
        f"{len(gt_df[gt_df['decision']=='no'])} no) | Splits: 30 standardized MCCV folds")

    all_arm_results = {}
    for arm_name, is_fuzzy in [("hard", False), ("soft", True)]:
        log(f"\n{'='*100}\n[ARM: {arm_name.upper()}] Supervision Mode: {'Fuzzy Targets' if is_fuzzy else 'Hard Binary Labels'}\n{'='*100}")

        # 2. Phase 1: Per-Split Brent Bandwidth Optimization
        log(f"\n[2/6] [{arm_name}] Executing Phase 1: Per-Split Brent σ Optimization (MCCV-30)...")
        phase_res = run_brent_unimodal_phase(df_tab, df_mri, df_text, gt_df, splits_df, is_fuzzy)

        # 3. Phase 2: Late Fusion Simplex Sweep
        log(f"\n[3/6] [{arm_name}] Executing Phase 2: Late Fusion Simplex Sweep...")
        c5_best, grid_fusion = evaluate_late_fusion_grid(phase_res, gt_df)

        # 4. Phase 3: Benchmark & Wilcoxon
        log(f"\n[4/6] [{arm_name}] Executing Phase 3: Full Evaluation and Wilcoxon Testing...")
        agg_metrics, split_metrics, split_preds, wilcoxon_tests = run_full_evaluation_and_wilcoxon(
            phase_res, gt_df, c0_preds, c5_best
        )

        # 5. Persist Results
        log(f"\n[5/6] [{arm_name}] Serializing structured benchmark artifacts to disk...")
        arm_dir = RESULTS_DIR / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(arm_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
            json.dump(agg_metrics, f, indent=2)
        with open(arm_dir / "split_metrics.json", "w", encoding="utf-8") as f:
            json.dump(split_metrics, f, indent=2)
        with open(arm_dir / "split_predictions.json", "w", encoding="utf-8") as f:
            json.dump(split_preds, f, indent=2)
        with open(arm_dir / "wilcoxon_tests.json", "w", encoding="utf-8") as f:
            json.dump(wilcoxon_tests, f, indent=2)
        with open(arm_dir / "optimized_configs.json", "w", encoding="utf-8") as f:
            json.dump({
                "supervision_mode": arm_name,
                "C5_LateFusion_Optimal": c5_best,
                "per_split_sigmas": {
                    str(r["split_idx"]): r["sigmas"] for r in phase_res["per_split"]
                },
            }, f, indent=2)

        all_arm_results[arm_name] = {"aggregate": agg_metrics, "wilcoxon": wilcoxon_tests}

        # 6. Report Summary Table
        log(f"\n[6/6] [{arm_name}] Benchmark Summary Results (MCCV-30):")
        log("=" * 100)
        header = (f"{'Condition':<22} | {'Macro-F1 (Mean±Std)':<22} | {'F1-Yes':<8} | "
                  f"{'F1-No':<8} | {'S_gate':<8} | {'Sens/Spec':<14}")
        log(header)
        log("-" * 100)
        for cond in agg_metrics.keys():
            m = agg_metrics[cond]
            row = (f"{cond:<22} | {m['macro_f1']['mean']:.4f} ± {m['macro_f1']['std']:.4f}   | "
                   f"{m['f1_yes']['mean']:.4f}  | {m['f1_no']['mean']:.4f}  | "
                   f"{m['s_gate']['mean']:.4f} | {m['sensitivity']['mean']:.2f}/{m['specificity']['mean']:.2f}")
            log(row)
        log("=" * 100)
        log("\nPaired Wilcoxon Signed-Rank Hypothesis Tests vs C0_Baseline:")
        for comp, res in wilcoxon_tests.items():
            sig = "SIG (p < 0.05)" if res["is_significant_alpha_0_05"] else "NS (p >= 0.05)"
            log(f"  • {comp:<28}: diff = {res['mean_difference']:+.4f} | W = {res['statistic']:<6.1f} | p = {res['p_value']:.4e} [{sig}]")

    total_min = (time.time() - t_start) / 60.0
    log(f"\nExperiment exp_5 successfully completed in {total_min:.1f} minutes.")


if __name__ == "__main__":
    main()
