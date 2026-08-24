"""
Benchmark Runner for exp_9: Joint Feature-Attribution Fusion (Raw Clinical Features + Coalesced SHAP 23D).
Evaluates Subtask 1.3 across 30 MCCV Splits with Real-Time Terminal Progress.
"""

from __future__ import annotations
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, wilcoxon

from models import (
    CLINICAL_VARIABLES,
    ORDINAL_WEIGHT_MAP,
    TabularPreprocessor,
    FuzzyKNNRegressor,
    ShapAttributionExtractor,
    JointFeatureAugmenter,
    JointRelevanceMetaPredictor,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT_DIR / "experiments" / "exp_9"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    {"name": "C1_SHAP_Only_13D", "space": "shap_13d"},
    {"name": "C2_Raw_Only_10D", "space": "raw_10d"},
    {"name": "C3_Joint_Forest_23D", "space": "joint_23d"},
    {"name": "C4_Joint_ExtraTrees_23D", "space": "joint_23d"},
    {"name": "C5_Targeted_Hybrid_23D", "space": "joint_23d"},
]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tab_path = ROOT_DIR / "data" / "processed" / "task1" / "clinical_tabular.csv"
    gt_path = ROOT_DIR / "data" / "processed" / "task1" / "ground_truth_task1.csv"
    splits_path = ROOT_DIR / "data" / "processed" / "task1" / "mccv_splits_30.csv"

    tab_df = pd.read_csv(tab_path)
    gt_df = pd.read_csv(gt_path)
    splits_df = pd.read_csv(splits_path)

    merged = pd.merge(tab_df, gt_df, on=["pid", "case_id"], suffixes=("", "_gt"))
    return merged, splits_df


def map_fuzzy_decision_target(decision: str, confidence: str) -> float:
    d = str(decision).strip().lower()
    c = str(confidence).strip().lower()
    if d == "yes":
        if c == "clear":
            return 1.00
        elif c == "borderline":
            return 0.75
        else:
            return 0.50
    else:
        if c == "uncertain":
            return 0.50
        elif c == "borderline":
            return 0.25
        else:
            return 0.00


def compute_metrics_for_condition(
    preds_dict: Dict[str, List[str]],
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    macro_f1s = []
    rhos = {}
    mae_ranks = []
    decisive_set_f1s = []

    for var_name in CLINICAL_VARIABLES:
        gt_col = f"weight_{var_name}"
        y_true_str = val_df[gt_col].fillna("not_used").astype(str).str.lower().tolist()
        y_pred_str = preds_dict[var_name]

        # Macro F1 per variable
        f1_v = float(
            f1_score(
                y_true_str,
                y_pred_str,
                labels=["not_used", "noted", "important", "decisive"],
                average="macro",
                zero_division=0,
            )
        )
        macro_f1s.append(f1_v)

        # Spearman rank correlation
        y_true_int = np.array([ORDINAL_WEIGHT_MAP.get(s, 0) for s in y_true_str])
        y_pred_int = np.array([ORDINAL_WEIGHT_MAP.get(s, 0) for s in y_pred_str])

        # Compute ordinal rank MAE for S_weights
        diff = np.abs(y_true_int - y_pred_int) / 3.0
        mae_ranks.extend(diff.tolist())

        if np.std(y_true_int) > 1e-6 and np.std(y_pred_int) > 1e-6:
            r_val, p_val = spearmanr(y_true_int, y_pred_int)
            rhos[var_name] = float(r_val) if not np.isnan(r_val) else 0.0
        else:
            rhos[var_name] = 0.0

    mean_macro_f1 = float(np.mean(macro_f1s))
    s_weights = float(1.0 - np.mean(mae_ranks))

    # Decisive factors Set-F1 (for {important, decisive})
    for i in range(len(val_df)):
        true_decisive = set()
        pred_decisive = set()
        for var_name in CLINICAL_VARIABLES:
            gt_val = str(val_df[f"weight_{var_name}"].iloc[i]).lower()
            if gt_val in ["important", "decisive"]:
                true_decisive.add(var_name)
            if preds_dict[var_name][i] in ["important", "decisive"]:
                pred_decisive.add(var_name)

        inter = len(true_decisive.intersection(pred_decisive))
        prec = inter / max(len(pred_decisive), 1) if pred_decisive else 1.0
        rec = inter / max(len(true_decisive), 1) if true_decisive else 1.0
        f1_case = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        decisive_set_f1s.append(f1_case)

    mean_decisive_f1 = float(np.mean(decisive_set_f1s))

    return {
        "mean_macro_f1": mean_macro_f1,
        "s_weights": s_weights,
        "decisive_factors_f1": mean_decisive_f1,
        "rhos": rhos,
        "macro_f1_per_var": {v: float(f1) for v, f1 in zip(CLINICAL_VARIABLES, macro_f1s)},
    }


def record_git_version():
    version_file = RESULTS_DIR / "version.txt"
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        version_text = f"Commit: {commit_hash}\nTimestamp: {datetime.now().isoformat()}"
    except Exception:
        version_text = f"exp_9_run_{int(time.time())}\nTimestamp: {datetime.now().isoformat()}"
    version_file.write_text(version_text)


def main():
    print("\n" + "=" * 95)
    print("  EXPERIMENT 9: Joint Feature-Attribution Fusion (Raw Clinical Features + Coalesced SHAP 23D)")
    print("  Subtask 1.3: Clinical Relevance Weighting | Standardized MCCV-30 (N=73 train / 18 val)")
    print("=" * 95)

    record_git_version()
    df, splits_df = load_data()
    n_splits = 30

    results_by_condition = {c["name"]: [] for c in CONDITIONS}
    split_predictions_log = {c["name"]: {} for c in CONDITIONS}

    print(f"\n[INFO] Loaded cohort: {len(df)} labeled cases | Evaluating {n_splits} MCCV splits...")
    print("-" * 95)
    header = f"{'Split':^7} | {'C1_SHAP13D':^11} | {'C2_Raw10D':^11} | {'C3_Joint23D':^11} | {'C4_ExTree23':^11} | {'C5_Hybrid':^11} | {'rho(PSA)':^9} | {'rho(PIR)':^9}"
    print(header)
    print("-" * 95)

    for split_idx in range(n_splits):
        col_name = f"split_{split_idx}"
        split_map = dict(zip(splits_df["pid"], splits_df[col_name]))

        train_mask = df["pid"].map(lambda p: split_map.get(p) == "train")
        val_mask = df["pid"].map(lambda p: split_map.get(p) == "val")

        train_sub = df[train_mask].copy().reset_index(drop=True)
        val_sub = df[val_mask].copy().reset_index(drop=True)

        # 1. Fit Preprocessor
        preprocessor = TabularPreprocessor().fit(train_sub)
        X_train_knn = preprocessor.transform_knn(train_sub)
        X_val_knn = preprocessor.transform_knn(val_sub)

        X_train_raw10 = preprocessor.transform_raw_10d(train_sub)
        X_val_raw10 = preprocessor.transform_raw_10d(val_sub)

        # 2. Fit Base Tabular Champion (C1* Fuzzy-KNN)
        y_train_fuzzy = np.array([
            map_fuzzy_decision_target(r["decision"], r["confidence"]) for _, r in train_sub.iterrows()
        ], dtype=np.float32)
        base_model = FuzzyKNNRegressor(k=5).fit(X_train_knn, y_train_fuzzy)

        # 3. Extract 10D SHAP & 13D Coalesced
        shap_extractor = ShapAttributionExtractor(base_model, X_train_knn, n_background=40)
        shap_train_10d = shap_extractor.extract_shap_10d(X_train_knn)
        shap_val_10d = shap_extractor.extract_shap_10d(X_val_knn)

        shap_train_13d = JointFeatureAugmenter.augment_coalesced_13d(shap_train_10d)
        shap_val_13d = JointFeatureAugmenter.augment_coalesced_13d(shap_val_10d)

        # 4. Construct 23D Joint Representation
        z_train_23d = JointFeatureAugmenter.build_joint_23d(X_train_raw10, shap_train_13d)
        z_val_23d = JointFeatureAugmenter.build_joint_23d(X_val_raw10, shap_val_13d)

        split_results_map = {}

        # 5. Train & Evaluate each Condition
        for cond in CONDITIONS:
            c_name = cond["name"]
            c_space = cond["space"]

            if c_space == "shap_13d":
                X_tr = shap_train_13d
                X_va = shap_val_13d
            elif c_space == "raw_10d":
                X_tr = X_train_raw10
                X_va = X_val_raw10
            else:  # joint_23d
                X_tr = z_train_23d
                X_va = z_val_23d

            meta_clf = JointRelevanceMetaPredictor(condition_name=c_name, random_state=42 + split_idx).fit(
                X_tr, train_sub
            )
            preds = meta_clf.predict(X_va)
            metrics = compute_metrics_for_condition(preds, val_sub)

            results_by_condition[c_name].append(metrics)
            split_results_map[c_name] = metrics

            # Log predictions per case
            split_key = f"split_{split_idx}"
            if split_key not in split_predictions_log[c_name]:
                split_predictions_log[c_name][split_key] = {}
            for case_i, case_id in enumerate(val_sub["case_id"]):
                split_predictions_log[c_name][split_key][str(case_id)] = {
                    var_name: preds[var_name][case_i] for var_name in CLINICAL_VARIABLES
                }

        c1_f1 = split_results_map["C1_SHAP_Only_13D"]["mean_macro_f1"]
        c2_f1 = split_results_map["C2_Raw_Only_10D"]["mean_macro_f1"]
        c3_f1 = split_results_map["C3_Joint_Forest_23D"]["mean_macro_f1"]
        c4_f1 = split_results_map["C4_Joint_ExtraTrees_23D"]["mean_macro_f1"]
        c5_f1 = split_results_map["C5_Targeted_Hybrid_23D"]["mean_macro_f1"]
        c4_rho_psa = split_results_map["C4_Joint_ExtraTrees_23D"]["rhos"].get("psa", 0.0)
        c4_rho_pir = split_results_map["C4_Joint_ExtraTrees_23D"]["rhos"].get("pirads", 0.0)

        print(
            f"Fold {split_idx+1:02d} |   {c1_f1:.4f}    |   {c2_f1:.4f}   |   {c3_f1:.4f}   |   {c4_f1:.4f}   |   {c5_f1:.4f}   |  {c4_rho_psa:6.3f}   |  {c4_rho_pir:6.3f}"
        )

    print("-" * 95)
    print("\n" + "=" * 95)
    print("  FINAL AGGREGATE RESULTS ACROSS 30 MCCV SPLITS (MEAN ± STD)")
    print("=" * 95)

    aggregate_summary = {}
    for c_name, split_list in results_by_condition.items():
        f1s = [m["mean_macro_f1"] for m in split_list]
        s_weights = [m["s_weights"] for m in split_list]
        decisives = [m["decisive_factors_f1"] for m in split_list]
        
        rho_psa = [m["rhos"].get("psa", 0.0) for m in split_list]
        rho_pir = [m["rhos"].get("pirads", 0.0) for m in split_list]
        rho_dre = [m["rhos"].get("dre", 0.0) for m in split_list]
        rho_bx = [m["rhos"].get("bx", 0.0) for m in split_list]
        rho_psad = [m["rhos"].get("psad", 0.0) for m in split_list]

        aggregate_summary[c_name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "s_weights_mean": float(np.mean(s_weights)),
            "s_weights_std": float(np.std(s_weights)),
            "decisive_set_f1_mean": float(np.mean(decisives)),
            "decisive_set_f1_std": float(np.std(decisives)),
            "rho_psa_mean": float(np.mean(rho_psa)),
            "rho_pirads_mean": float(np.mean(rho_pir)),
            "rho_dre_mean": float(np.mean(rho_dre)),
            "rho_bx_mean": float(np.mean(rho_bx)),
            "rho_psad_mean": float(np.mean(rho_psad)),
            "per_var_f1_mean": {
                v: float(np.mean([m["macro_f1_per_var"][v] for m in split_list])) for v in CLINICAL_VARIABLES
            },
        }

    header_summary = f"{'Condition':<28} | {'Mean Macro-F1':^16} | {'S_weights':^15} | {'Decisive Set-F1':^16} | {'rho(PSA)':^9} | {'rho(PIRADS)':^11}"
    print(header_summary)
    print("-" * 95)
    for c_name, stats in aggregate_summary.items():
        print(
            f"{c_name:<28} | {stats['mean_macro_f1']:.4f} ± {stats['std_macro_f1']:.4f}  | {stats['s_weights_mean']:.4f} ± {stats['s_weights_std']:.4f} | {stats['decisive_set_f1_mean']:.4f} ± {stats['decisive_set_f1_std']:.4f} |  {stats['rho_psa_mean']:6.3f}   |   {stats['rho_pirads_mean']:6.3f}"
        )
    print("-" * 95)

    # 6. Statistical Significance vs C1 (SHAP-Only Control)
    c1_f1_scores = [m["mean_macro_f1"] for m in results_by_condition["C1_SHAP_Only_13D"]]
    c2_f1_scores = [m["mean_macro_f1"] for m in results_by_condition["C2_Raw_Only_10D"]]
    wilcoxon_tests = {}

    for c_name in ["C2_Raw_Only_10D", "C3_Joint_Forest_23D", "C4_Joint_ExtraTrees_23D", "C5_Targeted_Hybrid_23D"]:
        cand_scores = [m["mean_macro_f1"] for m in results_by_condition[c_name]]
        stat, pval = wilcoxon(cand_scores, c1_f1_scores)
        wilcoxon_tests[f"{c_name}_vs_C1_SHAP"] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "delta_f1": float(np.mean(cand_scores) - np.mean(c1_f1_scores)),
        }

    # Also test C4 vs C2 (Raw Only)
    c4_scores = [m["mean_macro_f1"] for m in results_by_condition["C4_Joint_ExtraTrees_23D"]]
    stat, pval = wilcoxon(c4_scores, c2_f1_scores)
    wilcoxon_tests["C4_Joint_ExtraTrees_vs_C2_Raw"] = {
        "statistic": float(stat),
        "p_value": float(pval),
        "delta_f1": float(np.mean(c4_scores) - np.mean(c2_f1_scores)),
    }

    # 7. Write Results Artifacts
    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    with open(RESULTS_DIR / "split_metrics.json", "w") as f:
        json.dump(results_by_condition, f, indent=2)

    with open(RESULTS_DIR / "split_predictions.json", "w") as f:
        json.dump(split_predictions_log, f, indent=2)

    with open(RESULTS_DIR / "wilcoxon_tests.json", "w") as f:
        json.dump(wilcoxon_tests, f, indent=2)

    print(f"\n[SUCCESS] Artifacts written to {RESULTS_DIR}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
