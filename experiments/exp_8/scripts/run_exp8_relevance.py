"""
Benchmark Runner for exp_8: Multivariate Balanced Decision Trees over Full 10D SHAP & Coalesced Feature Spaces.
Evaluates Subtask 1.3 across 30 MCCV Splits with Real-Time Terminal Progress.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score

from models import (
    CLINICAL_VARIABLES_10,
    FUZZY_TARGET_MAP,
    WEIGHT_INT_TO_STR,
    WEIGHT_STR_TO_INT,
    CoalescedAugmenter,
    FuzzyKNNRegressor,
    RelevanceMetaPredictor,
    ShapAttributionExtractor,
    TabularPreprocessor,
)

# Constantes de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task1"
RESULTS_DIR = BASE_DIR / "results"

CONDITIONS = [
    {"name": "C1_10D_Raw_Tree", "features": "10d", "class_weight": None, "model": "tree", "depth": 3},
    {"name": "C2_10D_Balanced_Tree", "features": "10d", "class_weight": "balanced", "model": "tree", "depth": 3},
    {"name": "C3_Coalesced_Balanced_Tree", "features": "13d", "class_weight": "balanced", "model": "tree", "depth": 3},
    {"name": "C4_Coalesced_Balanced_Forest", "features": "13d", "class_weight": "balanced", "model": "forest", "depth": 3},
    {"name": "C5_Hybrid_Regularized", "features": "13d", "class_weight": "balanced", "model": "hybrid", "depth": 3},
]


def load_data():
    tab_df = pd.read_csv(DATA_DIR / "clinical_tabular.csv")
    gt_df = pd.read_csv(DATA_DIR / "ground_truth_task1.csv")
    splits_df = pd.read_csv(DATA_DIR / "mccv_splits_30.csv")

    # Merge on case_id
    merged = pd.merge(tab_df, gt_df, on=["case_id", "pid"], suffixes=("", "_gt"))
    # Filter only labeled cases
    merged = merged[merged["is_labeled"] == 1].reset_index(drop=True)

    # Compute fuzzy target for base C1* model
    fuzzy_targets = []
    for _, row in merged.iterrows():
        dec = str(row["decision"]).strip().lower()
        conf = str(row["confidence"]).strip().lower()
        fuzzy_targets.append(FUZZY_TARGET_MAP.get((dec, conf), 0.5))
    merged["target_fuzzy_risk"] = fuzzy_targets

    return merged, splits_df


def compute_metrics_for_predictions(
    y_true_weights: pd.DataFrame,
    preds_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    n_samples = len(y_true_weights)
    var_macro_f1s = {}
    var_rhos = {}
    var_pvals = {}

    all_labels = ["not_used", "noted", "important", "decisive"]
    mae_list = []
    tp_imp_dec = 0
    fp_imp_dec = 0
    fn_imp_dec = 0

    for var in CLINICAL_VARIABLES_10:
        target_col = f"weight_{var}"
        true_str = y_true_weights[target_col].astype(str).str.lower().values
        pred_str = preds_dict[var]

        # 1. Macro-F1 per variable
        mf1 = f1_score(true_str, pred_str, labels=all_labels, average="macro", zero_division=0)
        var_macro_f1s[var] = float(mf1)

        # 2. Spearman Correlation
        true_int = np.array([WEIGHT_STR_TO_INT.get(s, 1) for s in true_str])
        pred_int = np.array([WEIGHT_STR_TO_INT.get(s, 1) for s in pred_str])

        if len(np.unique(true_int)) > 1 and len(np.unique(pred_int)) > 1:
            rho, pval = spearmanr(true_int, pred_int)
            var_rhos[var] = float(rho) if not np.isnan(rho) else 0.0
            var_pvals[var] = float(pval) if not np.isnan(pval) else 1.0
        else:
            var_rhos[var] = 0.0
            var_pvals[var] = 1.0

        # 3. Reasoning Weight MAE
        for ti, pi in zip(true_int, pred_int):
            mae_list.append(abs(ti - pi) / 3.0)

        # 4. Decisive / Important Set-F1 accumulation
        for ti, pi in zip(true_int, pred_int):
            is_true_imp = ti in [2, 3]
            is_pred_imp = pi in [2, 3]
            if is_true_imp and is_pred_imp:
                tp_imp_dec += 1
            elif not is_true_imp and is_pred_imp:
                fp_imp_dec += 1
            elif is_true_imp and not is_pred_imp:
                fn_imp_dec += 1

    mean_macro_f1 = float(np.mean(list(var_macro_f1s.values())))
    s_weights = float(1.0 - np.mean(mae_list)) if mae_list else 0.0

    prec_imp = tp_imp_dec / (tp_imp_dec + fp_imp_dec) if (tp_imp_dec + fp_imp_dec) > 0 else 0.0
    rec_imp = tp_imp_dec / (tp_imp_dec + fn_imp_dec) if (tp_imp_dec + fn_imp_dec) > 0 else 0.0
    decisive_set_f1 = float(2 * prec_imp * rec_imp / (prec_imp + rec_imp)) if (prec_imp + rec_imp) > 0 else 0.0

    return {
        "mean_macro_f1": mean_macro_f1,
        "s_weights": s_weights,
        "decisive_set_f1": decisive_set_f1,
        "var_macro_f1s": var_macro_f1s,
        "var_rhos": var_rhos,
        "var_pvals": var_pvals,
    }


def record_git_version():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    version_file = RESULTS_DIR / "version.txt"
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True).strip().splitlines()[0]
        version_text = f"{commit_sha}  {commit_msg}\nTimestamp: {datetime.now().isoformat()}"
    except Exception:
        version_text = f"exp_8_run_{int(time.time())}\nTimestamp: {datetime.now().isoformat()}"
    version_file.write_text(version_text)


def main():
    print("\n" + "=" * 90)
    print("  EXPERIMENT 8: Multivariate Decision Trees over 10D Full SHAP & Coalesced Spaces")
    print("  Subtask 1.3: Clinical Relevance Weighting | MCCV-30 (N=73 train / 18 val)")
    print("=" * 90)

    record_git_version()
    df, splits_df = load_data()
    n_splits = 30

    # Contenedores de resultados
    results_by_condition = {c["name"]: [] for c in CONDITIONS}
    split_predictions_log = {c["name"]: {} for c in CONDITIONS}

    print(f"\n[INFO] Loaded cohort: {len(df)} labeled cases | Evaluating {n_splits} MCCV splits...")
    print("-" * 90)
    header = f"{'Split':^7} | {'C1_Raw F1':^11} | {'C2_Bal F1':^11} | {'C3_Coal F1':^11} | {'C4_Forest':^11} | {'C5_Hybrid':^11} | {'rho(PSA)':^9} | {'rho(PIR)':^9}"
    print(header)
    print("-" * 90)

    for split_idx in range(n_splits):
        col_name = f"split_{split_idx}"
        split_map = dict(zip(splits_df["pid"], splits_df[col_name]))

        train_mask = df["pid"].map(lambda p: split_map.get(p) == "train")
        val_mask = df["pid"].map(lambda p: split_map.get(p) == "val")

        train_sub = df[train_mask].copy().reset_index(drop=True)
        val_sub = df[val_mask].copy().reset_index(drop=True)

        # 1. Preprocesar datos tabulares sin fuga
        preprocessor = TabularPreprocessor()
        X_train_tab = preprocessor.fit_transform(train_sub)
        X_val_tab = preprocessor.transform(val_sub)

        # 2. Entrenar modelo base C1* FuzzyKNN
        y_train_fuzzy = train_sub["target_fuzzy_risk"].values
        base_model = FuzzyKNNRegressor(k=5, metric="euclidean", weights="distance")
        base_model.fit(X_train_tab, y_train_fuzzy)

        # 3. Extracción de SHAP 10D por split
        shap_extractor = ShapAttributionExtractor(base_model, preprocessor)
        shap_extractor.fit_explainer(X_train_tab)
        phi_train_10d = shap_extractor.extract_10d_shap(X_train_tab, train_sub)
        phi_val_10d = shap_extractor.extract_10d_shap(X_val_tab, val_sub)

        # 4. Aumentación coalescente 13D
        phi_train_13d = CoalescedAugmenter.augment(phi_10d=phi_train_10d)
        phi_val_13d = CoalescedAugmenter.augment(phi_10d=phi_val_10d)

        # Ground truth de pesos
        y_train_weights = train_sub[[f"weight_{v}" for v in CLINICAL_VARIABLES_10]]
        y_val_weights = val_sub[[f"weight_{v}" for v in CLINICAL_VARIABLES_10]]

        split_line_metrics = {}

        for cond in CONDITIONS:
            c_name = cond["name"]
            is_13d = cond["features"] == "13d"
            X_tr = phi_train_13d if is_13d else phi_train_10d
            X_va = phi_val_13d if is_13d else phi_val_10d

            meta_predictor = RelevanceMetaPredictor(
                condition=c_name,
                max_depth=cond["depth"],
                class_weight=cond["class_weight"],
                random_state=42 + split_idx,
            )
            meta_predictor.fit(X_tr, y_train_weights, raw_train_df=train_sub)
            preds_dict = meta_predictor.predict(X_va, raw_test_df=val_sub)

            metrics = compute_metrics_for_predictions(y_val_weights, preds_dict)
            results_by_condition[c_name].append(metrics)
            split_line_metrics[c_name] = metrics

            # Log predictions
            split_predictions_log[c_name][f"split_{split_idx}"] = {
                case_id: {var: preds_dict[var][i] for var in CLINICAL_VARIABLES_10}
                for i, case_id in enumerate(val_sub["case_id"].values)
            }

        # Imprimir fila en tiempo real
        c1_f1 = split_line_metrics["C1_10D_Raw_Tree"]["mean_macro_f1"]
        c2_f1 = split_line_metrics["C2_10D_Balanced_Tree"]["mean_macro_f1"]
        c3_f1 = split_line_metrics["C3_Coalesced_Balanced_Tree"]["mean_macro_f1"]
        c4_f1 = split_line_metrics["C4_Coalesced_Balanced_Forest"]["mean_macro_f1"]
        c5_f1 = split_line_metrics["C5_Hybrid_Regularized"]["mean_macro_f1"]
        psa_rho = split_line_metrics["C3_Coalesced_Balanced_Tree"]["var_rhos"]["psa"]
        pirads_rho = split_line_metrics["C3_Coalesced_Balanced_Tree"]["var_rhos"]["pirads"]

        print(
            f"Fold {split_idx+1:02d} | {c1_f1:^11.4f} | {c2_f1:^11.4f} | {c3_f1:^11.4f} | {c4_f1:^11.4f} | {c5_f1:^11.4f} | {psa_rho:^9.3f} | {pirads_rho:^9.3f}"
        )

    print("-" * 90)

    # 5. Agregación y Pruebas Estadísticas
    aggregate_summary = {}
    wilcoxon_summary = {}

    print("\n" + "=" * 90)
    print("  FINAL AGGREGATE RESULTS ACROSS 30 MCCV SPLITS (Mean ± Std)")
    print("=" * 90)
    res_header = f"{'Condition':<28} | {'Mean Macro-F1':<16} | {'S_weights':<14} | {'Decisive Set-F1':<16} | {'rho(PSA)':<11} | {'rho(PIRADS)':<11}"
    print(res_header)
    print("-" * 90)

    for cond in CONDITIONS:
        c_name = cond["name"]
        m_list = results_by_condition[c_name]

        f1_vals = [m["mean_macro_f1"] for m in m_list]
        sw_vals = [m["s_weights"] for m in m_list]
        dec_vals = [m["decisive_set_f1"] for m in m_list]
        psa_rhos = [m["var_rhos"]["psa"] for m in m_list]
        pir_rhos = [m["var_rhos"]["pirads"] for m in m_list]

        mean_f1, std_f1 = float(np.mean(f1_vals)), float(np.std(f1_vals))
        mean_sw, std_sw = float(np.mean(sw_vals)), float(np.std(sw_vals))
        mean_dec, std_dec = float(np.mean(dec_vals)), float(np.std(dec_vals))
        mean_psa_rho = float(np.mean(psa_rhos))
        mean_pir_rho = float(np.mean(pir_rhos))

        # Macro-F1 por variable
        var_f1_means = {}
        var_rho_means = {}
        for var in CLINICAL_VARIABLES_10:
            var_f1_means[var] = float(np.mean([m["var_macro_f1s"][var] for m in m_list]))
            var_rho_means[var] = float(np.mean([m["var_rhos"][var] for m in m_list]))

        aggregate_summary[c_name] = {
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "mean_s_weights": mean_sw,
            "std_s_weights": std_sw,
            "mean_decisive_set_f1": mean_dec,
            "std_decisive_set_f1": std_dec,
            "mean_psa_rho": mean_psa_rho,
            "mean_pirads_rho": mean_pir_rho,
            "var_macro_f1_means": var_f1_means,
            "var_rho_means": var_rho_means,
        }

        print(
            f"{c_name:<28} | {mean_f1:.4f} ± {std_f1:.4f}  | {mean_sw:.4f} ± {std_sw:.4f} | {mean_dec:.4f} ± {std_dec:.4f}  | {mean_psa_rho:^11.3f} | {mean_pir_rho:^11.3f}"
        )

    print("-" * 90)

    # Pruebas de Wilcoxon vs C1_10D_Raw_Tree
    base_f1s = [m["mean_macro_f1"] for m in results_by_condition["C1_10D_Raw_Tree"]]
    for cond in CONDITIONS:
        c_name = cond["name"]
        if c_name == "C1_10D_Raw_Tree":
            continue
        cand_f1s = [m["mean_macro_f1"] for m in results_by_condition[c_name]]
        diff = np.array(cand_f1s) - np.array(base_f1s)
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = wilcoxon(cand_f1s, base_f1s, alternative="two-sided")
        wilcoxon_summary[f"{c_name}_vs_C1"] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "delta_f1": float(np.mean(cand_f1s) - np.mean(base_f1s)),
        }

    # 6. Guardar artefactos
    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    with open(RESULTS_DIR / "split_metrics.json", "w") as f:
        json.dump(results_by_condition, f, indent=2)

    with open(RESULTS_DIR / "split_predictions.json", "w") as f:
        json.dump(split_predictions_log, f, indent=2)

    with open(RESULTS_DIR / "wilcoxon_tests.json", "w") as f:
        json.dump(wilcoxon_summary, f, indent=2)

    print(f"\n[SUCCESS] Artifacts written to {RESULTS_DIR.resolve()}/")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
