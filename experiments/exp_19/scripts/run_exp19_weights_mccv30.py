#!/usr/bin/env python3
"""
Systematic Hyperparameter Search for Task 2 Variable Weights Meta-Models (exp_19).
Protocol: Multi-subtask MCCV-30 (data/processed/task2/mccv_splits_30.csv).
Metrics: chimera_metrics.task1.reasoning.compute_variable_weights_score & compute_important_decisive_f1.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

# Workspace and chimera_metrics paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train_release" / "task2"
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_19"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHIMERA_METRICS_PATH = PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"
if str(CHIMERA_METRICS_PATH) not in sys.path:
    sys.path.insert(0, str(CHIMERA_METRICS_PATH))

from chimera_metrics.task1.reasoning import (  # noqa: E402
    compute_important_decisive_f1,
    compute_variable_weights_score,
)

VARIABLES_11 = [
    "age",
    "bx_gl_prim",
    "bx_gl_sec",
    "bx_isup",
    "comorbidity",
    "cspca",
    "ct",
    "fh",
    "pirads",
    "psa",
    "psad",
]
WEIGHT_ORDER = ["not_used", "noted", "important", "decisive"]
IDX_W = {c: i for i, c in enumerate(WEIGHT_ORDER)}

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
        self.scaler: MinMaxScaler | None = None

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                med = df[col].median()
                self.impute_medians[col] = 0.0 if pd.isna(med) else float(med)
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                st = df[col].mode()
                self.cat_modes[col] = str(st.iloc[0]) if not st.empty else "unknown"
            else:
                self.cat_modes[col] = "unknown"
        X_train = self._build_features(df, fit_cats=True)
        self.scaler = MinMaxScaler().fit(X_train)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df, fit_cats=False)
        assert self.scaler is not None
        return np.clip(self.scaler.transform(X), 0.0, 1.0)

    def _build_features(self, df: pd.DataFrame, fit_cats: bool) -> np.ndarray:
        num_parts: List[np.ndarray] = []
        for col in T2_NUMERIC_COLS:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                num_parts.append(s.fillna(self.impute_medians.get(col, 0.0)).values)
            else:
                num_parts.append(np.zeros(len(df)))
        cat_parts: List[pd.DataFrame] = []
        for col in T2_CATEGORICAL_COLS:
            if col in df.columns:
                s = df[col].fillna(self.cat_modes.get(col, "unknown")).astype(str)
            else:
                s = pd.Series(["unknown"] * len(df))
            dummies = pd.get_dummies(s, prefix=col, drop_first=False)
            if fit_cats:
                self.cat_columns_[col] = dummies.columns.tolist()
            else:
                dummies = dummies.reindex(columns=self.cat_columns_.get(col, []), fill_value=0.0)
            cat_parts.append(dummies.astype(float))
        X = np.column_stack(num_parts)
        if cat_parts:
            X = np.hstack([X, pd.concat(cat_parts, axis=1).values])
        return X


class MRIPreprocessor:
    def __init__(self, n_components: int | None = 16):
        self.n_components = n_components
        self.pca: PCA | None = None
        self.mean_vector: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MRIPreprocessor":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.mean_vector = np.mean(X_arr, axis=0)
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0):
                X_arr[i] = self.mean_vector
        if self.n_components is not None:
            max_comp = min(X_arr.shape[0] - 1, X_arr.shape[1], self.n_components)
            self.pca = PCA(n_components=max(1, max_comp), random_state=42).fit(X_arr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        for i in range(len(X_arr)):
            if np.all(X_arr[i] == 0.0) and self.mean_vector is not None:
                X_arr[i] = self.mean_vector
        if self.pca is not None:
            return self.pca.transform(X_arr)
        return X_arr


class TextPreprocessor:
    def __init__(self, max_features: int | None = 50):
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: List[str]) -> "TextPreprocessor":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b\w+(?:-\w+)*\b",
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        assert self.vectorizer is not None
        return self.vectorizer.transform(texts).toarray()


def load_ground_truth_weights(pids: List[str]) -> Dict[str, Dict[str, str]]:
    vw_map: Dict[str, Dict[str, str]] = {}
    for pid in pids:
        reasoning_path = RAW_DIR / pid / "prostate-treatment-decision-reasoning.json"
        if reasoning_path.exists():
            with open(reasoning_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            vw_map[pid] = d.get("variable_weights", {})
        else:
            vw_map[pid] = {}
    return vw_map


def main() -> None:
    print("=" * 80)
    print("RUNNING TASK 2 SUBTASK 2.3 (VARIABLE RELEVANCE WEIGHTS) MCCV-30 BENCHMARK (exp_19)")
    print("=" * 80)

    # Load canonical datasets
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_splits = pd.read_csv(DATA_DIR / "mccv_splits_30.csv")

    pids = df_gt["pid"].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    vw_gt = load_ground_truth_weights(pids)

    mri_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    mri_arr_full = df_mri[mri_cols].values.astype(float)
    texts_full = df_text["full_encounter_narrative"].fillna("").tolist()

    # Pre-cache fold representations (Joint 98D space: Tabular 32D + MRI PCA 16D + Text TFIDF 50D)
    fold_cache = []
    print("\nExtracting fold representations across 30 MCCV splits...")
    for split_id in range(30):
        col_name = f"split_{split_id}"
        train_pids = df_splits[df_splits[col_name] == "train"]["pid"].tolist()
        val_pids = df_splits[df_splits[col_name] == "val"]["pid"].tolist()

        train_indices = [pid_to_idx[p] for p in train_pids]
        val_indices = [pid_to_idx[p] for p in val_pids]

        # Tabular
        tab_prep = Task2TabularPreprocessor().fit(df_tab.iloc[train_indices])
        X_tab_tr = tab_prep.transform(df_tab.iloc[train_indices])
        X_tab_val = tab_prep.transform(df_tab.iloc[val_indices])

        # MRI
        mri_prep = MRIPreprocessor(n_components=16).fit(mri_arr_full[train_indices])
        X_mri_tr = mri_prep.transform(mri_arr_full[train_indices])
        X_mri_val = mri_prep.transform(mri_arr_full[val_indices])

        # Text
        txt_prep = TextPreprocessor(max_features=50).fit([texts_full[i] for i in train_indices])
        X_txt_tr = txt_prep.transform([texts_full[i] for i in train_indices])
        X_txt_val = txt_prep.transform([texts_full[i] for i in val_indices])

        # Joint 98D
        X_joint_tr = np.hstack([X_tab_tr, X_mri_tr, X_txt_tr])
        X_joint_val = np.hstack([X_tab_val, X_mri_val, X_txt_val])

        fold_cache.append({
            "split_id": split_id,
            "train_pids": train_pids,
            "val_pids": val_pids,
            "X_tr": X_joint_tr,
            "X_val": X_joint_val,
        })

    # -------------------------------------------------------------------------
    # Phase 1: Per-Variable Hyperparameter Optimization Grid Search
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Per-Variable Hyperparameter Grid Search ---")
    algorithms = {
        "rf": RandomForestClassifier,
        "et": ExtraTreesClassifier,
    }
    n_estimators_list = [100]
    criterions = ["gini"]
    class_weights = ["balanced", None]
    n_jobs_cfg = -1

    param_combinations = list(itertools.product(
        ["rf", "et"],
        n_estimators_list,
        criterions,
        class_weights,
    ))

    best_hyperparams_per_var: Dict[str, Dict[str, Any]] = {}

    for var in VARIABLES_11:
        best_score = -1.0
        best_cfg = None

        for alg_name, n_est, crit, cw in param_combinations:
            f1_scores = []
            clf_cls = algorithms[alg_name]
            for fc in fold_cache:
                X_tr, X_val = fc["X_tr"], fc["X_val"]
                train_pids, val_pids = fc["train_pids"], fc["val_pids"]

                y_tr = [IDX_W[vw_gt[pid].get(var, "not_used")] for pid in train_pids]
                y_val = [IDX_W[vw_gt[pid].get(var, "not_used")] for pid in val_pids]

                model = clf_cls(
                    n_estimators=n_est,
                    criterion=crit,
                    class_weight=cw,
                    max_features="sqrt",
                    random_state=42 + fc["split_id"],
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)

                f1 = f1_score(y_val, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
                f1_scores.append(f1)

            mean_f1 = float(np.mean(f1_scores))
            if mean_f1 > best_score:
                best_score = mean_f1
                best_cfg = {
                    "algorithm": alg_name,
                    "n_estimators": n_est,
                    "criterion": crit,
                    "class_weight": cw,
                    "mean_macro_f1": mean_f1,
                }

        best_hyperparams_per_var[var] = best_cfg
        print(f"  {var:<14} -> Best: {best_cfg['algorithm']} (n={best_cfg['n_estimators']}, crit={best_cfg['criterion']}, cw={best_cfg['class_weight']}) | Macro-F1: {best_score:.4f}")

    # Baseline exp_12 per-var configuration
    exp12_baseline_cfg = {
        "age": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "bx_gl_prim": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "bx_gl_sec": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "bx_isup": {"algorithm": "rf", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "comorbidity": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "cspca": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "ct": {"algorithm": "rf", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "fh": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "pirads": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "psa": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
        "psad": {"algorithm": "et", "n_estimators": 200, "criterion": "gini", "class_weight": "balanced"},
    }

    # -------------------------------------------------------------------------
    # Phase 2: Full MCCV-30 Benchmark & chimera_metrics Governance
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Evaluating Full MCCV-30 Benchmark with chimera_metrics ---")
    split_results: List[Dict[str, Any]] = []

    for fc in fold_cache:
        split_id = fc["split_id"]
        X_tr, X_val = fc["X_tr"], fc["X_val"]
        train_pids, val_pids = fc["train_pids"], fc["val_pids"]

        # Models for C1 (Baseline) and C2 (Tuned)
        preds_c1: Dict[str, Dict[str, str]] = {pid: {} for pid in val_pids}
        preds_c2: Dict[str, Dict[str, str]] = {pid: {} for pid in val_pids}

        for var in VARIABLES_11:
            y_tr = [IDX_W[vw_gt[pid].get(var, "not_used")] for pid in train_pids]

            # Fit C1
            b_cfg = exp12_baseline_cfg[var]
            clf_b = algorithms[b_cfg["algorithm"]](
                n_estimators=b_cfg["n_estimators"],
                criterion=b_cfg["criterion"],
                class_weight=b_cfg["class_weight"],
                max_features="sqrt",
                random_state=42 + split_id,
                n_jobs=-1,
            )
            clf_b.fit(X_tr, y_tr)
            y_pred_b = clf_b.predict(X_val)
            for idx, pid in enumerate(val_pids):
                preds_c1[pid][var] = WEIGHT_ORDER[int(y_pred_b[idx])]

            # Fit C2
            t_cfg = best_hyperparams_per_var[var]
            clf_t = algorithms[t_cfg["algorithm"]](
                n_estimators=t_cfg["n_estimators"],
                criterion=t_cfg["criterion"],
                class_weight=t_cfg["class_weight"],
                max_features="sqrt",
                random_state=42 + split_id,
                n_jobs=-1,
            )
            clf_t.fit(X_tr, y_tr)
            y_pred_t = clf_t.predict(X_val)
            for idx, pid in enumerate(val_pids):
                preds_c2[pid][var] = WEIGHT_ORDER[int(y_pred_t[idx])]

        # Evaluate using chimera_metrics for each validation case
        c1_s_weights_list = []
        c1_set_f1_list = []
        c2_s_weights_list = []
        c2_set_f1_list = []

        for pid in val_pids:
            gt_dict = {"variable_weights": vw_gt[pid]}
            pred_c1_dict = {"variable_weights": preds_c1[pid]}
            pred_c2_dict = {"variable_weights": preds_c2[pid]}

            # C1 metrics
            c1_s_weights_list.append(compute_variable_weights_score(gt_dict, pred_c1_dict))
            c1_set_f1_list.append(compute_important_decisive_f1(gt_dict, pred_c1_dict))

            # C2 metrics
            c2_s_weights_list.append(compute_variable_weights_score(gt_dict, pred_c2_dict))
            c2_set_f1_list.append(compute_important_decisive_f1(gt_dict, pred_c2_dict))

        split_results.append({
            "split_id": split_id,
            "C1_baseline": {
                "s_weights": float(np.mean(c1_s_weights_list)),
                "set_f1": float(np.mean(c1_set_f1_list)),
            },
            "C2_tuned": {
                "s_weights": float(np.mean(c2_s_weights_list)),
                "set_f1": float(np.mean(c2_set_f1_list)),
            },
        })

    # Aggregate metrics
    c1_sw = [s["C1_baseline"]["s_weights"] for s in split_results]
    c1_f1 = [s["C1_baseline"]["set_f1"] for s in split_results]
    c2_sw = [s["C2_tuned"]["s_weights"] for s in split_results]
    c2_f1 = [s["C2_tuned"]["set_f1"] for s in split_results]

    try:
        w_p_sw = float(wilcoxon(c1_sw, c2_sw).pvalue)
    except Exception:
        w_p_sw = 1.0

    try:
        w_p_f1 = float(wilcoxon(c1_f1, c2_f1).pvalue)
    except Exception:
        w_p_f1 = 1.0

    agg_results = {
        "C1_baseline_exp12": {
            "s_weights_mean": float(np.mean(c1_sw)),
            "s_weights_std": float(np.std(c1_sw)),
            "set_f1_mean": float(np.mean(c1_f1)),
            "set_f1_std": float(np.std(c1_f1)),
        },
        "C2_per_var_tuned": {
            "s_weights_mean": float(np.mean(c2_sw)),
            "s_weights_std": float(np.std(c2_sw)),
            "set_f1_mean": float(np.mean(c2_f1)),
            "set_f1_std": float(np.std(c2_f1)),
        },
        "wilcoxon_tests": {
            "s_weights_p_value": w_p_sw,
            "set_f1_p_value": w_p_f1,
        },
        "best_hyperparams_per_var": best_hyperparams_per_var,
    }

    print("\n" + "=" * 80)
    print("FINAL MCCV-30 RESULTS SUMMARY (CHIMERA_METRICS EVALUATION)")
    print("=" * 80)
    print(f"C1 (Baseline exp_12):")
    print(f"  S_weights: {agg_results['C1_baseline_exp12']['s_weights_mean']:.4f} +/- {agg_results['C1_baseline_exp12']['s_weights_std']:.4f}")
    print(f"  Set-F1   : {agg_results['C1_baseline_exp12']['set_f1_mean']:.4f} +/- {agg_results['C1_baseline_exp12']['set_f1_std']:.4f}")
    print(f"\nC2 (Per-Variable Tuned):")
    print(f"  S_weights: {agg_results['C2_per_var_tuned']['s_weights_mean']:.4f} +/- {agg_results['C2_per_var_tuned']['s_weights_std']:.4f}")
    print(f"  Set-F1   : {agg_results['C2_per_var_tuned']['set_f1_mean']:.4f} +/- {agg_results['C2_per_var_tuned']['set_f1_std']:.4f}")
    print(f"\nStatistical Significance:")
    print(f"  p-value (S_weights): {w_p_sw:.6f}")
    print(f"  p-value (Set-F1)   : {w_p_f1:.6f}")

    # Persist artifacts
    with open(RESULTS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_hyperparams_per_var, f, indent=2)

    with open(RESULTS_DIR / "split_metrics.json", "w", encoding="utf-8") as f:
        json.dump(split_results, f, indent=2)

    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2)

    print(f"\n[+] All artifacts successfully saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
