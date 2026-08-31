#!/usr/bin/env python3
"""
Systematic Hyperparameter and Late-Fusion Search for Task 2 Treatment Decision (exp_17).
Protocol: Multi-subtask MCCV-30 (data/processed/task2/mccv_splits_30.csv).
Metrics: Strictly evaluated via utils/pkg/chimera_metrics.
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score

# Ensure chimera_metrics is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "utils" / "pkg" / "chimera_metrics"))

from chimera_metrics.core.base import norm_treatment_decision
from chimera_metrics.task2.decision import compute_treatment_decision_gate

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "task2"
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_17"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["watchful_waiting", "active_surveillance", "continued_surveillance", "active_treatment"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
N_CLASSES = len(CLASS_ORDER)
N_SPLITS = 30

CONF_MAP = {"clear": 1.0, "borderline": 0.5, "uncertain": 0.0}

TAB_CAT_COLS = ["bx_gl_prim", "bx_gl_sec", "bx_isup", "ct", "fh", "pirads"]
TAB_NUM_COLS = ["age", "psa", "psad", "comorbidity", "cspca"]


class Task2TabularPreprocessor:
    def __init__(self):
        self.cat_modes: Dict[str, str] = {}
        self.num_medians: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> "Task2TabularPreprocessor":
        df = df.copy()
        for col in TAB_CAT_COLS:
            if col in df.columns:
                m = df[col].mode()
                self.cat_modes[col] = m.iloc[0] if not m.empty else "missing"
        for col in TAB_NUM_COLS:
            if col in df.columns:
                self.num_medians[col] = float(df[col].median()) if not df[col].dropna().empty else 0.0

        processed = self._impute_and_encode(df)
        self.feature_columns = processed.columns.tolist()
        self.scaler.fit(processed)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        processed = self._impute_and_encode(df)
        for col in self.feature_columns:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[self.feature_columns]
        return self.scaler.transform(processed)

    def _impute_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in TAB_CAT_COLS:
            if col in df.columns:
                val = self.cat_modes.get(col, "missing")
                df[col] = df[col].fillna(val).astype(str)
        for col in TAB_NUM_COLS:
            if col in df.columns:
                val = self.num_medians.get(col, 0.0)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

        cat_present = [c for c in TAB_CAT_COLS if c in df.columns]
        df_encoded = pd.get_dummies(df[cat_present], prefix=cat_present, drop_first=False)
        num_present = [c for c in TAB_NUM_COLS if c in df.columns]
        return pd.concat([df[num_present], df_encoded], axis=1)


def compute_soft_targets(gt_df: pd.DataFrame) -> np.ndarray:
    n_samples = len(gt_df)
    Y_soft = np.zeros((n_samples, N_CLASSES), dtype=float)
    for i, (_, row) in enumerate(gt_df.iterrows()):
        dec = norm_treatment_decision(row["treatment_decision"])
        k = CLASS_TO_IDX[dec]
        c = CONF_MAP.get(str(row["confidence"]).lower(), 1.0)
        p_k = 0.25 + 0.75 * c
        p_rest = (1.0 - p_k) / (N_CLASSES - 1)
        Y_soft[i, :] = p_rest
        Y_soft[i, k] = p_k
    return Y_soft


def compute_brier_multiclass(y_true_indices: np.ndarray, y_prob: np.ndarray) -> float:
    n = len(y_true_indices)
    one_hot = np.zeros((n, N_CLASSES))
    for i, idx in enumerate(y_true_indices):
        one_hot[i, idx] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def evaluate_predictions_with_chimera(y_val_str: List[str], preds_str: List[str], p_fused: np.ndarray) -> Dict[str, float]:
    gate_scores = []
    for gt_d, pred_d in zip(y_val_str, preds_str):
        score, _, _, _ = compute_treatment_decision_gate(gt_d, pred_d)
        gate_scores.append(score)

    gate_acc = float(np.mean(gate_scores))
    macro_f1 = float(f1_score(y_val_str, preds_str, labels=CLASS_ORDER, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_val_str, preds_str, labels=CLASS_ORDER, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_val_str, preds_str, labels=CLASS_ORDER, average="weighted", zero_division=0))

    per_class = f1_score(y_val_str, preds_str, labels=CLASS_ORDER, average=None, zero_division=0)
    per_class_dict = {c: float(per_class[i]) for i, c in enumerate(CLASS_ORDER)}

    y_val_idx = np.array([CLASS_TO_IDX[d] for d in y_val_str])
    brier = compute_brier_multiclass(y_val_idx, p_fused)

    one_hot = np.zeros((len(y_val_idx), N_CLASSES))
    for i, idx in enumerate(y_val_idx):
        one_hot[i, idx] = 1.0

    aucs = []
    for c_i in range(N_CLASSES):
        if len(np.unique(one_hot[:, c_i])) > 1:
            try:
                aucs.append(roc_auc_score(one_hot[:, c_i], p_fused[:, c_i]))
            except Exception:
                pass
    auroc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "gate_acc": gate_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "brier": brier,
        "auroc": auroc,
        **{f"f1_{k}": v for k, v in per_class_dict.items()},
    }


def generate_simplex_grid(step: float = 0.05) -> List[Tuple[float, float, float]]:
    n_steps = int(round(1.0 / step))
    weights = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w1 = round(i * step, 2)
            w2 = round(j * step, 2)
            w3 = round(k * step, 2)
            weights.append((w1, w2, w3))
    return weights


def l1_norm(arr: np.ndarray) -> np.ndarray:
    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return arr / s


def align_probs(prob_arr: np.ndarray, classes_present: np.ndarray) -> np.ndarray:
    full_prob = np.zeros((len(prob_arr), N_CLASSES))
    for local_i, global_c in enumerate(classes_present):
        full_prob[:, int(global_c)] = prob_arr[:, local_i]
    return full_prob


def main():
    print("=" * 80)
    print("STARTING TASK 2 (SUBTASK 2.1) SYSTEMATIC HYPERPARAMETER SEARCH (exp_17)")
    print("=" * 80)

    # 1. Load data
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_txt = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_splits = pd.read_csv(DATA_DIR / "mccv_splits_30.csv")

    mri_feature_cols = [c for c in df_mri.columns if c != "pid"]

    # Search grids
    hard_grid = list(itertools.product([0.1, 1.0, 10.0], [None, "balanced"]))
    fuzzy_grid = list(itertools.product([0.1, 1.0, 10.0], [0.05, 0.10]))

    # Data structures to store fold predictions
    # hard_val_preds[modality][config_tuple][split_idx] = prob_array (N_val x 4)
    modalities = ["tab", "mri", "txt"]
    hard_fold_probs: Dict[str, Dict[Tuple, List[np.ndarray]]] = {m: {cfg: [] for cfg in hard_grid} for m in modalities}
    fuzzy_fold_probs: Dict[str, Dict[Tuple, List[np.ndarray]]] = {m: {cfg: [] for cfg in fuzzy_grid} for m in modalities}

    split_gt_val_str: List[List[str]] = []

    print(f"[*] Training and evaluating unimodal configs across {N_SPLITS} splits...")

    for split_idx in range(N_SPLITS):
        split_col = f"split_{split_idx}"
        train_pids = set(df_splits[df_splits[split_col] == "train"]["pid"])
        val_pids = set(df_splits[df_splits[split_col] == "val"]["pid"])

        train_mask = df_gt["pid"].isin(train_pids)
        val_mask = df_gt["pid"].isin(val_pids)

        gt_train = df_gt[train_mask].copy()
        gt_val = df_gt[val_mask].copy()

        y_train_str = [norm_treatment_decision(d) for d in gt_train["treatment_decision"]]
        y_val_str = [norm_treatment_decision(d) for d in gt_val["treatment_decision"]]
        split_gt_val_str.append(y_val_str)
        y_train_idx = np.array([CLASS_TO_IDX[s] for s in y_train_str])
        Y_soft_train = compute_soft_targets(gt_train)

        # Tabular
        tab_prep = Task2TabularPreprocessor()
        X_tab_train = tab_prep.fit(df_tab[train_mask]).transform(df_tab[train_mask])
        X_tab_val = tab_prep.transform(df_tab[val_mask])

        # MRI
        mri_scaler = StandardScaler()
        X_mri_train = mri_scaler.fit_transform(df_mri[train_mask][mri_feature_cols].to_numpy(dtype=float))
        X_mri_val = mri_scaler.transform(df_mri[val_mask][mri_feature_cols].to_numpy(dtype=float))

        # Text
        tfidf = TfidfVectorizer(max_features=None, lowercase=True)
        txt_train_series: pd.Series = pd.Series(df_txt[train_mask]["full_encounter_narrative"])
        txt_val_series: pd.Series = pd.Series(df_txt[val_mask]["full_encounter_narrative"])
        X_txt_train = tfidf.fit_transform(txt_train_series.fillna("").tolist()).toarray()
        X_txt_val = tfidf.transform(txt_val_series.fillna("").tolist()).toarray()

        feat_dict_train = {"tab": X_tab_train, "mri": X_mri_train, "txt": X_txt_train}
        feat_dict_val = {"tab": X_tab_val, "mri": X_mri_val, "txt": X_txt_val}

        # Train & record Hard-SVC
        for m in modalities:
            for c_val, cw in hard_grid:
                svc = SVC(kernel="linear", C=c_val, class_weight=cw, probability=True, random_state=42)
                svc.fit(feat_dict_train[m], y_train_idx)
                raw_p = svc.predict_proba(feat_dict_val[m])
                aligned_p = align_probs(raw_p, svc.classes_)
                hard_fold_probs[m][(c_val, cw)].append(aligned_p)

        # Train & record Fuzzy-SVR
        for m in modalities:
            for c_val, eps in fuzzy_grid:
                svr = MultiOutputRegressor(LinearSVR(C=c_val, epsilon=eps, random_state=42))
                svr.fit(feat_dict_train[m], Y_soft_train)
                raw_out = np.clip(svr.predict(feat_dict_val[m]), 0, None)
                norm_p = l1_norm(raw_out)
                fuzzy_fold_probs[m][(c_val, eps)].append(norm_p)

    print("[*] Selecting best unimodal hyperparameter configurations based on mean Macro-F1...")
    best_hard_cfg: Dict[str, Tuple] = {}
    best_fuzzy_cfg: Dict[str, Tuple] = {}

    for m in modalities:
        # Hard
        hard_scores = {}
        for cfg in hard_grid:
            f1s = []
            for s in range(N_SPLITS):
                p = hard_fold_probs[m][cfg][s]
                preds = [CLASS_ORDER[i] for i in np.argmax(p, axis=1)]
                f1s.append(f1_score(split_gt_val_str[s], preds, labels=CLASS_ORDER, average="macro", zero_division=0))
            hard_scores[cfg] = np.mean(f1s)
        best_hard_cfg[m] = max(hard_scores, key=lambda k: hard_scores[k])
        print(f"  [Hard-SVC] Best {m}: C={best_hard_cfg[m][0]}, class_weight={best_hard_cfg[m][1]} (Macro-F1: {hard_scores[best_hard_cfg[m]]:.4f})")

        # Fuzzy
        fuzzy_scores = {}
        for cfg in fuzzy_grid:
            f1s = []
            for s in range(N_SPLITS):
                p = fuzzy_fold_probs[m][cfg][s]
                preds = [CLASS_ORDER[i] for i in np.argmax(p, axis=1)]
                f1s.append(f1_score(split_gt_val_str[s], preds, labels=CLASS_ORDER, average="macro", zero_division=0))
            fuzzy_scores[cfg] = np.mean(f1s)
        best_fuzzy_cfg[m] = max(fuzzy_scores, key=lambda k: fuzzy_scores[k])
        print(f"  [Fuzzy-SVR] Best {m}: C={best_fuzzy_cfg[m][0]}, epsilon={best_fuzzy_cfg[m][1]} (Macro-F1: {fuzzy_scores[best_fuzzy_cfg[m]]:.4f})")

    # Save best unimodal hyperparams
    best_hyperparams_export = {
        "Hard_SVC": {m: {"C": best_hard_cfg[m][0], "class_weight": best_hard_cfg[m][1]} for m in modalities},
        "Fuzzy_SVR": {m: {"C": best_fuzzy_cfg[m][0], "epsilon": best_fuzzy_cfg[m][1]} for m in modalities},
    }
    with open(RESULTS_DIR / "best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams_export, f, indent=2)

    # 2. Simplex search
    simplex_points = generate_simplex_grid(step=0.05)
    print(f"\n[*] Optimizing late-fusion weights across {len(simplex_points)} simplex combinations...")

    # Hard-SVC Simplex Search
    best_hard_weights = None
    best_hard_f1 = -1.0
    for w_tab, w_mri, w_txt in simplex_points:
        split_f1s = []
        for s in range(N_SPLITS):
            p_tab = hard_fold_probs["tab"][best_hard_cfg["tab"]][s]
            p_mri = hard_fold_probs["mri"][best_hard_cfg["mri"]][s]
            p_txt = hard_fold_probs["txt"][best_hard_cfg["txt"]][s]
            p_fused = w_tab * p_tab + w_mri * p_mri + w_txt * p_txt
            row_sums = p_fused.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            p_fused = p_fused / row_sums
            preds = [CLASS_ORDER[i] for i in np.argmax(p_fused, axis=1)]
            split_f1s.append(f1_score(split_gt_val_str[s], preds, labels=CLASS_ORDER, average="macro", zero_division=0))
        mean_f1 = np.mean(split_f1s)
        if mean_f1 > best_hard_f1:
            best_hard_f1 = mean_f1
            best_hard_weights = (w_tab, w_mri, w_txt)

    print(f"  [Hard-SVC Fused] Optimal weights (tab, mri, txt): {best_hard_weights} -> Mean Macro-F1: {best_hard_f1:.4f}")

    # Fuzzy-SVR Simplex Search
    best_fuzzy_weights = None
    best_fuzzy_f1 = -1.0
    for w_tab, w_mri, w_txt in simplex_points:
        split_f1s = []
        for s in range(N_SPLITS):
            p_tab = fuzzy_fold_probs["tab"][best_fuzzy_cfg["tab"]][s]
            p_mri = fuzzy_fold_probs["mri"][best_fuzzy_cfg["mri"]][s]
            p_txt = fuzzy_fold_probs["txt"][best_fuzzy_cfg["txt"]][s]
            p_fused = w_tab * p_tab + w_mri * p_mri + w_txt * p_txt
            row_sums = p_fused.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            p_fused = p_fused / row_sums
            preds = [CLASS_ORDER[i] for i in np.argmax(p_fused, axis=1)]
            split_f1s.append(f1_score(split_gt_val_str[s], preds, labels=CLASS_ORDER, average="macro", zero_division=0))
        mean_f1 = np.mean(split_f1s)
        if mean_f1 > best_fuzzy_f1:
            best_fuzzy_f1 = mean_f1
            best_fuzzy_weights = (w_tab, w_mri, w_txt)

    print(f"  [Fuzzy-SVR Fused] Optimal weights (tab, mri, txt): {best_fuzzy_weights} -> Mean Macro-F1: {best_fuzzy_f1:.4f}")

    # Save best fusion weights
    fusion_weights_export = {
        "Hard_SVC": {"weights": list(best_hard_weights), "optimal_macro_f1": float(best_hard_f1)},
        "Fuzzy_SVR": {"weights": list(best_fuzzy_weights), "optimal_macro_f1": float(best_fuzzy_f1)},
    }
    with open(RESULTS_DIR / "best_fusion_weights.json", "w") as f:
        json.dump(fusion_weights_export, f, indent=2)

    # 3. Final Evaluation of Tuned Systems using chimera_metrics
    print("\n[*] Evaluating final tuned systems with chimera_metrics...")
    hard_split_metrics = []
    fuzzy_split_metrics = []

    for s in range(N_SPLITS):
        y_val_str = split_gt_val_str[s]

        # Hard Fused
        w_tab, w_mri, w_txt = best_hard_weights
        p_tab = hard_fold_probs["tab"][best_hard_cfg["tab"]][s]
        p_mri = hard_fold_probs["mri"][best_hard_cfg["mri"]][s]
        p_txt = hard_fold_probs["txt"][best_hard_cfg["txt"]][s]
        p_fused_hard = w_tab * p_tab + w_mri * p_mri + w_txt * p_txt
        row_sums = p_fused_hard.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        p_fused_hard = p_fused_hard / row_sums
        preds_hard = [CLASS_ORDER[i] for i in np.argmax(p_fused_hard, axis=1)]
        m_hard = evaluate_predictions_with_chimera(y_val_str, preds_hard, p_fused_hard)
        hard_split_metrics.append(m_hard)

        # Fuzzy Fused
        w_tab, w_mri, w_txt = best_fuzzy_weights
        p_tab = fuzzy_fold_probs["tab"][best_fuzzy_cfg["tab"]][s]
        p_mri = fuzzy_fold_probs["mri"][best_fuzzy_cfg["mri"]][s]
        p_txt = fuzzy_fold_probs["txt"][best_fuzzy_cfg["txt"]][s]
        p_fused_fuzzy = w_tab * p_tab + w_mri * p_mri + w_txt * p_txt
        row_sums = p_fused_fuzzy.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        p_fused_fuzzy = p_fused_fuzzy / row_sums
        preds_fuzzy = [CLASS_ORDER[i] for i in np.argmax(p_fused_fuzzy, axis=1)]
        m_fuzzy = evaluate_predictions_with_chimera(y_val_str, preds_fuzzy, p_fused_fuzzy)
        fuzzy_split_metrics.append(m_fuzzy)

    # Aggregations
    def aggregate(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        keys = list(metrics_list[0].keys())
        out = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out

    agg_results = {
        "Hard_SVC_Tuned": aggregate(hard_split_metrics),
        "Fuzzy_SVR_Tuned": aggregate(fuzzy_split_metrics),
    }

    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(agg_results, f, indent=2)

    with open(RESULTS_DIR / "split_metrics.json", "w") as f:
        json.dump({"Hard_SVC_Tuned": hard_split_metrics, "Fuzzy_SVR_Tuned": fuzzy_split_metrics}, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL AGGREGATE PERFORMANCE (MCCV-30):")
    print("=" * 80)
    print(f"  [Hard-SVC Tuned]  Macro-F1: {agg_results['Hard_SVC_Tuned']['macro_f1']['mean']:.4f} ± {agg_results['Hard_SVC_Tuned']['macro_f1']['std']:.4f} | Brier: {agg_results['Hard_SVC_Tuned']['brier']['mean']:.4f} ± {agg_results['Hard_SVC_Tuned']['brier']['std']:.4f} | Gate Acc: {agg_results['Hard_SVC_Tuned']['gate_acc']['mean']:.4f}")
    print(f"  [Fuzzy-SVR Tuned] Macro-F1: {agg_results['Fuzzy_SVR_Tuned']['macro_f1']['mean']:.4f} ± {agg_results['Fuzzy_SVR_Tuned']['macro_f1']['std']:.4f} | Brier: {agg_results['Fuzzy_SVR_Tuned']['brier']['mean']:.4f} ± {agg_results['Fuzzy_SVR_Tuned']['brier']['std']:.4f} | Gate Acc: {agg_results['Fuzzy_SVR_Tuned']['gate_acc']['mean']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
