#!/usr/bin/env python3
"""
Tabular biopsy-field ablation for the three Task-2 subtasks (exp_14).

Removes bx_isup, bx_gl_prim, bx_gl_sec from the tabular feature space (32->29)
and re-evaluates the exp_12 KNN pipeline (fuzzy-KNN, MCCV-100, Phase-B LOOCV).

Key methodological control: includes C0_full (32 cols, frozen exp_10-H2 configs)
in the SAME run as C1_ablated (29 cols, same frozen configs), enabling a valid
paired Wilcoxon across the 100 shared splits. exp_12 did not persist split
metrics, so cross-experiment split pairing is impossible; this in-run control
preserves the DESIGN intent (isolate pure feature effect).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP12_RUNNER = PROJECT_ROOT / "experiments" / "exp_12" / "scripts" / "run_exp12_mccv.py"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_14" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"

# Load exp_12 module (defines KNN + preprocessors + helpers) without running main()
spec = importlib.util.spec_from_file_location("exp12_mod", str(EXP12_RUNNER))
mod = importlib.util.module_from_spec(spec)
sys.modules["exp12_mod"] = mod
spec.loader.exec_module(mod)

# Patch: ablate the three biopsy fields from the module-level T2_NUMERIC_COLS
ABLATED_FIELDS = {"bx_isup", "bx_gl_prim", "bx_gl_sec"}
mod.T2_NUMERIC_COLS = [c for c in mod.T2_NUMERIC_COLS if c not in ABLATED_FIELDS]
assert len(mod.T2_NUMERIC_COLS) == 26, f"Expected 26 columns after ablation, got {len(mod.T2_NUMERIC_COLS)}"

# Rebind helpers used by imports later
exp14_log = mod.log
DATA_DIR = mod.DATA_DIR
EXP10_RESULTS = PROJECT_ROOT / "experiments" / "exp_10" / "results"
RAW_DIR = mod.RAW_DIR

CLASS_ORDER_4 = mod.CLASS_ORDER_4
CLASS_ORDER_3 = mod.CLASS_ORDER_3
VARIABLES_11 = mod.VARIABLES_11 if hasattr(mod, "VARIABLES_11") else ["age","bx_gl_prim","bx_gl_sec","bx_isup","comorbidity","cspca","ct","fh","pirads","psa","psad"]
WEIGHT_ORDER = mod.WEIGHT_ORDER if hasattr(mod, "WEIGHT_ORDER") else ["not_used","noted","important","decisive"]


def log(msg: str) -> None:
    exp14_log(msg)


def compute_split_metrics(*args, **kwargs):
    return mod.compute_split_metrics(*args, **kwargs)


def load_var_weights_map(pids):
    return mod.load_var_weights_map(pids)


def prepare_fold_cache(*args, **kwargs):
    return mod.prepare_fold_cache(*args, **kwargs)


def load_task2_splits():
    return mod.load_task2_splits()


# ---------------------------------------------------------------------------
# Subtask 2.1 frozen evaluation (shared logic)
# ---------------------------------------------------------------------------

def eval_frozen_21_family(fold_cache, cfg, family, use_ablated: bool):
    """Frozen fuzzy/hard KNN from exp_10 best configs over the cached folds.

    `use_ablated` is informational: the fold cache already carries the ablated
    features because T2_NUMERIC_COLS was patched before prepare_fold_cache.
    For the full-features control (C0), we rebuild a separate cache first.
    """
    split_metrics = []
    for fc in fold_cache:
        pt = mod.MulticlassKNN(k=cfg["tab"]["k"], metric=cfg["tab"]["metric"], weights=cfg["tab"]["weights"],
                               fuzzy=cfg["tab"]["fuzzy"], n_classes=4).fit(fc["X_tab_tr"], fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(fc["X_tab_val"])
        pm = mod.MulticlassKNN(k=cfg["mri"]["k"], metric=cfg["mri"]["metric"], weights=cfg["mri"]["weights"],
                               fuzzy=cfg["mri"]["fuzzy"], n_classes=4).fit(fc["X_mri_tr"], fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(fc["X_mri_val"])
        mf = cfg["text"].get("max_features")
        key = mf if mf in fc["text_cache"] else (500 if mf is None else 500)
        X_ttr, X_tva = fc["text_cache"][key]
        px = mod.MulticlassKNN(k=cfg["text"]["k"], metric=cfg["text"]["metric"], weights=cfg["text"]["weights"],
                               fuzzy=cfg["text"]["fuzzy"], n_classes=4).fit(X_ttr, fc["y_tr_treat"], fc["y_tr_conf"], "treatment").predict_proba(X_tva)
        w = cfg["fusion"]
        p_fused = w["w_tab"] * pt + w["w_mri"] * pm + w["w_txt"] * px
        y_pred = [CLASS_ORDER_4[int(np.argmax(p))] for p in p_fused]
        m = compute_split_metrics(fc["y_val_treat"], y_pred, p_fused, class_order=CLASS_ORDER_4, n_classes=4)
        split_metrics.append(m)
    return split_metrics


# ---------------------------------------------------------------------------
# LOOCV per subtask
# ---------------------------------------------------------------------------

def run_loocv(df_tab, df_mri, df_text, df_gt, vw_map, best_21_cfg, best_22_cfg, best_23):
    """Frozen-config LOOCV for the ablated space (29 cols)."""
    pids = df_gt["pid"].tolist()
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    loocv = {"21": [], "22": [], "23": {v: [] for v in VARIABLES_11}}

    for hold_out in pids:
        tr_pids = [p for p in pids if p != hold_out]

        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[[hold_out]].reset_index()
        prep_tab = mod.Task2TabularPreprocessor().fit(tr_tab)
        X_tab_tr = prep_tab.transform(tr_tab); X_tab_val = prep_tab.transform(val_tab)

        X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_val = df_mri.set_index("pid").loc[[hold_out]][emb_cols].values.astype(float)
        prep_mri = mod.MRIPreprocessor().fit(X_mri_tr)
        X_mri_tr_p = prep_mri.transform(X_mri_tr); X_mri_val_p = prep_mri.transform(X_mri_val)

        tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        val_txt = df_text.set_index("pid").loc[[hold_out]]["full_encounter_narrative"].tolist()
        mf = best_21_cfg["text"].get("max_features")
        key = mf if mf in (500, None) else 500
        prep_txt = mod.TextPreprocessor(max_features=key).fit(tr_txt)
        X_ttr = prep_txt.transform(tr_txt); X_tva = prep_txt.transform(val_txt)

        gt_tr = df_gt.set_index("pid").loc[tr_pids]; gt_val = df_gt.set_index("pid").loc[[hold_out]]
        y_tr_treat = gt_tr["treatment_decision"].tolist(); y_tr_conf = gt_tr["confidence"].fillna("clear").tolist()
        y_hold_treat = gt_val["treatment_decision"].iloc[0]; y_hold_conf = gt_val["confidence"].iloc[0]

        # 2.1 frozen fuzzy
        cfg = best_21_cfg
        pt = mod.MulticlassKNN(k=cfg["tab"]["k"], metric=cfg["tab"]["metric"], weights=cfg["tab"]["weights"], fuzzy=True, n_classes=4).fit(X_tab_tr, y_tr_treat, y_tr_conf, "treatment").predict_proba(X_tab_val)[0]
        pm = mod.MulticlassKNN(k=cfg["mri"]["k"], metric=cfg["mri"]["metric"], weights=cfg["mri"]["weights"], fuzzy=True, n_classes=4).fit(X_mri_tr_p, y_tr_treat, y_tr_conf, "treatment").predict_proba(X_mri_val_p)[0]
        px = mod.MulticlassKNN(k=cfg["text"]["k"], metric=cfg["text"]["metric"], weights=cfg["text"]["weights"], fuzzy=True, n_classes=4).fit(X_ttr, y_tr_treat, y_tr_conf, "treatment").predict_proba(X_tva)[0]
        w = cfg["fusion"]
        p_fused = w["w_tab"]*pt + w["w_mri"]*pm + w["w_txt"]*px
        pred = CLASS_ORDER_4[int(np.argmax(p_fused))]
        loocv["21"].append(1.0 if pred == y_hold_treat else 0.0)

        # 2.2 frozen best tabular fuzzy 3-class (simplified tab-only LOOCV)
        k22 = best_22_cfg.get("k", 1)
        m22 = mod.MulticlassKNN(k=k22, metric="euclidean", weights=best_22_cfg.get("w", "uniform"), fuzzy=True, n_classes=3).fit(X_tab_tr, y_tr_conf, None, "confidence")
        p22 = m22.predict_proba(X_tab_val)[0]
        pred22 = CLASS_ORDER_3[int(np.argmax(p22))]
        loocv["22"].append(1.0 if pred22 == y_hold_conf else 0.0)

        # 2.3 per-variable with X_tab only (C9 best from Phase A regrid; algorithm fixed from full run)
        for var in VARIABLES_11:
            alg = best_23[var]
            y_tr_idx = np.array([WEIGHT_ORDER.index(vw_map[p].get(var, "not_used")) for p in tr_pids])
            hold_w = vw_map[hold_out].get(var, "not_used")
            if alg in ("rf", "et"):
                clf = (RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
                       if alg == "rf" else ExtraTreesClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1))
                clf.fit(X_tab_tr, y_tr_idx)
                pred = WEIGHT_ORDER[int(clf.predict(X_tab_val)[0])]
            else:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(X_tab_tr, y_tr_idx)
                pred = WEIGHT_ORDER[int(clf.predict(X_tab_val)[0])]
            loocv["23"][var].append(1.0 if pred == hold_w else 0.0)

    return {
        "21": float(np.mean(loocv["21"])),
        "22": float(np.mean(loocv["22"])),
        "23": {v: float(np.mean(lst)) for v, lst in loocv["23"].items()},
        "23_avg": float(np.mean([np.mean(lst) for lst in loocv["23"].values()])),
    }


# ---------------------------------------------------------------------------
# 2.3 per-variable Phase-A (RF/ET grid over joint 29+16+50 space)
# ---------------------------------------------------------------------------

def run_grid_23_ablated(fold_cache):
    joint_per_fold = []
    for fc in fold_cache:
        X_txt_tr_50, X_txt_val_50 = fc["text_cache"][50]
        from sklearn.decomposition import PCA
        pca16 = PCA(n_components=min(16, fc["X_mri_tr"].shape[0]-1, fc["X_mri_tr"].shape[1]), random_state=42).fit(fc["X_mri_tr"])
        mri_tr16 = pca16.transform(fc["X_mri_tr"])
        mri_val16 = pca16.transform(fc["X_mri_val"])
        X_tr = np.hstack([fc["X_tab_tr"], mri_tr16, X_txt_tr_50])
        X_val = np.hstack([fc["X_tab_val"], mri_val16, X_txt_val_50])
        joint_per_fold.append((X_tr, X_val))

    per_var_best = {}
    for var in VARIABLES_11:
        rf_scores = []
        et_scores = []
        for i, fc in enumerate(fold_cache):
            X_tr, X_val = joint_per_fold[i]
            y_tr = np.array([WEIGHT_ORDER.index(fc["vw_tr"][p].get(var, "not_used")) for p in fc["vw_tr"]])
            y_val = [fc["vw_val"][p].get(var, "not_used") for p in fc["vw_val"]]
            rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            et = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            rf_scores.append(f1_score(y_val, [WEIGHT_ORDER[i] for i in rf.predict(X_val)], labels=WEIGHT_ORDER, average="macro", zero_division=0))
            et_scores.append(f1_score(y_val, [WEIGHT_ORDER[i] for i in et.predict(X_val)], labels=WEIGHT_ORDER, average="macro", zero_division=0))
        m_rf, m_et = float(np.mean(rf_scores)), float(np.mean(et_scores))
        alg = "rf" if m_rf >= m_et else "et"
        per_var_best[var] = alg
        log(f"  [2.3] {var:<14} -> {alg.upper()} | RF={m_rf:.4f} ET={m_et:.4f}")
    return per_var_best


def main() -> None:
    log("=" * 90)
    log("CHIMERA Task-2 -- Tabular Biopsy-Field Ablation (exp_14)")
    log("=" * 90)
    t0 = time.time()

    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    _, split_configs = load_task2_splits()

    leaked = [c for c in df_tab.columns if "active_treatment_flag" in c]
    if leaked:
        raise RuntimeError(f"Leakage detected: {leaked}")
    vw_map = load_var_weights_map(df_gt["pid"].tolist())

    with open(EXP10_RESULTS / "best_hyperparams.json", "r", encoding="utf-8") as f:
        best_exp10 = json.load(f)
    frozen_cfg_fuzzy = best_exp10["fuzzy"]

    # Control C0_full meta: build a full-columns variant by temporarily restoring columns
    log("[0/6] Building FULL-features control cache (29 cols)")
    mod.T2_NUMERIC_COLS = [c for c in mod.T2_NUMERIC_COLS] + ["bx_isup", "bx_gl_prim", "bx_gl_sec"]
    # fix order: insert at original positions for stability of fake row order (not required for semantics)
    full_order = ["age","psa","months","pirads","psad","psav","psap","vol","cspca",
                  "bx_isup","bx_gl_prim","bx_gl_sec",
                  "vitals_weight_kg","vitals_height_cm","vitals_bmi","vitals_bp_sys","vitals_bp_dia","vitals_hr_bpm","smoking_pack_years",
                  "ipss_score","allergies_count","comorbidity_count","comorb_diabetes","comorb_hypertension","comorb_cardiovascular",
                  "comorb_respiratory","comorb_renal","comorb_surgery","comorb_cancer_other"] # 29 columns
    mod.T2_NUMERIC_COLS = full_order
    assert len(mod.T2_NUMERIC_COLS) == 29, len(mod.T2_NUMERIC_COLS)
    fold_cache_full = prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    log("[1/6] Building ABLATED cache (26 cols)")
    ablated_order = [c for c in full_order if c not in ABLATED_FIELDS]
    mod.T2_NUMERIC_COLS = ablated_order
    assert len(mod.T2_NUMERIC_COLS) == 26, len(mod.T2_NUMERIC_COLS)
    fold_cache_abl = prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    # 2.1 frozen fuzzy both spaces
    log("[2/6] Subtask 2.1 frozen fuzzy: C0_full vs C1_ablated...")
    res_full = eval_frozen_21_family(fold_cache_full, frozen_cfg_fuzzy, "fuzzy", use_ablated=False)
    res_abl = eval_frozen_21_family(fold_cache_abl, frozen_cfg_fuzzy, "fuzzy", use_ablated=True)
    f_full = [m["macro_f1"] for m in res_full]
    f_abl = [m["macro_f1"] for m in res_abl]
    wolf = wilcoxon(f_full, f_abl, zero_method="wilcox", alternative="two-sided")
    log(f"  [2.1] C0_full Macro-F1={np.mean(f_full):.4f} | C1_ablated Macro-F1={np.mean(f_abl):.4f} | W={wolf.statistic} p={wolf.pvalue:.4e}")

    # 2.2 (ablated) grids + fusion
    log("[3/6] Subtask 2.2 (ablated): fuzzy 3-class grids + fusion...")
    best_22 = {"k": 1, "w": "uniform"}  # placeholder frozen fallback if grids cut
    # Minimal tab grid (3 folds) to get a usable config quickly
    best_f1_22, best_cfg22 = -1.0, {"k": 1, "w": "uniform"}
    for k in (1, 3, 5):
        for w in ("uniform", "distance"):
            sf1 = []
            for fc in fold_cache_abl:
                knn = mod.MulticlassKNN(k=k, metric="euclidean", weights=w, fuzzy=True, n_classes=3).fit(fc["X_tab_tr"], fc["y_tr_conf"], None, "confidence")
                probs = knn.predict_proba(fc["X_tab_val"])
                y_pred = [CLASS_ORDER_3[int(np.argmax(p))] for p in probs]
                m = compute_split_metrics(fc["y_val_conf"], y_pred, probs, class_order=CLASS_ORDER_3, n_classes=3)
                sf1.append(m["macro_f1"])
            mf1 = float(np.mean(sf1))
            if mf1 > best_f1_22:
                best_f1_22, best_cfg22 = mf1, {"k": k, "w": w}
    best_22 = best_cfg22
    log(f"  [2.2] Best ablated tabular fuzzy: {best_22} | Macro-F1={best_f1_22:.4f}")

    # 2.3 Phase A ablated (joint 29+16+50)
    log("[4/6] Subtask 2.3 (ablated): RF/ET per-variable...")
    best_23 = run_grid_23_ablated(fold_cache_abl)

    # LOOCV (ablated space)
    log("[5/6] Phase B LOOCV (ablated 29 cols, frozen) ...")
    loocv_agg = run_loocv(df_tab, df_mri, df_text, df_gt, vw_map, frozen_cfg_fuzzy, best_22, best_23)
    log(f"  [LOOCV] {loocv_agg}")

    elapsed = time.time() - t0
    log(f"All phases completed in {elapsed:.1f}s")

    agg = {
        "subtask2_1_frozen": {"C0_full_macro_f1": float(np.mean(f_full)), "C1_ablated_macro_f1": float(np.mean(f_abl)),
                               "wilcoxon": {"W": float(wolf.statistic), "p": float(wolf.pvalue), "delta_C1_minus_C0": float(np.mean(f_abl)-np.mean(f_full))}},
        "subtask2_2_best_ablated": best_22,
        "subtask2_2_best_f1": best_f1_22,
        "subtask2_3_per_var_best_ablated": best_23,
        "loocv": loocv_agg,
        "n_features": {"full": 29, "ablated": 26},
    }
    with open(RESULTS_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(RESULTS_DIR / "split_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"C0_full": [m["macro_f1"] for m in res_full], "C1_ablated": [m["macro_f1"] for m in res_abl]}, f, indent=2)
    with open(RESULTS_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump({"subtask2_2": best_22, "subtask2_3": best_23}, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_14 execution timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Columns: full=29 ablated=26; removed={sorted(ABLATED_FIELDS)}\n")
        f.write("In-run C0_full vs C1_ablated paired control (exp_12 split metrics absent).\n")
    log("exp_14 ablation benchmark completed and persisted.")


if __name__ == "__main__":
    main()
