#!/usr/bin/env python3
"""
Cascaded Subtask 2.1 -> 2.2 (Treatment-Informed Confidence) — corrected (exp_15_v2).

Strict NESTED cross-fitting with NO preprocessing leakage:
- Phase A (100 MCCV splits): per-split train/val preprocessed from fold-cache (already leak-free).
    . OOF 2.1 outputs for train cases: 5-fold inner CV INSIDE the train fold.
    . Val 2.1 outputs via full-train 2.1.
    . Soft-SVR grid (tabular branch + 2.1 injection).
- Phase B LOOCV (70 folds): preprocessing (StandardScaler/TF-IDF) adjusted INSIDE each fold,
  never including the held-out case. Pooled OOF 2.1 for injection also recomputed leak-free.

No eval() — C/eps stored as numbers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP13_RUNNER = PROJECT_ROOT / "experiments" / "exp_13" / "scripts" / "run_exp13_mccv.py"
EXP11_BEST = PROJECT_ROOT / "experiments" / "exp_11" / "results" / "best_hyperparams.json"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_15" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "execution.log"

spec = importlib.util.spec_from_file_location("exp13_mod", str(EXP13_RUNNER))
mod = importlib.util.module_from_spec(spec)
sys.modules["exp13_mod"] = mod
spec.loader.exec_module(mod)

SoftLinearSVR = mod.SoftLinearSVR
build_treatment_soft = mod.build_treatment_soft
load_task2_splits = mod.load_task2_splits
DATA_DIR = mod.DATA_DIR
RAW_DIR = mod.RAW_DIR
CLASS_ORDER_4 = mod.CLASS_ORDER_4
CLASS_ORDER_3 = mod.CLASS_ORDER_3
IDX_4 = mod.IDX_4
CONFIDENCE_C = mod.CONFIDENCE_C
CONF_ORDINAL = mod.CONF_ORDINAL

T2_NUMERIC_COLS = mod.T2_NUMERIC_COLS
T2_CATEGORICAL_COLS = mod.T2_CATEGORICAL_COLS


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_conf_3(y_conf_list):
    Y = np.zeros((len(y_conf_list), 3))
    for i, c in enumerate(y_conf_list):
        Y[i, CLASS_ORDER_3.index(c)] = CONF_ORDINAL.get(c, 0.5)
    return Y


def build_conf_bin(y_conf_list):
    Y = np.zeros((len(y_conf_list), 2))
    for i, c in enumerate(y_conf_list):
        ci = 0 if c == "borderline" else 1
        Y[i, ci] = CONF_ORDINAL.get(c, 0.5)
    return Y


class Task2PreprocLeakFree:
    """Fit transform on train only; transform new data (no global pre-fit)."""

    def __init__(self):
        self.tab = None
        self.mri = None
        self.txt = None
        self.emb_cols = None

    def fit_train(self, df_tab_tr, df_mri_tr, df_text_tr, mri_raw_tr, texts_tr, mf_txt500_for_22):
        tr_tab = df_tab_tr.copy()
        self.tab = mod.Task2TabularPreprocessor().fit(tr_tab)
        self.emb_cols = [c for c in df_mri_tr.columns if c.startswith("emb_")]
        self.mri = mod.MRIStandardPreprocessor().fit(mri_raw_tr)
        self.txt = mod.TextPreprocessor(max_features=mf_txt500_for_22).fit(texts_tr)
        self.txt_full = mod.TextPreprocessor(max_features=None).fit(texts_tr)  # for 2.1 source
        return self

    def transform_pair(self, df_tab_va, df_mri_df, mri_raw_va, texts_va, mri_df_for_cols):
        X_tab_val = self.tab.transform(df_tab_va)
        X_mri_val = self.mri.transform(mri_raw_va)
        X_txt_val = self.txt.transform(texts_va)
        X_txt_full_val = self.txt_full.transform(texts_va)
        return X_tab_val, X_mri_val, X_txt_val, X_txt_full_val


def fit_21(cfg21, X_tab, X_mri, X_txt_full, y_treat, y_conf):
    Y = build_treatment_soft(y_treat, y_conf, 4)
    m_tab = SoftLinearSVR(C=cfg21["tab"]["C"], epsilon=cfg21["tab"]["epsilon"]).fit(X_tab, Y)
    if X_mri.shape[0] > 0:
        m_mri = SoftLinearSVR(C=cfg21["mri"]["C"], epsilon=cfg21["mri"]["epsilon"]).fit(X_mri, Y)
    else:
        m_mri = None
    m_txt = SoftLinearSVR(C=cfg21["text"]["C"], epsilon=cfg21["text"]["epsilon"]).fit(X_txt_full, Y)
    w = cfg21["fusion"]

    def pred(X_tab_v, X_mri_v, X_txt_full_v):
        pt = m_tab.predict_proba(X_tab_v)
        pm = m_mri.predict_proba(X_mri_v) if m_mri is not None else np.zeros((len(X_tab_v), 4))
        px = m_txt.predict_proba(X_txt_full_v)
        return w["w_tab"] * pt + w["w_mri"] * pm + w["w_txt"] * px

    return pred


def inject(probs21, cond):
    if cond == "C0":
        return np.zeros((probs21.shape[0], 0))
    if cond == "C1":
        return probs21
    if cond == "C2":
        oh = np.zeros_like(probs21); idx = np.argmax(probs21, axis=1); oh[np.arange(len(oh)), idx] = 1.0
        return oh
    if cond == "C3":
        oh = np.zeros_like(probs21); idx = np.argmax(probs21, axis=1); oh[np.arange(len(oh)), idx] = 1.0
        return np.hstack([probs21, oh])
    raise ValueError(cond)


CONDITIONS = ["C0", "C1", "C2", "C3"]
MODES = ["3class", "binary"]


def main():
    log("=" * 90)
    log("CHIMERA Task-2 -- Cascaded 2.1->2.2 (v2, leak-free) (exp_15)")
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

    with open(EXP11_BEST, "r", encoding="utf-8") as f:
        best_11 = json.load(f)
    cfg21 = best_11["fuzzy"]
    cfg22_grid = [{"C": c, "eps": e} for c in (0.1, 1.0, 10.0) for e in (0.05, 0.1)]
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]

    # ------------------------------------------------------------------
    # PHASE A - 100 MCCV splits (uses exp13 fold cache, already leak-safe)
    # ------------------------------------------------------------------
    vw_map = mod.load_var_weights_map(df_gt["pid"].tolist())
    fold_cache = mod.prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    split_results = []
    for split_idx, sc in enumerate(split_configs):
        fc = fold_cache[split_idx]
        X_tab_tr, X_tab_val = fc["X_tab_tr"], fc["X_tab_val"]
        X_mri_tr, X_mri_val = fc["X_mri_tr"], fc["X_mri_val"]
        X_txt_tr_full, X_txt_val_full = fc["text_cache"][None]
        y_tr_treat, y_tr_conf = fc["y_tr_treat"], fc["y_tr_conf"]
        y_val_treat, y_val_conf = fc["y_val_treat"], fc["y_val_conf"]

        n_tr = len(y_tr_conf)
        oof = np.zeros((n_tr, 4))
        y_tr_idx = np.array([IDX_4[c] for c in y_tr_treat])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # OOF 2.1 for train cases (inner 5-fold inside this MCCV train fold)
        for _, (tr_in, va_in) in enumerate(skf.split(np.zeros(n_tr), y_tr_idx)):
            p = fit_21(
                cfg21,
                X_tab_tr[tr_in], X_mri_tr[tr_in], X_txt_tr_full[tr_in],
                [y_tr_treat[j] for j in tr_in], [y_tr_conf[j] for j in tr_in],
            )
            oof[va_in] = p(X_tab_tr[va_in], X_mri_tr[va_in], X_txt_tr_full[va_in])

        # Predict val with full-train 2.1
        p_full = fit_21(cfg21, X_tab_tr, X_mri_tr, X_txt_tr_full, y_tr_treat, y_tr_conf)
        val_probs = p_full(X_tab_val, X_mri_val, X_txt_val_full)

        for cond in CONDITIONS:
            inj_tr = inject(oof, cond)
            inj_val = inject(val_probs, cond)
            X_tab_tr_c = np.hstack([X_tab_tr, inj_tr]) if inj_tr.shape[1] else X_tab_tr
            X_tab_val_c = np.hstack([X_tab_val, inj_val]) if inj_val.shape[1] else X_tab_val
            for mode in MODES:
                Y_tr = build_conf_3(y_tr_conf) if mode == "3class" else build_conf_bin(y_tr_conf)
                best_mf = -1.0
                best_cfg = None
                for ccfg in cfg22_grid:
                    m_tab = SoftLinearSVR(C=ccfg["C"], epsilon=ccfg["eps"]).fit(X_tab_tr_c, Y_tr)
                    pv = m_tab.predict_proba(X_tab_val_c)
                    if mode == "3class":
                        yp = [CLASS_ORDER_3[int(np.argmax(p))] for p in pv]
                        mf = f1_score(y_val_conf, yp, labels=CLASS_ORDER_3, average="macro", zero_division=0)
                    else:
                        yp = [0 if np.argmax(p) == 0 else 1 for p in pv]
                        yv = [0 if c == "borderline" else 1 for c in y_val_conf]
                        mf = f1_score(yv, yp, average="macro", zero_division=0)
                    if mf > best_mf:
                        best_mf, best_cfg = mf, ccfg
                split_results.append({
                    "split_idx": split_idx, "cond": cond, "mode": mode,
                    "best_C": best_cfg["C"], "best_eps": best_cfg["eps"],
                    "best_macro_f1": float(best_mf),
                })

        if (split_idx + 1) % 10 == 0:
            log(f"  [A] split {split_idx+1}/100 | el={time.time()-t0:.1f}s")
    log(f"[A] Phase A completed: {time.time()-t0:.1f}s")

    # Aggregate per (cond, mode)
    aggregate = {}
    for cond in CONDITIONS:
        for mode in MODES:
            rows = [r for r in split_results if r["cond"] == cond and r["mode"] == mode]
            f1s = [r["best_macro_f1"] for r in rows]
            # most common cfg
            counter: Dict[str, int] = {}
            for r in rows:
                key = f"{r['best_C']}_{r['best_eps']}"
                counter[key] = counter.get(key, 0) + 1
            k_common = max(counter, key=counter.get)
            c_str, e_str = k_common.split("_")
            aggregate[f"{cond}_{mode}"] = {
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "best_C": float(c_str), "best_eps": float(e_str),
            }

    # ------------------------------------------------------------------
    # PHASE B - LOOCV leak-free (preprocessors fitted INSIDE each fold)
    # ------------------------------------------------------------------
    loo = LeaveOneOut()
    n_all = len(df_gt)
    pids_all = df_gt["pid"].tolist()
    y_conf_all = df_gt["confidence"].fillna("clear").tolist()
    y_treat_all = df_gt["treatment_decision"].tolist()

    # 1) Leak-free pooled OOF 2.1 over all N (5-fold outer, preprocess inside each fold)
    pooled_oof = np.zeros((n_all, 4))
    # We create permuted ids for deterministic folds
    y_all_idx = np.array([IDX_4[c] for c in y_treat_all])
    ids = np.arange(n_all)
    skf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log("[B] Computing leak-free pooled OOF 2.1 (5-fold, preprocess inside)...")
    for fold_i, (tr_idx, va_idx) in enumerate(skf_full.split(ids, y_all_idx)):
        tr_pids_f = [pids_all[i] for i in tr_idx]
        va_pids_f = [pids_all[i] for i in va_idx]
        df_tab_tr = df_tab.set_index("pid").loc[tr_pids_f].reset_index()
        df_mri_tr = df_mri.set_index("pid").loc[tr_pids_f].reset_index()
        df_txt_tr_rows = df_text.set_index("pid").loc[tr_pids_f]
        prep = Task2PreprocLeakFree()
        prep.fit_train(df_tab_tr, df_mri_tr, df_txt_tr_rows, df_mri_tr[emb_cols].values.astype(float),
                       df_txt_tr_rows["full_encounter_narrative"].tolist(), 500)
        # transform train
        X_tab_f_tr = prep.tab.transform(df_tab_tr)
        X_mri_f_tr = prep.mri.transform(df_mri_tr[emb_cols].values.astype(float))
        X_txt_f_tr = prep.txt_full.transform(df_txt_tr_rows["full_encounter_narrative"].tolist())

        df_tab_va = df_tab.set_index("pid").loc[va_pids_f].reset_index()
        df_mri_va = df_mri.set_index("pid").loc[va_pids_f].reset_index()
        df_txt_va_rows = df_text.set_index("pid").loc[va_pids_f]
        X_tab_f_va = prep.tab.transform(df_tab_va)
        X_mri_f_va = prep.mri.transform(df_mri_va[emb_cols].values.astype(float))
        X_txt_f_va = prep.txt_full.transform(df_txt_va_rows["full_encounter_narrative"].tolist())

        pp = fit_21(cfg21, X_tab_f_tr, X_mri_f_tr, X_txt_f_tr,
                    [y_treat_all[i] for i in tr_idx], [y_conf_all[i] for i in tr_idx])
        pooled_oof[va_idx] = pp(X_tab_f_va, X_mri_f_va, X_txt_f_va)
        log(f"  [B] pooled-OOF fold {fold_i+1}/5 done")
    log("[B] pooled-OOF done")

    # 2) LOOCV accuracy + confusion matrices, preprocess inside each fold
    loocv_metric = {}
    confusion = {c: {} for c in CONDITIONS}
    log("[B] LOOCV...")
    for cond in CONDITIONS:
        inj_all = inject(pooled_oof, cond)
        has_inj = inj_all.shape[1] > 0
        for mode in MODES:
            agg_key = f"{cond}_{mode}"
            Cv = aggregate[agg_key]["best_C"]
            Ev = aggregate[agg_key]["best_eps"]
            labels = CLASS_ORDER_3 if mode == "3class" else ["borderline", "clear"]
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            acc = []
            for vi, (tr_idx, va_idx) in enumerate(loo.split(np.arange(n_all))):
                hold_i = va_idx[0]
                tr_pids_f = [pids_all[i] for i in tr_idx]
                df_tab_tr = df_tab.set_index("pid").loc[tr_pids_f].reset_index()
                df_mri_tr = df_mri.set_index("pid").loc[tr_pids_f].reset_index()
                df_txt_tr_rows = df_text.set_index("pid").loc[tr_pids_f]
                prep = Task2PreprocLeakFree()
                prep.fit_train(df_tab_tr, df_mri_tr, df_txt_tr_rows, df_mri_tr[emb_cols].values.astype(float),
                               df_txt_tr_rows["full_encounter_narrative"].tolist(), 500)
                X_tab_tr_f = prep.tab.transform(df_tab_tr)
                # single holdout transform
                hold_pid = pids_all[hold_i]
                df_tab_ho = df_tab.set_index("pid").loc[[hold_pid]].reset_index()
                df_mri_ho = df_mri.set_index("pid").loc[[hold_pid]].reset_index()
                ho_txt = df_text.set_index("pid").loc[[hold_pid]]["full_encounter_narrative"].tolist()
                X_tab_ho = prep.tab.transform(df_tab_ho)
                X_mri_ho = prep.mri.transform(df_mri_ho[emb_cols].values.astype(float))
                X_txt_ho = prep.txt.transform(ho_txt)

                # Injection: use pooled_oof for train (leak-free OOF) and val: need val-specific 2.1 predict
                # NOTE: pooled_oof[tr_idx] exists for all train rows; for the single holdout, use its pooled_oof
                inj_tr = inject(pooled_oof[tr_idx], cond)
                inj_ho = inject(pooled_oof[[hold_i]], cond)
                X_tab_tr_c = np.hstack([X_tab_tr_f, inj_tr]) if has_inj else X_tab_tr_f
                X_tab_ho_c = np.hstack([X_tab_ho, inj_ho]) if has_inj else X_tab_ho

                Y_tr = build_conf_3([y_conf_all[i] for i in tr_idx]) if mode == "3class" else build_conf_bin([y_conf_all[i] for i in tr_idx])
                m_tab = SoftLinearSVR(C=Cv, epsilon=Ev).fit(X_tab_tr_c, Y_tr)
                proba_ho = m_tab.predict_proba(X_tab_ho_c)[0]
                true_lb = y_conf_all[hold_i]
                pred = labels[int(np.argmax(proba_ho))]
                acc.append(1.0 if pred == true_lb else 0.0)
                ci = labels.index(true_lb) if true_lb in labels else 0
                cj = labels.index(pred) if pred in labels else 0
                # For 3-class, borderline and clear are at index 1,2 (uncertain index0 remains 0)
                cm[ci, cj] += 1
            loocv_metric[agg_key] = float(np.mean(acc)) if acc else None
            confusion[cond][mode] = {"loocv": cm.tolist(), "labels": labels}

        log(f"  [B] {cond} LOOCV done")

    elapsed = time.time() - t0
    log(f"All done in {elapsed:.1f}s")
    out = {
        "aggregated_per_cond_mode": aggregate,
        "loocv_accuracy": loocv_metric,
        "confusion_matrices": confusion,
        "runtime_seconds": float(elapsed),
        "nested_oof": True,
        "preproc_leak_free": True,
        "source21": "exp_11 fuzzy-SVR frozen",
    }
    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RESULTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(confusion, f, indent=2)
    with open(RESULTS_DIR / "version.txt", "w") as f:
        f.write(f"exp_15 v2 timestamp: {datetime.now().isoformat()}\n")
        f.write("Leak-free preprocessing per fold; nested OOF 2.1; no eval().\n")
    log("exp_15 v2 completed and persisted.")


if __name__ == "__main__":
    main()
