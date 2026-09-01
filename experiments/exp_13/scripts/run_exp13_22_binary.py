#!/usr/bin/env python3
"""
Binary confidence (borderline vs clear) benchmark for Task-2 Subtask 2.2 (exp_13-bin).

Drops the phantom 'uncertain' class (n=0), training exactly two classes:
  borderline (0) vs clear (1)

Method: linear-SVM (exp_13 style) on the three modalities with grids + late fusion + LOOCV.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP13 = PROJECT_ROOT / "experiments" / "exp_13" / "scripts" / "run_exp13_mccv.py"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp_13" / "results"
BIN_RESULTS = RESULTS_DIR / "binary_22"
BIN_RESULTS.mkdir(parents=True, exist_ok=True)
LOG_FILE = BIN_RESULTS / "execution.log"

# Load exp_13 module without executing main()
spec = importlib.util.spec_from_file_location("exp13_mod", str(EXP13))
mod = importlib.util.module_from_spec(spec)
sys.modules["exp13_mod"] = mod
spec.loader.exec_module(mod)

DATA_DIR = mod.DATA_DIR
log = mod.log


def main() -> None:
    log("=" * 90)
    log("Task-2 Subtask 2.2 BINARY (borderline vs clear) — linear-SVM")
    log("=" * 90)
    t0 = time.time()

    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")
    _, split_configs = mod.load_task2_splits()
    # We do not need variable weights; reuse exp_13 prepare_fold_cache via a dependency injection
    import json as _json
    raw_dir = mod.RAW_DIR
    vw_map = mod.load_var_weights_map(df_gt["pid"].tolist())
    fold_cache = mod.prepare_fold_cache(df_tab, df_mri, df_text, df_gt, split_configs, vw_map)

    # Binary target per fold
    for fc in fold_cache:
        fc["y_tr_bin"] = [1 if c == "clear" else 0 for c in fc["y_tr_conf"]]
        fc["y_val_bin"] = [1 if c == "clear" else 0 for c in fc["y_val_conf"]]

    grids = {
        "tab": {"hard": [{"C": C, "cw": cw} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced")],
                 "soft": [{"C": C, "eps": e} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1)]},
        "mri": {"hard": [{"C": C, "cw": cw} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced")],
                 "soft": [{"C": C, "eps": e} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1)]},
        "text": {"hard": [{"C": C, "cw": cw, "mf": mf} for C in (0.1, 1.0, 10.0) for cw in (None, "balanced") for mf in (500, None)],
                  "soft": [{"C": C, "eps": e, "mf": mf} for C in (0.1, 1.0, 10.0) for e in (0.05, 0.1) for mf in (500, None)]},
    }

    def build_soft_binary(y_bin):
        Y = np.zeros((len(y_bin), 2))
        for i, c in enumerate(y_bin):
            Y[i, c] = 1.0  # 0 -> [1,0], 1 -> [0,1]
        return Y

    best_cfg: Dict[str, Any] = {}
    probs_cache: Dict[str, Dict[str, List[np.ndarray]]] = {"hard": {}, "soft": {}}

    for family in ("hard", "soft"):
        for modality in ("tab", "mri", "text"):
            best_f1 = -1.0
            best_c = None
            best_probs = None
            for cand in grids[modality][family]:
                split_f1s = []
                split_accs = []
                split_probs = []
                for fc in fold_cache:
                    if len(set(fc["y_tr_bin"])) < 2 or len(set(fc["y_val_bin"])) < 2:
                        continue
                    if modality == "tab":
                        X_tr, X_val = fc["X_tab_tr"], fc["X_tab_val"]
                    elif modality == "mri":
                        X_tr, X_val = fc["X_mri_tr"], fc["X_mri_val"]
                    else:
                        key = cand.get("mf", 500) if cand.get("mf") in fc["text_cache"] else 500
                        X_tr, X_val = fc["text_cache"][key]
                    if family == "hard":
                        y_idx = np.array(fc["y_tr_bin"])
                        mdl = mod.HardLinearSVM(C=cand["C"], class_weight=cand["cw"], n_classes=2).fit(X_tr, y_idx)
                    else:
                        Y_tr = build_soft_binary(fc["y_tr_bin"])
                        mdl = mod.SoftLinearSVR(C=cand["C"], epsilon=cand["eps"]).fit(X_tr, Y_tr)
                    probs = mdl.predict_proba(X_val)
                    y_pred = [int(np.argmax(p)) for p in probs]
                    split_f1s.append(f1_score(fc["y_val_bin"], y_pred, average="macro", zero_division=0))
                    split_accs.append(float(np.mean(np.array(y_pred) == np.array(fc["y_val_bin"]))))
                    split_probs.append(probs)
                if not split_f1s:
                    continue
                mean_f1 = float(np.mean(split_f1s))
                if mean_f1 > best_f1:
                    best_f1, best_c, best_probs = mean_f1, cand, split_probs
            best_cfg[f"{family}_{modality}"] = best_c
            probs_cache[family][modality] = best_probs
            log(f"  [bin] Best {family}/{modality}: {best_c} | Macro-F1={best_f1:.4f}")

    # Fusion per family
    step = 0.05
    w_cands = []
    for w1_i in range(int(1.0 / step) + 1):
        for w2_i in range(int(1.0 / step) + 1 - w1_i):
            w1 = round(w1_i * step, 2)
            w2 = round(w2_i * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= -1e-6:
                w_cands.append((w1, w2, max(0.0, w3)))
    for family in ("hard", "soft"):
        best_w = None
        best_f1w = -1.0
        for (wt, wm, wx) in w_cands:
            fs = []
            for i, fc in enumerate(fold_cache):
                if len(set(fc["y_val_bin"])) < 2:
                    continue
                try:
                    fused = wt * probs_cache[family]["tab"][i] + wm * probs_cache[family]["mri"][i] + wx * probs_cache[family]["text"][i]
                except (TypeError, IndexError):
                    continue
                y_pred = [int(np.argmax(p)) for p in fused]
                fs.append(f1_score(fc["y_val_bin"], y_pred, average="macro", zero_division=0))
            if not fs:
                continue
            mean_f = float(np.mean(fs))
            if mean_f > best_f1w:
                best_f1w, best_w = mean_f, (wt, wm, wx)
        best_cfg[f"{family}_fusion"] = {"w_tab": best_w[0], "w_mri": best_w[1], "w_txt": best_w[2], "mean_macro_f1": best_f1w}
        log(f"  [bin] {family} fusion: w={best_w} | Macro-F1={best_f1w:.4f}")

    # LOOCV with frozen best fusion (full 3 modalities)
    pids = df_gt["pid"].tolist()
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    loocv = {"hard": [], "soft": []}
    for hold_out in pids:
        tr_pids = [p for p in pids if p != hold_out]
        gt_tr = df_gt.set_index("pid").loc[tr_pids]
        gt_val = df_gt.set_index("pid").loc[[hold_out]]
        y_tr_bin = [1 if c == "clear" else 0 for c in gt_tr["confidence"].fillna("clear").tolist()]
        y_hold_bin = 1 if gt_val["confidence"].iloc[0] == "clear" else 0
        if len(set(y_tr_bin)) < 2:
            continue
        tr_tab = df_tab.set_index("pid").loc[tr_pids].reset_index()
        val_tab = df_tab.set_index("pid").loc[[hold_out]].reset_index()
        prep_tab = mod.Task2TabularPreprocessor().fit(tr_tab)
        X_tab_tr = prep_tab.transform(tr_tab); X_tab_val = prep_tab.transform(val_tab)
        X_mri_tr = df_mri.set_index("pid").loc[tr_pids][emb_cols].values.astype(float)
        X_mri_val = df_mri.set_index("pid").loc[[hold_out]][emb_cols].values.astype(float)
        prep_mri = mod.MRIStandardPreprocessor().fit(X_mri_tr)
        X_mri_tr_p = prep_mri.transform(X_mri_tr); X_mri_val_p = prep_mri.transform(X_mri_val)
        tr_txt = df_text.set_index("pid").loc[tr_pids]["full_encounter_narrative"].tolist()
        val_txt = df_text.set_index("pid").loc[[hold_out]]["full_encounter_narrative"].tolist()

        for family in ("hard", "soft"):
            mf = best_cfg[f"{family}_text"].get("mf", 500)
            key = mf if mf in (500, None) else 500
            prep_txt = mod.TextPreprocessor(max_features=key).fit(tr_txt)
            X_ttr = prep_txt.transform(tr_txt); X_tva = prep_txt.transform(val_txt)
            if family == "hard":
                y_idx = np.array(y_tr_bin)
                m_tab = mod.HardLinearSVM(C=best_cfg["hard_tab"]["C"], class_weight=best_cfg["hard_tab"].get("cw"), n_classes=2).fit(X_tab_tr, y_idx)
                m_mri = mod.HardLinearSVM(C=best_cfg["hard_mri"]["C"], class_weight=best_cfg["hard_mri"].get("cw"), n_classes=2).fit(X_mri_tr_p, y_idx)
                m_txt = mod.HardLinearSVM(C=best_cfg["hard_text"]["C"], class_weight=best_cfg["hard_text"].get("cw"), n_classes=2).fit(X_ttr, y_idx)
            else:
                Y_tr = build_soft_binary(y_tr_bin)
                m_tab = mod.SoftLinearSVR(C=best_cfg["soft_tab"]["C"], epsilon=best_cfg["soft_tab"]["eps"]).fit(X_tab_tr, Y_tr)
                m_mri = mod.SoftLinearSVR(C=best_cfg["soft_mri"]["C"], epsilon=best_cfg["soft_mri"]["eps"]).fit(X_mri_tr_p, Y_tr)
                m_txt = mod.SoftLinearSVR(C=best_cfg["soft_text"]["C"], epsilon=best_cfg["soft_text"]["eps"]).fit(X_ttr, Y_tr)
            pt = m_tab.predict_proba(X_tab_val); pm = m_mri.predict_proba(X_mri_val_p); px = m_txt.predict_proba(X_tva)
            w = best_cfg[f"{family}_fusion"]
            fused = w["w_tab"]*pt + w["w_mri"]*pm + w["w_txt"]*px
            pred = int(np.argmax(fused[0]))
            loocv[family].append(1.0 if pred == y_hold_bin else 0.0)

    loocv_agg = {fam: float(np.mean(v)) if v else None for fam, v in loocv.items()}
    log(f"  [bin] LOOCV accuracy: {loocv_agg}")

    hours = (time.time() - t0)
    summary = {
        "family_best": best_cfg,
        "loocv_accuracy": loocv_agg,
        "runtime_seconds": float(hours),
    }
    with open(BIN_RESULTS / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(BIN_RESULTS / "version.txt", "w", encoding="utf-8") as f:
        f.write(f"exp_13 binary 2.2 timestamp: {datetime.now().isoformat()}\n")
        f.write("Binary borderline-vs-clear SVM benchmark (uncertain dropped).\n")
    log(f"Binary 2.2 completed in {hours:.1f}s and persisted.")


if __name__ == "__main__":
    main()
