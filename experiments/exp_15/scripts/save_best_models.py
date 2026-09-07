#!/usr/bin/env python3
"""Persist the best exp_15 cascade model (C3: 2.1 probability vector + argmax one-hot) as pickle."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP13_RUNNER = PROJECT_ROOT / "experiments" / "exp_13" / "scripts" / "run_exp13_mccv.py"
EXP11_BEST = PROJECT_ROOT / "experiments" / "exp_11" / "results" / "best_hyperparams.json"
MODELS_DIR = PROJECT_ROOT / "experiments" / "exp_15" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("exp13_mod", str(EXP13_RUNNER))
mod = importlib.util.module_from_spec(spec)
sys.modules["exp13_mod"] = mod
spec.loader.exec_module(mod)

SoftLinearSVR = mod.SoftLinearSVR
Task2TabularPreprocessor = mod.Task2TabularPreprocessor
MRIStandardPreprocessor = mod.MRIStandardPreprocessor
TextPreprocessor = mod.TextPreprocessor
build_treatment_soft = mod.build_treatment_soft
DATA_DIR = mod.DATA_DIR
CLASS_ORDER_4 = mod.CLASS_ORDER_4
CLASS_ORDER_3 = mod.CLASS_ORDER_3
IDX_4 = mod.IDX_4
CONF_ORDINAL = mod.CONF_ORDINAL

CONF3_ORDER = {"uncertain": 0, "borderline": 1, "clear": 2}


def build_conf_3(y_conf: List[str]) -> np.ndarray:
    Y = np.zeros((len(y_conf), 3))
    for i, c in enumerate(y_conf):
        Y[i, CONF3_ORDER[c]] = CONF_ORDINAL.get(c, 0.5)
    return Y


def main() -> None:
    print("=" * 80)
    print("Saving best exp_15 cascade model (C3: probs + argmax one-hot) as pickle")
    print("=" * 80)

    with open(EXP11_BEST, "r", encoding="utf-8") as f:
        best_11 = json.load(f)
    cfg21 = best_11["fuzzy"]

    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")

    y_treat = df_gt["treatment_decision"].tolist()
    y_conf = df_gt["confidence"].fillna("clear").tolist()
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]

    # Preprocess ALL N=70
    prep_tab = Task2TabularPreprocessor().fit(df_tab)
    X_tab_all = prep_tab.transform(df_tab)
    prep_mri = MRIStandardPreprocessor().fit(df_mri[emb_cols].values.astype(float))
    X_mri_all = prep_mri.transform(df_mri[emb_cols].values.astype(float))
    txt_all = df_text["full_encounter_narrative"].tolist()
    prep_txt = TextPreprocessor(max_features=None).fit(txt_all)
    X_txt_all = prep_txt.transform(txt_all)

    # Fit the 2.1 source (fuzzy-SVR) on ALL cases for inference-time injection
    Y21 = build_treatment_soft(y_treat, y_conf, 4)
    src_tab = SoftLinearSVR(C=cfg21["tab"]["C"], epsilon=cfg21["tab"]["epsilon"]).fit(X_tab_all, Y21)
    src_mri = SoftLinearSVR(C=cfg21["mri"]["C"], epsilon=cfg21["mri"]["epsilon"]).fit(X_mri_all, Y21)
    src_txt = SoftLinearSVR(C=cfg21["text"]["C"], epsilon=cfg21["text"]["epsilon"]).fit(X_txt_all, Y21)
    w21 = cfg21["fusion"]

    # Build C3 injection: probs + argmax-onehot (8 dims)
    pt = src_tab.predict_proba(X_tab_all)
    pm = src_mri.predict_proba(X_mri_all)
    px = src_txt.predict_proba(X_txt_all)
    probs21 = w21["w_tab"] * pt + w21["w_mri"] * pm + w21["w_txt"] * px
    oh = np.zeros_like(probs21)
    idx = np.argmax(probs21, axis=1)
    oh[np.arange(len(oh)), idx] = 1.0
    inj_all = np.hstack([probs21, oh])

    X_tab_c3 = np.hstack([X_tab_all, inj_all])

    # Best 2.2 config from aggregate: C=0.1, eps=0.05 (3-class)
    Y_conf_target = build_conf_3(y_conf)
    model_22 = SoftLinearSVR(C=0.1, epsilon=0.05).fit(X_tab_c3, Y_conf_target)

    bundle = {
        "experiment": "exp_15",
        "condition": "C3_both_probs_argmax",
        "created_at": datetime.now().isoformat(),
        "preprocessors": {"tab": prep_tab, "mri": prep_mri, "text": prep_txt},
        "source_21": {
            "models": {"tab": src_tab, "mri": src_mri, "text": src_txt},
            "fusion_weights": {"w_tab": w21["w_tab"], "w_mri": w21["w_mri"], "w_txt": w21["w_txt"]},
        },
        "model_22": model_22,
        "model_22_scope": "tabular_branch_only",
        "injection": {"type": "C3", "dims": 8, "order": "[probs21(4), argmax_onehot(4)]"},
        "best_cfg_22": {"C": 0.1, "eps": 0.05},
        "class_order_3": CLASS_ORDER_3,
        "class_order_21": CLASS_ORDER_4,
    }

    pkl_path = MODELS_DIR / "subtask2_confidence_cascade_svm.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved: {pkl_path} ({pkl_path.stat().st_size} bytes)")
    print("exp_15 C3 cascade pickle persisted successfully.")


if __name__ == "__main__":
    main()
