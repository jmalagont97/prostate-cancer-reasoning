#!/usr/bin/env python3
"""Persist the binary (borderline vs clear) Subtask 2.2 SVM bundle for inference."""

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
EXP13 = PROJECT_ROOT / "experiments" / "exp_13" / "scripts" / "run_exp13_mccv.py"
BIN_RESULTS = PROJECT_ROOT / "experiments" / "exp_13" / "results" / "binary_22"
MODELS_DIR = PROJECT_ROOT / "experiments" / "exp_13" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("exp13_mod", str(EXP13))
mod = importlib.util.module_from_spec(spec)
sys.modules["exp13_mod"] = mod
spec.loader.exec_module(mod)

DATA_DIR = mod.DATA_DIR


def build_soft_binary(y_bin: List[int]) -> np.ndarray:
    Y = np.zeros((len(y_bin), 2))
    for i, c in enumerate(y_bin):
        Y[i, c] = 1.0
    return Y


def main() -> None:
    print("=" * 80)
    print("Saving exp_13 binary Subtask 2.2 SVM bundle (retrained on all N=70)")
    print("=" * 80)

    with open(BIN_RESULTS / "aggregate_metrics.json", "r", encoding="utf-8") as f:
        family_best = json.load(f)["family_best"]

    df_gt = pd.read_csv(DATA_DIR / "ground_truth_task2.csv")
    df_tab = pd.read_csv(DATA_DIR / "clinical_tabular_task2.csv")
    df_mri = pd.read_csv(DATA_DIR / "mri_embeddings_task2.csv")
    df_text = pd.read_csv(DATA_DIR / "clinical_texts_task2.csv")

    y_bin = [1 if c == "clear" else 0 for c in df_gt["confidence"].fillna("clear").tolist()]
    emb_cols = [c for c in df_mri.columns if c.startswith("emb_")]
    mri_raw = df_mri[emb_cols].values.astype(float)
    texts = df_text["full_encounter_narrative"].tolist()

    prep_tab = mod.Task2TabularPreprocessor().fit(df_tab)
    X_tab_all = prep_tab.transform(df_tab)
    prep_mri = mod.MRIStandardPreprocessor().fit(mri_raw)
    X_mri_all = prep_mri.transform(mri_raw)

    bundles: Dict[str, Any] = {}
    for family in ("hard", "soft"):
        mf = family_best[f"{family}_text"].get("mf", 500)
        key = mf if mf in (500, None) else 500
        prep_txt = mod.TextPreprocessor(max_features=key).fit(texts)
        X_txt_all = prep_txt.transform(texts)

        if family == "hard":
            y_idx = np.array(y_bin)
            m_tab = mod.HardLinearSVM(C=family_best["hard_tab"]["C"], class_weight=family_best["hard_tab"].get("cw"), n_classes=2).fit(X_tab_all, y_idx)
            m_mri = mod.HardLinearSVM(C=family_best["hard_mri"]["C"], class_weight=family_best["hard_mri"].get("cw"), n_classes=2).fit(X_mri_all, y_idx)
            m_txt = mod.HardLinearSVM(C=family_best["hard_text"]["C"], class_weight=family_best["hard_text"].get("cw"), n_classes=2).fit(X_txt_all, y_idx)
        else:
            Y_tr = build_soft_binary(y_bin)
            m_tab = mod.SoftLinearSVR(C=family_best["soft_tab"]["C"], epsilon=family_best["soft_tab"]["eps"]).fit(X_tab_all, Y_tr)
            m_mri = mod.SoftLinearSVR(C=family_best["soft_mri"]["C"], epsilon=family_best["soft_mri"]["eps"]).fit(X_mri_all, Y_tr)
            m_txt = mod.SoftLinearSVR(C=family_best["soft_text"]["C"], epsilon=family_best["soft_text"]["eps"]).fit(X_txt_all, Y_tr)

        w = family_best[f"{family}_fusion"]
        bundles[family] = {
            "models": {"tab": m_tab, "mri": m_mri, "text": m_txt},
            "fusion_weights": {"w_tab": w["w_tab"], "w_mri": w["w_mri"], "w_txt": w["w_txt"]},
        }

    full_bundle = {
        "bundles": bundles,
        "preprocessors": {"tab": prep_tab, "mri": prep_mri,
                           "text_500": None, "text_none": None},
        # text preprocessors are family-specific; store them inside bundles
        "class_map": {"borderline": 0, "clear": 1},
        "class_order": ["borderline", "clear"],
        "experiment": "exp_13 binary 2.2",
        "created_at": datetime.now().isoformat(),
    }
    # Store text preprocessors per family
    full_bundle["bundles"]["hard"]["text_preprocessor"] = mod.TextPreprocessor(max_features=500).fit(texts)
    full_bundle["bundles"]["soft"]["text_preprocessor"] = mod.TextPreprocessor(max_features=500).fit(texts)

    pkl_path = MODELS_DIR / "subtask2_confidence_binary_svm.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(full_bundle, f)
    print(f"  Saved: {pkl_path} ({pkl_path.stat().st_size} bytes)")

    with open(MODELS_DIR / "subtask2_binary_fusion_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "hard": bundles["hard"]["fusion_weights"],
            "soft": bundles["soft"]["fusion_weights"],
        }, f, indent=2)
    print("  Saved: subtask2_binary_fusion_weights.json")
    print("\nBinary 2.2 SVM bundle persisted successfully.")


if __name__ == "__main__":
    main()
