"""Backfill macro-F1 into exp_2's already-completed results, per this project's cross-experiment
macro-F1 reporting initiative, extended back to exp_1-exp_5 on 2026-08-13.

Reproduces each condition's exact model/feature-frame configuration from
experiments/exp_2/scripts/run_{decision,confidence,weights,reveal}.py (unchanged, no edits) to
recompute out-of-fold predictions and derive macro-F1, merged into the existing metrics.json via
read-modify-write. Same definitions and same per-repeat CV discipline note as
experiments/exp_1/scripts/backfill_macro_f1.py.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_2/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_official_feature_frame
from chimera_task1.reasoning_labels import (
    CONFIDENCE_RANK,
    REVEAL_SECTIONS,
    TASK1_FACTORS,
    WEIGHT_RANK,
    parse_reveal_sequences,
    weight_col,
)
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import RANDOM_STATE, load_labeled_data
from chimera_task1.train_reasoning import load_annotated, make_classifier, repeated_out_of_fold_predict

N_SPLITS = 5
N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]


def merge_metrics(condition: str, new_fields: dict) -> None:
    path = RESULTS_DIR / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


def decision_sklearn_macro_f1(feature_frame, y, clf) -> tuple[float, float]:
    preprocessor = build_preprocessor(feature_frame)
    per_repeat = []
    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in skf.split(feature_frame, y):
            pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
            pipe.fit(feature_frame.iloc[train_idx], y[train_idx])
            preds[test_idx] = pipe.predict(feature_frame.iloc[test_idx])
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1], zero_division=0))
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def kdm_decision_macro_f1(feature_frame, y) -> tuple[float, float]:
    preprocessor = build_preprocessor(feature_frame)
    X_pre = preprocessor.fit_transform(feature_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)
    per_repeat = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in kf.split(X):
            test_probs = fit_predict_kdm(X[train_idx], y[train_idx], X[test_idx], n_classes=2)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1], zero_division=0))
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def kdm_confidence_macro_f1(feature_frame, y_rank) -> tuple[float, float]:
    preprocessor = build_preprocessor(feature_frame)
    X_pre = preprocessor.fit_transform(feature_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)
    per_repeat = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y_rank), dtype=int)
        for train_idx, test_idx in kf.split(X):
            test_probs = fit_predict_kdm(X[train_idx], y_rank[train_idx], X[test_idx], n_classes=3)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def run_decision() -> None:
    _, inp, df = load_labeled_data()
    y = (df["target_biopsy_decision"] == "yes").astype(int).values
    for treatment in ("count", "flags"):
        X = select_official_feature_frame(inp, comorbidity_treatment=treatment)
        logistic = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
        hgb = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=1.0, max_iter=100, class_weight="balanced",
        )
        m, s = decision_sklearn_macro_f1(X, y, logistic)
        merge_metrics(f"decision_logistic_{treatment}", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})
        m, s = decision_sklearn_macro_f1(X, y, hgb)
        merge_metrics(f"decision_hgb_{treatment}", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})
        m, s = kdm_decision_macro_f1(X, y)
        merge_metrics(f"decision_kdm_{treatment}", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})


def run_confidence() -> None:
    ann, inp_ann = load_annotated()
    y_labels = ann["target_confidence"].values
    y_rank = np.array([CONFIDENCE_RANK[label] for label in y_labels])
    for treatment in ("count", "flags"):
        X = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)
        preprocessor = build_preprocessor(X)

        per_repeat = []
        for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
            pred_rank = np.array([CONFIDENCE_RANK[p] for p in preds])
            per_repeat.append(f1_score(y_rank, pred_rank, average="macro", labels=[0, 1, 2], zero_division=0))
        merge_metrics(f"confidence_logistic_{treatment}", {
            "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
        })

        m, s = kdm_confidence_macro_f1(X, y_rank)
        merge_metrics(f"confidence_kdm_{treatment}", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})


def eval_factor_macro_f1(X, y_labels, preprocessor) -> float:
    y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
    per_repeat = []
    for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
        pred_rank = np.array([WEIGHT_RANK[p] for p in preds])
        per_repeat.append(f1_score(y_rank, pred_rank, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    return float(np.mean(per_repeat))


def run_weights() -> None:
    ann, inp_ann = load_annotated()
    for treatment in ("count", "flags"):
        full_frame = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)
        preprocessor_official = build_preprocessor(full_frame)

        per_factor_official = {}
        for factor in IN_SCOPE_FACTORS:
            y_labels = ann[weight_col(factor)].values
            per_factor_official[factor] = round(eval_factor_macro_f1(full_frame, y_labels, preprocessor_official), 3)
        merge_metrics(f"weights_official_{treatment}", {
            "mean_macro_f1": round(float(np.mean(list(per_factor_official.values()))), 3),
            "per_factor_macro_f1": per_factor_official,
        })

        per_factor_restricted = {}
        for factor in IN_SCOPE_FACTORS:
            cols = restricted_feature_group(factor, treatment)
            X = full_frame[cols]
            preprocessor = build_preprocessor(X)
            y_labels = ann[weight_col(factor)].values
            per_factor_restricted[factor] = round(eval_factor_macro_f1(X, y_labels, preprocessor), 3)
        merge_metrics(f"weights_restricted_{treatment}", {
            "mean_macro_f1": round(float(np.mean(list(per_factor_restricted.values()))), 3),
            "per_factor_macro_f1": per_factor_restricted,
        })


def section_f1(true_col, pred_col) -> float:
    tp = int(np.sum((true_col == 1) & (pred_col == 1)))
    fp = int(np.sum((true_col == 0) & (pred_col == 1)))
    fn = int(np.sum((true_col == 1) & (pred_col == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def run_reveal() -> None:
    ann, inp_ann = load_annotated()
    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])

    for treatment in ("count", "flags"):
        X = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)
        preprocessor = build_preprocessor(X)

        per_repeat_macro_f1 = []
        per_section_f1s = {s: [] for s in sections}
        for repeat in range(N_REPEATS):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
            preds = np.zeros_like(Y)
            for train_idx, test_idx in kf.split(X):
                pipe = Pipeline([("prep", preprocessor), ("clf", MultiOutputClassifier(make_classifier(), n_jobs=-1))])
                pipe.fit(X.iloc[train_idx], Y[train_idx])
                preds[test_idx] = pipe.predict(X.iloc[test_idx])
            this_repeat = []
            for i, s in enumerate(sections):
                f1 = section_f1(Y[:, i], preds[:, i])
                per_section_f1s[s].append(f1)
                this_repeat.append(f1)
            per_repeat_macro_f1.append(float(np.mean(this_repeat)))

        per_section_mean = {s: round(float(np.mean(v)), 3) for s, v in per_section_f1s.items()}
        merge_metrics(f"reveal_{treatment}", {
            "macro_f1_mean": round(float(np.mean(per_repeat_macro_f1)), 3),
            "macro_f1_std": round(float(np.std(per_repeat_macro_f1)), 3),
            "per_section_macro_f1": per_section_mean,
            "macro_f1_definition": "per-section binary F1 (revealed vs. not), macro-averaged across the modeled sections",
        })


def main() -> None:
    print("=== decision ===")
    run_decision()
    print("\n=== confidence ===")
    run_confidence()
    print("\n=== weights ===")
    run_weights()
    print("\n=== reveal ===")
    run_reveal()


if __name__ == "__main__":
    main()
