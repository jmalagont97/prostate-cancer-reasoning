"""Backfill macro-F1 into exp_3's already-completed results, per this project's cross-experiment
macro-F1 reporting initiative, extended back to exp_1-exp_5 on 2026-08-13.

Reproduces each condition's exact model/feature-frame configuration from
experiments/exp_3/scripts/run_{decision,confidence,weights,reveal}.py (unchanged, no edits),
reusing cv_utils.repeated_cv_proba (already generic over n_classes) directly rather than
reimplementing a CV loop -- so decision (2-class) and confidence (3-class) both get bit-identical
fold membership to the original run this time (unlike exp_1/exp_2's backfill, which had to
approximate `RepeatedStratifiedKFold`/plain-loop CV with an explicit per-repeat loop).

Structured as reusable functions so experiments/exp_4/scripts/backfill_macro_f1.py can import and
reuse them with exp_4's own (no-MRI) feature frame, exactly as exp_4's own run_*.py scripts already
import exp_3's cv_utils.py/models.py unchanged.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_3/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cv_utils import N_SPLITS, RANDOM_STATE, repeated_cv_proba  # noqa: E402
from models import build_sklearn_models  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group
from chimera_task1.reasoning_labels import (
    CONFIDENCE_LEVELS,
    CONFIDENCE_RANK,
    REVEAL_SECTIONS,
    TASK1_FACTORS,
    WEIGHT_RANK,
    parse_reveal_sequences,
    weight_col,
)
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import load_data, load_labeled_data, mri_pca_features
from chimera_task1.train_reasoning import load_annotated, make_classifier, repeated_out_of_fold_predict

N_REPEATS = 10
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]


def merge_metrics(results_dir: Path, condition: str, new_fields: dict) -> None:
    path = results_dir / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


def section_f1(true_col, pred_col) -> float:
    tp = int(np.sum((true_col == 1) & (pred_col == 1)))
    fp = int(np.sum((true_col == 0) & (pred_col == 1)))
    fn = int(np.sum((true_col == 1) & (pred_col == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def run_decision(results_dir: Path, X, y: np.ndarray) -> None:
    for name, (clf, use_sample_weight) in build_sklearn_models().items():
        all_probas = repeated_cv_proba(X, y, clf, use_sample_weight, N_REPEATS, stratified=True)
        per_repeat = [f1_score(y, p.argmax(axis=1), average="macro", labels=[0, 1], zero_division=0) for p in all_probas]
        merge_metrics(results_dir, f"decision_{name}", {
            "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
        })

    preprocessor = build_preprocessor(X)
    X_pre = preprocessor.fit_transform(X)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X_scaled = StandardScaler().fit_transform(X_pre)
    per_repeat = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in kf.split(X_scaled):
            test_probs = fit_predict_kdm(X_scaled[train_idx], y[train_idx], X_scaled[test_idx], n_classes=2)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1], zero_division=0))
    merge_metrics(results_dir, "decision_kdm", {
        "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
    })


def run_confidence(results_dir: Path, X, y: np.ndarray) -> None:
    for name, (clf, use_sample_weight) in build_sklearn_models().items():
        all_probas = repeated_cv_proba(X, y, clf, use_sample_weight, N_REPEATS, stratified=True)
        per_repeat = [f1_score(y, p.argmax(axis=1), average="macro", labels=[0, 1, 2], zero_division=0) for p in all_probas]
        merge_metrics(results_dir, f"confidence_{name}", {
            "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
        })

    preprocessor = build_preprocessor(X)
    X_pre = preprocessor.fit_transform(X)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X_scaled = StandardScaler().fit_transform(X_pre)
    n_classes = len(CONFIDENCE_LEVELS)
    per_repeat = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in kf.split(X_scaled):
            test_probs = fit_predict_kdm(X_scaled[train_idx], y[train_idx], X_scaled[test_idx], n_classes)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    merge_metrics(results_dir, "confidence_kdm", {
        "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
    })


def eval_factor_macro_f1(X, y_labels, preprocessor) -> float:
    y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
    per_repeat = []
    for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
        pred_rank = np.array([WEIGHT_RANK[p] for p in preds])
        per_repeat.append(f1_score(y_rank, pred_rank, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    return float(np.mean(per_repeat))


def run_weights(results_dir: Path, ann, X_full, comorbidity_treatment: str = "flags") -> None:
    preprocessor_official = build_preprocessor(X_full)
    per_factor_official = {}
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        per_factor_official[factor] = round(eval_factor_macro_f1(X_full, y_labels, preprocessor_official), 3)
    merge_metrics(results_dir, "weights_official", {
        "mean_macro_f1": round(float(np.mean(list(per_factor_official.values()))), 3),
        "per_factor_macro_f1": per_factor_official,
    })

    per_factor_restricted = {}
    for factor in IN_SCOPE_FACTORS:
        cols = restricted_feature_group(factor, comorbidity_treatment)
        X = X_full[cols]
        preprocessor = build_preprocessor(X)
        y_labels = ann[weight_col(factor)].values
        per_factor_restricted[factor] = round(eval_factor_macro_f1(X, y_labels, preprocessor), 3)
    merge_metrics(results_dir, "weights_restricted", {
        "mean_macro_f1": round(float(np.mean(list(per_factor_restricted.values()))), 3),
        "per_factor_macro_f1": per_factor_restricted,
    })


def run_reveal(results_dir: Path, ann, X) -> None:
    preprocessor = build_preprocessor(X)
    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])

    per_repeat_macro_f1 = []
    per_section_f1s = {s: [] for s in sections}
    N_REPEATS_REVEAL = 8  # train_reasoning.py's own N_REPEATS, which run_reveal.py imports unchanged
    for repeat in range(N_REPEATS_REVEAL):
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
    merge_metrics(results_dir, "reveal", {
        "macro_f1_mean": round(float(np.mean(per_repeat_macro_f1)), 3),
        "macro_f1_std": round(float(np.std(per_repeat_macro_f1)), 3),
        "per_section_macro_f1": per_section_mean,
        "macro_f1_definition": "per-section binary F1 (revealed vs. not), macro-averaged across the modeled sections",
    })


def main() -> None:
    from chimera_task1.features import select_exp3_feature_frame

    results_dir = Path(__file__).parent.parent / "results"

    _, full_inp, _ = load_data()
    _, inp, df = load_labeled_data()
    y_decision = (df["target_biopsy_decision"] == "yes").astype(int).values
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp["case_id"]].reset_index(drop=True)
    X_decision = select_exp3_feature_frame(inp, mri_pca)
    print("=== decision ===")
    run_decision(results_dir, X_decision, y_decision)

    ann, inp_ann = load_annotated()
    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])
    full_inp2 = pd.read_csv("data/inputs.csv")
    mri_pca_full2 = mri_pca_features(full_inp2, n_components=2)
    mri_pca_full2["case_id"] = full_inp2["case_id"].values
    mri_pca2 = mri_pca_full2.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_ann = select_exp3_feature_frame(inp_ann, mri_pca2)
    print("\n=== confidence ===")
    run_confidence(results_dir, X_ann, y_conf_rank)

    print("\n=== weights ===")
    run_weights(results_dir, ann, X_ann)

    print("\n=== reveal ===")
    run_reveal(results_dir, ann, X_ann)


if __name__ == "__main__":
    main()
