"""Backfill macro-F1 into exp_1's already-completed results, per this project's cross-experiment
macro-F1 reporting initiative (2026-08-12), extended back to exp_1-exp_5 on 2026-08-13 for a full
"every model, every experiment" comparison table.

exp_1 predates any of this project's scripts/ directories -- its conditions were run directly via
`python -m chimera_task1.train_decision` / `train_reasoning` / `train_confidence_kdm`, which only
ever printed scores (never wrote metrics.json at all -- those files were written by hand from the
printed output, per exp_1/reports/summary.md). This script reproduces each condition's exact model/
feature-frame/CV configuration from those modules (unchanged, no edits) to recompute out-of-fold
predictions and derive macro-F1, merged into the existing metrics.json via read-modify-write.

Definitions (same as every other experiment's backfill):
- Decision: standard binary macro-F1 (average of "yes"/"no" class F1). NOTE: exp_1's stored
  `f1_mean` is the POSITIVE-CLASS ("yes") F1 only (`sklearn.metrics.f1_score` default
  `average="binary"`), not macro-F1 -- confirmed by reading train_decision.py's `evaluate()`. The
  two are genuinely different numbers, not just a renaming.
- Confidence: standard 3-class macro-F1 (uncertain/borderline/clear).
- Weights: per-factor 4-class macro-F1 (not_used/noted/important/decisive, labels=[0,1,2,3]
  explicit), averaged across factors. exp_1 scored ALL of TASK1_FACTORS (10, including "fh") --
  unlike every later experiment, which excludes "fh" onto its own tool-revealed path. Reproduced
  as originally run, not retroactively filtered.
- Reveal: per-section binary F1 (revealed vs. not), macro-averaged across the modeled sections.

CV discipline note: the original decision/confidence/weights/reveal code paths use sklearn's
`RepeatedStratifiedKFold`/`KFold` objects directly (a single CV object spanning all repeats, not a
per-repeat loop with `random_state=BASE+repeat`). This script instead loops repeats explicitly with
`random_state=RANDOM_STATE+repeat` per repeat (the same discipline already used for exp_6-exp_8's
backfills) to get one clean out-of-fold prediction set per repeat -- this reproduces the *same CV
protocol* (5-fold, 10 or 8 repeats, same RANDOM_STATE seed family) but not bit-identical fold
membership to the original run. Aggregate macro-F1 across ~50-80 folds is not meaningfully sensitive
to this distinction; the original ordinal_distance/f1_mean numbers are left untouched either way.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_1/scripts/backfill_macro_f1.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from chimera_task1.features import build_preprocessor, select_feature_frame
from chimera_task1.reasoning_labels import (
    CONFIDENCE_RANK,
    REVEAL_SECTIONS,
    TASK1_FACTORS,
    WEIGHT_RANK,
    parse_reveal_sequences,
    weight_col,
)
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import RANDOM_STATE, load_data, load_labeled_data, mri_pca_features
from chimera_task1.train_reasoning import load_annotated, make_classifier

N_SPLITS = 5
N_REPEATS_DECISION = 10
N_REPEATS_REASONING = 8  # train_reasoning.py's own N_REPEATS
RESULTS_DIR = Path(__file__).parent.parent / "results"


def merge_metrics(condition: str, new_fields: dict) -> None:
    path = RESULTS_DIR / condition / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    payload.update(new_fields)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] merged: {new_fields}")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decision_oof_macro_f1(feature_frame: pd.DataFrame, y: np.ndarray, clf, n_repeats: int) -> tuple[float, float]:
    preprocessor = build_preprocessor(feature_frame)
    per_repeat = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in skf.split(feature_frame, y):
            pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
            pipe.fit(feature_frame.iloc[train_idx], y[train_idx])
            preds[test_idx] = pipe.predict(feature_frame.iloc[test_idx])
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1], zero_division=0))
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def dummy_oof_macro_f1(y: np.ndarray, strategy: str, constant: int | None, n_repeats: int) -> tuple[float, float]:
    per_repeat = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        for train_idx, test_idx in skf.split(np.zeros((len(y), 1)), y):
            dummy = DummyClassifier(strategy=strategy, constant=constant, random_state=RANDOM_STATE)
            dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
            preds[test_idx] = dummy.predict(np.zeros((len(test_idx), 1)))
        per_repeat.append(f1_score(y, preds, average="macro", labels=[0, 1], zero_division=0))
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def run_decision() -> None:
    _, full_inp, _ = load_data()
    gt, inp, df = load_labeled_data()
    y = (df["target_biopsy_decision"] == "yes").astype(int).values

    always_no_m, always_no_s = dummy_oof_macro_f1(y, "constant", 0, N_REPEATS_DECISION)
    always_yes_m, always_yes_s = dummy_oof_macro_f1(y, "constant", 1, N_REPEATS_DECISION)
    random_m, random_s = dummy_oof_macro_f1(y, "stratified", None, N_REPEATS_DECISION)
    merge_metrics("decision_baseline", {
        "always_no_macro_f1_mean": round(always_no_m, 3), "always_no_macro_f1_std": round(always_no_s, 3),
        "always_yes_macro_f1_mean": round(always_yes_m, 3), "always_yes_macro_f1_std": round(always_yes_s, 3),
        "random_class_proportional_macro_f1_mean": round(random_m, 3),
        "random_class_proportional_macro_f1_std": round(random_s, 3),
    })

    clinical_only = select_feature_frame(inp)
    logistic = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    hgb = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, max_leaf_nodes=7, min_samples_leaf=20,
        l2_regularization=1.0, max_iter=100, class_weight="balanced",
    )
    m, s = decision_oof_macro_f1(clinical_only, y, logistic, N_REPEATS_DECISION)
    merge_metrics("decision_logistic_clinical", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})
    m, s = decision_oof_macro_f1(clinical_only, y, hgb, N_REPEATS_DECISION)
    merge_metrics("decision_hgb_clinical", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})

    mri_pca_full = mri_pca_features(full_inp, n_components=10)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp["case_id"]].reset_index(drop=True)
    with_mri = select_feature_frame(inp, include_mri_pca=mri_pca)
    m, s = decision_oof_macro_f1(with_mri, y, logistic, N_REPEATS_DECISION)
    merge_metrics("decision_logistic_mri_pca", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})
    m, s = decision_oof_macro_f1(with_mri, y, hgb, N_REPEATS_DECISION)
    merge_metrics("decision_hgb_mri_pca", {"macro_f1_mean": round(m, 3), "macro_f1_std": round(s, 3)})


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def run_confidence() -> None:
    ann, inp_ann = load_annotated()
    X = select_feature_frame(inp_ann)
    preprocessor = build_preprocessor(X)
    y_labels = ann["target_confidence"].values
    y_rank = np.array([CONFIDENCE_RANK[label] for label in y_labels])
    majority_rank = np.bincount(y_rank).argmax()

    # baseline: always predict the majority class (closed form, no CV needed)
    baseline_preds = np.full(len(y_rank), majority_rank)
    baseline_f1 = f1_score(y_rank, baseline_preds, average="macro", labels=[0, 1, 2], zero_division=0)
    merge_metrics("confidence_baseline", {"macro_f1": round(float(baseline_f1), 3)})

    # logistic (OvR): reuse repeated_out_of_fold_predict for identical fold/model logic
    from chimera_task1.train_reasoning import repeated_out_of_fold_predict

    per_repeat = []
    for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
        pred_rank = np.array([CONFIDENCE_RANK[p] for p in preds])
        per_repeat.append(f1_score(y_rank, pred_rank, average="macro", labels=[0, 1, 2], zero_division=0))
    merge_metrics("confidence_logistic", {
        "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
    })

    # kdm: reproduce train_confidence_kdm.py's exact fit_predict_kdm loop
    from sklearn.preprocessing import StandardScaler

    X_pre = preprocessor.fit_transform(X)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X_scaled = StandardScaler().fit_transform(X_pre)
    n_classes = 3
    per_repeat = []
    for repeat in range(10):  # train_confidence_kdm.py's own N_REPEATS
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y_rank), dtype=int)
        for train_idx, test_idx in kf.split(X_scaled):
            test_probs = fit_predict_kdm(X_scaled[train_idx], y_rank[train_idx], X_scaled[test_idx], n_classes)
            preds[test_idx] = test_probs.argmax(axis=1)
        per_repeat.append(f1_score(y_rank, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    merge_metrics("confidence_kdm", {
        "macro_f1_mean": round(float(np.mean(per_repeat)), 3), "macro_f1_std": round(float(np.std(per_repeat)), 3),
    })


# ---------------------------------------------------------------------------
# Weights (ALL of TASK1_FACTORS, including "fh", as exp_1 originally ran)
# ---------------------------------------------------------------------------

def run_weights() -> None:
    from chimera_task1.train_reasoning import repeated_out_of_fold_predict

    ann, inp_ann = load_annotated()
    X = select_feature_frame(inp_ann)
    preprocessor = build_preprocessor(X)

    baseline_factor_f1, logistic_factor_f1 = {}, {}
    for factor in TASK1_FACTORS:
        y_labels = ann[weight_col(factor)].values
        y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
        majority_rank = np.bincount(y_rank).argmax()

        baseline_preds = np.full(len(y_rank), majority_rank)
        baseline_factor_f1[factor] = round(
            float(f1_score(y_rank, baseline_preds, average="macro", labels=[0, 1, 2, 3], zero_division=0)), 3
        )

        per_repeat = []
        for preds in repeated_out_of_fold_predict(X, y_labels, preprocessor):
            pred_rank = np.array([WEIGHT_RANK[p] for p in preds])
            per_repeat.append(f1_score(y_rank, pred_rank, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        logistic_factor_f1[factor] = round(float(np.mean(per_repeat)), 3)

    merge_metrics("weights_baseline", {
        "mean_macro_f1": round(float(np.mean(list(baseline_factor_f1.values()))), 3),
        "per_factor_macro_f1": baseline_factor_f1,
    })
    merge_metrics("weights_logistic", {
        "mean_macro_f1": round(float(np.mean(list(logistic_factor_f1.values()))), 3),
        "per_factor_macro_f1": logistic_factor_f1,
    })


# ---------------------------------------------------------------------------
# Reveal
# ---------------------------------------------------------------------------

def run_reveal() -> None:
    ann, inp_ann = load_annotated()
    X = select_feature_frame(inp_ann)
    preprocessor = build_preprocessor(X)

    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])
    mode_pattern = Counter(tuple(row) for row in Y).most_common(1)[0][0]

    def section_f1(true_col, pred_col) -> float:
        tp = int(np.sum((true_col == 1) & (pred_col == 1)))
        fp = int(np.sum((true_col == 0) & (pred_col == 1)))
        fn = int(np.sum((true_col == 1) & (pred_col == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # baseline: always predict the mode pattern (closed form)
    baseline_pred = np.tile(np.array(mode_pattern), (len(Y), 1))
    baseline_per_section = {s: round(section_f1(Y[:, i], baseline_pred[:, i]), 3) for i, s in enumerate(sections)}
    merge_metrics("reveal_baseline", {
        "macro_f1": round(float(np.mean(list(baseline_per_section.values()))), 3),
        "per_section_macro_f1": baseline_per_section,
        "macro_f1_definition": "per-section binary F1 (revealed vs. not), macro-averaged across the modeled sections",
    })

    # logistic: reproduce train_reasoning.eval_reveal's exact MultiOutputClassifier loop
    n = len(Y)
    per_repeat_macro_f1 = []
    per_section_f1s = {s: [] for s in sections}
    for repeat in range(N_REPEATS_REASONING):
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
    merge_metrics("reveal_logistic", {
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
