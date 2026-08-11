"""Step 4: train + cross-validate the confidence / variable-weight / reveal-sequence models.

Only the 91 annotated cases carry these labels. Models are deliberately simple
(multinomial logistic regression) given that N — a complex model would just
memorize. Evaluated via repeated out-of-fold CV using the *official rubric's own
metrics* (ordinal distance, set F1, reveal precision), not plain accuracy, so
the numbers here are directly comparable to what the leaderboard would show.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe -m chimera_task1.train_reasoning
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from chimera_task1.features import build_preprocessor, select_feature_frame
from chimera_task1.reasoning_labels import (
    CONFIDENCE_RANK,
    REVEAL_SECTIONS,
    TASK1_FACTORS,
    WEIGHT_RANK,
    decisive_set_f1,
    ordinal_distance,
    parse_reveal_sequences,
    reveal_set_precision,
    weight_col,
)

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 8


def make_classifier() -> OneVsRestClassifier:
    # liblinear converges fast and reliably at this N (the lbfgs runs in train_decision.py hit
    # pathological slow-convergence warnings even at max_iter=5000 on similarly small folds).
    # liblinear itself no longer auto-handles 3+ classes (sklearn now raises), so wrap in
    # explicit one-vs-rest -- fine for a simple ordinal-ish classifier at this N.
    return OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="liblinear")
    )


def load_annotated() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv("data/ground_truth.csv")
    inp = pd.read_csv("data/inputs.csv")
    ann_mask = gt["target_confidence"].notna()
    ann = gt.loc[ann_mask].reset_index(drop=True)
    inp_ann = inp.loc[inp["case_id"].isin(ann["case_id"])].set_index("case_id").loc[ann["case_id"]].reset_index()
    assert (inp_ann["case_id"].values == ann["case_id"].values).all()
    return ann, inp_ann


def repeated_out_of_fold_predict(X: pd.DataFrame, y: np.ndarray, preprocessor) -> list[np.ndarray]:
    """Return one array of out-of-fold predictions per repeat (each covering the full N rows)."""
    n = len(y)
    all_repeats = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(n, dtype=object)
        for train_idx, test_idx in kf.split(X):
            pipe = Pipeline(
                [("prep", preprocessor), ("clf", make_classifier())]
            )
            pipe.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = pipe.predict(X.iloc[test_idx])
        all_repeats.append(preds)
    return all_repeats


def eval_confidence(X: pd.DataFrame, ann: pd.DataFrame, preprocessor) -> None:
    y = ann["target_confidence"].values
    majority = pd.Series(y).mode()[0]

    dists, majority_dists = [], []
    for preds in repeated_out_of_fold_predict(X, y, preprocessor):
        dists.append(ordinal_distance(list(y), list(preds), CONFIDENCE_RANK))
        majority_dists.append(ordinal_distance(list(y), [majority] * len(y), CONFIDENCE_RANK))
    print(
        f"confidence          ordinal_distance = {np.mean(dists):.3f} +/- {np.std(dists):.3f}"
        f"   (always-'{majority}' baseline = {np.mean(majority_dists):.3f})"
    )


def eval_weights(X: pd.DataFrame, ann: pd.DataFrame, preprocessor) -> None:
    factor_dists, factor_f1s = [], []
    factor_dists_base, factor_f1s_base = [], []
    for factor in TASK1_FACTORS:
        y = ann[weight_col(factor)].values
        majority = pd.Series(y).mode()[0]

        dists, f1s, dists_base, f1s_base = [], [], [], []
        for preds in repeated_out_of_fold_predict(X, y, preprocessor):
            dists.append(ordinal_distance(list(y), list(preds), WEIGHT_RANK))
            f1s.append(decisive_set_f1(list(y), list(preds)))
            dists_base.append(ordinal_distance(list(y), [majority] * len(y), WEIGHT_RANK))
            f1s_base.append(decisive_set_f1(list(y), [majority] * len(y)))
        print(
            f"  {factor:12s} ordinal_error = {np.mean(dists):.3f} (baseline {np.mean(dists_base):.3f})"
            f"   decisive-set F1 = {np.mean(f1s):.3f} (baseline {np.mean(f1s_base):.3f})"
        )
        factor_dists.append(np.mean(dists))
        factor_f1s.append(np.mean(f1s))
        factor_dists_base.append(np.mean(dists_base))
        factor_f1s_base.append(np.mean(f1s_base))

    print(
        f"variable_weights     mean ordinal_error = {np.mean(factor_dists):.3f}"
        f" (baseline {np.mean(factor_dists_base):.3f})"
        f"   mean decisive-set F1 = {np.mean(factor_f1s):.3f} (baseline {np.mean(factor_f1s_base):.3f})"
    )


def eval_reveal(X: pd.DataFrame, ann: pd.DataFrame, preprocessor) -> None:
    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])

    # Baseline: always predict the single most common section-subset (mode pattern), ignoring features.
    from collections import Counter

    mode_pattern = Counter(tuple(row) for row in Y).most_common(1)[0][0]

    precisions, precisions_base = [], []
    n = len(Y)
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.zeros_like(Y)
        for train_idx, test_idx in kf.split(X):
            pipe = Pipeline(
                [
                    ("prep", preprocessor),
                    ("clf", MultiOutputClassifier(make_classifier(), n_jobs=-1)),
                ]
            )
            pipe.fit(X.iloc[train_idx], Y[train_idx])
            preds[test_idx] = pipe.predict(X.iloc[test_idx])

        fold_precisions = [
            reveal_set_precision(
                [s for s, v in zip(sections, true_row) if v], [s for s, v in zip(sections, pred_row) if v]
            )
            for true_row, pred_row in zip(Y, preds)
        ]
        precisions.append(np.mean(fold_precisions))

        base_precisions = [
            reveal_set_precision([s for s, v in zip(sections, true_row) if v], [s for s, v in zip(sections, mode_pattern) if v])
            for true_row in Y
        ]
        precisions_base.append(np.mean(base_precisions))

    print(
        f"reveal_sequence      set precision = {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}"
        f"   (always-mode-pattern baseline = {np.mean(precisions_base):.3f})"
    )
    print(f"  sections modeled: {sections}")
    print(f"  mode pattern: {[s for s, v in zip(sections, mode_pattern) if v]}")


def main() -> None:
    ann, inp_ann = load_annotated()
    print(f"n annotated = {len(ann)}\n")

    X = select_feature_frame(inp_ann)
    preprocessor = build_preprocessor(X)

    eval_confidence(X, ann, preprocessor)
    print()
    eval_weights(X, ann, preprocessor)
    print()
    eval_reveal(X, ann, preprocessor)


if __name__ == "__main__":
    main()
