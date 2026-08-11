"""exp_4: reveal-sequence model, 1 condition, on the 16-column clinical-only frame (no MRI).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_4/scripts/run_reveal.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from chimera_task1.features import build_preprocessor, select_exp4_feature_frame
from chimera_task1.reasoning_labels import REVEAL_SECTIONS, parse_reveal_sequences, reveal_set_precision
from chimera_task1.train_reasoning import N_REPEATS, N_SPLITS, RANDOM_STATE, load_annotated, make_classifier

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_condition(X, ann) -> dict:
    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])
    preprocessor = build_preprocessor(X)

    precisions = []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.zeros_like(Y)
        for train_idx, test_idx in kf.split(X):
            pipe = Pipeline([("prep", preprocessor), ("clf", MultiOutputClassifier(make_classifier(), n_jobs=-1))])
            pipe.fit(X.iloc[train_idx], Y[train_idx])
            preds[test_idx] = pipe.predict(X.iloc[test_idx])

        fold_precisions = [
            reveal_set_precision(
                [s for s, v in zip(sections, true_row) if v], [s for s, v in zip(sections, pred_row) if v]
            )
            for true_row, pred_row in zip(Y, preds)
        ]
        precisions.append(np.mean(fold_precisions))

    mode_pattern = Counter(tuple(row) for row in Y).most_common(1)[0][0]
    return {
        "model": "MultiOutputClassifier(OvR logistic regression)",
        "sections_modeled": sections,
        "set_precision_mean": round(float(np.mean(precisions)), 3),
        "set_precision_std": round(float(np.std(precisions)), 3),
        "mode_pattern": [s for s, v in zip(sections, mode_pattern) if v],
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "reveal_sequence", "features": "exp_4 16-column frame", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] {metrics}")


def main() -> None:
    ann, inp_ann = load_annotated()
    print(f"n annotated = {len(ann)}\n")

    X = select_exp4_feature_frame(inp_ann)
    write_metrics("reveal", run_condition(X, ann))


if __name__ == "__main__":
    main()
