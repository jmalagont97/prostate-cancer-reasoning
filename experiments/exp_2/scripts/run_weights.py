"""exp_2: variable-weight models, 4 conditions = {official, restricted} x {count, flags}.

`fh` is excluded from all 4 conditions (separate tool-revealed source, not
part of the 11-variable official table -- see features.py). Reuses
chimera_task1.train_reasoning.repeated_out_of_fold_predict for the actual
CV/scoring (no changes to train_reasoning.py). "official" trains every
factor's classifier on the same full 11-variable frame, exactly as exp_1's
weights_logistic did; "restricted" trains each factor's classifier on just
that factor's own column(s), via features.restricted_feature_group().

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_2/scripts/run_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_official_feature_frame
from chimera_task1.reasoning_labels import TASK1_FACTORS, WEIGHT_RANK, decisive_set_f1, ordinal_distance, weight_col
from chimera_task1.train_reasoning import load_annotated, repeated_out_of_fold_predict

RESULTS_DIR = Path(__file__).parent.parent / "results"
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]  # fh stays on its separate tool-revealed path


def eval_one_factor(X, y: np.ndarray, preprocessor) -> tuple[float, float, float, float]:
    """Returns (ordinal_error, decisive_set_f1, baseline_ordinal_error, baseline_decisive_set_f1)."""
    import pandas as pd

    majority = pd.Series(y).mode()[0]
    dists, f1s, dists_base, f1s_base = [], [], [], []
    for preds in repeated_out_of_fold_predict(X, y, preprocessor):
        dists.append(ordinal_distance(list(y), list(preds), WEIGHT_RANK))
        f1s.append(decisive_set_f1(list(y), list(preds)))
        dists_base.append(ordinal_distance(list(y), [majority] * len(y), WEIGHT_RANK))
        f1s_base.append(decisive_set_f1(list(y), [majority] * len(y)))
    return float(np.mean(dists)), float(np.mean(f1s)), float(np.mean(dists_base)), float(np.mean(f1s_base))


def run_official(ann, inp_ann, treatment: str) -> dict:
    X = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)
    preprocessor = build_preprocessor(X)

    per_factor, dists, f1s, dists_base, f1s_base = {}, [], [], [], []
    for factor in IN_SCOPE_FACTORS:
        y = ann[weight_col(factor)].values
        d, f1, db, f1b = eval_one_factor(X, y, preprocessor)
        per_factor[factor] = {"ordinal_error": round(d, 3), "decisive_set_f1": round(f1, 3)}
        dists.append(d)
        f1s.append(f1)
        dists_base.append(db)
        f1s_base.append(f1b)

    return {
        "model": "OvR logistic regression, all 9 in-scope factors on the same frame",
        "mean_ordinal_error": round(float(np.mean(dists)), 3),
        "mean_decisive_set_f1": round(float(np.mean(f1s)), 3),
        "mean_ordinal_error_baseline": round(float(np.mean(dists_base)), 3),
        "mean_decisive_set_f1_baseline": round(float(np.mean(f1s_base)), 3),
        "per_factor": per_factor,
    }


def run_restricted(ann, inp_ann, treatment: str) -> dict:
    full_frame = select_official_feature_frame(inp_ann, comorbidity_treatment=treatment)

    per_factor, dists, f1s, dists_base, f1s_base = {}, [], [], [], []
    for factor in IN_SCOPE_FACTORS:
        cols = restricted_feature_group(factor, treatment)
        X = full_frame[cols]
        preprocessor = build_preprocessor(X)
        y = ann[weight_col(factor)].values
        d, f1, db, f1b = eval_one_factor(X, y, preprocessor)
        per_factor[factor] = {"ordinal_error": round(d, 3), "decisive_set_f1": round(f1, 3), "columns": cols}
        dists.append(d)
        f1s.append(f1)
        dists_base.append(db)
        f1s_base.append(f1b)

    return {
        "model": "OvR logistic regression, one classifier per factor on its own restricted column(s)",
        "mean_ordinal_error": round(float(np.mean(dists)), 3),
        "mean_decisive_set_f1": round(float(np.mean(f1s)), 3),
        "mean_ordinal_error_baseline": round(float(np.mean(dists_base)), 3),
        "mean_decisive_set_f1_baseline": round(float(np.mean(f1s_base)), 3),
        "per_factor": per_factor,
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "variable_weights", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] mean_ordinal_error={metrics['mean_ordinal_error']} "
          f"(baseline {metrics['mean_ordinal_error_baseline']})  "
          f"mean_decisive_set_f1={metrics['mean_decisive_set_f1']} "
          f"(baseline {metrics['mean_decisive_set_f1_baseline']})")


def main() -> None:
    ann, inp_ann = load_annotated()
    print(f"n annotated = {len(ann)}, in-scope factors = {IN_SCOPE_FACTORS}\n")

    for treatment in ("count", "flags"):
        write_metrics(f"weights_official_{treatment}",
                      {"comorbidity_treatment": treatment, "features": "11 official Task-1 variables",
                       **run_official(ann, inp_ann, treatment)})
        write_metrics(f"weights_restricted_{treatment}",
                      {"comorbidity_treatment": treatment, "features": "per-factor restricted group",
                       **run_restricted(ann, inp_ann, treatment)})


if __name__ == "__main__":
    main()
