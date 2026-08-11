"""exp_5: variable-weight models, 16 conditions = 8 models x {official, restricted} scope.

Reuses experiments/exp_3/scripts/{models.py,cv_utils.py} unchanged. cv_utils.repeated_cv_proba
is already generic over n_classes (inferred from y), so the 4-class weight target needs no new
plumbing -- same code path exp_3's run_confidence.py already used for its 3-class target.

N_REPEATS reduced to 5 (from exp_3's 10) given the much higher fit count here: 8 models x 9
factors x 2 scopes, vs. 8 models x 1 condition for decision/confidence. Restricted scope alone is
up to 8 x 9 x 5 x 5 = 1800 individual fits.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_5/scripts/run_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
from cv_utils import N_SPLITS, RANDOM_STATE, repeated_cv_proba  # noqa: E402
from models import NO_IMBALANCE_HANDLING, build_sklearn_models  # noqa: E402

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_exp3_feature_frame
from chimera_task1.reasoning_labels import TASK1_FACTORS, WEIGHT_LEVELS, WEIGHT_RANK, decisive_set_f1, ordinal_distance, weight_col
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

N_REPEATS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]


def eval_one_factor(X: pd.DataFrame, y_labels: np.ndarray, clf, use_sample_weight: bool) -> tuple[float, float, float, float]:
    y = np.array([WEIGHT_RANK[label] for label in y_labels])
    majority = pd.Series(y_labels).mode()[0]
    # Some factors don't have every one of the 4 weight levels represented at all (e.g. pirads
    # has zero "noted" cases among all 91) -- cv_utils.repeated_cv_proba densely re-indexes its
    # output columns to whatever classes ARE present (see its own comment), so column i means
    # WEIGHT_LEVELS[global_classes[i]] here, not WEIGHT_LEVELS[i] directly.
    global_classes = np.unique(y)

    all_probas = repeated_cv_proba(X, y, clf, use_sample_weight, N_REPEATS, stratified=False)
    dists, f1s = [], []
    for proba in all_probas:
        preds = [WEIGHT_LEVELS[global_classes[i]] for i in proba.argmax(axis=1)]
        dists.append(ordinal_distance(list(y_labels), preds, WEIGHT_RANK))
        f1s.append(decisive_set_f1(list(y_labels), preds))

    dist_base = ordinal_distance(list(y_labels), [majority] * len(y_labels), WEIGHT_RANK)
    f1_base = decisive_set_f1(list(y_labels), [majority] * len(y_labels))
    return float(np.mean(dists)), float(np.mean(f1s)), dist_base, f1_base


def eval_one_factor_kdm(X: pd.DataFrame, y_labels: np.ndarray) -> tuple[float, float, float, float]:
    y = np.array([WEIGHT_RANK[label] for label in y_labels])
    majority = pd.Series(y_labels).mode()[0]
    n_classes = len(WEIGHT_LEVELS)

    preprocessor = build_preprocessor(X)
    X_pre = preprocessor.fit_transform(X)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X_scaled = StandardScaler().fit_transform(X_pre)

    dists, f1s = [], []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=object)
        for train_idx, test_idx in kf.split(X_scaled):
            test_probs = fit_predict_kdm(X_scaled[train_idx], y[train_idx], X_scaled[test_idx], n_classes)
            preds[test_idx] = [WEIGHT_LEVELS[i] for i in test_probs.argmax(axis=1)]
        dists.append(ordinal_distance(list(y_labels), list(preds), WEIGHT_RANK))
        f1s.append(decisive_set_f1(list(y_labels), list(preds)))

    dist_base = ordinal_distance(list(y_labels), [majority] * len(y_labels), WEIGHT_RANK)
    f1_base = decisive_set_f1(list(y_labels), [majority] * len(y_labels))
    return float(np.mean(dists)), float(np.mean(f1s)), dist_base, f1_base


def run_condition(ann, X_full_or_groups, model_name: str, clf, use_sample_weight: bool, restricted: bool) -> dict:
    per_factor, dists, f1s, dists_base, f1s_base = {}, [], [], [], []
    for factor in IN_SCOPE_FACTORS:
        y_labels = ann[weight_col(factor)].values
        if restricted:
            cols = restricted_feature_group(factor, "flags")
            X = X_full_or_groups[cols]
        else:
            X = X_full_or_groups
            cols = None

        try:
            if model_name == "kdm":
                d, f1, db, f1b = eval_one_factor_kdm(X, y_labels)
            else:
                d, f1, db, f1b = eval_one_factor(X, y_labels, clf, use_sample_weight)
        except ValueError as e:
            # Some factors have a class with only 1-2 examples total (e.g. psa's "not_used"),
            # so most CV folds have zero of it in training. sklearn's own estimators degrade
            # gracefully (that class just never gets predicted); XGBoost's stricter internal
            # class validation raises instead. Recorded as a skip, not forced past -- this is a
            # genuine data-scarcity limit for this factor/model combination, not a code bug.
            per_factor[factor] = {"error": f"{type(e).__name__}: {e}", "skipped": True}
            print(f"  [SKIP] {model_name} / {factor}: {e}")
            continue

        entry = {"ordinal_error": round(d, 3), "decisive_set_f1": round(f1, 3)}
        if cols is not None:
            entry["columns"] = cols
        per_factor[factor] = entry
        dists.append(d)
        f1s.append(f1)
        dists_base.append(db)
        f1s_base.append(f1b)

    if not dists:
        return {
            "model": model_name,
            "scope": "restricted" if restricted else "official",
            "mean_ordinal_error": None,
            "mean_decisive_set_f1": None,
            "note": "all factors skipped -- see per_factor errors",
            "per_factor": per_factor,
        }

    result = {
        "model": model_name,
        "scope": "restricted" if restricted else "official",
        "n_factors_included": len(dists),
        "n_factors_skipped": len(IN_SCOPE_FACTORS) - len(dists),
        "mean_ordinal_error": round(float(np.mean(dists)), 3),
        "mean_decisive_set_f1": round(float(np.mean(f1s)), 3),
        "mean_ordinal_error_baseline": round(float(np.mean(dists_base)), 3),
        "mean_decisive_set_f1_baseline": round(float(np.mean(f1s_base)), 3),
        "per_factor": per_factor,
    }
    if model_name == "kdm" or model_name in NO_IMBALANCE_HANDLING:
        result["class_imbalance_handling"] = "none"
    elif use_sample_weight:
        result["class_imbalance_handling"] = "sample_weight (balanced)"
    else:
        result["class_imbalance_handling"] = "class_weight (balanced)"
    return result


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

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_full = select_exp3_feature_frame(inp_ann, mri_pca)
    print(f"feature frame: {X_full.shape}\n")

    models = build_sklearn_models()
    models["kdm"] = (None, False)  # handled specially in run_condition -- not a plain sklearn estimator

    print("=== official scope ===")
    for name, (clf, use_sw) in models.items():
        write_metrics(f"weights_official_{name}", run_condition(ann, X_full, name, clf, use_sw, restricted=False))

    print("\n=== restricted scope ===")
    for name, (clf, use_sw) in models.items():
        write_metrics(f"weights_restricted_{name}", run_condition(ann, X_full, name, clf, use_sw, restricted=True))


if __name__ == "__main__":
    main()
