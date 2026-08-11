"""exp_2: decision model, 6 conditions = {logistic, hgb, kdm} x {count, flags}.

Reuses chimera_task1.train_decision's CV/pipeline pattern and
chimera_task1.train_confidence_kdm.fit_predict_kdm (generic over n_classes,
so n_classes=2 works unmodified) -- no changes to either module. Feature
frame comes from features.select_official_feature_frame (the 11-variable,
schema-faithful path), not exp_1's 47-column select_feature_frame.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_2/scripts/run_decision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chimera_task1.features import build_preprocessor, select_official_feature_frame
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import RANDOM_STATE, load_labeled_data

N_SPLITS = 5
N_REPEATS_SKLEARN = 10
N_REPEATS_KDM = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_sklearn_condition(feature_frame, y: np.ndarray, model_name: str, clf) -> dict:
    preprocessor = build_preprocessor(feature_frame)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_SKLEARN, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    f1_scores = cross_val_score(pipe, feature_frame, y, cv=cv, scoring="f1", n_jobs=-1)

    cv_single = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipe, feature_frame, y, cv=cv_single, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)

    return {
        "model": model_name,
        "f1_mean": round(float(f1_scores.mean()), 3),
        "f1_std": round(float(f1_scores.std()), 3),
        "n_folds": len(f1_scores),
        "roc_auc": round(float(auc), 3),
        "pr_auc": round(float(ap), 3),
    }


def run_kdm_condition(feature_frame, y: np.ndarray) -> dict:
    preprocessor = build_preprocessor(feature_frame)
    X_pre = preprocessor.fit_transform(feature_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)

    f1_scores, entropies = [], []
    proba_last_repeat = None
    for repeat in range(N_REPEATS_KDM):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=int)
        proba = np.zeros((len(y), 2))
        for train_idx, test_idx in kf.split(X):
            test_probs = fit_predict_kdm(X[train_idx], y[train_idx], X[test_idx], n_classes=2)
            proba[test_idx] = test_probs
            preds[test_idx] = test_probs.argmax(axis=1)
        f1_scores.append(f1_score(y, preds))
        entropy = -(proba * np.log(np.clip(proba, 1e-7, 1))).sum(axis=1)
        entropies.append(entropy.mean())
        proba_last_repeat = proba

    auc = roc_auc_score(y, proba_last_repeat[:, 1])
    ap = average_precision_score(y, proba_last_repeat[:, 1])

    return {
        "model": "KDM (memory-based, sigma-only trained)",
        "f1_mean": round(float(np.mean(f1_scores)), 3),
        "f1_std": round(float(np.std(f1_scores)), 3),
        "n_folds": N_REPEATS_KDM * N_SPLITS,
        "roc_auc": round(float(auc), 3),
        "pr_auc": round(float(ap), 3),
        "mean_predictive_entropy": round(float(np.mean(entropies)), 3),
        "max_possible_entropy": round(float(np.log(2)), 3),
        "note": "roc_auc/pr_auc computed from the last of the N_REPEATS_KDM out-of-fold probability sets "
        "(not averaged, unlike f1) -- KDM has no predict_proba-style single-pass CV helper here.",
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "decision", "features": "11 official Task-1 variables", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] {metrics}")


def main() -> None:
    _, inp, df = load_labeled_data()  # bugfix 2026-08-10: filters to the 91 actually-labeled cases
    y = (df["target_biopsy_decision"] == "yes").astype(int).values
    print(f"n={len(df)}, positive rate={y.mean():.2%}\n")

    for treatment in ("count", "flags"):
        X = select_official_feature_frame(inp, comorbidity_treatment=treatment)

        logistic = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
        write_metrics(f"decision_logistic_{treatment}", {"comorbidity_treatment": treatment,
                       **run_sklearn_condition(X, y, "logistic_regression", logistic)})

        hgb = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=1.0, max_iter=100, class_weight="balanced",
        )
        write_metrics(f"decision_hgb_{treatment}", {"comorbidity_treatment": treatment,
                       **run_sklearn_condition(X, y, "hist_gradient_boosting", hgb)})

        write_metrics(f"decision_kdm_{treatment}", {"comorbidity_treatment": treatment,
                       **run_kdm_condition(X, y)})


if __name__ == "__main__":
    main()
