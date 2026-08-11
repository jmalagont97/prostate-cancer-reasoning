"""exp_3: decision model, 8 conditions = 7 sklearn/xgboost models + KDM, on the 19-column frame.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_3/scripts/run_decision.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_utils import N_SPLITS, RANDOM_STATE, repeated_cv_proba  # noqa: E402
from models import NO_IMBALANCE_HANDLING, build_sklearn_models  # noqa: E402

from chimera_task1.features import select_exp3_feature_frame
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import load_data, load_labeled_data, mri_pca_features

N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_sklearn_condition(feature_frame, y: np.ndarray, model_name: str, clf, use_sample_weight: bool) -> dict:
    all_probas = repeated_cv_proba(feature_frame, y, clf, use_sample_weight, N_REPEATS, stratified=True)

    f1_scores = [f1_score(y, p.argmax(axis=1)) for p in all_probas]
    aucs = [roc_auc_score(y, p[:, 1]) for p in all_probas]
    aps = [average_precision_score(y, p[:, 1]) for p in all_probas]

    result = {
        "model": model_name,
        "f1_mean": round(float(np.mean(f1_scores)), 3),
        "f1_std": round(float(np.std(f1_scores)), 3),
        "n_folds": N_REPEATS * N_SPLITS,
        "roc_auc_mean": round(float(np.mean(aucs)), 3),
        "pr_auc_mean": round(float(np.mean(aps)), 3),
    }
    if model_name in NO_IMBALANCE_HANDLING:
        result["class_imbalance_handling"] = "none"
    elif use_sample_weight:
        result["class_imbalance_handling"] = "sample_weight (balanced)"
    else:
        result["class_imbalance_handling"] = "class_weight (balanced)"
    return result


def run_kdm_condition(feature_frame, y: np.ndarray) -> dict:
    from sklearn.preprocessing import StandardScaler

    from chimera_task1.features import build_preprocessor

    preprocessor = build_preprocessor(feature_frame)
    X_pre = preprocessor.fit_transform(feature_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)

    f1_scores, entropies = [], []
    proba_last_repeat = None
    for repeat in range(N_REPEATS):
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

    return {
        "model": "KDM (memory-based, sigma-only trained)",
        "f1_mean": round(float(np.mean(f1_scores)), 3),
        "f1_std": round(float(np.std(f1_scores)), 3),
        "n_folds": N_REPEATS * N_SPLITS,
        "roc_auc_mean": round(float(roc_auc_score(y, proba_last_repeat[:, 1])), 3),
        "pr_auc_mean": round(float(average_precision_score(y, proba_last_repeat[:, 1])), 3),
        "mean_predictive_entropy": round(float(np.mean(entropies)), 3),
        "max_possible_entropy": round(float(np.log(2)), 3),
        "class_imbalance_handling": "none",
        "note": "roc_auc/pr_auc from the last of N_REPEATS out-of-fold probability sets, not averaged, "
        "same as exp_2's KDM decision conditions.",
    }


def write_metrics(condition: str, metrics: dict) -> None:
    out_dir = RESULTS_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "target": "decision", "features": "exp_3 19-column frame (PSA-reduced + MRI-PCA)", **metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{condition}] {metrics}")


def main() -> None:
    # bugfix 2026-08-10: target_biopsy_decision is NaN for 104/195 cases; load_labeled_data()
    # filters to the 91 actually-labeled cases instead of silently coding NaN as y=0. MRI-PCA is
    # still fit on the full 195-case embedding population (unsupervised, doesn't need a decision
    # label), then aligned to the 91 labeled cases by case_id -- same pattern already used in
    # run_confidence.py/run_weights.py/run_reveal.py.
    _, full_inp, _ = load_data()
    _, inp, df = load_labeled_data()
    y = (df["target_biopsy_decision"] == "yes").astype(int).values
    print(f"n={len(df)}, positive rate={y.mean():.2%}\n")

    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp["case_id"]].reset_index(drop=True)
    X = select_exp3_feature_frame(inp, mri_pca)
    print(f"feature frame: {X.shape}\n")

    for name, (clf, use_sample_weight) in build_sklearn_models().items():
        write_metrics(f"decision_{name}", run_sklearn_condition(X, y, name, clf, use_sample_weight))

    write_metrics("decision_kdm", run_kdm_condition(X, y))


if __name__ == "__main__":
    main()
