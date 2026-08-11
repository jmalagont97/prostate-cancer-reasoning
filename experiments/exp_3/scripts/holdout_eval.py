"""Genuine held-out test evaluation of the leading models from exp_1-3.

Everything reported in exp_1/exp_2/exp_3 was repeated cross-validation on the training set
alone -- no split has ever been held out and left untouched. This script does that once:
~20% of the 91 labeled cases (stratified by decision) are carved out and never used for fitting
or feature-engineering choices (MRI-PCA is fit on the train portion only, not test); the leading
models per target are refit on the remaining ~80% and scored once on the held-out portion.

N=~18 for the test split -- a single point estimate at this size is noisy; report accordingly.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_3/scripts/holdout_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from models import build_sklearn_models  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_confidence_kdm import fit_predict_kdm
from chimera_task1.train_decision import MRI_EMB_PREFIX
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2


def mri_pca_train_only(inp_train: pd.DataFrame, inp_all: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Fit MRI-PCA on the train split's embeddings only, then transform every case in inp_all.

    Stricter than exp_1-3's pattern (which fit on the full 195-case pool regardless of any
    train/test split, since no split existed yet) -- here train/test separation should hold at
    the feature-engineering step too, not just at model-fitting.
    """
    emb_cols = [c for c in inp_all.columns if c.startswith(MRI_EMB_PREFIX)]
    train_emb = inp_train[emb_cols]
    has_mri_train = ~train_emb.isna().any(axis=1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(train_emb.loc[has_mri_train].values)
    print(f"MRI-PCA fit on {has_mri_train.sum()} train cases with MRI: "
          f"{pca.explained_variance_ratio_.sum():.1%} variance explained")

    all_emb = inp_all[emb_cols]
    has_mri_all = ~all_emb.isna().any(axis=1)
    coords = np.zeros((len(inp_all), n_components))
    coords[has_mri_all.values] = pca.transform(all_emb.loc[has_mri_all].values)
    cols = [f"mri_pca_{i}" for i in range(n_components)]
    out = pd.DataFrame(coords, columns=cols, index=inp_all.index)
    out["mri_missing"] = (~has_mri_all).astype("int64")
    return out


def fit_transform_features(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Impute + scale, fit on train only, applied to both -- no leakage from test into the fit."""
    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values
    y_conf_labels = ann["target_confidence"].values
    y_conf = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE
    )
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train, inp_test = inp_ann.iloc[train_idx].reset_index(drop=True), inp_ann.iloc[test_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]
    y_conf_train, y_conf_test = y_conf[train_idx], y_conf[test_idx]
    y_conf_labels_test = y_conf_labels[test_idx]

    mri_pca_test_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_all = select_exp3_feature_frame(inp_ann, mri_pca_test_aligned)
    X_train_raw, X_test_raw = X_all.iloc[train_idx].reset_index(drop=True), X_all.iloc[test_idx].reset_index(drop=True)

    print(f"held-out decision positive rate: {y_dec_test.mean():.2%} (train: {y_dec_train.mean():.2%})\n")

    # ---- Decision: knn (best CV), svm, kdm as runners-up for context ----
    print("=" * 70)
    print("DECISION (held-out test, n=%d)" % len(test_idx))
    print("=" * 70)
    X_train, X_test = fit_transform_features(X_train_raw, X_test_raw)
    models = build_sklearn_models()
    for name in ["knn", "svm"]:
        clf, use_sw = models[name]
        fit_kwargs = {}
        if use_sw:
            from sklearn.utils.class_weight import compute_sample_weight

            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_dec_train)
        clf.fit(X_train, y_dec_train, **fit_kwargs)
        proba = clf.predict_proba(X_test)
        preds = proba.argmax(axis=1)
        f1 = f1_score(y_dec_test, preds)
        try:
            auc = roc_auc_score(y_dec_test, proba[:, 1])
        except ValueError:
            auc = float("nan")  # can happen if the tiny test split is single-class
        print(f"\n--- decision_{name} ---  F1={f1:.3f}  ROC-AUC={auc:.3f}")
        print(classification_report(y_dec_test, preds, target_names=["no", "yes"], digits=3, zero_division=0))

    kdm_probs = fit_predict_kdm(X_train, y_dec_train, X_test, n_classes=2)
    kdm_preds = kdm_probs.argmax(axis=1)
    f1_kdm = f1_score(y_dec_test, kdm_preds)
    print(f"\n--- decision_kdm ---  F1={f1_kdm:.3f}")
    print(classification_report(y_dec_test, kdm_preds, target_names=["no", "yes"], digits=3, zero_division=0))

    # ---- Confidence: svm (best CV), kdm as runner-up ----
    print("=" * 70)
    print("CONFIDENCE (held-out test, n=%d)" % len(test_idx))
    print("=" * 70)
    clf_svm, use_sw_svm = models["svm"]
    clf_svm.fit(X_train, y_conf_train)
    proba_svm = clf_svm.predict_proba(X_test)
    preds_svm_idx = proba_svm.argmax(axis=1)
    preds_svm = [CONFIDENCE_LEVELS[i] for i in preds_svm_idx]
    dist_svm = ordinal_distance(list(y_conf_labels_test), preds_svm, CONFIDENCE_RANK)
    print(f"\n--- confidence_svm ---  ordinal_distance={dist_svm:.3f}")
    print(classification_report(y_conf_test, preds_svm_idx, target_names=CONFIDENCE_LEVELS, digits=3, zero_division=0))

    kdm_conf_probs = fit_predict_kdm(X_train, y_conf_train, X_test, n_classes=3)
    kdm_conf_preds_idx = kdm_conf_probs.argmax(axis=1)
    kdm_conf_preds = [CONFIDENCE_LEVELS[i] for i in kdm_conf_preds_idx]
    dist_kdm = ordinal_distance(list(y_conf_labels_test), kdm_conf_preds, CONFIDENCE_RANK)
    print(f"\n--- confidence_kdm ---  ordinal_distance={dist_kdm:.3f}")
    print(classification_report(y_conf_test, kdm_conf_preds_idx, target_names=CONFIDENCE_LEVELS, digits=3, zero_division=0))

    # naive baselines on this exact test split, for a fair local comparison
    majority_dec = int(y_dec_train.mean() >= 0.5)
    baseline_f1 = f1_score(y_dec_test, np.full_like(y_dec_test, majority_dec))
    majority_conf = pd.Series(y_conf_labels[train_idx]).mode()[0]
    baseline_dist = ordinal_distance(list(y_conf_labels_test), [majority_conf] * len(y_conf_labels_test), CONFIDENCE_RANK)
    print("=" * 70)
    print("NAIVE BASELINES on this exact held-out split")
    print("=" * 70)
    print(f"decision: always '{'yes' if majority_dec else 'no'}' (train majority)  F1={baseline_f1:.3f}")
    print(f"confidence: always '{majority_conf}' (train majority)  ordinal_distance={baseline_dist:.3f}")


if __name__ == "__main__":
    main()
