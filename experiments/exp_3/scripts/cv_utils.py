"""Shared repeated-CV harness for exp_3's 8-model comparison (decision + confidence).

Scales features uniformly (StandardScaler) for every model -- doesn't hurt tree-based models
and keeps one consistent preprocessing path instead of a per-model conditional (see
experiments/exp_3/DESIGN.md Section 9). Each fold clones the estimator fresh via sklearn's
clone() so no state leaks across folds/repeats.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from chimera_task1.features import build_preprocessor

RANDOM_STATE = 0
N_SPLITS = 5


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def repeated_cv_proba(
    feature_frame,
    y: np.ndarray,
    clf,
    use_sample_weight: bool,
    n_repeats: int,
    stratified: bool = True,
) -> list[np.ndarray]:
    """Repeated K-fold CV producing one out-of-fold predicted-probability array per repeat.

    Each returned array has shape (n_samples, n_classes), covering every row exactly once per
    repeat (out-of-fold). Preprocessing (impute/one-hot -- degenerates to pure imputation for
    exp_3's frame, which has no remaining categorical columns) + StandardScaler are fit fresh on
    each fold's training data only, never on the held-out fold.
    """
    # global_classes: the actual label VALUES present anywhere in y, not assumed to be a
    # contiguous 0..n-1 range -- e.g. weights' `pirads` factor has values {0, 2, 3} with zero
    # "noted" (=1) cases among all 91, so np.unique(y) is [0, 2, 3], not [0, 1, 2]. Mapping each
    # actual value to a dense column index (not using the value itself as the index) is what
    # makes the fold-level class alignment below correct in that case.
    global_classes = np.unique(y)
    n_classes = len(global_classes)
    class_to_col = {c: i for i, c in enumerate(global_classes)}
    all_probas = []
    for repeat in range(n_repeats):
        splitter_cls = StratifiedKFold if stratified else KFold
        splitter = splitter_cls(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        probas = np.zeros((len(y), n_classes))

        for train_idx, test_idx in splitter.split(feature_frame, y):
            X_train_raw = feature_frame.iloc[train_idx]
            X_test_raw = feature_frame.iloc[test_idx]

            preprocessor = build_preprocessor(feature_frame)
            X_train = _to_dense(preprocessor.fit_transform(X_train_raw))
            X_test = _to_dense(preprocessor.transform(X_test_raw))

            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            fold_clf = clone(clf)
            fit_kwargs = {}
            if use_sample_weight:
                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y[train_idx])
            fold_clf.fit(X_train, y[train_idx], **fit_kwargs)
            # BUGFIX (found during exp_5): a fold's training set may not contain every class --
            # e.g. weights' rarest per-factor classes have as few as 1 example total out of 91,
            # so some fold's train split has zero of it. predict_proba() then returns fewer
            # columns than the global n_classes (only for classes it saw), which would silently
            # misalign into the wrong global columns (or crash) if assigned directly. Map each of
            # fold_clf.classes_ (actual label values, not necessarily contiguous from 0 -- see
            # global_classes above) through class_to_col to get the correct dense column;
            # anything unseen during training correctly stays 0 probability. Never triggered for
            # exp_3/exp_4's decision/confidence results (well-balanced, contiguous classes), so
            # this fix doesn't change any previously-reported number for those targets.
            fold_proba = fold_clf.predict_proba(X_test)
            cols = [class_to_col[c] for c in fold_clf.classes_]
            probas[np.ix_(test_idx, cols)] = fold_proba

        all_probas.append(probas)
    return all_probas
