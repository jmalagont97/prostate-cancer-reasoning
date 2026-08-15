"""exp_11: small shared helpers for scoring a >2-class target with the full metric suite
(experiments/INDEX.md's "full metric-suite reporting initiative" note) -- the first time this
project has needed multiclass AUROC/Brier score (every prior AUROC/Brier addition, exp_9/exp_10,
was decision, a binary target).

accuracy_score, f1_score(average="macro"), and ordinal_distance() are all reused directly from
sklearn/reasoning_labels.py at each call site -- only the two metrics with no ready-made multiclass
form live here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Mean over samples of sum_k (p_k - 1[y=k])^2 -- the standard multiclass generalization of
    Brier score. sklearn's brier_score_loss is binary-only, hence this small explicit function.
    """
    onehot = np.eye(n_classes)[y_true]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def safe_multiclass_auroc(y_true: np.ndarray, proba: np.ndarray, labels: list[int]) -> float | None:
    """One-vs-rest macro AUROC, guarded against the degenerate case where the scored set is
    missing a class entirely (e.g. a single-row LOO fold, or an unlucky small CV fold) -- returns
    None (caller logs and skips) rather than crashing, same discipline as this project's other
    per-fold degenerate-case handling (exp_5's ValueError-catch precedent for rare per-factor
    classes).

    NOTE (found during exp_11's smoke tests): sklearn's roc_auc_score(multi_class="ovr") does NOT
    raise ValueError when a class is missing from y_true despite an explicit `labels=` -- it just
    emits UndefinedMetricWarning and folds a silent NaN into the macro average, so a naive
    try/except ValueError lets a NaN through uncaught. Checking class presence explicitly up front
    is the only reliable guard.
    """
    if not set(labels) <= set(np.unique(y_true)):
        return None
    try:
        value = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        return None
    return None if np.isnan(value) else value
