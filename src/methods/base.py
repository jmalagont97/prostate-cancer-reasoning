"""Shared contract for classification methods (`src/methods/*.py`).

Every method module (currently only `mem_kdm.py`; `knn.py`/`fuzzy_knn.py` later)
implements the `Method` protocol below. This module carries only what is
genuinely shared across methods:

  - `Modalities` / `Targets` — the data shapes every method's `fit`/`predict`
    exchange with `src/evaluation`.
  - `Method` — the fit/predict/confidence contract.
  - Meta-threshold plumbing (`fit_meta_thresholds`, `apply_meta_thresholds`,
    `fit_predict_heldout_trees`) — lifted from `exp_23`/`exp_24`'s confidence
    heads, generalized over an explicit `splits` list instead of a
    `df_design_labeled` frame + column-name convention, so a second method can
    reuse them without inheriting anything and without importing `src/evaluation`.

This module must never import `src/evaluation` or `kdm` — methods are model
code; the harness (splits, cohort loading, scoring) is a separate concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr

# {"tab": ..., "mri": ..., "txt": ...} — one array/frame per modality, same
# row order and row count within one `fit`/`predict` call.
Modalities = dict


@dataclass
class Targets:
    """Supervision passed to `Method.fit`.

    Supervision is soft throughout; a hard-labeled arm is simply
    `y_soft = y_binary.astype(float)` — there is no separate hard code path.
    """

    y_binary: np.ndarray
    """(n,) int in {0,1} — evaluation ground truth ONLY. Never a fitting signal."""

    y_soft: np.ndarray
    """(n,) float in [0,1] — THE supervision signal `fit` consumes."""

    y_conf: np.ndarray | None = None
    """(n,) int in {0,1,2} — diagnostic-confidence label. Never used to fit the
    biopsy-decision model; only used by `fit_confidence`."""

    soft_from_confidence: bool = False
    """True when `y_soft` was derived from the confidence annotation (e.g. via
    `data.build_targets(..., certainty_map=...)`). Any model fitted on such
    targets is target-informed for the confidence-prediction task, since
    `y_soft` leaks information about `y_conf`."""


@runtime_checkable
class Method(Protocol):
    """Contract every `src/methods/*.py` classifier satisfies.

    All hyperparameters live in `__init__`; nothing but data is passed to
    `fit`. `fit` builds the model from scratch on its own data (no partial_fit,
    no cross-call state) — "memory = the data passed to this `fit` call" falls
    out of this by construction.
    """

    target_informed: bool
    """True iff this instance was `fit` with `Targets.soft_from_confidence=True`.
    Confidence-signal selectors must filter this out by default (see
    `mem_kdm.best_signal`) — a model fit on confidence-derived targets cannot
    validly be used to predict confidence."""

    # --- biopsy decision (binary) ---
    def fit(self, X: Modalities, targets: Targets) -> "Method": ...

    def predict_proba(self, X: Modalities) -> np.ndarray:
        """(n, 2) class probabilities."""
        ...

    def predict(self, X: Modalities, threshold: float = 0.50) -> np.ndarray:
        """(n,) int in {0,1}."""
        ...

    # --- diagnostic confidence (3-class), derived from THIS method's internals ---
    def uncertainty_signals(self, X: Modalities) -> dict:
        """Method-specific uncertainty signals, e.g. the particle-set signals
        for `MemKDM` (see `mem_kdm.extract_particle_signals`)."""
        ...

    def fit_confidence(self, y_conf: np.ndarray, splits, head: str) -> "Method":
        """Phase-A operation: fits and freezes a confidence head on this
        instance's own `uncertainty_signals` of its training data. Requires
        `splits`; there is no single-shot / whole-cohort variant, so exp_17's
        collapse of Phase A and Phase B is not expressible here."""
        ...

    def predict_confidence(self, X: Modalities) -> np.ndarray:
        """(n,) int in {0,1,2}, using the frozen head from `fit_confidence`."""
        ...


# ---------------------------------------------------------------------------
# Meta-threshold plumbing — shared, method-agnostic. Lifted from
# exp_23/scripts/train.py:506-587 and exp_24/scripts/train.py:543-599,
# generalized to accept `splits: list[tuple[np.ndarray, np.ndarray]]`
# (train_idx, val_idx) instead of a df_design_labeled frame.
# ---------------------------------------------------------------------------
def fit_meta_thresholds(
    signal: np.ndarray,
    y_conf: np.ndarray,
    splits: list,
    max_depth: int = 2,
    random_state: int = 42,
) -> dict:
    """Per-split `DecisionTreeClassifier(max_depth=2)` cut-point extraction,
    averaged over splits, with a monotone-direction vote and a percentile
    fallback for degenerate splits. Raises RuntimeError if every split is
    degenerate (see `fit_meta_thresholds_safe` for a fallback wrapper)."""
    thresholds_t1, thresholds_t2, directions = [], [], []
    fallback_count = degenerate_count = nonmonotone_count = 0

    lo, hi = float(signal.min()), float(signal.max())
    sweep = np.linspace(lo, hi, 50).reshape(-1, 1)

    for train_idx, _val_idx in splits:
        X_tr = signal[train_idx].reshape(-1, 1)
        y_tr = y_conf[train_idx]

        dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=random_state)
        dt.fit(X_tr, y_tr)

        tree_thresholds = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2])

        if len(tree_thresholds) >= 2:
            t1, t2 = float(tree_thresholds[0]), float(tree_thresholds[1])
        elif len(tree_thresholds) == 1:
            fallback_count += 1
            t1 = float(tree_thresholds[0])
            p67 = float(np.percentile(X_tr, 67))
            t2 = max(p67, t1 + (float(X_tr.max()) - t1) / 2)
        else:
            fallback_count += 1
            t1, t2 = (float(v) for v in np.percentile(X_tr, [33, 67]))

        thresholds_t1.append(t1)
        thresholds_t2.append(t2)

        pred_sweep = dt.predict(sweep)
        if len(np.unique(pred_sweep)) == 1:
            degenerate_count += 1
            continue
        rho, _ = spearmanr(np.arange(len(sweep)), pred_sweep)
        if np.isnan(rho):
            degenerate_count += 1
            continue
        if abs(rho) < 0.1:
            nonmonotone_count += 1
        directions.append(np.sign(rho))

    meta_t1 = float(np.mean(thresholds_t1))
    meta_t2 = float(np.mean(thresholds_t2))
    if len(directions) == 0:
        raise RuntimeError("all splits degenerate — cannot determine signal direction")
    direction = int(np.sign(np.nansum(directions)))
    assert direction in (1, -1), f"ambiguous direction: {direction}"

    return {
        "meta_threshold_1": meta_t1,
        "meta_threshold_2": meta_t2,
        "direction": direction,
        "fallback_count": fallback_count,
        "degenerate_count": degenerate_count,
        "nonmonotone_count": nonmonotone_count,
        "n_splits": len(splits),
        "degenerate_fallback": False,
    }


def fit_meta_thresholds_safe(signal: np.ndarray, y_conf: np.ndarray, splits: list, **kwargs) -> dict:
    """`fit_meta_thresholds` wrapped with a whole-signal percentile fallback
    for the rare case every split degenerates. Absorbs exp_24's
    `fit_1d_confidence_signal_safe`."""
    try:
        return fit_meta_thresholds(signal, y_conf, splits, **kwargs)
    except (RuntimeError, AssertionError):
        t1, t2 = (float(v) for v in np.percentile(signal, [33, 67]))
        return {
            "meta_threshold_1": t1,
            "meta_threshold_2": t2,
            "direction": 1,
            "fallback_count": len(splits),
            "degenerate_count": len(splits),
            "nonmonotone_count": 0,
            "n_splits": len(splits),
            "degenerate_fallback": True,
        }


def apply_meta_thresholds(signal: np.ndarray, thr: dict) -> np.ndarray:
    """Applies frozen thresholds from `fit_meta_thresholds[_safe]`. No
    refitting — this is the Phase-B side."""
    t1, t2, direction = thr["meta_threshold_1"], thr["meta_threshold_2"], thr["direction"]
    if direction == 1:
        return np.where(signal < t1, 0, np.where(signal < t2, 1, 2))
    return np.where(signal < t1, 2, np.where(signal < t2, 1, 0))


def fit_predict_heldout_trees(
    signal_matrix: np.ndarray,
    y_conf: np.ndarray,
    splits: list,
    max_depth: int = 3,
    random_state: int = 42,
):
    """One `DecisionTreeClassifier` per split, trained on that split's train
    rows only. Patient i's prediction is the majority vote over ONLY the
    trees whose split had i in validation — trees that never saw patient i
    during fitting. Lifted from exp_24's `fit_multivariate_confidence_head`.

    This protocol is self-referential: it only produces predictions for the
    exact cohort it was fit on (each row needs to have appeared in at least
    one split's validation set). It has no `predict`-on-new-data mode — that
    is a design choice inherited from exp_24, not an omission.

    Returns (final_pred, votes_per_patient).
    """
    n = signal_matrix.shape[0]
    votes = [[] for _ in range(n)]
    for train_idx, val_idx in splits:
        dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=random_state)
        dt.fit(signal_matrix[train_idx], y_conf[train_idx])
        preds = dt.predict(signal_matrix[val_idx])
        for i, p in zip(val_idx, preds):
            votes[i].append(int(p))
    assert all(len(v) > 0 for v in votes), "a patient received zero held-out votes"
    final_pred = np.array([int(np.bincount(v, minlength=3).argmax()) for v in votes])
    votes_per_patient = np.array([len(v) for v in votes])
    return final_pred, votes_per_patient
