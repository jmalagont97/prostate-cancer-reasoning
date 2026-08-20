"""BrentMemKDM — a `MemKDM` whose only fitted quantity is the RBF bandwidth
`sigma`, one per modality, chosen by a GLOBAL derivative-free search of a
cross-validated metric (default: mean Macro-F1 over a set of MCCV folds).

This is deliberately not another `sigma_mult` grid point and not per-fold
Adam training (see `mem_kdm.py`'s module docstring and `experiments/exp_27`,
which showed per-fold gradient training makes Phase B 88 independent small
fits rather than a frozen-parameter evaluation). Here `x`, `y`, `w` are never
trained anywhere; the search fits exactly one scalar per modality, and it is
fit ONCE, globally, against the mean metric over all folds — never per fold.

For >1 modality the search is nested Brent: an outer 1-D Brent search over
`sigma_m1` whose every evaluation runs a full 1-D Brent search over
`sigma_m2` (recursively, over `sigma_m3`, ...). `strategy="coordinate"`
offers cyclic per-modality Brent sweeps as a cheaper (non-nested) fallback.

--------------------------------------------------------------------------
Why the objective can be evaluated ~1000x faster than a real `MemKDM.fit`
--------------------------------------------------------------------------
With `x_train=y_train=w_train=False`, identity encoders, and
`init_kdm_layer`'s data-driven init (`c_x = X_train`,
`c_y = to_amplitude(smooth(y_soft, label_smoothing))`, `c_w` uniform),
`MemKDM.predict_proba` collapses EXACTLY to a Nadaraya-Watson estimator.
Tracing `kdm.layers.kdm_layer.KDMLayer.forward`/`_compute_mixture` and
`kdm.utils.dm2discrete` for this parameter regime, component by component:

  in_w = 1                       (pure2dm: one input "component")
  k_j  = kernel(x, c_x_j)        (plain RBF value: exp(-d_j^2/(2*sigma^2))
                                   per leaf; a CrossProductKernelLayer takes
                                   the PRODUCT of per-modality plain values)
  raw_j = (1/n_train) * k_j^2    (KDMLayer squares the (product) kernel value
                                   and multiplies by the uniform comp_w)
  clamped_j = max(raw_j, 1e-12)  (KDMLayer's `out_w.clamp(min=self.eps)`,
                                   applied to raw_j, i.e. AFTER the 1/n_train
                                   scaling — not to k_j^2 directly; this is
                                   the exact point the fast and slow paths
                                   would diverge if this scaling were skipped)
  w_j  = clamped_j / sum_j clamped_j
  p1   = sum_j w_j * y_eff_j     (dm2discrete; c_y rows are unit-norm so its
                                   own L2-normalize is a no-op, and squaring
                                   the amplitude recovers y_eff_j exactly)

  where y_eff = smooth(y_soft_train, label_smoothing) (mem_kdm.smooth — the
  same function `MemKDM.fit` applies to build c_y's init) and, per leaf,
  k_j^2 for the PRODUCT kernel is `exp(-sum_m d_mj^2 / sigma_m^2)` — note the
  effective exponent is `-d^2/sigma^2`, NOT `-d^2/(2*sigma^2)`: squaring a
  `exp(-d^2/(2*sigma^2))` value doubles the exponent's denominator's inverse.

`_FoldCache` precomputes `d_mj^2` once per fold per modality (float32, same
`A_norm + B_norm - 2AB` expression as `RBFKernelLayer.forward`) and replays
exactly this sequence — division by n_train, THEN the 1e-12 clamp, THEN
normalize — for every sigma the search tries. `scripts/verify_brent_mem_kdm.py`
checks this reduction against a real `MemKDM.predict_proba` at both search
bounds (where the clamp is most likely to bite) and the center.

--------------------------------------------------------------------------
Optional k-NN truncation (`knn_k`)
--------------------------------------------------------------------------
`BrentMemKDM(knn_k=k)` retrieves, per query, the `k` memory points with the
largest product-kernel value (smallest `expo = sum_m d_mj^2/sigma_m^2` —
`knn_metric="kernel"`, the only value implemented) and applies the exact same
BrentMemKDM computation to only those `k`. Two things change relative to the
whole-memory path above, both load-bearing:

  - The divisor. `raw_j = k_j^2 / n_train` is computed BEFORE the `KDM_EPS`
    clamp (module docstring above), so truncating the sum to `k` points
    without also changing the divisor to `k_eff = min(k, n_train)` would
    silently renormalize over the wrong population whenever the clamp is
    live (`k2 < KDM_EPS * divisor`) — exactly the regime `exp_28`'s search
    landed in (`sigma_mult` 0.20-0.23, below every prior discrete grid
    point). `k_eff` is what makes the fast path equal a sub-`MemKDM` built
    with `n_comp = k_eff`, and what makes `knn_k >= n_train` identical to
    `knn_k=None`.
  - The neighbor set must be the SAME set in the fast (`_FoldCache`) and
    exact per-query (`_knn_submodel`) paths, or the two would only agree by
    coincidence. Both call `_topk_neighbors(expo, k)` — one helper, one
    tie-breaking rule (index order via `np.argpartition`, deterministic).

At `k=1`, `label_smoothing=0`, hard targets: the truncated mixture has a
single weight that normalizes to 1, so `p1 = y_soft[nearest]` exactly,
independent of sigma — i.e. this reduces exactly to a 1-NN classifier, the
same computation `exp_28`'s KNN reference arms compute. `knn_k` is therefore
a continuous family with `k=1` (1-NN) and `k=n_train` (the model above) as
its two ends.

`_sigma_ref_per_modality` (the search-bounds anchor) is NOT k-subsetted — it
stays a property of the full training fold, both because the leak-free
bounds argument depends on the full-fold scale and because `_sigma_from_knn`
(mean distance to the 3rd-NN) is undefined for `k < 4`. `k` itself is never
Brent-searched (Brent is a 1-D continuous method; `k` is discrete) — a
caller sweeps `k` in an outer loop, running one Brent sigma-search per `k`.

This module must never import `src/evaluation` (same rule as `mem_kdm.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from kdm.init import _sigma_from_knn

from .base import (
    Modalities,
    Targets,
    apply_meta_thresholds,
    fit_meta_thresholds_safe,
    fit_predict_heldout_trees,
)
from .mem_kdm import PARTICLE_SIGNAL_NAMES, EncoderSpec, KernelSpec, MemKDM, _best_1d_key, smooth

MIN_SIGMA = 1e-3
"""`RBFKernelLayer`'s structural floor (`sigma = softplus(raw) + min_sigma`);
it RAISES if constructed/assigned a sigma <= this. Search bounds are clipped
above it with margin (see `_bounds_for_modality`)."""

KDM_EPS = 1e-12
"""`KDMLayer.eps` — the clamp applied to `(1/n_train) * k^2` before
normalizing (see module docstring)."""


# ---------------------------------------------------------------------------
# Fold container — the unit of data the search consumes. Deliberately dumb:
# the caller is responsible for building X_train/X_val with per-split-fit
# preprocessors (e.g. `src/evaluation/data.py`'s `build_*_features`, each
# fit on `train_idx` only) so nothing here can leak. This module caches only
# the OUTPUTS of those transforms, never a fitted transformer.
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    X_train: Modalities
    y_soft_train: np.ndarray
    X_val: Modalities
    y_val: np.ndarray
    """(n_val,) int in {0,1} — evaluation ground truth for this fold's held-out rows."""

    def __post_init__(self) -> None:
        self.y_soft_train = np.asarray(self.y_soft_train, dtype=np.float32)
        self.y_val = np.asarray(self.y_val, dtype=np.int64)


def _validate_folds(folds: list, modality_order: list) -> None:
    """Checks shapes are internally consistent WITHIN each fold. Deliberately
    does NOT require a modality's dim to match across folds: per-split-fit
    transforms (`build_mri_features`/`build_text_features` with a variance-
    target PCA) legitimately yield a different component count per split —
    that's fine here since each fold's squared-distance matrix is computed
    independently (`_sq_dist_rbf`), never stacked across folds by dim."""
    if not folds:
        raise ValueError("at least one Fold is required")
    for i, f in enumerate(folds):
        for m in modality_order:
            if m not in f.X_train or m not in f.X_val:
                raise ValueError(f"fold {i} is missing modality {m!r}")
            d_tr = np.asarray(f.X_train[m]).shape[1]
            d_va = np.asarray(f.X_val[m]).shape[1]
            if d_tr != d_va:
                raise ValueError(f"fold {i} modality {m!r}: train dim {d_tr} != val dim {d_va}")
        first_mod = modality_order[0]
        if len(f.y_soft_train) != np.asarray(f.X_train[first_mod]).shape[0]:
            raise ValueError(f"fold {i}: y_soft_train length mismatches X_train row count")
        if len(f.y_val) != np.asarray(f.X_val[first_mod]).shape[0]:
            raise ValueError(f"fold {i}: y_val length mismatches X_val row count")


# ---------------------------------------------------------------------------
# Metrics — a scoring function is (y_true, p1_or_probs) -> float, higher is
# better. Named metrics get BOTH a vectorized batch form (used by the fast
# path's grouped scoring) and a per-fold scalar form (used by the torch
# backend and any metric without a vectorized form, e.g. AUROC). A custom
# metric is any `callable(y_true, probs_2col) -> float`; it always goes
# through the scalar per-fold path (no vectorized form is assumed for it).
# ---------------------------------------------------------------------------
def _to_2col(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1)
    return np.stack([1.0 - p1, p1], axis=-1)


def _binary_macro_f1_arrays(Y: np.ndarray, Pred: np.ndarray) -> np.ndarray:
    """Vectorized macro-F1 over the last axis, batched over any leading
    shape. Matches `sklearn.metrics.f1_score(y_true, y_pred,
    average="macro", zero_division=0)` EXACTLY for binary {0,1} labels,
    including sklearn's default `labels=None` behavior: a class absent from
    both y_true and y_pred (e.g. an all-class-1 fold) is excluded from the
    average rather than contributing a 0 — averaging over 1 class, not 2.
    Verified against sklearn on random + degenerate inputs in
    `scripts/verify_brent_mem_kdm.py`.
    """
    def per_class(cls: int):
        is_true, is_pred = (Y == cls), (Pred == cls)
        tp = (is_true & is_pred).sum(-1).astype(np.float64)
        fp = (~is_true & is_pred).sum(-1).astype(np.float64)
        fn = (is_true & ~is_pred).sum(-1).astype(np.float64)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), 0.0)
        present = is_true.any(-1) | is_pred.any(-1)
        return f1, present

    f1_0, present0 = per_class(0)
    f1_1, present1 = per_class(1)
    macro = np.zeros_like(f1_0)
    both = present0 & present1
    macro = np.where(both, (f1_0 + f1_1) / 2.0, macro)
    macro = np.where(present0 & ~present1, f1_0, macro)
    macro = np.where(present1 & ~present0, f1_1, macro)
    return macro


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scalar convenience wrapper around `_binary_macro_f1_arrays`, for
    direct comparison against sklearn in the verification script."""
    Y = np.asarray(y_true, dtype=np.int64)[None, :]
    P = np.asarray(y_pred, dtype=np.int64)[None, :]
    return float(_binary_macro_f1_arrays(Y, P)[0])


def _f1_macro_vec(Y: np.ndarray, P1: np.ndarray, threshold: float) -> np.ndarray:
    return _binary_macro_f1_arrays(Y, (P1 >= threshold).astype(np.int64))


def _f1_macro_scalar(y_true: np.ndarray, p1: np.ndarray, threshold: float) -> float:
    return binary_macro_f1(y_true, (np.asarray(p1) >= threshold).astype(np.int64))


def _accuracy_vec(Y: np.ndarray, P1: np.ndarray, threshold: float) -> np.ndarray:
    pred = (P1 >= threshold).astype(np.int64)
    return (pred == Y).mean(-1)


def _accuracy_scalar(y_true: np.ndarray, p1: np.ndarray, threshold: float) -> float:
    pred = (np.asarray(p1) >= threshold).astype(int)
    return float(accuracy_score(y_true, pred))


def _neg_brier_vec(Y: np.ndarray, P1: np.ndarray, threshold: float) -> np.ndarray:
    return -((P1 - Y.astype(np.float64)) ** 2).mean(-1)


def _neg_brier_scalar(y_true: np.ndarray, p1: np.ndarray, threshold: float) -> float:
    return -float(brier_score_loss(y_true, p1))


def _neg_log_loss_vec(Y: np.ndarray, P1: np.ndarray, threshold: float, eps: float = 1e-12) -> np.ndarray:
    p1c = np.clip(P1, eps, 1 - eps)
    y = Y.astype(np.float64)
    return (y * np.log(p1c) + (1 - y) * np.log(1 - p1c)).mean(-1)


def _neg_log_loss_scalar(y_true: np.ndarray, p1: np.ndarray, threshold: float, eps: float = 1e-12) -> float:
    p1c = np.clip(np.asarray(p1, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean(y * np.log(p1c) + (1 - y) * np.log(1 - p1c)))


def _auroc_scalar(y_true: np.ndarray, p1: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5  # undefined on a single-class fold; chance-level fallback (documented, not silent)
    return float(roc_auc_score(y_true, p1))


@dataclass
class _Metric:
    vectorized: Callable | None  # (Y: (G,n), P1: (G,n), threshold) -> (G,) | None if unavailable
    scalar: Callable  # (y_true: (n,), p1: (n,), threshold) -> float


METRICS: dict[str, _Metric] = {
    "macro_f1": _Metric(_f1_macro_vec, _f1_macro_scalar),
    "accuracy": _Metric(_accuracy_vec, _accuracy_scalar),
    "neg_brier": _Metric(_neg_brier_vec, _neg_brier_scalar),
    "neg_log_loss": _Metric(_neg_log_loss_vec, _neg_log_loss_scalar),
    "auroc": _Metric(None, _auroc_scalar),  # not batch-vectorized: falls back to a per-fold loop
}


# ---------------------------------------------------------------------------
# Fast backend — the Nadaraya-Watson reduction from the module docstring,
# vectorized over folds grouped by (n_val, n_train) shape.
# ---------------------------------------------------------------------------
def _sq_dist_rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`||a-b||^2` for every (a in A, b in B) pair, replicating
    `RBFKernelLayer.forward`'s exact expression and dtype (float32) so the
    fast path's rounding matches the torch path's, not just its formula."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    AB = A @ B.T
    a_norm = (A ** 2).sum(-1, keepdims=True)
    b_norm = (B ** 2).sum(-1)[None, :]
    dist2 = a_norm + b_norm - 2.0 * AB
    np.clip(dist2, 0.0, None, out=dist2)
    return dist2.astype(np.float32)


def _topk_neighbors(expo: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` smallest entries of `expo` along the last axis —
    equivalently the `k` memory points with the largest product-kernel value
    (largest `exp(-expo)`). Single source of truth for "which points are the
    k nearest neighbors" (module docstring's k-NN section): both
    `_FoldCache.probs`'s knn branch (fast path) and `_knn_submodel` (exact
    per-query path) call this, so the two paths always agree on the neighbor
    set rather than only on the formula. `np.argpartition`'s tie-breaking is
    by index — an arbitrary but deterministic choice among equidistant
    points, and both paths make the SAME choice because both call this."""
    n_train = expo.shape[-1]
    k = min(int(k), n_train)
    if k == n_train:
        return np.argsort(expo, axis=-1)
    part = np.argpartition(expo, k - 1, axis=-1)[..., :k]
    part_expo = np.take_along_axis(expo, part, axis=-1)
    order = np.argsort(part_expo, axis=-1)
    return np.take_along_axis(part, order, axis=-1)


def _h_b(p: float) -> float:
    """Binary entropy in nats, exact zero at p=0/1 (no log(0) via clipping —
    matters for exp_30's G3 exact-degeneracy check at knn_k=1)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _neighborhood_signals(y_binary_train: np.ndarray, nbr: np.ndarray, expo_nbr: np.ndarray) -> dict:
    """Family-C (exp_30): signals definable only because `knn_k` retrieves a
    literal, finite neighbor set. Both inputs are non-target-informed even
    when the underlying model was fit on confidence-derived soft targets:
    `y_binary_train` is the raw biopsy label (never `confidence`), and
    `expo_nbr` is a pure input-space kernel exponent, independent of
    `y_soft`/`c_y`. `expo_nbr` is ascending (nearest first, from
    `_topk_neighbors`), so its last entry is the farthest retrieved
    neighbor — the k-th-neighbor distance in kernel-exponent units."""
    p = float(y_binary_train[nbr].mean())
    return {
        "nbr_label_entropy": _h_b(p),
        "nbr_kth_expo": float(expo_nbr[-1]),
    }


def _knn_submodel(
    X_train: dict, y_soft_train: np.ndarray, modality_order: list, sigmas: dict, k: int, x_row: dict,
    label_smoothing: float = 0.0, seed: int = 0,
) -> tuple:
    """Builds and fits a `k_eff`-memory `MemKDM` for ONE query row `x_row`
    (a dict of `(1, dim)` arrays), retrieving its nearest neighbors by the
    same `expo` rule `_FoldCache.probs`'s knn branch uses (via
    `_topk_neighbors`, so the fast and exact per-query paths always agree on
    the neighbor set). Shared by `_TorchScorer.score(knn_k=...)` (the exact
    reference `scripts/verify_brent_mem_kdm.py` checks the fast path
    against) and `BrentMemKDM._knn_signals` (the actual knn-mode prediction
    path) — "apply BrentMemKDM to the k retrieved neighbors" means exactly
    the same computation in both places. Returns `(fitted MemKDM,
    neighbor_indices, expo_at_neighbors)` — the third element is the
    per-neighbor kernel exponent (ascending, nearest first), consumed by
    `_neighborhood_signals` (exp_30's family-C signals)."""
    n_train = len(y_soft_train)
    k_eff = min(int(k), n_train)
    expo = np.zeros((1, n_train), dtype=np.float32)
    for m in modality_order:
        d2 = _sq_dist_rbf(np.asarray(x_row[m]), np.asarray(X_train[m]))
        expo += d2 / (float(sigmas[m]) ** 2)
    nbr = _topk_neighbors(expo, k_eff)[0]
    expo_nbr = expo[0, nbr]
    kernels = {m: KernelSpec(sigma=float(sigmas[m]), trainable=False) for m in modality_order}
    encoders = {m: EncoderSpec("identity") for m in modality_order}
    sub_X = {m: np.asarray(X_train[m])[nbr] for m in modality_order}
    sub_y = np.asarray(y_soft_train)[nbr]
    model = MemKDM(kernels=kernels, encoders=encoders, x_train=False, y_train=False, w_train=False,
                    label_smoothing=label_smoothing, seed=seed)
    model.fit(sub_X, Targets(y_binary=np.zeros(len(sub_y), dtype=int), y_soft=sub_y))
    return model, nbr, expo_nbr


class _FoldCache:
    """Precomputes, once, the per-fold per-modality squared-distance
    matrices `(n_val, n_train)`; every subsequent sigma evaluation is a
    handful of vectorized array ops over folds grouped by matching
    `(n_val, n_train)` shape (ragged shapes just form singleton groups —
    correct, only less vectorized)."""

    def __init__(self, folds: list, modality_order: list, label_smoothing: float = 0.0):
        self.modality_order = list(modality_order)
        self.n_folds = len(folds)
        self.y_val = [f.y_val for f in folds]
        self.n_train = [len(f.y_soft_train) for f in folds]
        self.y_eff = [smooth(f.y_soft_train, label_smoothing).astype(np.float64) for f in folds]
        self.dist2: dict[str, list] = {m: [] for m in self.modality_order}
        for f in folds:
            for m in self.modality_order:
                self.dist2[m].append(_sq_dist_rbf(f.X_val[m], f.X_train[m]))

        self._groups: dict[tuple, list] = {}
        for i, f in enumerate(folds):
            shape = (len(f.y_val), len(f.y_soft_train))
            self._groups.setdefault(shape, []).append(i)

    def probs(self, sigmas: dict, knn_k: int | None = None) -> list:
        """Per-fold class-1 probability arrays `(n_val,)`, via the fast
        Nadaraya-Watson reduction (module docstring). The single place the
        fast-path math lives — `score()` calls this, and
        `scripts/verify_brent_mem_kdm.py` compares it directly against a
        real `MemKDM.predict_proba`.

        `knn_k=None` (default) takes the whole-memory path below, byte-for-
        byte unchanged. `knn_k=k` truncates each query's mixture to its
        `k_eff = min(k, n_train)` nearest memory points (module docstring's
        k-NN section) — `_topk_neighbors` on `expo` selects them, the
        `1/n_train` divisor becomes `1/k_eff`, and the `KDM_EPS` clamp /
        normalize / contraction against `y_eff` are applied to the truncated
        set instead of the full one."""
        out = [None] * self.n_folds
        for (n_val, n_train), idxs in self._groups.items():
            expo = np.zeros((len(idxs), n_val, n_train), dtype=np.float32)
            for m in self.modality_order:
                sigma2 = float(sigmas[m]) ** 2
                stacked = np.stack([self.dist2[m][i] for i in idxs], axis=0)
                expo += stacked / sigma2
            y_eff_stack = np.stack([self.y_eff[i] for i in idxs], axis=0)

            if knn_k is None:
                k2 = np.exp(-expo)
                raw = k2 / n_train  # the uniform comp_w=1/n_comp factor, applied BEFORE the clamp below
                np.maximum(raw, KDM_EPS, out=raw)
                w = raw / raw.sum(-1, keepdims=True)
                p1 = np.clip(np.einsum("gvt,gt->gv", w, y_eff_stack), 0.0, 1.0)
            else:
                k_eff = min(int(knn_k), n_train)
                nbr_idx = _topk_neighbors(expo, k_eff)  # (len(idxs), n_val, k_eff)
                expo_k = np.take_along_axis(expo, nbr_idx, axis=-1)
                k2 = np.exp(-expo_k)
                raw = k2 / k_eff  # the k_eff-uniform comp_w factor, applied BEFORE the clamp below
                np.maximum(raw, KDM_EPS, out=raw)
                w = raw / raw.sum(-1, keepdims=True)
                g_idx = np.arange(len(idxs))[:, None, None]
                y_eff_k = y_eff_stack[g_idx, nbr_idx]  # (len(idxs), n_val, k_eff)
                p1 = np.clip(np.einsum("gvk,gvk->gv", w, y_eff_k), 0.0, 1.0)

            for j, i in enumerate(idxs):
                out[i] = p1[j]
        return out

    def score(self, sigmas: dict, metric, threshold: float, aggregate: str, knn_k: int | None = None):
        p1_per_fold = self.probs(sigmas, knn_k=knn_k)
        per_fold = np.empty(self.n_folds, dtype=np.float64)
        is_named = isinstance(metric, str)
        entry = METRICS[metric] if is_named else None

        for (n_val, n_train), idxs in self._groups.items():
            p1 = np.stack([p1_per_fold[i] for i in idxs], axis=0)
            y_true_stack = np.stack([self.y_val[i] for i in idxs], axis=0)

            if is_named and entry.vectorized is not None:
                scores = entry.vectorized(y_true_stack, p1, threshold)
            else:
                scores = np.empty(len(idxs))
                for g in range(len(idxs)):
                    if is_named:
                        scores[g] = entry.scalar(y_true_stack[g], p1[g], threshold)
                    else:
                        scores[g] = metric(y_true_stack[g], _to_2col(p1[g]))
            for j, i in enumerate(idxs):
                per_fold[i] = scores[j]

        return _aggregate(per_fold, aggregate), per_fold


class _TorchScorer:
    """Exact-but-slow fold scorer: builds and fits a real `MemKDM` per fold
    per evaluation. Never on the hot path of a real search (`backend="fast"`
    covers that) — used for `backend="torch"` and as the reference the
    verification script checks the fast path against."""

    def __init__(self, folds: list, modality_order: list, label_smoothing: float = 0.0, seed: int = 0):
        self.folds = folds
        self.modality_order = list(modality_order)
        self.label_smoothing = label_smoothing
        self.seed = seed

    def score(self, sigmas: dict, metric, threshold: float, aggregate: str, knn_k: int | None = None):
        per_fold = np.empty(len(self.folds), dtype=np.float64)
        is_named = isinstance(metric, str)
        entry = METRICS[metric] if is_named else None
        for i, f in enumerate(self.folds):
            if knn_k is None:
                kernels = {m: KernelSpec(sigma=float(sigmas[m]), trainable=False) for m in self.modality_order}
                encoders = {m: EncoderSpec("identity") for m in self.modality_order}
                model = MemKDM(kernels=kernels, encoders=encoders, x_train=False, y_train=False, w_train=False,
                                label_smoothing=self.label_smoothing, seed=self.seed)
                # `Targets.y_binary` is evaluation-only and never read by `MemKDM.fit` (see base.py); this
                # dummy same-length-as-y_soft array is filler, not a semantic input.
                dummy_y_binary = np.zeros(len(f.y_soft_train), dtype=int)
                model.fit(f.X_train, Targets(y_binary=dummy_y_binary, y_soft=f.y_soft_train))
                probs = model.predict_proba(f.X_val)
                p1 = probs[:, 1]
            else:
                # Exact per-query knn path: one fresh k_eff-memory MemKDM per val row, via the same
                # `_knn_submodel` helper the fast path's `probs(knn_k=...)` is checked against.
                n_val = len(f.y_val)
                p1 = np.empty(n_val, dtype=np.float64)
                for j in range(n_val):
                    x_row = {m: np.asarray(f.X_val[m])[j:j + 1] for m in self.modality_order}
                    sub_model, _nbr, _expo_nbr = _knn_submodel(f.X_train, f.y_soft_train, self.modality_order, sigmas,
                                                                knn_k, x_row, label_smoothing=self.label_smoothing,
                                                                seed=self.seed)
                    p1[j] = sub_model.predict_proba(x_row)[0, 1]
                probs = _to_2col(p1)
            if is_named:
                per_fold[i] = entry.scalar(f.y_val, p1, threshold)
            else:
                per_fold[i] = metric(f.y_val, probs)
        return _aggregate(per_fold, aggregate), per_fold


def _aggregate(per_fold: np.ndarray, aggregate: str) -> float:
    if aggregate == "mean":
        return float(per_fold.mean())
    if aggregate == "median":
        return float(np.median(per_fold))
    raise ValueError(f"unknown aggregate: {aggregate!r}")


# ---------------------------------------------------------------------------
# Bounds — anchored to the same data-driven scale MemKDM itself uses
# (`kdm.init._sigma_from_knn`, mean distance to the 3rd-NN), computed on
# TRAIN blocks only so the search bounds are leak-free.
# ---------------------------------------------------------------------------
def _sigma_ref_per_modality(folds: list, modality_order: list) -> dict:
    refs = {m: [] for m in modality_order}
    for f in folds:
        for m in modality_order:
            refs[m].append(_sigma_from_knn(np.asarray(f.X_train[m]), 1.0))
    return {m: float(np.mean(v)) for m, v in refs.items()}


def _bounds_for_modality(sigma_ref: float, bounds_mult: tuple) -> tuple:
    lo = max(sigma_ref * bounds_mult[0], 1.05 * MIN_SIGMA)
    hi = sigma_ref * bounds_mult[1]
    if hi <= lo:
        hi = lo * 2.0
    return lo, hi


# ---------------------------------------------------------------------------
# 1-D bounded Brent with a log-sigma pre-scan bracket. Mean Macro-F1 (the
# default metric) is piecewise-constant in sigma — predictions only change
# when some p1 crosses the decision threshold — so plain Brent can converge
# to an arbitrary point on a flat plateau. The pre-scan turns "assume
# unimodal" into "unimodal within the winning coarse bracket", and the final
# `max(prescan_best, brent_result)` means a degenerate/flat bracket can never
# make the 1-D search regress below its own coarse grid.
# ---------------------------------------------------------------------------
def _brent_1d(objective: Callable, lo: float, hi: float, n_prescan: int, xatol: float, maxiter: int):
    if n_prescan < 2:
        raise ValueError("n_prescan must be >= 2")
    log_lo, log_hi = math.log(lo), math.log(hi)
    xs = np.linspace(log_lo, log_hi, n_prescan)
    scores = np.array([objective(math.exp(x)) for x in xs])
    best_i = int(np.argmax(scores))
    a = xs[max(best_i - 1, 0)]
    b = xs[min(best_i + 1, n_prescan - 1)]
    if a >= b:
        a, b = log_lo, log_hi

    res = minimize_scalar(lambda log_sigma: -objective(math.exp(log_sigma)),
                           bounds=(a, b), method="bounded",
                           options={"xatol": xatol, "maxiter": maxiter})
    cand_sigma, cand_score = math.exp(float(res.x)), -float(res.fun)

    if scores[best_i] >= cand_score:
        return math.exp(float(xs[best_i])), float(scores[best_i])
    return cand_sigma, cand_score


class _BestTracker:
    """Records every LEAF evaluation (a full sigma vector + its score), and
    returns the global argmax over all of them. Required under nesting: the
    outer Brent's own returned optimum is only as good as the candidate
    points it happened to try, which need not include the true best combo
    the inner searches encountered along the way."""

    def __init__(self):
        self.trace: list = []
        self._best = None  # (score, sigmas, per_fold)

    def record(self, sigmas: dict, score: float, per_fold: np.ndarray, level: int):
        self.trace.append({"sigmas": dict(sigmas), "score": float(score), "level": int(level)})
        if self._best is None or score > self._best[0]:
            self._best = (float(score), dict(sigmas), np.array(per_fold, copy=True))

    def best(self):
        if self._best is None:
            raise RuntimeError("no evaluations recorded")
        return self._best


def _nested_search(remaining: list, sigmas_so_far: dict, cache_score_fn: Callable, bounds: dict,
                    n_prescan: int, xatol: float, maxiter: int, tracker: _BestTracker, level: int) -> float:
    name = remaining[0]
    lo, hi = bounds[name]
    if len(remaining) == 1:
        def objective(sigma):
            sigmas = {**sigmas_so_far, name: sigma}
            score, per_fold = cache_score_fn(sigmas)
            tracker.record(sigmas, score, per_fold, level)
            return score
    else:
        rest = remaining[1:]

        def objective(sigma):
            sigmas = {**sigmas_so_far, name: sigma}
            return _nested_search(rest, sigmas, cache_score_fn, bounds, n_prescan, xatol, maxiter, tracker, level + 1)

    _, best_score = _brent_1d(objective, lo, hi, n_prescan, xatol, maxiter)
    return best_score


def _coordinate_search(modality_order: list, bounds: dict, cache_score_fn: Callable,
                        n_prescan: int, xatol: float, maxiter: int, max_rounds: int, tracker: _BestTracker) -> dict:
    sigmas = {m: math.sqrt(bounds[m][0] * bounds[m][1]) for m in modality_order}  # geometric-mean init
    prev_score = None
    for rnd in range(max_rounds):
        for name in modality_order:
            lo, hi = bounds[name]

            def objective(sigma, _name=name):
                trial = {**sigmas, _name: sigma}
                score, per_fold = cache_score_fn(trial)
                tracker.record(trial, score, per_fold, rnd)
                return score

            best_sigma, _ = _brent_1d(objective, lo, hi, n_prescan, xatol, maxiter)
            sigmas[name] = best_sigma
        cur_score, _ = cache_score_fn(sigmas)
        if prev_score is not None and cur_score <= prev_score + 1e-12:
            break
        prev_score = cur_score
    return sigmas


# ---------------------------------------------------------------------------
# SigmaSearchResult / run_brent_search — the module-level search entry point.
# ---------------------------------------------------------------------------
@dataclass
class SigmaSearchResult:
    sigmas: dict
    score: float
    per_fold_scores: dict  # {"mean": ..., "std": ..., "values": [...]}
    sigma_ref: dict
    sigma_mult: dict  # sigmas[m] / sigma_ref[m] — comparable to exp_27's grid winners
    bounds: dict
    n_evals: int
    strategy: str
    metric: str
    trace: list = field(default_factory=list)
    knn_k: int | None = None
    """`None` = whole-memory (default); an int = per-query k-NN truncation
    (module docstring's k-NN section) used while fitting this result."""


def run_brent_search(
    folds: list,
    modality_order: list,
    metric="macro_f1",
    strategy: str = "nested",
    bounds_mult: tuple = (1.0 / 32, 32.0),
    n_prescan: int = 15,
    xatol: float = 1e-2,
    maxiter: int = 20,
    max_rounds: int = 5,
    aggregate: str = "mean",
    label_smoothing: float = 0.0,
    threshold: float = 0.50,
    backend: str = "auto",
    knn_k: int | None = None,
) -> SigmaSearchResult:
    """Global Brent search for one sigma per modality, maximizing the mean
    (or `aggregate`) `metric` over `folds`. `backend="auto"`/`"fast"` uses the
    Nadaraya-Watson reduction (module docstring); `"torch"` fits a real
    `MemKDM` per fold per evaluation — exact, ~1000x slower, intended for
    verification, not routine searches. `knn_k` (module docstring's k-NN
    section): `None` scores every sigma candidate against the whole memory
    (unchanged); an int truncates each query to its `k_eff = min(knn_k,
    n_train)` nearest memory points before scoring — `k` itself is not
    searched here, sweep it in an outer loop.
    """
    modality_order = list(modality_order)
    if isinstance(metric, str) and metric not in METRICS:
        raise ValueError(f"unknown metric: {metric!r}; known: {sorted(METRICS)}")
    _validate_folds(folds, modality_order)

    if backend in ("auto", "fast"):
        scorer = _FoldCache(folds, modality_order, label_smoothing=label_smoothing)
    elif backend == "torch":
        scorer = _TorchScorer(folds, modality_order, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    def cache_score_fn(sigmas):
        return scorer.score(sigmas, metric, threshold, aggregate, knn_k=knn_k)

    sigma_ref = _sigma_ref_per_modality(folds, modality_order)
    bounds = {m: _bounds_for_modality(sigma_ref[m], bounds_mult) for m in modality_order}

    tracker = _BestTracker()
    if strategy == "nested":
        _nested_search(modality_order, {}, cache_score_fn, bounds, n_prescan, xatol, maxiter, tracker, level=0)
    elif strategy == "coordinate":
        _coordinate_search(modality_order, bounds, cache_score_fn, n_prescan, xatol, maxiter, max_rounds, tracker)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    best_score, best_sigmas, best_per_fold = tracker.best()
    return SigmaSearchResult(
        sigmas=best_sigmas,
        score=best_score,
        per_fold_scores={"mean": float(best_per_fold.mean()), "std": float(best_per_fold.std()),
                          "values": best_per_fold.tolist()},
        sigma_ref=sigma_ref,
        sigma_mult={m: best_sigmas[m] / sigma_ref[m] for m in modality_order},
        bounds=bounds,
        n_evals=len(tracker.trace),
        strategy=strategy,
        metric=metric if isinstance(metric, str) else "custom",
        trace=tracker.trace,
        knn_k=knn_k,
    )


# ---------------------------------------------------------------------------
# BrentMemKDM — the `Method`-protocol wrapper (see base.Method).
# ---------------------------------------------------------------------------
class BrentMemKDM:
    """`search(folds)` is the global Phase-A operation (picks and freezes
    `self.sigmas_`); `fit(X, targets)` builds and fits the frozen-sigma
    `MemKDM` (a pure init — `x_train=y_train=w_train=False` and every kernel
    `trainable=False` means `MemKDM.fit` takes the zero-trainable-parameters
    branch, mem_kdm.py's `has_trainable` check — so `fit` is deterministic
    and seed-independent; a consuming experiment needs no seed averaging).

    `knn_k` (module docstring's k-NN section, default `None`): when set,
    `fit` does not build one whole-memory `MemKDM` — instead it stores the
    training data, and every predict/uncertainty call retrieves each query's
    `k_eff = min(knn_k, n_train)` nearest memory points (by kernel value,
    `knn_metric="kernel"` — the only value implemented) and fits a fresh
    `k_eff`-memory `MemKDM` on just that subset (`_knn_submodel`). `knn_k`
    must be set in `__init__` (not `search()`): `experiments/exp_28` builds
    instances via `m = BrentMemKDM(); m.sigmas_ = {...}; m.modality_order =
    [...]; m.fit(...)`, bypassing `search()` entirely, so any state `search()`
    alone set would silently leave that call pattern unrestricted.
    """

    def __init__(
        self,
        metric="macro_f1",
        strategy: str = "nested",
        bounds_mult: tuple = (1.0 / 32, 32.0),
        n_prescan: int = 15,
        xatol: float = 1e-2,
        maxiter: int = 20,
        max_rounds: int = 5,
        aggregate: str = "mean",
        label_smoothing: float = 0.0,
        threshold: float = 0.50,
        modality_order: list | None = None,
        backend: str = "auto",
        seed: int = 0,
        knn_k: int | None = None,
        knn_metric: str = "kernel",
    ):
        self.metric = metric
        self.strategy = strategy
        self.bounds_mult = bounds_mult
        self.n_prescan = n_prescan
        self.xatol = xatol
        self.maxiter = maxiter
        self.max_rounds = max_rounds
        self.aggregate = aggregate
        self.label_smoothing = label_smoothing
        self.threshold = threshold
        self.modality_order = list(modality_order) if modality_order is not None else None
        self.backend = backend
        self.seed = seed
        self.knn_k = knn_k
        if knn_metric != "kernel":
            raise ValueError(f"unknown knn_metric: {knn_metric!r}; only 'kernel' is implemented")
        self.knn_metric = knn_metric

        self.sigmas_: dict | None = None
        self.result_: SigmaSearchResult | None = None
        self._inner: MemKDM | None = None
        self._confidence: dict | None = None
        self.target_informed = False

        # knn-mode fitted state (self._inner stays None in this mode).
        self._train_X: Modalities | None = None
        self._train_y_binary: np.ndarray | None = None
        self._train_y_soft: np.ndarray | None = None

    # ---------------------------------------------------------------- search
    def search(self, folds: list, modality_order: list | None = None) -> SigmaSearchResult:
        order = list(modality_order) if modality_order is not None else self.modality_order
        if order is None:
            order = list(folds[0].X_train.keys())
        self.modality_order = order
        self.result_ = run_brent_search(
            folds, order, metric=self.metric, strategy=self.strategy, bounds_mult=self.bounds_mult,
            n_prescan=self.n_prescan, xatol=self.xatol, maxiter=self.maxiter, max_rounds=self.max_rounds,
            aggregate=self.aggregate, label_smoothing=self.label_smoothing, threshold=self.threshold,
            backend=self.backend, knn_k=self.knn_k,
        )
        self.sigmas_ = dict(self.result_.sigmas)
        return self.result_

    # ------------------------------------------------------------------ fit
    def to_memkdm(self, **kwargs) -> MemKDM:
        if self.sigmas_ is None:
            raise RuntimeError("BrentMemKDM.search() must be called before to_memkdm()/fit()")
        kernels = {m: KernelSpec(sigma=self.sigmas_[m], trainable=False) for m in self.modality_order}
        encoders = {m: EncoderSpec("identity") for m in self.modality_order}
        params = dict(x_train=False, y_train=False, w_train=False,
                      label_smoothing=self.label_smoothing, seed=self.seed)
        params.update(kwargs)
        return MemKDM(kernels=kernels, encoders=encoders, **params)

    def fit(self, X: Modalities, targets: Targets) -> "BrentMemKDM":
        if self.knn_k is None:
            self._inner = self.to_memkdm().fit(X, targets)
            self.target_informed = self._inner.target_informed
            return self
        if self.sigmas_ is None:
            raise RuntimeError("BrentMemKDM.search() must be called before fit() (or sigmas_/modality_order set directly)")
        self._inner = None
        self._train_X = {m: np.asarray(X[m]) for m in self.modality_order}
        self._train_y_binary = np.asarray(targets.y_binary, dtype=np.int64)
        self._train_y_soft = np.clip(np.asarray(targets.y_soft, dtype=np.float32), 0.0, 1.0)
        self.target_informed = bool(targets.soft_from_confidence)
        self._confidence = None
        return self

    def _require_fit(self) -> MemKDM:
        if self._inner is None:
            raise RuntimeError("BrentMemKDM.fit() must be called before predict/uncertainty methods")
        return self._inner

    def _require_knn_fit(self) -> tuple:
        if self._train_X is None:
            raise RuntimeError("BrentMemKDM.fit() must be called before predict/uncertainty methods")
        return self._train_X, self._train_y_soft

    # -------------------------------------------------------------- knn mode
    def _knn_signals(self, X: Modalities) -> dict:
        """knn-mode `uncertainty_signals`: for each query row, retrieves its
        `k_eff` nearest memory points and fits a fresh `k_eff`-memory
        `MemKDM` on just that subset (`_knn_submodel`), then reads that
        submodel's own `uncertainty_signals` on the single row. Concatenated
        over rows into the same dict shape `mem_kdm.extract_particle_signals`
        returns, plus two exp_30 family-C keys (`nbr_label_entropy`,
        `nbr_kth_expo`, see `_neighborhood_signals`) that only exist in this
        knn-mode branch — the whole-memory path (`knn_k=None`) has no
        literal neighbor set to define them over. `log_marginal` here is
        over `k_eff` components rather than `n_train`, so its scale differs
        from non-knn mode — the meta-threshold heads below refit thresholds
        against this model's own signals regardless, so that is a
        documentation note, not a correctness issue."""
        X_train, y_soft_train = self._require_knn_fit()
        y_binary_train = self._train_y_binary
        n_val = len(np.asarray(X[self.modality_order[0]]))
        fam_c_keys = ("nbr_label_entropy", "nbr_kth_expo")
        collected = {key: [] for key in ("probs",) + PARTICLE_SIGNAL_NAMES + fam_c_keys}
        for i in range(n_val):
            x_row = {m: np.asarray(X[m])[i:i + 1] for m in self.modality_order}
            model, nbr, expo_nbr = _knn_submodel(X_train, y_soft_train, self.modality_order, self.sigmas_,
                                                  self.knn_k, x_row, label_smoothing=self.label_smoothing,
                                                  seed=self.seed)
            sig = model.uncertainty_signals(x_row)
            for key in ("probs",) + PARTICLE_SIGNAL_NAMES:
                collected[key].append(sig[key][0])
            fam_c = _neighborhood_signals(y_binary_train, nbr, expo_nbr)
            for key in fam_c_keys:
                collected[key].append(fam_c[key])
        return {key: (np.stack(vals, axis=0) if key == "probs" else np.array(vals))
                for key, vals in collected.items()}

    # ------------------------------------------------------------- predict
    def predict_proba(self, X: Modalities) -> np.ndarray:
        if self.knn_k is None:
            return self._require_fit().predict_proba(X)
        return self._knn_signals(X)["probs"]

    def predict(self, X: Modalities, threshold: float = 0.50) -> np.ndarray:
        if self.knn_k is None:
            return self._require_fit().predict(X, threshold=threshold)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def uncertainty_signals(self, X: Modalities) -> dict:
        if self.knn_k is None:
            return self._require_fit().uncertainty_signals(X)
        return self._knn_signals(X)

    # ---------------------------------------------------------- confidence
    def fit_confidence(self, y_conf: np.ndarray, splits: list, head: str = "meta_threshold_1d",
                        X: Modalities | None = None, key: str | None = None) -> "BrentMemKDM":
        if self.knn_k is None:
            self._require_fit().fit_confidence(y_conf, splits, head=head, X=X, key=key)
            return self
        if X is None:
            raise ValueError("fit_confidence requires X (the data to compute uncertainty_signals on)")
        signals = self._knn_signals(X)
        if head == "meta_threshold_1d":
            sig_key = key or _best_1d_key(signals, y_conf, splits)
            thr = fit_meta_thresholds_safe(signals[sig_key], y_conf, splits)
            self._confidence = {"head": head, "key": sig_key, "thr": thr}
        elif head == "multivariate_heldout":
            keys = sorted(k for k in signals if k != "probs")
            S = np.stack([signals[k] for k in keys], axis=1)
            pred, votes = fit_predict_heldout_trees(S, y_conf, splits)
            self._confidence = {"head": head, "keys": keys, "_heldout_pred": pred, "_heldout_votes": votes}
        else:
            raise ValueError(f"unknown confidence head: {head!r}")
        return self

    def predict_confidence(self, X: Modalities | None = None) -> np.ndarray:
        if self.knn_k is None:
            return self._require_fit().predict_confidence(X)
        if self._confidence is None:
            raise RuntimeError("BrentMemKDM.fit_confidence() must be called before predict_confidence")
        head = self._confidence["head"]
        if head == "meta_threshold_1d":
            if X is None:
                raise ValueError("predict_confidence requires X for head='meta_threshold_1d'")
            signal = self._knn_signals(X)[self._confidence["key"]]
            return apply_meta_thresholds(signal, self._confidence["thr"])
        if head == "multivariate_heldout":
            return self._confidence["_heldout_pred"]
        raise ValueError(f"unknown confidence head: {head!r}")

    # --------------------------------------------------------------- kernel
    def kernel_params(self) -> dict:
        if self.knn_k is None:
            return self._require_fit().kernel_params()
        if self.sigmas_ is None:
            raise RuntimeError("BrentMemKDM.search() must be called before kernel_params() (or sigmas_ set directly)")
        return {m: KernelSpec(sigma=self.sigmas_[m], trainable=False) for m in self.modality_order}
