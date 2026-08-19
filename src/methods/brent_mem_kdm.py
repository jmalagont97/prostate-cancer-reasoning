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

from .base import Modalities, Targets
from .mem_kdm import EncoderSpec, KernelSpec, MemKDM, smooth

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

    def probs(self, sigmas: dict) -> list:
        """Per-fold class-1 probability arrays `(n_val,)`, via the fast
        Nadaraya-Watson reduction (module docstring). The single place the
        fast-path math lives — `score()` calls this, and
        `scripts/verify_brent_mem_kdm.py` compares it directly against a
        real `MemKDM.predict_proba`."""
        out = [None] * self.n_folds
        for (n_val, n_train), idxs in self._groups.items():
            expo = np.zeros((len(idxs), n_val, n_train), dtype=np.float32)
            for m in self.modality_order:
                sigma2 = float(sigmas[m]) ** 2
                stacked = np.stack([self.dist2[m][i] for i in idxs], axis=0)
                expo += stacked / sigma2
            k2 = np.exp(-expo)
            raw = k2 / n_train  # the uniform comp_w=1/n_comp factor, applied BEFORE the clamp below
            np.maximum(raw, KDM_EPS, out=raw)
            w = raw / raw.sum(-1, keepdims=True)
            y_eff_stack = np.stack([self.y_eff[i] for i in idxs], axis=0)
            p1 = np.clip(np.einsum("gvt,gt->gv", w, y_eff_stack), 0.0, 1.0)
            for j, i in enumerate(idxs):
                out[i] = p1[j]
        return out

    def score(self, sigmas: dict, metric, threshold: float, aggregate: str):
        p1_per_fold = self.probs(sigmas)
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

    def score(self, sigmas: dict, metric, threshold: float, aggregate: str):
        per_fold = np.empty(len(self.folds), dtype=np.float64)
        is_named = isinstance(metric, str)
        entry = METRICS[metric] if is_named else None
        for i, f in enumerate(self.folds):
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
) -> SigmaSearchResult:
    """Global Brent search for one sigma per modality, maximizing the mean
    (or `aggregate`) `metric` over `folds`. `backend="auto"`/`"fast"` uses the
    Nadaraya-Watson reduction (module docstring); `"torch"` fits a real
    `MemKDM` per fold per evaluation — exact, ~1000x slower, intended for
    verification, not routine searches.
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
        return scorer.score(sigmas, metric, threshold, aggregate)

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

        self.sigmas_: dict | None = None
        self.result_: SigmaSearchResult | None = None
        self._inner: MemKDM | None = None
        self.target_informed = False

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
            backend=self.backend,
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
        self._inner = self.to_memkdm().fit(X, targets)
        self.target_informed = self._inner.target_informed
        return self

    def _require_fit(self) -> MemKDM:
        if self._inner is None:
            raise RuntimeError("BrentMemKDM.fit() must be called before predict/uncertainty methods")
        return self._inner

    # ------------------------------------------------------------- predict
    def predict_proba(self, X: Modalities) -> np.ndarray:
        return self._require_fit().predict_proba(X)

    def predict(self, X: Modalities, threshold: float = 0.50) -> np.ndarray:
        return self._require_fit().predict(X, threshold=threshold)

    def uncertainty_signals(self, X: Modalities) -> dict:
        return self._require_fit().uncertainty_signals(X)

    # ---------------------------------------------------------- confidence
    def fit_confidence(self, y_conf: np.ndarray, splits: list, head: str = "meta_threshold_1d",
                        X: Modalities | None = None, key: str | None = None) -> "BrentMemKDM":
        self._require_fit().fit_confidence(y_conf, splits, head=head, X=X, key=key)
        return self

    def predict_confidence(self, X: Modalities | None = None) -> np.ndarray:
        return self._require_fit().predict_confidence(X)

    # --------------------------------------------------------------- kernel
    def kernel_params(self) -> dict:
        return self._require_fit().kernel_params()
