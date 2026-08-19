"""Memory-based Kernel Density Matrix classifier(s), single-file method module.

Generalizes exp_23 ("Tabular KDM Biopsy Decision Prediction") and exp_24
("Particle-Set Uncertainty Decomposition for the Tabular KDM") from one
tabular-only model into `MemKDM`: one class covering 1..N modalities, each
carrying its own RBF-kernel bandwidth, combined as a product kernel
`k(x, c) = Pi_m k_m(x_m, c_m)`. A unimodal `MemKDM` is the one-entry case —
not a different code path — so it reproduces exp_23's numbers bit-for-bit.

"Memory-based" is the exp_23/24 sense: `KDMLayer` with `n_comp = n_train`,
prototypes initialised from the entire training fold via `init_kdm_layer`.
Not `MemKDMClassModelWrapper`/faiss (rejected in exp_23/DESIGN.md Sec 2.4:
faiss segfaults alongside torch, approximate NN is pointless at N=88, and
`MemKDMClassModel.forward` hardcodes `F.one_hot` which silently truncates
soft labels).

Supervision is soft throughout; a "hard" arm is simply `y_soft` valued in
{0, 1}. There is one amplitude encoding, `to_amplitude(y) = [sqrt(1-y),
sqrt(y)]`, and one loss (soft cross-entropy) — exp_23's hard/soft arms and
exp_24's ("Arm C") ~a1-smoothed arm become target vectors plus the
`label_smoothing` hyperparameter, not three code paths.

Cosine-similarity behaviour for MRI/text is obtained by L2-normalizing those
feature blocks in preprocessing (see `src/evaluation/data.py`), not by a
cosine kernel: `KDMLayer` squares kernel values, so a parameterless cosine
kernel ties a perfectly anti-aligned prototype (cos=-1) with a perfectly
aligned one (cos=1) at weight 1.0 either way. RBF on unit-norm vectors is
monotone in cosine similarity (||a-b||^2 = 2 - 2*cos(a,b)) and keeps a single
tunable bandwidth per modality — the quantity `from_unimodal` transfers.

This module must never import `src/evaluation`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist

from kdm.layers import CrossProductKernelLayer, KDMLayer, RBFKernelLayer
from kdm.init import _sigma_from_knn, init_kdm_layer
from kdm.utils import dm2comp, dm2discrete, pure2dm

from .base import (
    Modalities,
    Targets,
    apply_meta_thresholds,
    fit_meta_thresholds_safe,
    fit_predict_heldout_trees,
)

PARTICLE_SIGNAL_NAMES = (
    "h_total", "h_aleatoric", "h_epistemic", "h_weights", "log_ess", "w_max", "log_marginal",
)


# ---------------------------------------------------------------------------
# Target amplitude encoding — the single path replacing exp_23's
# to_amplitude_hard/to_amplitude_soft and exp_24's to_amplitude_hard_smoothed.
# ---------------------------------------------------------------------------
def to_amplitude(y_soft: np.ndarray) -> np.ndarray:
    """`[sqrt(1-y), sqrt(y)]` amplitude (Born-rule) encoding.

    At y in {0,1} this is exactly `F.one_hot(y,2).float()` (exp_23's
    `to_amplitude_hard`); at continuous y it is exp_23's `to_amplitude_soft`.
    One formula, no arm branch.
    """
    y = np.clip(np.asarray(y_soft, dtype=np.float32), 0.0, 1.0)
    return np.stack([np.sqrt(1 - y), np.sqrt(y)], axis=1).astype(np.float32)


def smooth(y_soft: np.ndarray, label_smoothing: float) -> np.ndarray:
    """`y <- y*(1-eps) + eps/2`. At `eps=0` this is the identity — the
    structural check exp_24 used to confirm its Arm C reduces exactly to
    Arm A. Generalizes exp_24's `to_amplitude_hard_smoothed`, which applied
    the same affine map to a binary y, to any y in [0,1].

    NOTE: `label_smoothing` perturbs only the memory-prototype INIT target
    (`c_y`), never the training loss target. exp_24's Arm C trains against
    the raw (unsmoothed) hard labels (`train_kdm(..., "hard",
    y_binary_tr=y_binary_tr)`) — smoothing only moves the init away from the
    exact one-hot corner that is `c_y`'s gradient fixed point (exp_24
    Sec 2.1). `MemKDM.fit` applies this function to build `c_y`'s init only;
    the loss target is always the caller's raw `Targets.y_soft`.
    """
    if label_smoothing == 0.0:
        return np.asarray(y_soft, dtype=np.float32)
    y = np.clip(np.asarray(y_soft, dtype=np.float32), 0.0, 1.0)
    return (y * (1.0 - label_smoothing) + label_smoothing / 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-modality specs
# ---------------------------------------------------------------------------
@dataclass
class KernelSpec:
    """One modality's RBF kernel. RBF is the only kernel kind (see module
    docstring) — cosine behaviour comes from L2-normalizing that modality's
    feature block in preprocessing, not from a kernel choice here."""

    sigma_mult: float = 1.0
    """Bandwidth init multiplier: sigma = mean-dist-to-3rd-NN * sigma_mult,
    via `kdm.init._sigma_from_knn` — ignored when `sigma` is given."""

    trainable: bool = True
    """Whether this block's sigma is learnable during `fit`."""

    sigma: float | None = None
    """Explicit sigma, overriding data-driven init. Set by `MemKDM.kernel_params()`
    for the unimodal-to-multimodal transfer path (`MemKDM.from_unimodal`)."""


@dataclass
class EncoderSpec:
    """One modality's frozen-family encoder. `"identity"` (no parameters) or
    `"linear"` (an `nn.Linear`, trained jointly with the KDM prototypes —
    "frozen" here means the encoder FAMILY is fixed, not that its parameters
    never move; see exp_23/DESIGN.md Sec 2.4, where the linear-encoder arm's
    weights ARE included in the same Adam optimizer as the KDM prototypes)."""

    kind: str = "identity"  # "identity" | "linear"
    out_dim: int | None = None  # required when kind == "linear"


def _make_encoder(spec: EncoderSpec, in_dim: int) -> tuple[nn.Module, int]:
    if spec.kind == "identity":
        return nn.Identity(), in_dim
    if spec.kind == "linear":
        if spec.out_dim is None:
            raise ValueError("EncoderSpec(kind='linear') requires out_dim")
        return nn.Linear(in_dim, spec.out_dim), spec.out_dim
    raise ValueError(f"unknown encoder kind: {spec.kind!r}")


# ---------------------------------------------------------------------------
# Kernel tree construction
# ---------------------------------------------------------------------------
def _build_kernel(encoded_dims: dict, kernels: dict, order: list):
    """Builds the product kernel `Pi_m RBF_m` over modalities in `order`.

    1 modality -> that modality's RBFKernelLayer directly (no wrapper), so
    exp_23's single-block numbers reproduce bit-for-bit.
    N modalities -> right-nested CrossProductKernelLayer, column ranges
    matching the concatenation order `order` (see CrossProductKernelLayer's
    `A[:, :, :dim1]` / `A[:, :, dim1:]` split, which assumes exactly this).

    Returns (kernel, leaves) where leaves is
    [(name, col_offset, dim, RBFKernelLayer), ...] in `order`.
    """
    offsets = {}
    off = 0
    for name in order:
        offsets[name] = off
        off += encoded_dims[name]

    def make_leaf(name):
        spec = kernels[name]
        # sigma=0.5 is a placeholder (always overwritten by the data-driven
        # or transferred init below); RBFKernelLayer's constructor consumes
        # no RNG regardless of the value, so this choice does not affect the
        # torch.manual_seed(seed)-pinned reproduction of encoder/KDMLayer init.
        return RBFKernelLayer(sigma=0.5, dim=encoded_dims[name], trainable=spec.trainable, min_sigma=1e-3)

    if len(order) == 1:
        name = order[0]
        rbf = make_leaf(name)
        return rbf, [(name, offsets[name], encoded_dims[name], rbf)]

    leaves = []
    last_name = order[-1]
    kernel = make_leaf(last_name)
    leaves.append((last_name, offsets[last_name], encoded_dims[last_name], kernel))
    for name in reversed(order[:-1]):
        rbf = make_leaf(name)
        leaves.append((name, offsets[name], encoded_dims[name], rbf))
        kernel = CrossProductKernelLayer(dim1=encoded_dims[name], kernel1=rbf, kernel2=kernel)
    leaves.reverse()
    return kernel, leaves


def _iter_kernel_leaves(kernel):
    if isinstance(kernel, CrossProductKernelLayer):
        yield from _iter_kernel_leaves(kernel.kernel1)
        yield from _iter_kernel_leaves(kernel.kernel2)
    else:
        yield kernel


class _MemKDMCore(nn.Module):
    """Encoder(s) -> concatenated pure KDM state -> product-kernel KDMLayer
    -> class probabilities. For one modality this constructs and calls
    exactly what `KDMClassModel` would (same RBFKernelLayer, same KDMLayer),
    so RNG consumption and the forward computation are identical."""

    def __init__(self, encoders: dict, modality_order: list, kdm_layer: KDMLayer):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modality_order = list(modality_order)
        self.kdm = kdm_layer

    def _encode(self, x_dict: dict) -> torch.Tensor:
        return torch.cat([self.encoders[m](x_dict[m]) for m in self.modality_order], dim=-1)

    def forward(self, x_dict: dict) -> torch.Tensor:
        rho_x = pure2dm(self._encode(x_dict))
        rho_y = self.kdm(rho_x)
        return dm2discrete(rho_y)


def _as_tensor_dict(X: dict, modality_order: list) -> dict:
    return {m: torch.as_tensor(np.asarray(X[m]), dtype=torch.float32) for m in modality_order}


def _check_amplitude_roundtrip(model: _MemKDMCore, Xt: dict, c_y: np.ndarray) -> None:
    """Port of exp_23's `init_and_check` round-trip probe, generalized over
    the (possibly composed) kernel. Narrows every leaf's sigma far below the
    data's own minimum pairwise distance in the full encoded space, so each
    training point's own prototype should dominate its own kernel weight;
    asserts the resulting class-1 probability recovers c_y[:,1]**2 (the
    Born-rule amplitude-squared) to within 1e-4. A naive raw-probability
    encoding `[1-y, y]` fails this at ~0.15 error; the sqrt encoding passes
    at ~0.0 (see exp_23/DESIGN.md Sec 2.4)."""
    probe = copy.deepcopy(model)
    probe.eval()
    with torch.no_grad():
        enc_np = probe._encode(Xt).numpy()
    pairwise = pdist(enc_np)
    pairwise = pairwise[pairwise > 0]
    min_d = float(pairwise.min()) if len(pairwise) > 0 else 1.0
    leaves = list(_iter_kernel_leaves(probe.kdm.kernel))
    min_sigma = min(float(leaf.min_sigma) for leaf in leaves)
    probe_sigma = max(min_sigma * 1.5, min_d / 50.0)
    for leaf in leaves:
        leaf.sigma = probe_sigma
    with torch.no_grad():
        probs = probe(Xt)
    target = c_y[:, 1] ** 2
    err = float(np.abs(probs[:, 1].numpy() - target).max())
    assert err < 1e-4, (
        f"amplitude round-trip failed: max err {err} "
        f"(probe_sigma={probe_sigma:.2e}, min_pairwise_dist={min_d:.4f})"
    )


# ---------------------------------------------------------------------------
# MemKDM — the method
# ---------------------------------------------------------------------------
class MemKDM:
    """Memory-based KDM classifier over 1..N modalities.

    There is no separate unimodal/multimodal class: `kernels={"tab": ...}`
    IS the unimodal model, and its numbers reproduce exp_23 bit-for-bit
    (single-block path uses the block kernel directly, no
    CrossProductKernelLayer wrapper — see `_build_kernel`).
    """

    def __init__(
        self,
        kernels: dict,
        encoders: dict | None = None,
        label_smoothing: float = 0.0,
        x_train: bool = False,
        y_train: bool = False,
        w_train: bool = True,
        lr: float = 1e-3,
        epochs: int = 300,
        seed: int = 0,
        check_roundtrip: bool = False,
    ):
        if not kernels:
            raise ValueError("MemKDM requires at least one modality in `kernels`")
        self.kernels = dict(kernels)
        self.modality_names = list(kernels.keys())
        self.encoders = dict(encoders) if encoders is not None else {
            m: EncoderSpec("identity") for m in self.modality_names
        }
        self.label_smoothing = label_smoothing
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.check_roundtrip = check_roundtrip

        self.target_informed = False
        self._model: _MemKDMCore | None = None
        self._kernel_leaves = None  # [(name, offset, dim, RBFKernelLayer), ...]
        self._confidence: dict | None = None
        self.cy_drift_max_: float | None = None  # max|c_y_after - c_y_before|, set iff y_train

    # ------------------------------------------------------------------ fit
    def fit(self, X: Modalities, targets: Targets) -> "MemKDM":
        torch.manual_seed(self.seed)

        raw_dims = {m: np.asarray(X[m]).shape[1] for m in self.modality_names}
        n_comp = len(targets.y_soft)

        # 1. Encoders, constructed in modality order (consumes RNG only for
        #    "linear" specs — nn.Linear's kaiming_uniform_ init).
        encoder_modules, encoded_dims = {}, {}
        for m in self.modality_names:
            mod, dim = _make_encoder(self.encoders[m], raw_dims[m])
            encoder_modules[m] = mod
            encoded_dims[m] = dim

        # 2. Kernel tree (no RNG consumed).
        kernel, leaves = _build_kernel(encoded_dims, self.kernels, self.modality_names)
        self._kernel_leaves = leaves

        # 3. KDMLayer (consumes RNG: c_x = torch.randn(n_comp, dim_x) * 0.05,
        #    later overwritten by init_kdm_layer below).
        dim_x = sum(encoded_dims.values())
        kdm_layer = KDMLayer(
            kernel=kernel, dim_x=dim_x, dim_y=2, n_comp=n_comp,
            x_train=self.x_train, y_train=self.y_train, w_train=self.w_train,
        )
        model = _MemKDMCore(encoder_modules, self.modality_names, kdm_layer)

        # 4. Data-driven init. label_smoothing perturbs the INIT target only
        #    (see `smooth`'s docstring) — the loss target below stays raw.
        Xt = _as_tensor_dict(X, self.modality_names)
        with torch.no_grad():
            enc_full = model._encode(Xt)

        y_soft = np.clip(np.asarray(targets.y_soft, dtype=np.float32), 0.0, 1.0)
        y_init = smooth(y_soft, self.label_smoothing)
        c_y = to_amplitude(y_init)

        init_kdm_layer(kdm_layer, enc_full.detach(), torch.as_tensor(c_y, dtype=torch.float32), init_sigma=False)
        enc_full_np = enc_full.detach().numpy()
        for name, offset, dim, leaf in leaves:
            spec = self.kernels[name]
            if spec.sigma is not None:
                leaf.sigma = spec.sigma
            else:
                block = enc_full_np[:, offset:offset + dim]
                leaf.sigma = _sigma_from_knn(block, spec.sigma_mult)

        if self.check_roundtrip:
            _check_amplitude_roundtrip(model, Xt, c_y)

        # 5. Train. Loss target is ALWAYS the raw y_soft (never smoothed) —
        #    matches exp_23 Arm A/B exactly (label_smoothing=0, no-op) and
        #    exp_24 Arm C (label_smoothing>0 only moved the init above).
        c_y_before = kdm_layer.c_y.detach().clone() if self.y_train else None
        t = torch.as_tensor(np.stack([1 - y_soft, y_soft], axis=1), dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(self.epochs):
            probs = model(Xt)
            loss = -(t * torch.log(probs.clamp_min(1e-7))).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if c_y_before is not None:
            self.cy_drift_max_ = float((kdm_layer.c_y.detach() - c_y_before).abs().max().item())

        self._model = model
        self.target_informed = bool(targets.soft_from_confidence)
        self._confidence = None
        return self

    # ------------------------------------------------------------- predict
    def _require_fit(self) -> _MemKDMCore:
        if self._model is None:
            raise RuntimeError("MemKDM.fit() must be called before predict/uncertainty methods")
        return self._model

    def predict_proba(self, X: Modalities) -> np.ndarray:
        return self.uncertainty_signals(X)["probs"]

    def predict(self, X: Modalities, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def uncertainty_signals(self, X: Modalities) -> dict:
        return extract_particle_signals(self._require_fit(), X)

    # ---------------------------------------------------------- confidence
    def fit_confidence(self, y_conf: np.ndarray, splits: list, head: str = "meta_threshold_1d",
                        X: Modalities | None = None, key: str | None = None) -> "MemKDM":
        """Fits a confidence head on this model's own uncertainty signals of
        its training data (pass the same `X` `fit` was called with, or rely
        on `predict_confidence` computing them fresh — for `MemKDM` these are
        the same call since there's no separate memory-vs-query distinction
        at this layer; the harness in `src/evaluation/protocol.py` is what
        enforces the train/val split).
        """
        if X is None:
            raise ValueError("fit_confidence requires X (the data to compute uncertainty_signals on)")
        signals = self.uncertainty_signals(X)
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
        if self._confidence is None:
            raise RuntimeError("MemKDM.fit_confidence() must be called before predict_confidence")
        head = self._confidence["head"]
        if head == "meta_threshold_1d":
            if X is None:
                raise ValueError("predict_confidence requires X for head='meta_threshold_1d'")
            signal = self.uncertainty_signals(X)[self._confidence["key"]]
            return apply_meta_thresholds(signal, self._confidence["thr"])
        if head == "multivariate_heldout":
            # Self-referential protocol (see base.fit_predict_heldout_trees):
            # only valid for the exact cohort fit_confidence was called on.
            return self._confidence["_heldout_pred"]
        raise ValueError(f"unknown confidence head: {head!r}")

    # --------------------------------------------------------------- kernel
    def kernel_params(self) -> dict:
        """{modality: KernelSpec(..., sigma=<fitted sigma>)} — for transfer
        into a joint model via `MemKDM.from_unimodal`."""
        self._require_fit()
        out = {}
        for name, _offset, _dim, leaf in self._kernel_leaves:
            spec = self.kernels[name]
            out[name] = KernelSpec(sigma_mult=spec.sigma_mult, trainable=spec.trainable, sigma=float(leaf.sigma.detach()))
        return out

    @classmethod
    def from_unimodal(cls, models: dict, freeze_kernels: bool = True, **kwargs) -> "MemKDM":
        """Builds a joint multimodal `MemKDM` seeded with per-modality
        bandwidths tuned by fitting `models` unimodally first. With
        `freeze_kernels=True`, those bandwidths are held fixed and only the
        shared memory (`c_x`/`c_y`/`c_w`) adapts jointly — turning the
        per-modality search into cheap independent unimodal sweeps instead
        of a combinatorial joint grid."""
        kernels, encoders = {}, {}
        for name, m in models.items():
            spec = m.kernel_params()[name]
            kernels[name] = KernelSpec(
                sigma_mult=spec.sigma_mult, sigma=spec.sigma,
                trainable=(spec.trainable and not freeze_kernels),
            )
            encoders[name] = m.encoders[name]
        return cls(kernels=kernels, encoders=encoders, **kwargs)


def _best_1d_key(signals: dict, y_conf: np.ndarray, splits: list) -> str:
    """Default 1D-head signal selection when no `key` is given: the particle
    signal with the highest Phase-A meta-threshold Macro-F1. Kept local and
    simple (no import of src/evaluation/metrics) — a thin re-derivation, not
    the canonical scorer used for reporting."""
    from sklearn.metrics import f1_score

    best_key, best_score = None, -1.0
    for key in PARTICLE_SIGNAL_NAMES:
        thr = fit_meta_thresholds_safe(signals[key], y_conf, splits)
        pred = apply_meta_thresholds(signals[key], thr)
        score = f1_score(y_conf, pred, average="macro", zero_division=0)
        if score > best_score:
            best_key, best_score = key, score
    return best_key


# ---------------------------------------------------------------------------
# Particle-set signal extraction — generalizes exp_24's extract_particle_signals
# (exp_24/scripts/train.py:234-264) over the model's own modality dict input.
# ---------------------------------------------------------------------------
def extract_particle_signals(model: _MemKDMCore, X: Modalities, eps: float = 1e-7) -> dict:
    """Per-sample signals from the KDM's output density matrix, before
    dm2discrete's clamp collapses it. `probs` here is the canonical
    prediction path for this module (see module docstring "one probability
    path") — computed via the same w/p decomposition as the other six
    signals, agreeing with `dm2discrete(rho_y)` to ~1e-9 (verified in the
    verification script), well within any downstream decision threshold.

    h_aleatoric is identically 0 whenever every particle's own predictive
    distribution p_j is one-hot, i.e. whenever `Targets.y_soft` is binary AND
    `label_smoothing == 0` (exp_24 Sec 2.1: the Born-rule map has a gradient
    fixed point at a one-hot c_y, so y_train=True does not escape it either).
    Either genuinely soft targets or label_smoothing > 0 is required for a
    non-degenerate aleatoric/epistemic split.
    """
    model.eval()
    with torch.no_grad():
        Xt = _as_tensor_dict(X, model.modality_order)
        rho_x = pure2dm(model._encode(Xt))
        rho_y = model.kdm(rho_x)
        w, v = dm2comp(rho_y)
        w = w / w.sum(-1, keepdim=True)
        p = F.normalize(v, p=2, dim=-1, eps=1e-12) ** 2

        p_mean = (w.unsqueeze(-1) * p).sum(dim=1)
        h_total = -(p_mean * torch.log(p_mean.clamp_min(eps))).sum(-1)

        h_particles = -(p * torch.log(p.clamp_min(eps))).sum(-1)
        h_aleatoric = (w * h_particles).sum(dim=1)
        h_epistemic = h_total - h_aleatoric

        h_weights = -(w * torch.log(w.clamp_min(eps))).sum(-1)
        log_ess = -torch.log((w ** 2).sum(-1).clamp_min(eps))
        w_max = w.max(dim=-1).values

        log_marginal = model.kdm.log_marginal(rho_x)

    return {
        "probs": p_mean.numpy(),
        "h_total": h_total.numpy(),
        "h_aleatoric": h_aleatoric.numpy(),
        "h_epistemic": h_epistemic.numpy(),
        "h_weights": h_weights.numpy(),
        "log_ess": log_ess.numpy(),
        "w_max": w_max.numpy(),
        "log_marginal": log_marginal.numpy(),
    }


# ---------------------------------------------------------------------------
# Composite Reliability Index — two named definitions (see exp_17/DESIGN.md
# vs. exp_17's actual code; they disagree, so both are named and neither is
# a silent default).
# ---------------------------------------------------------------------------
def composite_reliability_index(P: np.ndarray):
    """`(2*|p_mean-0.5|) * (1 - 2*std)`, clipped to [0,1] — what exp_9/10/11/
    12/17/19 ACTUALLY compute (six byte-identical copies, deduplicated here).
    `P`: (n, n_modalities) per-modality P(biopsy=yes). Returns
    (ici, p_mean, p_std, margin)."""
    p_mean = P.mean(axis=1)
    p_std = P.std(axis=1)
    margin = np.abs(p_mean - 0.50)
    ici = np.clip((2.0 * margin) * (1.0 - 2.0 * p_std), 0.0, 1.0)
    return ici, p_mean, p_std, margin


def inter_modality_variance(P: np.ndarray) -> np.ndarray:
    """Plain inter-modality variance — what `exp_17/DESIGN.md` Sec 2
    SPECIFIES (and the code never actually computed)."""
    return P.var(axis=1)


def simplex_grid(n_modalities: int, step: float = 0.05):
    """Reproduces exp_16's 231-point simplex grid search: all weight vectors
    on the `n_modalities`-simplex at the given step size."""
    if n_modalities != 3:
        raise NotImplementedError("simplex_grid currently supports exactly 3 modalities (as in exp_16)")
    grid = []
    for wt in np.arange(0.0, 1.0 + step / 2, step):
        for wm in np.arange(0.0, 1.0 - wt + step / 2, step):
            wx = round(1.0 - wt - wm, 4)
            if wx >= -1e-5:
                grid.append((round(float(wt), 4), round(float(wm), 4), max(0.0, wx)))
    return grid


def soft_vote(probs: dict, weights: dict) -> np.ndarray:
    """Sum_m weights[m] * probs[m]. `probs[m]`: (n, 2)."""
    names = list(probs.keys())
    out = np.zeros_like(probs[names[0]])
    for m in names:
        out = out + weights[m] * probs[m]
    return out


def search_fusion_weights(probs: dict, y_true: np.ndarray, splits: list, step: float = 0.05) -> dict:
    """Phase-A fusion-weight search: grid-searches the simplex, scored by
    mean MCCV Macro-F1 over `splits` — NOT the evaluation set. `splits`
    is required with no whole-cohort default, which is the fix for exp_16's
    biggest defect (its weights were selected by maximising Macro-F1 on the
    same LOOCV out-of-fold predictions it then reported)."""
    from sklearn.metrics import f1_score

    names = list(probs.keys())
    if len(names) != 3:
        raise NotImplementedError("search_fusion_weights currently supports exactly 3 modalities")
    grid = simplex_grid(3, step)
    best_w, best_score = None, -1.0
    for wt, wm, wx in grid:
        weights = dict(zip(names, (wt, wm, wx)))
        scores = []
        for train_idx, val_idx in splits:
            p = soft_vote({m: probs[m][val_idx] for m in names}, weights)
            pred = (p[:, 1] >= 0.50).astype(int)
            scores.append(f1_score(y_true[val_idx], pred, average="macro", zero_division=0))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_w, best_score = weights, mean_score
    return {"weights": best_w, "mean_macro_f1": best_score}


# ---------------------------------------------------------------------------
# LateFusionMemKDM — the exp_17-comparable soft-voting baseline.
# ---------------------------------------------------------------------------
class LateFusionMemKDM:
    """Holds one single-modality `MemKDM` per modality; combines their
    probabilities by weighted soft voting. Kept alongside `MemKDM`'s
    product-kernel joint model as the exp_17-comparable baseline — it does
    NOT share a particle set, so its `uncertainty_signals` composite is
    necessarily different in kind from `MemKDM`'s."""

    def __init__(self, members: dict, weights: dict | None = None):
        self.members = dict(members)
        self.modality_names = list(members.keys())
        n = len(self.modality_names)
        self.weights = dict(weights) if weights is not None else {m: 1.0 / n for m in self.modality_names}
        self.target_informed = False
        self._confidence: dict | None = None

    def fit(self, X: Modalities, targets: Targets) -> "LateFusionMemKDM":
        for name, member in self.members.items():
            member.fit({name: X[name]}, targets)
        self.target_informed = any(m.target_informed for m in self.members.values())
        self._confidence = None
        return self

    def predict_proba(self, X: Modalities) -> np.ndarray:
        probs = {name: self.members[name].predict_proba({name: X[name]}) for name in self.modality_names}
        return soft_vote(probs, self.weights)

    def predict(self, X: Modalities, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def uncertainty_signals(self, X: Modalities) -> dict:
        out = {}
        p_yes = {}
        for name in self.modality_names:
            sig = self.members[name].uncertainty_signals({name: X[name]})
            p_yes[name] = sig["probs"][:, 1]
            for k, v in sig.items():
                out[f"{name}__{k}"] = v
        P = np.stack([p_yes[m] for m in self.modality_names], axis=1)
        ici, p_mean, p_std, margin = composite_reliability_index(P)
        out["composite_ici"] = ici
        out["inter_modality_variance"] = inter_modality_variance(P)
        out["p_mean"], out["p_std"], out["margin"] = p_mean, p_std, margin
        return out

    def fit_confidence(self, y_conf: np.ndarray, splits: list, head: str = "meta_threshold_1d",
                        X: Modalities | None = None, key: str = "composite_ici") -> "LateFusionMemKDM":
        if X is None:
            raise ValueError("fit_confidence requires X")
        signals = self.uncertainty_signals(X)
        if head == "meta_threshold_1d":
            thr = fit_meta_thresholds_safe(signals[key], y_conf, splits)
            self._confidence = {"head": head, "key": key, "thr": thr}
        elif head == "multivariate_heldout":
            keys = sorted(k for k in signals if k not in ("p_mean", "p_std", "margin"))
            S = np.stack([signals[k] for k in keys], axis=1)
            pred, votes = fit_predict_heldout_trees(S, y_conf, splits)
            self._confidence = {"head": head, "keys": keys, "_heldout_pred": pred, "_heldout_votes": votes}
        else:
            raise ValueError(f"unknown confidence head: {head!r}")
        return self

    def predict_confidence(self, X: Modalities | None = None) -> np.ndarray:
        if self._confidence is None:
            raise RuntimeError("fit_confidence() must be called before predict_confidence")
        head = self._confidence["head"]
        if head == "meta_threshold_1d":
            if X is None:
                raise ValueError("predict_confidence requires X for head='meta_threshold_1d'")
            signal = self.uncertainty_signals(X)[self._confidence["key"]]
            return apply_meta_thresholds(signal, self._confidence["thr"])
        if head == "multivariate_heldout":
            return self._confidence["_heldout_pred"]
        raise ValueError(f"unknown confidence head: {head!r}")
