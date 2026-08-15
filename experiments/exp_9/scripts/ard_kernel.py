"""exp_9: ARD (Automatic Relevance Determination) kernel bandwidth for the KDM backbone -- one
trained sigma_j per input dimension instead of exp_6/exp_7/exp_8's single shared scalar sigma.

Grounded in three facts confirmed against the installed kdm library this session (see
experiments/exp_9/IMPLEMENTATION.md for the full derivation):
  1. The ARD reformulation is a rescale-then-reuse of RBFKernelLayer's existing norm-trick
     distance computation, not a rewrite.
  2. kdm.init.init_kdm_layer() already works unchanged for per-dimension initialization, as long
     as raw_sigma is built as torch.full((dim,), value) -- never .expand(), which would alias
     memory and make every sigma_j move in lockstep during training.
  3. kdm.utils.dm_rbf_variance()'s isotropic d*sigma**2 term is wrong for ARD's anisotropic
     diag(sigma_1**2, ..., sigma_d**2) -- needs the small correction dm_rbf_variance_ard() below.
     Everything else in kdm_backbone.py (entropy, participation, occlusion_delta,
     kernel_distance_contribution) is sigma-shape-agnostic and reused unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kdm.base import Kernel
from kdm.init import init_kdm_layer
from kdm.layers import KDMLayer
from kdm.models import KDMClassModel
from kdm.utils import comp2dm, dm2comp, dm2discrete, pure2dm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
from kdm_backbone import kernel_distance_contribution  # noqa: E402,F401
# NOTE: kdm_backbone.occlusion_delta is NOT reused here, despite IMPLEMENTATION.md finding #3's
# claim that it's sigma-shape-agnostic -- that claim was wrong. occlusion_delta() internally
# calls the *full* compute_signals() (not just a probs-only computation) to get p(yes), and that
# scalar compute_signals() unconditionally computes the dispersion signal via the library's
# dm_rbf_variance(dm, sigma), which crashes when sigma is a (dim,) vector instead of scalar --
# found when run_signals_19col.py actually executed this path. kernel_distance_contribution IS
# still safely reused unchanged (confirmed: never references model.kernel.sigma at all).

RANDOM_STATE = 0


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


class ARDRBFKernelLayer(Kernel):
    """RBF kernel with one trainable bandwidth per input dimension, instead of
    kdm.layers.rbf_kernel_layer.RBFKernelLayer's single shared scalar.

    sigma is parameterized as softplus(raw_sigma) + min_sigma per-dimension, same
    reparameterization as the parent class, just shaped (dim,) instead of scalar.
    """

    def __init__(self, sigma, dim: int, trainable: bool = True, min_sigma: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.min_sigma = min_sigma
        sigma_t = torch.as_tensor(float(sigma))
        if sigma_t <= min_sigma:
            raise ValueError(f"Initial sigma ({float(sigma_t):g}) must be > min_sigma ({min_sigma:g})")
        raw_value = _softplus_inv(sigma_t - min_sigma)
        # torch.full allocates independent memory per element -- NOT .expand(), which would
        # return a view aliasing the same underlying value for every "dim" copy.
        raw = torch.full((dim,), float(raw_value))
        self.raw_sigma = nn.Parameter(raw, requires_grad=trainable)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.min_sigma  # shape (dim,)

    @sigma.setter
    def sigma(self, value) -> None:
        with torch.no_grad():
            value_t = torch.as_tensor(value, dtype=self.raw_sigma.dtype, device=self.raw_sigma.device)
            if (value_t <= self.min_sigma).any():
                raise ValueError(f"Assigned sigma must be > min_sigma ({self.min_sigma:g})")
            # copy_ broadcasts a scalar source into the (dim,) destination as a real per-element
            # copy (unlike .expand()), so every sigma_j starts at the same KNN-based value but is
            # independently updatable afterward.
            self.raw_sigma.copy_(_softplus_inv(value_t - self.min_sigma))

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            A: (bs, n, d)
            B: (m, d)
        Returns:
            K: (bs, n, m)
        """
        sigma = self.sigma  # (d,)
        A_scaled = A / sigma
        B_scaled = B / sigma
        AB = torch.matmul(A_scaled, B_scaled.transpose(-1, -2))
        A_norm = (A_scaled ** 2).sum(dim=-1, keepdim=True)
        B_norm = (B_scaled ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)
        dist2 = (A_norm + B_norm - 2.0 * AB).clamp(min=0.0)
        return torch.exp(-dist2 / 2.0)  # sigma already baked into the rescaled coordinates

    def log_weight(self) -> torch.Tensor:
        # Generalizes RBFKernelLayer.log_weight()'s `-dim*log(sigma) - dim*log(pi)/2` to
        # per-dimension sigma. Unexercised by this project's discriminative training loop
        # (only forward()+NLL is used, never log_marginal()), kept correct for completeness.
        return -torch.log(self.sigma + 1e-12).sum() - self.dim * math.log(math.pi) / 2


class ARDKDMClassModel(KDMClassModel):
    """KDMClassModel with ARDRBFKernelLayer swapped in for the hardcoded RBFKernelLayer. Only
    __init__'s kernel construction differs; self.kdm/forward() are inherited unchanged since
    KDMLayer accepts any kernel object satisfying forward(A, B)."""

    def __init__(self, encoded_size, dim_y, encoder, n_comp, sigma=0.1, sigma_trainable=True,
                 min_sigma=1e-3, x_train=True, y_train=True, w_train=True):
        nn.Module.__init__(self)
        self.encoded_size = encoded_size
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.encoder = encoder
        self.kernel = ARDRBFKernelLayer(sigma=sigma, dim=encoded_size, trainable=sigma_trainable, min_sigma=min_sigma)
        self.kdm = KDMLayer(
            kernel=self.kernel, dim_x=encoded_size, dim_y=dim_y, n_comp=n_comp,
            x_train=x_train, y_train=y_train, w_train=w_train,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm(rho_x)
        return dm2discrete(rho_y)


def fit_kdm_backbone_ard(
    X_train: np.ndarray, y_train: np.ndarray, n_classes: int = 2,
    n_epochs: int = 300, lr: float = 1e-2, sigma_mult: float = 1.0,
) -> ARDKDMClassModel:
    """Same memory-based fit as kdm_backbone.fit_kdm_backbone(), constructing an
    ARDKDMClassModel instead. No optimizer/weight_decay choice this round -- Adam only, per
    DESIGN.md's explicit no-hyperparameter-search guardrail for this experiment.
    """
    torch.manual_seed(RANDOM_STATE)
    dim_x = X_train.shape[1]
    n_comp = len(X_train)

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    y_onehot = F.one_hot(yt, n_classes).float()

    encoder = nn.Identity()
    model = ARDKDMClassModel(
        encoded_size=dim_x, dim_y=n_classes, encoder=encoder, n_comp=n_comp,
        sigma=0.5, sigma_trainable=True, x_train=False, y_train=False, w_train=False,
    )

    model.eval()
    with torch.no_grad():
        enc_sub = encoder(Xt)
    init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot, init_sigma=True, sigma_mult=sigma_mult)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(n_epochs):
        probs = model(Xt)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def dm_rbf_variance_ard(dm: torch.Tensor, sigma_vector: torch.Tensor) -> torch.Tensor:
    """ARD-correct generalization of kdm.utils.dm_rbf_variance(): the within-component term
    becomes sum_j(sigma_j^2) instead of d*sigma^2 (isotropic assumption), since ARD's
    per-component covariance is diag(sigma_1^2, ..., sigma_d^2), not sigma^2 * I.
    Reduces exactly to the library's formula when every sigma_j is equal.
    """
    sigma_vector = sigma_vector / math.sqrt(2)
    w, v = dm2comp(dm)
    squared_norms = (v ** 2).sum(dim=-1)
    weighted_squared_norms = torch.einsum("...i,...i->...", w, squared_norms)
    weighted_means = torch.einsum("...i,...ij->...j", w, v)
    squared_means = (weighted_means ** 2).sum(dim=-1)
    between_component_variance = weighted_squared_norms - squared_means
    within_component_variance = (sigma_vector ** 2).sum()
    return between_component_variance + within_component_variance


def _prototype_weights(model: ARDKDMClassModel, X: torch.Tensor) -> torch.Tensor:
    """Same replication of KDMLayer.forward()'s _compute_mixture -> clamp -> normalize as
    kdm_backbone._prototype_weights() -- duplicated here (not imported) since it's a private
    function name, not part of that module's public API."""
    rho_x = pure2dm(X)
    kdm_layer = model.kdm
    in_w, out_w = kdm_layer._compute_mixture(rho_x)
    out_w = out_w.clamp(min=kdm_layer.eps)
    out_w = out_w / out_w.sum(dim=2, keepdim=True)
    out_w = torch.einsum("...i,...ij->...j", in_w, out_w)
    return out_w


def compute_signals_ard(model: ARDKDMClassModel, X: np.ndarray) -> dict:
    """Same as kdm_backbone.compute_signals(), except the dispersion signal uses
    dm_rbf_variance_ard() (with the model's per-dimension sigma vector) instead of the library's
    isotropic dm_rbf_variance(). Entropy, participation, and the probs_check_ok cross-check are
    unchanged -- confirmed sigma-shape-agnostic (see IMPLEMENTATION.md finding #3).
    """
    Xt = torch.as_tensor(X, dtype=torch.float32)
    n = Xt.shape[0]

    with torch.no_grad():
        probs_forward = model(Xt)
        out_w = _prototype_weights(model, Xt)

        c_x = model.kdm.c_x
        c_y = model.kdm.c_y
        sigma_vector = model.kernel.sigma  # (dim,) now, not scalar

        probs_check = dm2discrete(comp2dm(out_w, c_y.unsqueeze(0).expand(n, -1, -1)))
        probs_check_ok = bool(torch.allclose(probs_check, probs_forward, atol=1e-5))

        entropy = -(probs_forward * torch.log(probs_forward.clamp_min(1e-7))).sum(dim=1)

        c_x_expanded = c_x.unsqueeze(0).expand(n, -1, -1)
        dispersion = dm_rbf_variance_ard(comp2dm(out_w, c_x_expanded), sigma_vector)

        participation = 1.0 / (out_w ** 2).sum(dim=1)

    return {
        "probs": probs_forward.numpy(),
        "out_w": out_w.numpy(),
        "entropy": entropy.numpy(),
        "dispersion": dispersion.numpy(),
        "participation": participation.numpy(),
        "probs_check_ok": probs_check_ok,
    }


def occlusion_delta_ard(model: ARDKDMClassModel, X: np.ndarray, col_idx: list[int], fill_values: np.ndarray) -> np.ndarray:
    """ARD-safe version of kdm_backbone.occlusion_delta() -- calls compute_signals_ard() instead
    of the scalar compute_signals(), which is required (not just preferred) for an ARD-fitted
    model: the scalar version's dispersion computation crashes on a vector-shaped sigma even
    though occlusion_delta only ever needed the 'probs' key.
    """
    original = compute_signals_ard(model, X)["probs"][:, 1]
    X_occ = X.copy()
    X_occ[:, col_idx] = fill_values
    occluded = compute_signals_ard(model, X_occ)["probs"][:, 1]
    return occluded - original
