"""exp_6: KDM diagnostic module -- fit once, derive decision + confidence + weights signals.

Reuses train_confidence_kdm.fit_predict_kdm's exact hyperparameters/training loop (300 epochs,
Adam lr=1e-2, sigma-only trainable, n_comp=len(X_train)) but returns the fitted model object
instead of just predictions, so downstream code can query its internal density-matrix structure
without retraining. `fit_predict_kdm` itself is left untouched (still used verbatim by
exp_2-exp_5) -- this module is purely additive, per this project's "shared code stays
reproducible" convention.

See experiments/exp_6/DESIGN.md for the derivation of each signal, and IMPLEMENTATION.md for the
library-source grounding (kdm/layers/kdm_layer.py, kdm/utils.py, kdm/init.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel
from kdm.utils import comp2dm, dm2discrete, dm_rbf_variance, pure2dm

RANDOM_STATE = 0
N_EPOCHS = 300


def fit_kdm_backbone(X_train: np.ndarray, y_train: np.ndarray, n_classes: int = 2) -> KDMClassModel:
    """Fit a memory-based KDM (frozen prototypes, sigma-only trained) and return the model itself.

    Identical to train_confidence_kdm.fit_predict_kdm's fit loop, except it returns the fitted
    model in eval mode instead of collapsing straight to a numpy prediction array -- callers need
    the model object to query out_w/c_x/sigma for the confidence/weights signals below.
    """
    torch.manual_seed(RANDOM_STATE)
    dim_x = X_train.shape[1]
    n_comp = len(X_train)  # memory-based: every training row is a KDM prototype

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    y_onehot = F.one_hot(yt, n_classes).float()

    encoder = nn.Identity()
    model = KDMClassModel(
        encoded_size=dim_x,
        dim_y=n_classes,
        encoder=encoder,
        n_comp=n_comp,
        sigma=0.5,
        sigma_trainable=True,
        x_train=False,
        y_train=False,
        w_train=False,
    )

    model.eval()
    with torch.no_grad():
        enc_sub = encoder(Xt)
    init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot, init_sigma=True)
    model.train()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(N_EPOCHS):
        probs = model(Xt)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _prototype_weights(model: KDMClassModel, X: torch.Tensor) -> torch.Tensor:
    """Replicate KDMLayer.forward()'s _compute_mixture -> clamp -> normalize steps exactly, up
    to (not including) the final matmul with c_y. Returns the normalized per-prototype posterior
    weight, shape (n, n_comp), summing to 1 across the n_comp axis for every row.
    """
    rho_x = pure2dm(X)  # (n, 1, dim_x + 1)
    kdm_layer = model.kdm
    in_w, out_w = kdm_layer._compute_mixture(rho_x)  # in_w: (n,1); out_w: (n,1,n_comp)
    out_w = out_w.clamp(min=kdm_layer.eps)
    out_w = out_w / out_w.sum(dim=2, keepdim=True)
    out_w = torch.einsum("...i,...ij->...j", in_w, out_w)  # (n, n_comp)
    return out_w


def compute_signals(model: KDMClassModel, X: np.ndarray) -> dict:
    """Given a fitted backbone and a batch of (already scaled) query rows, return every signal
    this experiment derives from it. Includes a built-in cross-check (see 'probs_check_ok')
    that the hand-replicated normalization in `_prototype_weights` matches the model's own
    forward() output -- must be True before any downstream signal is trusted.
    """
    Xt = torch.as_tensor(X, dtype=torch.float32)
    n = Xt.shape[0]

    with torch.no_grad():
        probs_forward = model(Xt)  # (n, n_classes) -- the model's own, authoritative output
        out_w = _prototype_weights(model, Xt)  # (n, n_comp)

        c_x = model.kdm.c_x  # (n_comp, dim_x), frozen training prototypes
        c_y = model.kdm.c_y  # (n_comp, n_classes), frozen one-hot labels
        sigma = model.kernel.sigma  # scalar tensor

        # Cross-check: dm2discrete(comp2dm(out_w, c_y)) should equal probs_forward exactly.
        probs_check = dm2discrete(comp2dm(out_w, c_y.unsqueeze(0).expand(n, -1, -1)))
        probs_check_ok = bool(torch.allclose(probs_check, probs_forward, atol=1e-5))

        entropy = -(probs_forward * torch.log(probs_forward.clamp_min(1e-7))).sum(dim=1)

        c_x_expanded = c_x.unsqueeze(0).expand(n, -1, -1)
        dispersion = dm_rbf_variance(comp2dm(out_w, c_x_expanded), sigma)

        participation = 1.0 / (out_w**2).sum(dim=1)

    return {
        "probs": probs_forward.numpy(),
        "out_w": out_w.numpy(),
        "entropy": entropy.numpy(),
        "dispersion": dispersion.numpy(),
        "participation": participation.numpy(),
        "probs_check_ok": probs_check_ok,
    }


def occlusion_delta(model: KDMClassModel, X: np.ndarray, col_idx: list[int], fill_values: np.ndarray) -> np.ndarray:
    """Signal D: replace columns `col_idx` with `fill_values` (that fold's training median --
    correct for binary/one-hot columns too, since the median of a skewed 0/1 column equals its
    mode), re-run the fitted backbone, return signed delta in p(class=1) (occluded - original).
    """
    original = compute_signals(model, X)["probs"][:, 1]
    X_occ = X.copy()
    X_occ[:, col_idx] = fill_values
    occluded = compute_signals(model, X_occ)["probs"][:, 1]
    return occluded - original


def kernel_distance_contribution(model: KDMClassModel, X: np.ndarray, col_idx: list[int]) -> np.ndarray:
    """Signal E: per-case, per-factor attribution read directly off the forward pass's own
    quantities -- no re-inference needed. sum_i out_w_i(x) * sum_{j in col_idx} (x_j - c_x[i,j])^2
    """
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        out_w = _prototype_weights(model, Xt)  # (n, n_comp)
        c_x = model.kdm.c_x  # (n_comp, dim_x)
        x_sub = Xt[:, col_idx]  # (n, |col_idx|)
        c_sub = c_x[:, col_idx]  # (n_comp, |col_idx|)
        # (n, n_comp, |col_idx|) squared diffs, summed over the factor's own columns
        sq_dist = (x_sub.unsqueeze(1) - c_sub.unsqueeze(0)) ** 2
        sq_dist = sq_dist.sum(dim=2)  # (n, n_comp)
        contribution = (out_w * sq_dist).sum(dim=1)  # (n,)
    return contribution.numpy()
