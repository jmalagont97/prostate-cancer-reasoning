"""exp_14: KDM regression backbone for weights -- trains directly on the ordinal weight rank
(0-3) as a continuous target via KDMRegressModel + dm_rbf_loglik, instead of KDMClassModel's
rank-blind classification loss. Mirrors experiments/exp_6/scripts/kdm_backbone.py's
fit_kdm_backbone/compute_signals shape as closely as the regression model's extra sigma_y
parameter allows. See DESIGN.md/IMPLEMENTATION.md for the full derivation and the three smoke
tests run against the installed kdm-torch API before this was written.

Serves both exp_14 conditions unchanged: dim_y=1 (per-factor, called once per factor) and
dim_y=9 (joint, called once for all factors at once) -- no duplicated fit/predict logic between
the two run scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from kdm.init import init_kdm_layer
from kdm.models import KDMRegressModel
from kdm.utils import dm_rbf_loglik

RANDOM_STATE = 0
N_EPOCHS = 300
N_LEVELS = 4  # not_used(0) < noted(1) < important(2) < decisive(3)


def fit_kdm_regress(X_train: np.ndarray, y_train_rank: np.ndarray, dim_y: int,
                     n_epochs: int = N_EPOCHS, lr: float = 1e-2) -> KDMRegressModel:
    """Fit a memory-based KDM regressor (frozen prototypes, sigma_x/sigma_y trained) on the
    ordinal rank(s) directly. y_train_rank must already be shaped (n, dim_y) -- a (n,1) column
    for the per-factor condition, a (n,9) matrix (every factor's rank at once) for the joint
    condition.
    """
    torch.manual_seed(RANDOM_STATE)
    dim_x = X_train.shape[1]
    n_comp = len(X_train)  # memory-based: every training row is a KDM prototype

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train_rank, dtype=torch.float32).reshape(n_comp, dim_y)

    encoder = nn.Identity()
    model = KDMRegressModel(
        encoded_size=dim_x, dim_y=dim_y, encoder=encoder, n_comp=n_comp,
        x_train=False, y_train=False, w_train=False,
    )

    model.eval()
    with torch.no_grad():
        enc_sub = encoder(Xt)
    # init_kdm_layer only sets kernel.sigma (input-side bandwidth) from the k-NN heuristic --
    # sigma_y (output-side kernel width) is left at its constructor default (0.1) and trained
    # from there via gradient descent. Verified in this session's smoke tests: this converges
    # cleanly (sigma_y shrinks 0.10 -> ~0.01 over 300 epochs on real-shaped data), not a bug.
    init_kdm_layer(model.kdm, enc_sub.detach(), yt, init_sigma=True)
    model.train()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(n_epochs):
        rho_y = model(Xt)
        loss = -dm_rbf_loglik(yt, rho_y, model.sigma_y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _normal_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via the error function -- torch has no built-in norm.cdf, and this
    keeps everything in torch rather than adding a scipy dependency to the pipeline."""
    z = (x - mean) / (std * (2 ** 0.5))
    return 0.5 * (1.0 + torch.special.erf(z))


def compute_signals_regress(model: KDMRegressModel, X: np.ndarray, n_levels: int = N_LEVELS) -> dict:
    """Given a fitted regressor and a batch of (already scaled) query rows, return the rounded
    ordinal prediction plus regression-derived pseudo-probabilities over the n_levels discrete
    ranks -- an APPROXIMATION (Normal(mean, variance) probability mass per rank bin), not a
    native classifier's calibrated probs. Every caller must treat probs_are_pseudo=True as a
    standing flag on the output, per DESIGN.md Section 2c.

    variance is a single scalar per case (confirmed by direct test against KDMRegressModel --
    predict_reg's dm_rbf_variance is inherently scalar-valued), broadcast across every output
    dimension j for dim_y>1 (the joint condition) -- only the per-dimension mean differs there.
    """
    Xt = torch.as_tensor(X, dtype=torch.float32)
    mean, variance = model.predict_reg(Xt)  # mean: (n, dim_y), variance: (n,)
    dim_y = mean.shape[1]
    std = variance.clamp_min(1e-6).sqrt()  # (n,)

    pred_rank = mean.round().clamp(0, n_levels - 1).long()  # (n, dim_y)

    # 4-bin Normal-CDF pseudo-probabilities per output dimension.
    edges = torch.arange(-0.5, n_levels - 0.5 + 1e-6, 1.0)  # [-0.5, 0.5, 1.5, 2.5, 3.5]
    probs = torch.zeros(mean.shape[0], dim_y, n_levels)
    for j in range(dim_y):
        cdf_hi = _normal_cdf(edges[1:].unsqueeze(0), mean[:, j:j + 1], std.unsqueeze(1))  # (n, n_levels)
        cdf_lo = _normal_cdf(edges[:-1].unsqueeze(0), mean[:, j:j + 1], std.unsqueeze(1))
        mass = (cdf_hi - cdf_lo).clamp_min(1e-9)
        probs[:, j, :] = mass / mass.sum(dim=1, keepdim=True)

    return {
        "mean": mean.detach().numpy(),
        "variance": variance.detach().numpy(),
        "pred_rank": pred_rank.detach().numpy(),
        "pseudo_probs": probs.detach().numpy(),  # (n, dim_y, n_levels)
        "probs_are_pseudo": True,
        "auroc_brier_note": "AUROC/Brier for this condition use regression-derived pseudo-"
                             "probabilities (Normal-CDF mass per rank bin from predict_reg's "
                             "mean/variance), not a native classifier's calibrated probs.",
    }
