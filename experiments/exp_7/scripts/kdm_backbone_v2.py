"""exp_7: configurable-hyperparameter KDM backbone fit + skew-aware log1p preprocessing helper.

compute_signals()/occlusion_delta()/kernel_distance_contribution() are re-exported unchanged from
exp_6/scripts/kdm_backbone.py -- they operate purely on an already-fitted model object and don't
care how it was trained, so there's nothing to duplicate here. Only the fit function itself needs
a configurable version (n_epochs/lr/sigma_mult/optimizer/weight_decay, none of which exp_6 ever
varied), plus the log1p transform for exp_6's most-skewed columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
from kdm_backbone import compute_signals, occlusion_delta, kernel_distance_contribution  # noqa: E402,F401

RANDOM_STATE = 0

# The 3 columns confirmed this session to be meaningfully right-skewed in exp_3's 19-column frame
# (cli_psa skew=4.28, cli_psad skew=4.24, cli_vol skew=1.29). cli_cspca (skew=-2.06) is
# deliberately excluded -- it's left-skewed and bounded near 1, a different shape that log1p
# doesn't fix. See experiments/exp_7/DESIGN.md Section 2.
LOG1P_COLUMNS = ["cli_psa", "cli_psad", "cli_vol"]


def apply_log1p_transform(X_pre: np.ndarray, col_idx: list[int]) -> np.ndarray:
    """Apply np.log1p to the given column indices of an already-imputed (not yet scaled) array.

    log1p has no fitted parameters (unlike the imputer's median or the scaler's mean/std), so
    it's safe to apply once before the CV split -- this project's leakage discipline concerns
    fitted statistics, not fixed deterministic transforms.
    """
    X = X_pre.copy()
    X[:, col_idx] = np.log1p(X[:, col_idx])
    return X


def fit_kdm_backbone(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 2,
    n_epochs: int = 300,
    lr: float = 1e-2,
    sigma_mult: float = 1.0,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
) -> KDMClassModel:
    """Same memory-based KDM fit as exp_6's fit_kdm_backbone, with 5 previously-hardcoded values
    now exposed: n_epochs, lr, sigma_mult (threaded into kdm.init.init_kdm_layer, which already
    accepts it -- no need to reimplement the KNN-based sigma-init logic), optimizer ("adam" or
    "adamw"), and weight_decay (meaningful only for adamw; adam is always called with
    weight_decay=0.0 per DESIGN.md's reasoning -- see experiments/exp_7/DESIGN.md Section 2).
    """
    torch.manual_seed(RANDOM_STATE)
    dim_x = X_train.shape[1]
    n_comp = len(X_train)

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
    init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot, init_sigma=True, sigma_mult=sigma_mult)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {optimizer!r}")

    for _ in range(n_epochs):
        probs = model(Xt)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model
