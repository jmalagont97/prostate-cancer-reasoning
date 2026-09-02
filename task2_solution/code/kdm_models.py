"""sklearn-style .fit(X,y)/.predict(X) wrappers around fagonzalezo/kdm's KDMClassModel and
KDMRegressModel (https://github.com/fagonzalezo/kdm), so they drop straight into the existing
model_factory slot (task2_variant_tester.py, subtask_config.py) unchanged -- same pattern as
mord.LogisticIT (exp_14).

Install note: `pip install git+https://github.com/fagonzalezo/kdm.git` needs a working PyTorch CPU
build. torch's latest release (2.13.0 at install time) fails to import on this machine with a
DLL-init error (WinError 1114, c10.dll) -- confirmed not a missing-VC++-redistributable issue
(msvcp140/vcruntime140 are present) and not fixed by the standard KMP_DUPLICATE_LIB_OK workaround.
torch==2.2.0+cpu (`pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu`)
imports and runs cleanly -- pinned as the working version for this environment.

KDM's own README example trains via a manual loop (encoder -> KDMClassModel -> nll_loss -> Adam) --
that loop is encapsulated here, with the encoder and n_comp chosen to fit this project's small
per-fold training sets (n_comp = n_train, matching init_kdm_layer's requirement that its component
count equal the number of initialization rows -- the standard "centers = training points" scheme
for kernel methods at this sample size).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel, KDMRegressModel
from kdm.utils import dm_rbf_loglik, pure2dm


def _make_encoder(n_features: int, encoded_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, encoded_size),
        nn.ReLU(),
        nn.Linear(encoded_size, encoded_size),
    )


class KDMClassifier:
    """Wraps KDMClassModel behind .fit(X, y)/.predict(X). y may be any hashable label (str/int) --
    internally one-hot encoded against the classes seen in .fit().

    `generative` (exp_22): optional co-training weight for KDMClassModel's own documented
    "generative co-training" recipe (kdm_class_model.py's docstring) -- adds
    `-generative * kdm.log_marginal(pure2dm(encoder(x))).mean()` to the loss for the back half of
    training, after freezing the encoder (`encoder.requires_grad_(False)`), so only the KDM layer's
    own components (c_x, c_y, c_w, kernel sigma) get refined by the extra density-estimation
    signal. Motivated by exp_20's likely diagnosis (a real neural encoder + 71 free components
    overfitting on ~70 training rows) -- this is specifically a regularizer for that regime, not an
    arbitrary knob. `generative=0.0` (default) reproduces exp_20's exact behaviour unchanged.
    """

    def __init__(self, encoded_size: int = 16, sigma: float = 0.5, epochs: int = 300,
                 lr: float = 1e-2, seed: int = 0, generative: float = 0.0,
                 freeze_epoch_frac: float = 0.5):
        self.encoded_size = encoded_size
        self.sigma = sigma
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.generative = generative
        self.freeze_epoch_frac = freeze_epoch_frac

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_train, n_features = X.shape
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y])
        y_onehot = np.eye(n_classes, dtype=np.float32)[y_idx]

        X_t = torch.as_tensor(X)
        y_onehot_t = torch.as_tensor(y_onehot)
        y_idx_t = torch.as_tensor(y_idx, dtype=torch.long)

        encoder = _make_encoder(n_features, self.encoded_size)
        self.model = KDMClassModel(
            encoded_size=self.encoded_size, dim_y=n_classes, encoder=encoder,
            n_comp=n_train, sigma=self.sigma,
        )
        with torch.no_grad():
            encoded_init = self.model.encoder(X_t)
        init_kdm_layer(self.model.kdm, encoded_init, y_onehot_t, init_sigma=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        freeze_at = int(self.epochs * self.freeze_epoch_frac) if self.generative > 0 else self.epochs
        frozen = False
        for epoch in range(self.epochs):
            if self.generative > 0 and epoch >= freeze_at and not frozen:
                self.model.encoder.requires_grad_(False)
                frozen = True
            probs = self.model(X_t)
            loss = nn.functional.nll_loss(torch.log(probs + 1e-8), y_idx_t)
            if self.generative > 0 and epoch >= freeze_at:
                encoded = self.model.encoder(X_t)
                rho_x = pure2dm(encoded)
                gen_loglik = self.model.kdm.log_marginal(rho_x).mean()
                loss = loss - self.generative * gen_loglik
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        probs = self.model(torch.as_tensor(X))
        idx = probs.argmax(dim=-1).numpy()
        return self.classes_[idx]

    @torch.no_grad()
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        return self.model(torch.as_tensor(X)).numpy()


class KDMRegressor:
    """Wraps KDMRegressModel behind .fit(X, y)/.predict(X) -- y is a 1-D continuous target,
    predict() returns the KDM's predictive mean (predict_reg's y_exp)."""

    def __init__(self, encoded_size: int = 16, sigma_x: float = 0.5, sigma_y: float = 0.5,
                 epochs: int = 300, lr: float = 1e-2, seed: int = 0):
        self.encoded_size = encoded_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_train, n_features = X.shape

        X_t = torch.as_tensor(X)
        y_t = torch.as_tensor(y)

        encoder = _make_encoder(n_features, self.encoded_size)
        self.model = KDMRegressModel(
            encoded_size=self.encoded_size, dim_y=1, encoder=encoder, n_comp=n_train,
            sigma_x=self.sigma_x, sigma_y=self.sigma_y,
        )
        with torch.no_grad():
            encoded_init = self.model.encoder(X_t)
        init_kdm_layer(self.model.kdm, encoded_init, y_t, init_sigma=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            rho_y = self.model(X_t)
            loglik = dm_rbf_loglik(y_t, rho_y, self.model.sigma_y)
            loss = -loglik.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        y_exp, _ = self.model.predict_reg(torch.as_tensor(X))
        return y_exp.numpy().flatten()
