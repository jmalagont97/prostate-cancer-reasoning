"""KDM-based confidence model — comparison against the plain OvR-logistic baseline.

Design choice for N=91 (≈73/fold in 5-fold CV): the KDM prototypes (c_x, c_y, c_w)
are FROZEN at the training data itself (x_train=y_train=w_train=False) — this is a
memory-based / Parzen-window-style classifier. Only the RBF kernel bandwidth
(sigma) is learned via gradient descent. This is deliberately the lowest-variance
KDM configuration: letting all of c_x/c_y/c_w train too would add ~n_comp free
parameters on top of an already-tiny N, defeating the point of trying KDM here
(better-calibrated uncertainty, not a higher-capacity model).

See .claude/skills/kdm/references/{classification,uncertainty}.md for the
library's own reference scaffolds this follows.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe -m chimera_task1.train_confidence_kdm
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from chimera_task1.features import build_preprocessor, select_feature_frame
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
N_EPOCHS = 300


def fit_predict_kdm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_classes: int) -> np.ndarray:
    torch.manual_seed(RANDOM_STATE)
    dim_x = X_train.shape[1]
    n_comp = len(X_train)  # memory-based: every training row is a KDM prototype

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    y_onehot = F.one_hot(yt, n_classes).float()

    encoder = nn.Identity()  # no learned projection at this N -- kernel acts on the raw scaled features
    model = KDMClassModel(
        encoded_size=dim_x,
        dim_y=n_classes,
        encoder=encoder,
        n_comp=n_comp,
        sigma=0.5,
        sigma_trainable=True,
        x_train=False,  # freeze prototypes = training data (memory-based / Parzen-window)
        y_train=False,
        w_train=False,  # uniform mixture weights, not learned
    )

    model.eval()
    with torch.no_grad():
        enc_sub = encoder(Xt)  # == Xt, but keeps the general init pattern from the skill
    init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot, init_sigma=True)
    model.train()

    # Only sigma has requires_grad=True given x_train=y_train=w_train=False.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(N_EPOCHS):
        probs = model(Xt)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_probs = model(torch.as_tensor(X_test, dtype=torch.float32))
    return test_probs.numpy()


def main() -> None:
    ann, inp_ann = load_annotated()
    X_frame = select_feature_frame(inp_ann)
    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    X = StandardScaler().fit_transform(X_pre)  # KDM's RBF kernel needs comparable feature scales

    y_labels = ann["target_confidence"].values
    y = np.array([CONFIDENCE_RANK[label] for label in y_labels])
    n_classes = len(CONFIDENCE_LEVELS)
    majority = CONFIDENCE_LEVELS[np.bincount(y).argmax()]

    dists, majority_dists, entropies = [], [], []
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        preds = np.empty(len(y), dtype=object)
        probs_all = np.zeros((len(y), n_classes))
        for train_idx, test_idx in kf.split(X):
            test_probs = fit_predict_kdm(X[train_idx], y[train_idx], X[test_idx], n_classes)
            probs_all[test_idx] = test_probs
            pred_idx = test_probs.argmax(axis=1)
            preds[test_idx] = [CONFIDENCE_LEVELS[i] for i in pred_idx]

        dists.append(ordinal_distance(list(y_labels), list(preds), CONFIDENCE_RANK))
        majority_dists.append(ordinal_distance(list(y_labels), [majority] * len(y_labels), CONFIDENCE_RANK))
        entropy = -(probs_all * np.log(np.clip(probs_all, 1e-7, 1))).sum(axis=1)
        entropies.append(entropy.mean())

    print(f"n annotated = {len(ann)}, classes = {CONFIDENCE_LEVELS}, majority = '{majority}'\n")
    print(
        f"KDM confidence       ordinal_distance = {np.mean(dists):.3f} +/- {np.std(dists):.3f}"
        f"   (always-'{majority}' baseline = {np.mean(majority_dists):.3f})"
    )
    print(f"  mean predictive entropy (out-of-fold): {np.mean(entropies):.3f}  (max possible = {np.log(n_classes):.3f})")
    print("  (compare against OvR-logistic result from `train_reasoning.py`'s eval_confidence)")


if __name__ == "__main__":
    main()
