#!/usr/bin/env python3
"""
deploy/subtask_1.1/knn_model.py

Self-contained ConfidenceWeightedKNN model class for pickling compatibility.
"""

import numpy as np
from numpy.linalg import norm


class ConfidenceWeightedKNN:
    def __init__(self, n_neighbors=1, metric="cosine", use_distance_weight=False, epsilon=1e-10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_distance_weight = use_distance_weight
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        self.conf_weights = None

    def fit(self, X, y, conf_weights):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        self.conf_weights = np.array(conf_weights, dtype=np.float64)

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        if self.metric == "cosine":
            X_norm = X / (norm(X, axis=1, keepdims=True) + self.epsilon)
            T_norm = self.X_train / (norm(self.X_train, axis=1, keepdims=True) + self.epsilon)
            dists = 1 - X_norm @ T_norm.T
            dists = np.clip(dists, 0, 2)
        else: # euclidean
            diff = X[:, None, :] - self.X_train[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))

        proba = np.zeros(len(X))
        for i in range(len(X)):
            nn_idx = np.argsort(dists[i])[:self.n_neighbors]
            d_nn = dists[i, nn_idx]
            y_nn = self.y_train[nn_idx]
            c_nn = self.conf_weights[nn_idx]

            if self.use_distance_weight and self.n_neighbors > 1:
                w_dist = 1.0 / (d_nn + self.epsilon)
            else:
                w_dist = np.ones_like(d_nn)

            q = 0.5 + c_nn * (y_nn - 0.5)
            proba[i] = np.sum(w_dist * q) / (np.sum(w_dist) + self.epsilon)

        return proba
