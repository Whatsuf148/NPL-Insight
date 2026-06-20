"""Minimal logistic regression via gradient descent, pure numpy.

scikit-learn/scipy are unusable on this machine (Windows Application
Control policy blocks scipy's compiled DLLs at the OS level — not a
code bug, see models/train_win_probability.py). Reimplementing this
narrow piece in numpy keeps the win-probability model runnable
everywhere without weakening the analytics — logistic regression on
four numeric features doesn't need scipy's machinery.
"""
from __future__ import annotations

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 2000, l2: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        X_norm = (X - self.feature_mean) / self.feature_std

        n_samples, n_features = X_norm.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear = X_norm @ self.weights + self.bias
            preds = self._sigmoid(linear)
            error = preds - y

            grad_w = (X_norm.T @ error) / n_samples + self.l2 * self.weights
            grad_b = error.mean()

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.feature_mean) / self.feature_std
        return self._sigmoid(X_norm @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
