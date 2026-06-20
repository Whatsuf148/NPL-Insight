"""Trains the win-probability model from the master dataset and saves it
to the path declared in config (models/win_probability_model.pkl).
Re-run any time the master dataset changes — the model is never
hand-tuned or hardcoded.

Uses src.logistic_regression (pure numpy) instead of scikit-learn:
scipy's compiled extensions are blocked by this machine's Windows
Application Control policy, which would make any scipy-backed model
untrainable here regardless of the code. Plain numpy keeps the model
trainable everywhere with no loss of capability for four numeric
features.
"""
from __future__ import annotations

import pickle

import numpy as np

from src.analytics import win_probability_features
from src.config import load_config, resolve_path
from src.logistic_regression import LogisticRegression
from src.storage import load_table

FEATURE_COLUMNS = ["runs", "wickets", "catches_dropped", "fielding_errors"]


def _train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train(config: dict | None = None) -> LogisticRegression:
    config = config or load_config()
    master_df = load_table("master_dataset", config)
    features = win_probability_features(master_df)

    X = features[FEATURE_COLUMNS].to_numpy()
    y = features["won"].to_numpy()

    X_train, X_test, y_train, y_test = _train_test_split(X, y)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Win probability model test accuracy: {accuracy:.2f}")

    model_path = resolve_path(config["analytics"]["win_probability_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLUMNS}, f)
    print(f"Saved model to {model_path}")
    return model


if __name__ == "__main__":
    train()
