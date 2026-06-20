"""Trains the win-probability model from the master dataset and saves it
to the path declared in config (models/win_probability_model.pkl).
Re-run any time the master dataset changes — the model is never
hand-tuned or hardcoded.
"""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.analytics import win_probability_features
from src.config import load_config, resolve_path
from src.storage import load_table

FEATURE_COLUMNS = ["runs", "wickets", "catches_dropped", "fielding_errors"]


def train(config: dict | None = None) -> LogisticRegression:
    config = config or load_config()
    master_df = load_table("master_dataset", config)
    features = win_probability_features(master_df)

    X = features[FEATURE_COLUMNS]
    y = features["won"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Win probability model test accuracy: {accuracy:.2f}")

    model_path = resolve_path(config["analytics"]["win_probability_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, model_path)
    print(f"Saved model to {model_path}")
    return model


if __name__ == "__main__":
    train()
