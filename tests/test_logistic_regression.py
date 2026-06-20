import numpy as np

from src.logistic_regression import LogisticRegression


def test_fits_linearly_separable_data_with_high_accuracy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    assert model.score(X, y) > 0.9


def test_predict_proba_is_between_zero_and_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_matches_threshold_on_proba():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)
    preds = model.predict(X)
    assert (preds == (probs >= 0.5).astype(int)).all()
