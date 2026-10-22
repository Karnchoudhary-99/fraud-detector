import numpy as np
import pytest

from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import make_synthetic_transactions


def test_engineered_columns_and_standardisation():
    ds = make_synthetic_transactions(n=200, seed=1)
    X = FeatureEngineer().fit_transform(ds)
    assert X.shape == (200, 4 + 8)
    # Standardised output: near-zero mean, unit variance per column.
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X.std(axis=0), 1, atol=1e-9)


def test_hour_encoding_is_cyclical():
    engineer = FeatureEngineer()
    ds = make_synthetic_transactions(n=100, seed=1)
    engineer.fit(ds)
    from fraud_detector.transactions import TransactionDataset

    late = TransactionDataset(
        np.array([23 * 3600.0]), np.array([10.0]),
        np.zeros((1, 8)), [f"V{i + 1}" for i in range(8)],
    )
    early = TransactionDataset(
        np.array([1 * 3600.0]), np.array([10.0]),
        np.zeros((1, 8)), [f"V{i + 1}" for i in range(8)],
    )
    a, b = engineer.transform(late)[0], engineer.transform(early)[0]
    # hour_sin/hour_cos (cols 1, 2) must be close for 23:00 vs 01:00.
    assert np.linalg.norm(a[1:3] - b[1:3]) < 1.0


def test_transform_requires_fit_and_matching_columns():
    engineer = FeatureEngineer()
    ds = make_synthetic_transactions(n=20, seed=1)
    with pytest.raises(RuntimeError, match="must be fitted"):
        engineer.transform(ds)
    engineer.fit(ds)
    wrong = make_synthetic_transactions(n=20, n_features=4, seed=1)
    with pytest.raises(ValueError, match="mismatch"):
        engineer.transform(wrong)
