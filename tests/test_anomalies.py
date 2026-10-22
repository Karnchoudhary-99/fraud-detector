import numpy as np
import pytest

from fraud_detector.anomalies import IsolationForestDetector, ZScoreDetector
from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import make_synthetic_transactions


def _engineered(seed=5):
    ds = make_synthetic_transactions(n=500, fraud_ratio=0.02, seed=seed)
    X = FeatureEngineer().fit_transform(ds)
    return ds, X


@pytest.mark.parametrize("detector_cls", [ZScoreDetector, IsolationForestDetector])
def test_detector_flags_expected_share_and_ranks_fraud_high(detector_cls):
    ds, X = _engineered()
    detector = detector_cls(contamination=0.02).fit(X)
    preds = detector.predict(X)
    # Quantile threshold flags ~2% (ties may push it slightly over).
    assert 8 <= int(preds.sum()) <= 15
    scores = detector.score(X)
    top = np.argsort(scores)[::-1][:20]
    # At least half of the top-20 should be true fraud: the synthetic
    # fraud cluster is shifted, but deliberately overlapping.
    assert int(ds.labels[top].sum()) >= 10


def test_zscore_flags_injected_outlier():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(200, 4))
    detector = ZScoreDetector(contamination=0.05).fit(normal)
    outlier = np.array([[50.0, 0.0, 0.0, 0.0]])
    assert detector.predict(outlier)[0] == 1
    assert detector.predict(np.array([[0.0, 0.0, 0.0, 0.0]]))[0] == 0


def test_isolation_scores_are_unit_interval():
    _, X = _engineered()
    detector = IsolationForestDetector(contamination=0.02).fit(X)
    scores = detector.score(X)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_detectors_reject_bad_input():
    with pytest.raises(ValueError, match="contamination"):
        ZScoreDetector(contamination=1.5)
    with pytest.raises(RuntimeError, match="must be fitted"):
        ZScoreDetector().score(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        IsolationForestDetector().fit(np.zeros((0, 2)))
