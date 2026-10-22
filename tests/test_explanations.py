import numpy as np
import pytest

from fraud_detector.anomalies import ZScoreDetector
from fraud_detector.explanations import (
    explain_flags,
    format_explanation,
    top_contributors,
)

NAMES = ["log_amount", "hour_sin", "V1", "V2"]


def test_top_contributors_ranked_by_magnitude_and_direction():
    row = np.array([0.1, -2.5, 3.0, 0.5])
    contributors = top_contributors(row, NAMES, top_n=2)
    assert [c.feature for c in contributors] == ["V1", "hour_sin"]
    assert contributors[0].direction == "above typical"
    assert contributors[1].direction == "below typical"
    assert contributors[0].magnitude == pytest.approx(3.0)


def test_top_contributors_validates_shape_and_names():
    with pytest.raises(ValueError):
        top_contributors(np.array([[1.0, 2.0]]), NAMES)
    with pytest.raises(ValueError):
        top_contributors(np.array([1.0, 2.0]), NAMES)
    with pytest.raises(ValueError):
        top_contributors(np.array([1.0, 2.0, 3.0, 4.0]), NAMES, top_n=0)


def test_explain_flags_ranks_rows_by_score():
    X = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
    explanations = explain_flags(X, np.array([0.1, 0.9]), NAMES, top_n_rows=1)
    assert len(explanations) == 1
    assert explanations[0].row_index == 1
    assert explanations[0].contributors[0].feature == "V1"
    assert "row 1 (score 0.900)" in format_explanation(explanations[0])
    assert "V1 +4.00 SD above typical" in format_explanation(explanations[0])


def test_zscore_contributions_match_score_inputs():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 3))
    X[0, 2] = 12.0  # one wild feature in row 0
    detector = ZScoreDetector(contamination=0.02, top_k=1).fit(X)
    contributions = detector.feature_contributions(X)
    assert contributions.shape == X.shape
    assert int(np.argmax(contributions[0])) == 2
    with pytest.raises(RuntimeError):
        ZScoreDetector().feature_contributions(X)
