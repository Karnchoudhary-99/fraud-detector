import numpy as np
import pytest

from fraud_detector.thresholds import best_f1_threshold


def test_tuning_beats_default_threshold_on_skewed_scores():
    # Fraud probabilities cluster just below 0.5: the default threshold
    # misses most of them, but a lower one recovers them.
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                        0.42, 0.44, 0.46, 0.48])
    choice = best_f1_threshold(y_true, y_proba)
    assert choice.threshold < 0.5
    assert choice.recall == 1.0
    assert choice.f1 == pytest.approx(0.8, abs=0.01)


def test_tuning_rejects_bad_inputs():
    with pytest.raises(ValueError, match="both classes"):
        best_f1_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="equal length"):
        best_f1_threshold(np.array([0, 1]), np.array([0.1]))
