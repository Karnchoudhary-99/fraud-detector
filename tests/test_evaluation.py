import numpy as np
import pytest

from fraud_detector.evaluation import evaluate_predictions


def test_perfect_and_all_legit_predictions():
    perfect = evaluate_predictions(
        np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert perfect.precision == 1.0 and perfect.recall == 1.0
    assert perfect.roc_auc == 1.0 and perfect.pr_auc == 1.0
    assert perfect.accuracy == 1.0
    # The trap case: predicting "legit" everywhere is 50% accurate here
    # but recall 0 — the metric that matters for fraud is zero.
    lazy = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))
    assert lazy.recall == 0.0 and lazy.f1 == 0.0
    assert lazy.roc_auc is None  # no probabilities supplied


def test_confusion_counts_and_mismatched_inputs():
    report = evaluate_predictions(
        np.array([0, 1, 1, 0, 1]), np.array([1, 1, 0, 0, 1])
    )
    assert (report.true_positives, report.false_positives,
            report.true_negatives, report.false_negatives) == (2, 1, 1, 1)
    assert report.fraud_count == 3
    with pytest.raises(ValueError, match="equal length"):
        evaluate_predictions(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="at least one row"):
        evaluate_predictions(np.array([]), np.array([]))
