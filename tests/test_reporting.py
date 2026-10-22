import numpy as np

from fraud_detector.evaluation import evaluate_predictions
from fraud_detector.reporting import format_report
from fraud_detector.thresholds import best_f1_threshold


def test_report_shows_fraud_metrics_and_confusion_first():
    report = evaluate_predictions(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]),
        np.array([0.1, 0.6, 0.7, 0.4]),
    )
    text = format_report("logistic", report)
    assert "model=logistic rows=4 fraud=2" in text
    assert "TP=1 FP=1 TN=1 FN=1" in text
    assert "precision=0.5000 recall=0.5000" in text


def test_report_handles_missing_ranking_metrics_and_threshold():
    report = evaluate_predictions(np.array([0, 1]), np.array([0, 1]))
    text = format_report("logistic", report)
    assert "roc_auc=n/a pr_auc=n/a" in text
    threshold = best_f1_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    text_with = format_report("logistic", report, threshold)
    assert "best_f1_threshold=" in text_with
