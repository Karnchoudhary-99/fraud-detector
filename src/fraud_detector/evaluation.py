"""Imbalance-aware evaluation for fraud predictions.

Accuracy is meaningless at 2% fraud — a model that predicts "legit"
for everything scores 98% and catches nothing.  Everything here is
reported per class, with precision/recall/F1 for the fraud class, the
confusion matrix, and ranking metrics (ROC-AUC, PR-AUC).  PR-AUC is the
headline ranking metric: unlike ROC-AUC it does not get flattered by
the huge number of easy true negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationReport:
    """One holdout evaluation, fraud class (1) front and centre."""

    n: int
    fraud_count: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.n if self.n else 0.0


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> EvaluationReport:
    """Score binary predictions (and optional fraud probabilities)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        raise ValueError("evaluate_predictions requires at least one row")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    roc_auc = pr_auc = None
    if y_proba is not None and len(set(y_true.tolist())) == 2:
        proba = np.asarray(y_proba, dtype=float)
        roc_auc = float(roc_auc_score(y_true, proba))
        pr_auc = float(average_precision_score(y_true, proba))
    return EvaluationReport(
        n=len(y_true),
        fraud_count=int((y_true == 1).sum()),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        precision=float(precision[0]),
        recall=float(recall[0]),
        f1=float(f1[0]),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )
