"""Decision-threshold tuning for fraud probabilities.

A classifier's default 0.5 threshold is calibrated for balanced
classes, not for 2% fraud.  Sweeping the threshold and picking the one
that maximises F1 on a held-out set trades a few extra false alarms
for fraud that would otherwise be missed — the trade a fraud team
actually wants, and one that must be chosen on validation data, never
on the rows the model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


@dataclass(frozen=True)
class ThresholdChoice:
    """Best threshold found, with its held-out fraud metrics."""

    threshold: float
    precision: float
    recall: float
    f1: float


def best_f1_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, steps: int = 101
) -> ThresholdChoice:
    """Return the threshold in [0, 1] maximising F1 for the fraud class."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must be non-empty and equal length")
    if len(set(y_true.tolist())) < 2:
        raise ValueError("threshold tuning requires both classes present")
    best = ThresholdChoice(0.5, 0.0, 0.0, -1.0)
    for threshold in np.linspace(0.0, 1.0, steps):
        y_pred = (y_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], zero_division=0
        )
        if f1[0] > best.f1:
            best = ThresholdChoice(
                float(threshold), float(precision[0]), float(recall[0]),
                float(f1[0]),
            )
    return best
