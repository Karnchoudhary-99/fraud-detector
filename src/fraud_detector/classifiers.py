"""Supervised fraud classifiers behind one small interface.

Two baselines, deliberately different in kind so their disagreement is
informative:

* ``logistic`` — linear, interpretable, ``class_weight="balanced"`` so
  the 2%-fraud minority is not simply ignored in favour of accuracy.
* ``gradient-boosting`` — histogram gradient boosting, which can learn
  the non-linear feature interactions a linear model cannot.

Both expose ``fit`` / ``predict`` / ``predict_proba`` over the already
engineered feature matrix, so the training pipeline and the CLI never
need to know which model they are driving.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class SupervisedClassifier:
    """A named sklearn estimator with fraud-probability output."""

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self._estimator = estimator
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedClassifier":
        if len(set(np.asarray(y).tolist())) < 2:
            raise ValueError(f"{self.name}: training data needs both classes")
        self._estimator.fit(X, y)
        self.classes_ = np.asarray(self._estimator.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the fraud class (label 1) for each row."""
        proba = np.asarray(self._estimator.predict_proba(X), dtype=float)
        fraud_col = list(self.classes_).index(1)
        return proba[:, fraud_col]


def make_classifier(name: str) -> SupervisedClassifier:
    """Build a classifier by name; unknown names fail loudly."""
    if name == "logistic":
        return SupervisedClassifier(
            "logistic",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    if name == "gradient-boosting":
        return SupervisedClassifier(
            "gradient-boosting",
            HistGradientBoostingClassifier(random_state=42),
        )
    raise ValueError(
        f"unknown classifier {name!r}; expected 'logistic' or 'gradient-boosting'"
    )


CLASSIFIER_NAMES = ("logistic", "gradient-boosting")
