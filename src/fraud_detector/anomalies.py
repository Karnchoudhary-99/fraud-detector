"""Unsupervised anomaly detectors for the cold-start case.

Both detectors share one interface: ``fit(X)`` learns the normal
profile, ``score(X)`` returns an anomaly score where higher means more
anomalous (isolation scores are scaled to [0, 1]; z-scores are mean
absolute z-values and are unbounded), and ``predict(X)`` flags the top
``contamination`` fraction of scores.  Using a quantile threshold (rather than a fixed
cut-off) keeps the flagged share stable as the score scale changes.

* :class:`ZScoreDetector` — per-feature standardisation fitted on the
  training rows; the score is the mean of the largest few absolute
  z-values, so a single wild feature flags a row without one noisy
  column dominating every score.
* :class:`IsolationForestDetector` — scikit-learn's isolation forest;
  its raw decision scores are mapped through the training-score range
  so they land on the same [0, 1] scale as the z-score detector.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def _to_unit_interval(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


class ZScoreDetector:
    """Mean-absolute-z anomaly detector over the top-k features."""

    def __init__(self, contamination: float = 0.02, top_k: int = 3) -> None:
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be in (0, 1)")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.contamination = contamination
        self.top_k = top_k
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._threshold: float | None = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        self._means = X.mean(axis=0)
        stds = X.std(axis=0)
        self._stds = np.where(stds == 0, 1.0, stds)
        scores = self.score(X)
        self._threshold = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._means is None or self._stds is None:
            raise RuntimeError("ZScoreDetector must be fitted before scoring")
        z = np.abs((np.asarray(X, dtype=float) - self._means) / self._stds)
        k = min(self.top_k, z.shape[1])
        top = np.sort(z, axis=1)[:, -k:]
        return top.mean(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._threshold is None:
            raise RuntimeError("ZScoreDetector must be fitted before predicting")
        return (self.score(X) >= self._threshold).astype(int)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Absolute per-feature z-values, i.e. what drives each score.

        Row ``i``'s score is the mean of its ``top_k`` largest values
        here, so the largest entries of a row are exactly the features
        that flagged it.  Raises if the detector is not fitted.
        """
        if self._means is None or self._stds is None:
            raise RuntimeError("ZScoreDetector must be fitted before scoring")
        return np.abs((np.asarray(X, dtype=float) - self._means) / self._stds)


class IsolationForestDetector:
    """Isolation-forest detector with [0, 1] scores."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42) -> None:
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be in (0, 1)")
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._threshold: float | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        self._model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        ).fit(X)
        # Raw decision_function is higher for normal rows; negate it so
        # higher means more anomalous, then scale by the training range.
        raw = -self._model.decision_function(X)
        self._lo, self._hi = float(raw.min()), float(raw.max())
        scores = self.score(X)
        self._threshold = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsolationForestDetector must be fitted before scoring")
        raw = -self._model.decision_function(np.asarray(X, dtype=float))
        return _to_unit_interval(raw, self._lo, self._hi)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._threshold is None:
            raise RuntimeError("IsolationForestDetector must be fitted before predicting")
        return (self.score(X) >= self._threshold).astype(int)
