"""Feature engineering for transaction rows.

Built from a fitted dataset only (never from the rows being scored), so
the same transform applies identically at train and inference time:

* ``log_amount`` — log1p of the amount, taming the heavy right tail.
* ``hour_sin`` / ``hour_cos`` — cyclical encoding of hour-of-day, so
  23:00 and 01:00 are neighbours instead of 22 hours apart.
* ``amount_z`` — amount standardised against the training distribution.
* every ``V*`` column — standardised with its training mean/std.

Columns with zero training variance get std 1.0, so they pass through
as centred constants instead of dividing by zero.
"""

from __future__ import annotations

import numpy as np

from fraud_detector.transactions import TransactionDataset

ENGINEERED_NAMES = ("log_amount", "hour_sin", "hour_cos", "amount_z")


class FeatureEngineer:
    """Fit-once, transform-anywhere transaction featuriser."""

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    @property
    def output_names(self) -> list[str]:
        return [*ENGINEERED_NAMES, *self.feature_names_]

    def fit(self, dataset: TransactionDataset) -> "FeatureEngineer":
        self.feature_names_ = list(dataset.feature_names)
        raw = self._raw(dataset)
        self._means = raw.mean(axis=0)
        stds = raw.std(axis=0)
        self._stds = np.where(stds == 0, 1.0, stds)
        return self

    def transform(self, dataset: TransactionDataset) -> np.ndarray:
        if self._means is None or self._stds is None:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        if list(dataset.feature_names) != self.feature_names_:
            raise ValueError(
                f"feature columns mismatch: expected {self.feature_names_}, "
                f"got {list(dataset.feature_names)}"
            )
        return (self._raw(dataset) - self._means) / self._stds

    def fit_transform(self, dataset: TransactionDataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    @staticmethod
    def _raw(dataset: TransactionDataset) -> np.ndarray:
        hours = (dataset.times % 86400) / 3600.0
        engineered = np.column_stack([
            np.log1p(dataset.amounts),
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            dataset.amounts,
        ])
        return np.hstack([engineered, dataset.features])
