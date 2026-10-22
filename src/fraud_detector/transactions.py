"""Transaction dataset loading and synthetic data generation.

The canonical CSV layout mirrors the well-known credit-card fraud
dataset shape without shipping any real cardholder data:

    time, amount, V1, V2, ..., Vn, is_fraud

``time`` is seconds since the first transaction in the file, ``amount``
is the transaction amount, ``V*`` are anonymised numeric features, and
``is_fraud`` (optional) is 0/1.  A ``class`` column is accepted as an
alias for ``is_fraud``.  Every row is validated: time and amount must be
finite, amount must be non-negative, and feature values must be finite.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = ("time", "amount")


@dataclass
class TransactionDataset:
    """An in-memory transaction table."""

    times: np.ndarray
    amounts: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.times)
        if len(self.amounts) != n or len(self.features) != n:
            raise ValueError("times, amounts and features must have equal length")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels must match the number of transactions")
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix width must match feature_names")

    def __len__(self) -> int:
        return len(self.times)

    @property
    def fraud_rate(self) -> float | None:
        if self.labels is None or len(self.labels) == 0:
            return None
        return float(np.mean(self.labels))

    def with_labels(self, labels: np.ndarray) -> "TransactionDataset":
        return TransactionDataset(
            self.times, self.amounts, self.features, list(self.feature_names),
            np.asarray(labels, dtype=int),
        )


def _parse_float(value: str, column: str, row_no: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_no}: {column} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row_no}: {column} must be finite, got {value!r}")
    return number


def load_transactions(path: str | Path) -> TransactionDataset:
    """Load and validate a transaction CSV file."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV file")
        header = [name.strip() for name in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")
        label_col = "is_fraud" if "is_fraud" in header else ("class" if "class" in header else None)
        feature_names = [c for c in header if c not in (*REQUIRED_COLUMNS, "is_fraud", "class")]
        times, amounts, rows, labels = [], [], [], []
        for row_no, raw in enumerate(reader, start=2):
            if raw is None or all((v or "").strip() == "" for v in raw.values()):
                continue  # skip blank lines
            times.append(_parse_float(raw["time"], "time", row_no))
            amount = _parse_float(raw["amount"], "amount", row_no)
            if amount < 0:
                raise ValueError(f"row {row_no}: amount must be >= 0, got {amount}")
            amounts.append(amount)
            rows.append([_parse_float(raw[name], name, row_no) for name in feature_names])
            if label_col is not None:
                label = _parse_float(raw[label_col], label_col, row_no)
                if label not in (0.0, 1.0):
                    raise ValueError(f"row {row_no}: {label_col} must be 0 or 1, got {label}")
                labels.append(int(label))
    if not times:
        raise ValueError(f"{path}: no data rows found")
    features = np.asarray(rows, dtype=float).reshape(len(times), len(feature_names))
    return TransactionDataset(
        np.asarray(times, dtype=float),
        np.asarray(amounts, dtype=float),
        features,
        feature_names,
        np.asarray(labels, dtype=int) if label_col is not None else None,
    )


def save_transactions(dataset: TransactionDataset, path: str | Path) -> None:
    """Write a dataset back to CSV in the canonical layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "amount", *dataset.feature_names, "is_fraud"])
        for i in range(len(dataset)):
            label = "" if dataset.labels is None else int(dataset.labels[i])
            writer.writerow([dataset.times[i], dataset.amounts[i],
                             *dataset.features[i], label])


def make_synthetic_transactions(
    n: int = 2000,
    fraud_ratio: float = 0.02,
    n_features: int = 8,
    seed: int = 42,
) -> TransactionDataset:
    """Generate a deterministic synthetic transaction set.

    Legitimate transactions cluster around modest amounts and centred
    features; fraudulent ones skew to larger amounts, a shifted feature
    mean, and bursts at odd hours.  The split is deliberately learnable
    but overlapping, so detectors must rank rather than separate cleanly.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= fraud_ratio < 1.0:
        raise ValueError("fraud_ratio must be in [0, 1)")
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_ratio))
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)

    times = np.sort(rng.uniform(0, 48 * 3600, size=n))
    amounts = np.where(
        labels == 1,
        rng.lognormal(mean=5.2, sigma=0.9, size=n),   # fraud: ~EUR 180 median
        rng.lognormal(mean=3.4, sigma=0.8, size=n),   # legit: ~EUR 30 median
    )
    features = rng.normal(0.0, 1.0, size=(n, n_features))
    fraud_rows = labels == 1
    features[fraud_rows] += rng.normal(1.2, 0.6, size=(int(fraud_rows.sum()), n_features))
    # Fraud bursts: compress a share of fraud times into night hours.
    night = fraud_rows & (rng.random(n) < 0.5)
    times[night] = (times[night] % 3600) + rng.choice([1, 2, 3, 4], size=int(night.sum())) * 3600
    order = np.argsort(times)
    return TransactionDataset(
        times[order],
        amounts[order],
        features[order],
        [f"V{i + 1}" for i in range(n_features)],
        labels[order],
    )
