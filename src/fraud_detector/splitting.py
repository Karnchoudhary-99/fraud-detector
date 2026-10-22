"""Stratified train/test splitting for labelled transaction data.

Fraud is rare (about 2% of the bundled sample), so a naive random split
can easily leave the test set with zero fraud rows — and a metric
computed on such a split says nothing.  The split here is stratified and
refuses to proceed unless *both* classes appear in *both* halves, so a
silent, uninformative evaluation is impossible.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from fraud_detector.transactions import TransactionDataset


def subset(dataset: TransactionDataset, indices) -> TransactionDataset:
    """Return the rows of ``dataset`` at ``indices``, labels included."""
    idx = list(indices)
    labels = None if dataset.labels is None else dataset.labels[idx]
    return TransactionDataset(
        dataset.times[idx],
        dataset.amounts[idx],
        dataset.features[idx],
        list(dataset.feature_names),
        labels,
    )


def stratified_split(
    dataset: TransactionDataset, test_size: float = 0.25, seed: int = 42
) -> tuple[TransactionDataset, TransactionDataset]:
    """Split a labelled dataset, preserving the fraud rate in both halves.

    Raises ``ValueError`` for unlabelled data, for a single-class
    dataset, or when either half would contain fewer than two rows of
    a class (stratification needs at least that many to be meaningful).
    """
    if dataset.labels is None:
        raise ValueError("stratified_split requires labelled data")
    if len(set(dataset.labels.tolist())) < 2:
        raise ValueError("stratified_split requires both classes present")
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    for label in (0, 1):
        if int((dataset.labels == label).sum()) < 2:
            raise ValueError(
                f"class {label} has fewer than 2 rows; cannot stratify"
            )
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=seed,
        stratify=dataset.labels,
    )
    return subset(dataset, train_idx), subset(dataset, test_idx)
