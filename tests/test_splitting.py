import numpy as np
import pytest

from fraud_detector.splitting import stratified_split, subset
from fraud_detector.transactions import make_synthetic_transactions


def test_split_preserves_fraud_rate_and_is_deterministic():
    ds = make_synthetic_transactions(n=400, fraud_ratio=0.05, seed=7)
    train, test = stratified_split(ds, test_size=0.25, seed=11)
    assert len(train) + len(test) == 400
    assert train.fraud_rate == pytest.approx(0.05, abs=0.02)
    assert test.fraud_rate == pytest.approx(0.05, abs=0.02)
    train2, test2 = stratified_split(ds, test_size=0.25, seed=11)
    assert train.labels.tolist() == train2.labels.tolist()
    assert test.times.tolist() == test2.times.tolist()


def test_split_rejects_unlabelled_and_single_class():
    ds = make_synthetic_transactions(n=50, fraud_ratio=0.1, seed=3)
    from fraud_detector.transactions import TransactionDataset

    bare = TransactionDataset(ds.times, ds.amounts, ds.features,
                              list(ds.feature_names), None)
    with pytest.raises(ValueError, match="labelled"):
        stratified_split(bare)
    only_legit = ds.with_labels(np.zeros(len(ds), dtype=int))
    with pytest.raises(ValueError, match="both classes"):
        stratified_split(only_legit)


def test_subset_keeps_rows_and_labels():
    ds = make_synthetic_transactions(n=20, fraud_ratio=0.2, seed=1)
    sub = subset(ds, [0, 5, 19])
    assert len(sub) == 3
    assert sub.labels.tolist() == [int(ds.labels[i]) for i in (0, 5, 19)]
    assert sub.amounts.tolist() == [float(ds.amounts[i]) for i in (0, 5, 19)]
