import numpy as np
import pytest

from fraud_detector.splitting import stratified_split
from fraud_detector.training import train_supervised
from fraud_detector.transactions import TransactionDataset, make_synthetic_transactions


def test_train_predict_round_trip_and_no_leak():
    ds = make_synthetic_transactions(n=400, fraud_ratio=0.1, seed=9)
    train, test = stratified_split(ds, test_size=0.25, seed=4)
    model = train_supervised(train, "logistic")
    preds = model.predict(test)
    proba = model.predict_proba(test)
    assert len(preds) == len(test) and len(proba) == len(test)
    assert proba[test.labels == 1].mean() > proba[test.labels == 0].mean()
    # The engineer is fitted on train only, so transforming the
    # training rows gives (near) zero mean by construction.
    assert np.abs(model.engineer.transform(train).mean(axis=0)).max() < 1e-6


def test_train_rejects_unlabelled_data():
    ds = make_synthetic_transactions(n=20, fraud_ratio=0.2, seed=1)
    bare = TransactionDataset(ds.times, ds.amounts, ds.features,
                              list(ds.feature_names), None)
    with pytest.raises(ValueError, match="labelled"):
        train_supervised(bare, "logistic")
