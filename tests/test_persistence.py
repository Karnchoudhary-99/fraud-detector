import pickle

import pytest

from fraud_detector.persistence import load_model, save_model
from fraud_detector.splitting import stratified_split
from fraud_detector.training import train_supervised
from fraud_detector.transactions import make_synthetic_transactions


def test_save_load_round_trip_preserves_predictions(tmp_path):
    ds = make_synthetic_transactions(n=300, fraud_ratio=0.1, seed=6)
    train, test = stratified_split(ds, test_size=0.25, seed=3)
    model = train_supervised(train, "logistic")
    path = save_model(model, tmp_path / "models" / "logistic.pkl")
    assert path.exists()
    reloaded = load_model(path)
    assert reloaded.name == "logistic"
    assert (reloaded.predict(test) == model.predict(test)).all()


def test_load_rejects_wrong_version_and_non_model(tmp_path):
    bad_version = tmp_path / "bad.pkl"
    bad_version.write_bytes(pickle.dumps({"format_version": 99}))
    with pytest.raises(ValueError, match="version mismatch"):
        load_model(bad_version)
    not_model = tmp_path / "notmodel.pkl"
    not_model.write_bytes(pickle.dumps({"format_version": 1, "model": "nope"}))
    with pytest.raises(ValueError, match="not a TrainedModel"):
        load_model(not_model)
