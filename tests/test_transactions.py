import numpy as np
import pytest

from fraud_detector.transactions import (
    load_transactions,
    make_synthetic_transactions,
    save_transactions,
)


def _write_csv(tmp_path, text):
    path = tmp_path / "tx.csv"
    path.write_text(text)
    return path


def test_load_valid_csv_with_class_alias(tmp_path):
    path = _write_csv(
        tmp_path,
        "time,amount,V1,class\n0,10.0,0.5,0\n3600,20.0,-1.0,1\n",
    )
    ds = load_transactions(path)
    assert len(ds) == 2
    assert ds.feature_names == ["V1"]
    assert ds.labels.tolist() == [0, 1]
    assert ds.fraud_rate == pytest.approx(0.5)


def test_load_rejects_missing_amount_and_bad_label(tmp_path):
    with pytest.raises(ValueError, match="missing required"):
        load_transactions(_write_csv(tmp_path, "time,V1\n0,1.0\n"))
    with pytest.raises(ValueError, match="must be 0 or 1"):
        load_transactions(_write_csv(tmp_path, "time,amount,is_fraud\n0,5.0,2\n"))
    with pytest.raises(ValueError, match="must be >= 0"):
        load_transactions(_write_csv(tmp_path, "time,amount\n0,-5.0\n"))


def test_synthetic_is_deterministic_and_labelled():
    a = make_synthetic_transactions(n=500, fraud_ratio=0.04, seed=7)
    b = make_synthetic_transactions(n=500, fraud_ratio=0.04, seed=7)
    assert a.labels.tolist() == b.labels.tolist()
    assert np.allclose(a.amounts, b.amounts)
    assert a.fraud_rate == pytest.approx(0.04)
    # Fraud amounts skew larger than legitimate ones on average.
    assert a.amounts[a.labels == 1].mean() > a.amounts[a.labels == 0].mean()


def test_save_load_round_trip(tmp_path):
    ds = make_synthetic_transactions(n=50, fraud_ratio=0.1, seed=3)
    path = tmp_path / "roundtrip.csv"
    save_transactions(ds, path)
    back = load_transactions(path)
    assert len(back) == 50
    assert back.labels.tolist() == ds.labels.tolist()
    assert np.allclose(back.amounts, ds.amounts)
