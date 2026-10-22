import numpy as np
import pytest

from fraud_detector.classifiers import CLASSIFIER_NAMES, make_classifier
from fraud_detector.features import FeatureEngineer
from fraud_detector.splitting import stratified_split
from fraud_detector.transactions import make_synthetic_transactions


def _split_xy():
    ds = make_synthetic_transactions(n=400, fraud_ratio=0.1, seed=5)
    train, test = stratified_split(ds, test_size=0.25, seed=2)
    eng = FeatureEngineer()
    X_train = eng.fit_transform(train)
    X_test = eng.transform(test)
    return X_train, train.labels, X_test, test.labels


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_each_classifier_beats_chance_and_returns_probabilities(name):
    X_train, y_train, X_test, y_test = _split_xy()
    clf = make_classifier(name).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    assert proba.shape == (len(y_test),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(preds.tolist()) <= {0, 1}
    # Fraud rows must score higher on average than legitimate rows.
    assert proba[y_test == 1].mean() > proba[y_test == 0].mean()


def test_unknown_classifier_and_single_class_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        make_classifier("random-forest")
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="both classes"):
        make_classifier("logistic").fit(X, np.zeros(4, dtype=int))
