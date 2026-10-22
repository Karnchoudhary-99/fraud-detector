"""End-to-end supervised training: engineer fitted on train only.

The single most common fraud-model leak is fitting the feature scaler
on the whole dataset and then "evaluating" on rows it has already seen.
This pipeline makes that impossible by construction: the
``FeatureEngineer`` is fitted on the training split alone, and the
returned ``TrainedModel`` carries that engineer with it, so inference
always uses the training-time transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_detector.classifiers import SupervisedClassifier, make_classifier
from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import TransactionDataset


@dataclass
class TrainedModel:
    """A fitted engineer + classifier pair, ready to score new rows."""

    name: str
    engineer: FeatureEngineer
    classifier: SupervisedClassifier

    def predict(self, dataset: TransactionDataset) -> np.ndarray:
        return self.classifier.predict(self.engineer.transform(dataset))

    def predict_proba(self, dataset: TransactionDataset) -> np.ndarray:
        return self.classifier.predict_proba(self.engineer.transform(dataset))


def train_supervised(
    train: TransactionDataset, classifier_name: str
) -> TrainedModel:
    """Fit ``classifier_name`` on ``train``; labels are required."""
    if train.labels is None:
        raise ValueError("train_supervised requires labelled training data")
    engineer = FeatureEngineer()
    X = engineer.fit_transform(train)
    classifier = make_classifier(classifier_name).fit(X, train.labels)
    return TrainedModel(classifier_name, engineer, classifier)
