"""Demo: score the bundled synthetic sample with both detectors.

Run from the repo root after ``pip install -e .``:

    python examples/demo.py

Prints how many of the top-20 scored rows are truly labelled fraud for
each detector.  Labels are used only for this reporting, never for
fitting — both detectors are unsupervised.
"""

from pathlib import Path

import numpy as np

from fraud_detector.anomalies import IsolationForestDetector, ZScoreDetector
from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import load_transactions

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    dataset = load_transactions(ROOT / "data" / "sample_transactions.csv")
    X = FeatureEngineer().fit_transform(dataset)
    for name, detector in [
        ("z-score", ZScoreDetector(contamination=0.02)),
        ("isolation forest", IsolationForestDetector(contamination=0.02)),
    ]:
        detector.fit(X)
        scores = detector.score(X)
        top = np.argsort(scores)[::-1][:20]
        hits = int(dataset.labels[top].sum()) if dataset.labels is not None else 0
        print(f"{name}: flagged={int(detector.predict(X).sum())}, "
              f"top-20 labelled fraud={hits}/20")


if __name__ == "__main__":
    main()
