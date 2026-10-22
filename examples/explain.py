"""Demo: explain why the top-flagged transactions were flagged.

Run from the repo root after installing the package (or with
``PYTHONPATH=src``):

    python examples/explain.py

Prints the three highest-scored rows under the z-score detector with
their three most deviant features each.  On the bundled synthetic
sample the fraud rows are amount-driven by construction, so amount_z
dominates — a property of the generator, not a claim about real fraud.
"""

from pathlib import Path

import numpy as np

from fraud_detector.anomalies import ZScoreDetector
from fraud_detector.explanations import explain_flags, format_explanation
from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import load_transactions

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    dataset = load_transactions(ROOT / "data" / "sample_transactions.csv")
    engineer = FeatureEngineer()
    X = engineer.fit_transform(dataset)
    detector = ZScoreDetector(contamination=0.02).fit(X)
    scores = detector.score(X)
    for explanation in explain_flags(X, scores, engineer.output_names, top_n_rows=3):
        print(format_explanation(explanation))
    top = np.argsort(scores)[::-1][:3]
    if dataset.labels is not None:
        print(f"top-3 labelled fraud: {int(dataset.labels[top].sum())}/3")


if __name__ == "__main__":
    main()
