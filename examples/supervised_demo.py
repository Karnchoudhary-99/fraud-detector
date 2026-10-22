"""Demo: train both supervised classifiers and compare them.

Run from the repo root after ``pip install -e .``:

    python examples/supervised_demo.py

Trains logistic regression and gradient boosting on a stratified
train split of the bundled synthetic sample, then prints each model's
holdout report and its best-F1 threshold.  Labels are used here for
training and reporting — unlike the unsupervised demo, which never
fits on labels.
"""

from pathlib import Path

from fraud_detector.classifiers import CLASSIFIER_NAMES
from fraud_detector.cli import _train_and_evaluate
from fraud_detector.reporting import format_report
from fraud_detector.transactions import load_transactions

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    dataset = load_transactions(ROOT / "data" / "sample_transactions.csv")
    for name in CLASSIFIER_NAMES:
        _, report, threshold = _train_and_evaluate(dataset, name, 0.25, 42)
        print(format_report(name, report, threshold))
        print()


if __name__ == "__main__":
    main()
