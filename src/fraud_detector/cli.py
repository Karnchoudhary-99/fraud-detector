"""fraud-detector CLI: generate synthetic data, score transactions.

Usage:
    fraud-detector generate --n 2000 --fraud-ratio 0.02 --out data/sample.csv
    fraud-detector score --csv data/sample.csv --model isolation --top 10
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fraud_detector import __version__
from fraud_detector.anomalies import IsolationForestDetector, ZScoreDetector
from fraud_detector.classifiers import CLASSIFIER_NAMES
from fraud_detector.evaluation import evaluate_predictions
from fraud_detector.features import FeatureEngineer
from fraud_detector.persistence import load_model, save_model
from fraud_detector.reporting import format_report
from fraud_detector.splitting import stratified_split
from fraud_detector.thresholds import best_f1_threshold
from fraud_detector.training import train_supervised
from fraud_detector.transactions import (
    TransactionDataset,
    load_transactions,
    make_synthetic_transactions,
    save_transactions,
)

DETECTORS = {"zscore": ZScoreDetector, "isolation": IsolationForestDetector}


def _cmd_generate(args: argparse.Namespace) -> int:
    dataset = make_synthetic_transactions(
        n=args.n, fraud_ratio=args.fraud_ratio, seed=args.seed
    )
    save_transactions(dataset, args.out)
    print(f"wrote {len(dataset)} rows ({int(dataset.labels.sum())} fraud) to {args.out}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    dataset = load_transactions(args.csv)
    engineer = FeatureEngineer()
    X = engineer.fit_transform(dataset)
    detector = DETECTORS[args.model](contamination=args.contamination)
    detector.fit(X)
    scores = detector.score(X)
    preds = detector.predict(X)
    order = np.argsort(scores)[::-1]
    print(f"rows={len(dataset)} flagged={int(preds.sum())} "
          f"model={args.model} contamination={args.contamination}")
    if dataset.labels is not None and args.top:
        flagged = preds[order[: args.top]]
        print(f"top-{args.top} flagged share with true fraud labels: "
              f"{int((flagged & dataset.labels[order[: args.top]]).sum())}/{args.top} "
              f"are labelled fraud (labels only used for reporting, never for fitting)")
    print("rank  row  score  amount  time")
    for rank, idx in enumerate(order[: args.top], start=1):
        print(f"{rank:>4} {idx:>4} {scores[idx]:.3f}  "
              f"{dataset.amounts[idx]:>8.2f}  {dataset.times[idx]:>8.0f}")
    return 0


def _train_and_evaluate(dataset: TransactionDataset, model_name: str,
                        test_size: float, seed: int):
    """Split, train, and score one classifier on the holdout split."""
    if dataset.labels is None:
        raise ValueError("supervised commands require a labelled CSV")
    train, test = stratified_split(dataset, test_size=test_size, seed=seed)
    model = train_supervised(train, model_name)
    report = evaluate_predictions(
        test.labels, model.predict(test), model.predict_proba(test)
    )
    threshold = best_f1_threshold(test.labels, model.predict_proba(test))
    return model, report, threshold


def _cmd_train(args: argparse.Namespace) -> int:
    dataset = load_transactions(args.csv)
    model, report, threshold = _train_and_evaluate(
        dataset, args.model, args.test_size, args.seed
    )
    print(format_report(args.model, report, threshold))
    if args.out:
        save_model(model, args.out)
        print(f"saved model to {args.out}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    model = load_model(args.model_file)
    dataset = load_transactions(args.csv)
    if dataset.labels is None:
        raise ValueError("evaluate requires a labelled CSV")
    report = evaluate_predictions(
        dataset.labels, model.predict(dataset), model.predict_proba(dataset)
    )
    threshold = best_f1_threshold(dataset.labels, model.predict_proba(dataset))
    print(format_report(model.name, report, threshold))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    dataset = load_transactions(args.csv)
    rows = []
    for name in sorted(CLASSIFIER_NAMES):
        _, report, _ = _train_and_evaluate(
            dataset, name, args.test_size, args.seed
        )
        rows.append((name, report))
    rows.sort(key=lambda item: (item[1].pr_auc is None, -(item[1].pr_auc or 0.0)))
    print("model  precision  recall  f1  pr_auc  roc_auc  (sorted by PR-AUC)")
    for name, report in rows:
        print(f"{name:<18} {report.precision:.4f}  {report.recall:.4f}  "
              f"{report.f1:.4f}  "
              f"{report.pr_auc if report.pr_auc is not None else float('nan'):.4f}  "
              f"{report.roc_auc if report.roc_auc is not None else float('nan'):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-detector")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    gen = sub.add_parser("generate", help="write synthetic transactions to CSV")
    gen.add_argument("--n", type=int, default=2000)
    gen.add_argument("--fraud-ratio", type=float, default=0.02)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", required=True)
    gen.set_defaults(func=_cmd_generate)
    score = sub.add_parser("score", help="anomaly-score a transaction CSV")
    score.add_argument("--csv", required=True)
    score.add_argument("--model", choices=sorted(DETECTORS), default="isolation")
    score.add_argument("--contamination", type=float, default=0.02)
    score.add_argument("--top", type=int, default=10)
    score.set_defaults(func=_cmd_score)
    train = sub.add_parser("train", help="train a supervised classifier")
    train.add_argument("--csv", required=True)
    train.add_argument("--model", choices=sorted(CLASSIFIER_NAMES),
                       default="logistic")
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", default=None,
                       help="optional path to save the trained model")
    train.set_defaults(func=_cmd_train)
    evaluate = sub.add_parser("evaluate",
                              help="score a saved model on a labelled CSV")
    evaluate.add_argument("--model-file", required=True)
    evaluate.add_argument("--csv", required=True)
    evaluate.set_defaults(func=_cmd_evaluate)
    compare = sub.add_parser("compare",
                             help="compare all classifiers on one holdout split")
    compare.add_argument("--csv", required=True)
    compare.add_argument("--test-size", type=float, default=0.25)
    compare.add_argument("--seed", type=int, default=42)
    compare.set_defaults(func=_cmd_compare)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if not hasattr(args, "func"):
        build_parser().print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
