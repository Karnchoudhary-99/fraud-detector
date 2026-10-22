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
from fraud_detector.features import FeatureEngineer
from fraud_detector.transactions import (
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
