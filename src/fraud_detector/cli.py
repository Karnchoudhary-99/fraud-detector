"""Placeholder CLI; the detection pipeline lands in the next session."""

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fraud-detector")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        from fraud_detector import __version__

        print(__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
