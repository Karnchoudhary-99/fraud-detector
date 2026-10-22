import re

from fraud_detector.cli import main

DATA = "data/sample_transactions.csv"


def test_score_explain_prints_contributors(capsys):
    assert main(["score", "--csv", DATA, "--model", "zscore",
                 "--top", "2", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "why flagged:" in out
    lines = [line for line in out.splitlines() if line.startswith("  row ")]
    assert len(lines) == 2
    assert re.search(r"row \d+ \(score \d+\.\d+\): .+ SD (above|below) typical",
                     lines[0])


def test_score_without_explain_prints_no_contributors(capsys):
    assert main(["score", "--csv", DATA, "--model", "zscore", "--top", "2"]) == 0
    assert "why flagged:" not in capsys.readouterr().out
