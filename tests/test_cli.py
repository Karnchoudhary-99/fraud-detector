from fraud_detector.cli import main
from fraud_detector.transactions import load_transactions


def test_cli_generate_and_score(tmp_path, capsys):
    out = tmp_path / "gen.csv"
    assert main(["generate", "--n", "200", "--fraud-ratio", "0.05", "--out", str(out)]) == 0
    ds = load_transactions(out)
    assert len(ds) == 200 and ds.fraud_rate == 0.05

    assert main(["score", "--csv", str(out), "--model", "zscore", "--top", "3"]) == 0
    captured = capsys.readouterr().out
    assert "flagged=" in captured and "rank" in captured


def test_cli_no_command_returns_usage_code(capsys):
    assert main([]) == 2
