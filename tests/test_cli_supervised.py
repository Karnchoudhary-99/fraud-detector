from fraud_detector.cli import main
from fraud_detector.transactions import make_synthetic_transactions, save_transactions


def _sample_csv(tmp_path, n=400):
    path = tmp_path / "sample.csv"
    save_transactions(make_synthetic_transactions(n=n, fraud_ratio=0.1, seed=8), path)
    return path


def test_cli_evaluate_scores_saved_model_on_labelled_csv(tmp_path, capsys):
    csv = _sample_csv(tmp_path)
    out = tmp_path / "model.pkl"
    assert main(["train", "--csv", str(csv), "--model", "logistic",
                 "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--model-file", str(out), "--csv", str(csv)]) == 0
    captured = capsys.readouterr().out
    assert "model=logistic rows=400" in captured
    assert "confusion:" in captured


def test_cli_compare_lists_both_classifiers_sorted(tmp_path, capsys):
    csv = _sample_csv(tmp_path)
    assert main(["compare", "--csv", str(csv)]) == 0
    captured = capsys.readouterr().out
    assert "logistic" in captured and "gradient-boosting" in captured
    assert "sorted by PR-AUC" in captured


def test_cli_train_reports_holdout_and_saves_model(tmp_path, capsys):
    csv = _sample_csv(tmp_path)
    out = tmp_path / "model.pkl"
    assert main(["train", "--csv", str(csv), "--model", "logistic",
                 "--out", str(out)]) == 0
    captured = capsys.readouterr().out
    assert "model=logistic" in captured and "best_f1_threshold=" in captured
    assert out.exists()

    from fraud_detector.persistence import load_model
    from fraud_detector.transactions import load_transactions

    model = load_model(out)
    assert len(model.predict(load_transactions(csv))) == 400
