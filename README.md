# Credit Card Fraud Detector

Detect fraudulent credit card transactions from tabular transaction
features: unsupervised anomaly scoring for the cold-start case, plus
supervised classifiers once labels exist — with precision/recall
evaluation that respects the extreme class imbalance of fraud data.

## Roadmap

- [x] Project scaffold (packaging, README, CI)
- [x] Transaction dataset loading + feature engineering
- [x] Anomaly detection (isolation forest, z-score baselines)
- [ ] Supervised classifiers (logistic regression, gradient boosting)
- [ ] Imbalance-aware evaluation (precision/recall/F1, PR-AUC)
- [ ] CLI + demo, README, then public

## Usage

```bash
pip install -e .
python -m pytest -q
```

## Dataset

CSV layout: `time, amount, V1..Vn, is_fraud` (`class` is accepted as
an alias for `is_fraud`; labels are optional for scoring). Rows are
validated on load: finite values, non-negative amounts, 0/1 labels.
`data/sample_transactions.csv` is a 1000-row synthetic set (20 fraud)
generated with `fraud-detector generate --n 1000 --out data/sample_transactions.csv`
— no real cardholder data is shipped.

Feature engineering (`FeatureEngineer`, fitted once on training data)
adds `log_amount`, cyclical hour-of-day (`hour_sin`/`hour_cos`),
`amount_z`, and standardises every `V*` column.

## Anomaly detection (unsupervised, cold-start)

```bash
fraud-detector score --csv data/sample_transactions.csv --model isolation --top 10
fraud-detector score --csv data/sample_transactions.csv --model zscore --top 10
python examples/demo.py
```

Both detectors flag the top `contamination` fraction (default 2%) of
scores. On the bundled sample, isolation forest puts 13/20 true fraud
rows in its top 20 and z-score puts 11/20 — honest baselines on
deliberately overlapping synthetic data, not a clean separation.

## License

MIT
