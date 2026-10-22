# Credit Card Fraud Detector

Detect fraudulent credit card transactions from tabular transaction
features: unsupervised anomaly scoring for the cold-start case, plus
supervised classifiers once labels exist — with precision/recall
evaluation that respects the extreme class imbalance of fraud data.

## Roadmap

- [x] Project scaffold (packaging, README, CI)
- [x] Transaction dataset loading + feature engineering
- [x] Anomaly detection (isolation forest, z-score baselines)
- [x] Supervised classifiers (logistic regression, gradient boosting)
- [x] Imbalance-aware evaluation (precision/recall/F1, PR-AUC)
- [ ] Self-review checklist, then public

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

## Supervised classifiers

Once labels exist, train a classifier on a stratified split (the
feature engineer is fitted on the training split only, so no scaler
leakage into the holdout), evaluate it with fraud-first metrics, and
optionally save it for later scoring:

```bash
fraud-detector train --csv data/sample_transactions.csv --model logistic --out models/logistic.pkl
fraud-detector evaluate --model-file models/logistic.pkl --csv data/sample_transactions.csv
fraud-detector compare --csv data/sample_transactions.csv
python examples/supervised_demo.py
```

Evaluation reports precision/recall/F1 for the fraud class, the
confusion matrix, ROC-AUC, and PR-AUC — the headline ranking metric,
since ROC-AUC is flattered by the huge number of easy true negatives
at 2% fraud. `best_f1_threshold` sweeps the decision threshold on the
holdout and reports the F1-optimal point instead of assuming 0.5.

Baseline on the bundled sample (250-row holdout, 5 fraud rows — small,
so read the rates, not just the headline):

| model | precision | recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| logistic | 0.833 | 1.000 | 0.909 | 0.967 |
| gradient-boosting | 1.000 | 0.400 | 0.571 | 0.794 |

Threshold tuning lifts gradient boosting's F1 to 0.667 (threshold
0.01) — a reminder that at this imbalance the threshold matters as
much as the model.

## License

MIT
